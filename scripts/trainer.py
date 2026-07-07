import os, glob, torch, secrets, shutil, json
import scripts.training_utils as dh
import sentencepiece as spm
import argparse

from collections import Counter
from pathlib import Path
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed, DataLoaderConfiguration
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from scripts.training_utils import NavaCodesDataset, collate_codes, LengthBucketDynamicBatchSampler
from src.Livmodel.Livo import LivoModel
from pathlib import Path
from safetensors.torch import load_file as safetensors_load_file

REPO_ROOT = Path(__file__).resolve().parents[1]

def R(root, *parts, mkdir=False, as_str=False):
    root = Path(root).expanduser().resolve()
    parts = [str(p).lstrip("/\\") for p in parts]
    p = root.joinpath(*parts)
    if mkdir:
        (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return str(p) if as_str else p

def resolve_cfg_path(paths_cfg, key, default_rel, mkdir=False):
    raw = (paths_cfg or {}).get(key, default_rel)
    p = Path(raw).expanduser()

    if not p.is_absolute():
        p = (REPO_ROOT / raw)

    p = p.resolve()

    if mkdir:
        target = p if p.suffix == "" else p.parent
        target.mkdir(parents=True, exist_ok=True)

    return p

def load_safetensors_non_strict(model: torch.nn.Module, ckpt_path: str, accelerator=None, max_list=200):
    pr = (accelerator.print if accelerator is not None else print)

    ckpt_path = str(Path(ckpt_path).expanduser().resolve())
    if accelerator is None or accelerator.is_main_process:
        pr(f"[TL] Loading pretrained safetensors: {ckpt_path}")

    sd_ckpt = safetensors_load_file(ckpt_path, device="cpu")
    sd_model = model.state_dict()

    model_keys = set(sd_model.keys())
    ckpt_keys  = list(sd_ckpt.keys())

    def try_map(keys, fn):
        mapped = [fn(k) for k in keys]
        hit = sum(1 for k in mapped if k in model_keys)
        return hit, mapped

    candidates = [
        ("identity", lambda k: k),
        ("strip_module", lambda k: k[7:] if k.startswith("module.") else k),
        ("strip_model",  lambda k: k[6:] if k.startswith("model.") else k),
        ("strip_net",    lambda k: k[4:] if k.startswith("net.") else k),
    ]

    best_name, best_keys, best_hit = None, None, -1
    for name, fn in candidates:
        hit, mapped = try_map(ckpt_keys, fn)
        if hit > best_hit:
            best_hit = hit
            best_name = name
            best_keys = mapped

    if accelerator is None or accelerator.is_main_process:
        pr(f"[TL] Key mapping used: {best_name} (matched {best_hit}/{len(ckpt_keys)} ckpt keys)")

    sd_ckpt_mapped = {}
    for old_k, new_k in zip(ckpt_keys, best_keys):
        if new_k not in sd_ckpt_mapped:
            sd_ckpt_mapped[new_k] = sd_ckpt[old_k]

    unexpected = []
    mismatched = []
    loadable = {}

    for k, v in sd_ckpt_mapped.items():
        if k not in sd_model:
            unexpected.append(k)
            continue
        if tuple(v.shape) != tuple(sd_model[k].shape):
            mismatched.append((k, tuple(v.shape), tuple(sd_model[k].shape)))
            continue
        tgt_dtype = sd_model[k].dtype
        if v.dtype != tgt_dtype:
            v = v.to(dtype=tgt_dtype)
        loadable[k] = v

    incompatible = model.load_state_dict(loadable, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected2 = list(getattr(incompatible, "unexpected_keys", []))

    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())

    loaded_param_elems = 0
    loaded_buf_elems = 0
    for k, v in loadable.items():
        if k in named_params:
            loaded_param_elems += v.numel()
        elif k in named_buffers:
            loaded_buf_elems += v.numel()

    total_param_elems = sum(p.numel() for p in model.parameters())
    pct = (100.0 * loaded_param_elems / max(1, total_param_elems))

    if accelerator is None or accelerator.is_main_process:
        pr(f"[TL] Loaded param elems: {loaded_param_elems:,} / {total_param_elems:,} ({pct:.2f}%)")
        pr(f"[TL] Loaded buffer elems: {loaded_buf_elems:,}")
        pr(f"[TL] Missing keys (not loaded): {len(missing)}")
        pr(f"[TL] Unexpected keys (ckpt-only): {len(unexpected)}")
        pr(f"[TL] Mismatched-shape keys: {len(mismatched)}")

        def _print_list(title, items):
            pr(f"[TL] {title}:")
            for x in items[:max_list]:
                pr(f"  - {x}")
            if len(items) > max_list:
                pr(f"  ... {len(items)-max_list} more")

        if missing:
            _print_list("Missing (model-only)", missing)
        if unexpected:
            _print_list("Unexpected (ckpt-only)", unexpected)
        if unexpected2:
            _print_list("Unexpected from load_state_dict()", unexpected2)
        if mismatched:
            pr("[TL] Mismatched shapes:")
            for (k, s_ckpt, s_model) in mismatched[:max_list]:
                pr(f"  - {k}: ckpt{s_ckpt} vs model{s_model}")
            if len(mismatched) > max_list:
                pr(f"  ... {len(mismatched)-max_list} more")

    return {
        "loaded_keys": list(loadable.keys()),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "loaded_param_elems": loaded_param_elems,
        "total_param_elems": total_param_elems,
    }


def training_loop(mixed_precision, config_path):

    CONFIG_JSON = Path(config_path).expanduser().resolve()

    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        CFG = json.load(f)

    TR = CFG["trainer"]
    keep_last = TR.get("checkpointing", {}).get("keep_last", 4)
    P  = CFG.get("paths", {})

    SPM_MODEL  = resolve_cfg_path(P, "spm_model",  None)
    SHARDS_DIR = resolve_cfg_path(P, "shards_dir", None)
    CKPT_DIR   = resolve_cfg_path(P, "ckpt_dir",   None, mkdir=True)

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=600))
    dataloader_config = DataLoaderConfiguration(
        use_seedable_sampler=True,
        even_batches=True,
    )

    cfg = ProjectConfiguration(
        project_dir=str(CKPT_DIR),
        automatic_checkpoint_naming=True
    )

    grad_accum_steps = TR.get("gradient_accumulation_steps", 1)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[process_group_kwargs],
        project_config=cfg,
        dataloader_config=dataloader_config,
        gradient_accumulation_steps=grad_accum_steps,
    )

    set_seed(TR["seed"])
    LivoTTS = LivoModel.from_json(str(CONFIG_JSON))
    #PRETRAINED_SFT = "/media/eri-4090/HDD/Saeed/LivoTTS/pretrained/lina/1280_16/model_stripped.safetensors"
    #load_safetensors_non_strict(LivoTTS, PRETRAINED_SFT, accelerator=accelerator)


    trunk_lr = TR["optimizer"]["trunk"]["lr"]

    trunk_params = [p for p in LivoTTS.parameters() if p.requires_grad]
    param_groups = [
        {
            "params": trunk_params,
            "lr": trunk_lr,
            "weight_decay": TR["optimizer"]["trunk"]["weight_decay"],
        },
    ]

    optimizer = torch.optim.AdamW(param_groups)

    total_params = sum(p.numel() for p in LivoTTS.parameters())
    trainable_params = sum(p.numel() for p in LivoTTS.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    accelerator.print(f"AR Total params:      {total_params:,}")
    accelerator.print(f"AR Trainable params:  {trainable_params:,}")
    accelerator.print(f"AR Frozen params:     {frozen_params:,}")
    accelerator.print(f"   → {100 * trainable_params/total_params:.2f}% trainable")

    warmup_steps = TR["scheduler"]["warmup_steps"]
    total_steps  = TR["scheduler"]["total_steps"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    ds_train = NavaCodesDataset(root=str(SHARDS_DIR), split="train")
    ds_eval = NavaCodesDataset(root=str(SHARDS_DIR), split="test")
    
    def _rec_spk(rec):
        return (rec.get("spk") or rec.get("speaker") or "<unk>").strip()

    def _rec_shard(rec):
        return rec.get("shard") or Path(rec.get("_tokens_path", "")).name or "<no_shard>"

    def summarize_ds(ds, name, shards_root: Path, accelerator, topn=10):
        n = len(ds)
        shards = [_rec_shard(r) for r in ds.items]
        shard_counts = Counter(shards)
        unique_shards = sorted(shard_counts.keys())

        secs = [float(r.get("seconds", 0.0) or 0.0) for r in ds.items]
        total_secs = sum(secs)
        total_hours = total_secs / 3600.0

        spks = [_rec_spk(r) for r in ds.items]
        n_spks = len(set(spks))

        accelerator.print(f"\n=== {name} ===")
        accelerator.print(f"rows: {n:,}")
        accelerator.print(f"unique speakers: {n_spks:,}")
        accelerator.print(f"unique shards: {len(unique_shards):,}")
        accelerator.print(f"total audio: {total_secs:,.2f} sec  ({total_hours:,.2f} hours)")

        missing = [s for s in unique_shards if not (shards_root / s).exists()]
        if missing:
            accelerator.print(f"WARNING: {len(missing)} shard token files missing under {shards_root}:")
            for s in missing[:10]:
                accelerator.print(f"  - {s}")
            if len(missing) > 10:
                accelerator.print(f"  ... {len(missing)-10} more")

        accelerator.print(f"top {topn} shards by rows:")
        for shard, cnt in shard_counts.most_common(topn):
            accelerator.print(f"  {shard}: {cnt:,}")
        if len(unique_shards) > topn:
            accelerator.print(f"  ... {len(unique_shards)-topn} more shards")

    meta_files = sorted(set(
        glob.glob(os.path.join(str(SHARDS_DIR), "shard-*.meta.jsonl")) +
        glob.glob(os.path.join(str(SHARDS_DIR), "shard-*.meta.jsonl.zst"))
    ))
    accelerator.print(f"\nMeta files found in {SHARDS_DIR}: {len(meta_files)}")
    for p in meta_files[:10]:
        accelerator.print(f"  - {os.path.basename(p)}")
    if len(meta_files) > 10:
        accelerator.print(f"  ... {len(meta_files)-10} more")

    summarize_ds(ds_train, "TRAIN", Path(SHARDS_DIR), accelerator, topn=10)
    summarize_ds(ds_eval,  "TEST",  Path(SHARDS_DIR), accelerator, topn=10)

    def make_collate_with_spk_ids(spk2id_map):
        def _collate(batch):
            out = collate_codes(batch)
            out["spk_ids"] = torch.tensor(
                [spk2id_map.get(b["spk"] or "<unk>", -1) for b in batch],
                dtype=torch.long
            )
            return out
        return _collate

    all_spks_train = [(rec.get("spk") or "<unk>").strip() for rec in ds_train.items]
    unique_spks = sorted(set(all_spks_train))
    spk2id = {s: i for i, s in enumerate(unique_spks)}

    collate_with_spk_ids_train = make_collate_with_spk_ids(spk2id)
    collate_with_spk_ids_eval  = make_collate_with_spk_ids(spk2id)

    def _sp_worker_init(_):
        dh._SP = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))
        dh._SP.set_encode_extra_options("bos:eos")

    lengths_train = [int(it.get("n_frames") or it["tok_len"]) for it in ds_train.items]
    lengths_eval = [int(it.get("n_frames") or it["tok_len"]) for it in ds_eval.items]

    sampler_train = LengthBucketDynamicBatchSampler(
        lengths=lengths_train,
        max_frames_per_batch=TR["dataloader"]["max_frames_per_batch"],
        n_buckets=TR["dataloader"]["n_buckets"],
        shuffle=TR["dataloader"]["shuffle"],
        drop_last=TR["dataloader"]["drop_last"],
        interleave=TR["dataloader"]["interleave"],
        bucket_temp=TR["dataloader"]["bucket_temp"],
        longest_first=TR["dataloader"]["longest_first"],
    )

    sampler_eval = LengthBucketDynamicBatchSampler(
        lengths=lengths_eval,
        max_frames_per_batch=TR["dataloader"]["max_frames_per_batch"],
        n_buckets=TR["dataloader"]["n_buckets"],
        shuffle=False,
        drop_last=False,
        interleave=False,
        bucket_temp=TR["dataloader"]["bucket_temp"],
        longest_first=TR["dataloader"]["longest_first"],
    )


    dl_train = DataLoader(
        ds_train,
        batch_sampler=sampler_train,
        num_workers=TR["dataloader"]["num_workers"],
        pin_memory=TR["dataloader"]["pin_memory"],
        collate_fn=collate_with_spk_ids_train,
        persistent_workers=TR["dataloader"]["persistent_workers"],
        worker_init_fn=_sp_worker_init,
    )
    
    dl_eval = DataLoader(
        ds_eval,
        batch_sampler=sampler_eval,
        num_workers=TR["dataloader"]["num_workers"],
        pin_memory=TR["dataloader"]["pin_memory"],
        collate_fn=collate_with_spk_ids_eval,
        persistent_workers=TR["dataloader"]["persistent_workers"],
        worker_init_fn=_sp_worker_init,
    )

    LivoTTS, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        LivoTTS, optimizer, dl_train, dl_eval, scheduler
    )

    ckpt_pattern = os.path.join(str(CKPT_DIR), "checkpoints", "checkpoint_*")
    existing = glob.glob(ckpt_pattern)
    if existing:
        accelerator.load_state()
        paths = glob.glob(os.path.join(str(CKPT_DIR), "checkpoints", "checkpoint_*"))
        epochs = [int(os.path.basename(p).split("_")[-1]) for p in paths]
        last_epoch = max(epochs) + 1

        cfg.iteration = last_epoch
        accelerator.print(f"▶ Resuming Training from epoch {last_epoch}")
        #seed_offset = secrets.randbelow(2**31)
        #set_seed(seed_offset)
    else:
        #seed_offset = secrets.randbelow(2**31)
        #set_seed(seed_offset)
        last_epoch = 0

    max_steps = TR["scheduler"]["total_steps"]

    inner_scheduler = scheduler.scheduler
    current_step = inner_scheduler.last_epoch
    if current_step < 0:
        current_step = 0

    accelerator.print(f"▶ Global step {current_step}, LR={optimizer.param_groups[0]['lr']}")

    epoch = last_epoch

    best_eval_loss = float("inf")
    early_stop = False

    while current_step < max_steps:
        train_loader.set_epoch(epoch)
        eval_loader.set_epoch(epoch)
        sampler_train.set_epoch(epoch)
        sampler_eval.set_epoch(epoch)

        LivoTTS.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in progress:
            if current_step >= max_steps:
                break
            
            with accelerator.accumulate(LivoTTS):
                text_token    = batch["text_token"]
                audio_token   = batch["audio_token"]
                encoder_mask  = batch["encoder_mask"]
                crossatt_mask = batch["crossatt_mask"]
                y_mask        = batch["logits_mask"]
                spk_ids       = batch["spk_ids"]

                with accelerator.autocast():
                    logits, loss, att, masked_logits, masked_target = LivoTTS(
                        text_token, audio_token, spk_ids, encoder_mask, crossatt_mask,
                        logits_mask=y_mask
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(LivoTTS.parameters(), TR["optimizer"]["clip_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                inner_scheduler = scheduler.scheduler
                current_step = inner_scheduler.last_epoch

                progress.set_postfix(
                    loss=loss.item(),
                    lr=inner_scheduler.get_last_lr()[0],
                    step=current_step
                )

        accelerator.wait_for_everyone()
        
        LivoTTS.eval()

        total_loss = 0.0
        n = 0

        eval_bar = tqdm(eval_loader, desc=f"Eval {epoch}", leave=False)

        for ebatch in eval_bar:
            with torch.no_grad():
                with accelerator.autocast():
                    _, eloss, *_ = LivoTTS(
                        ebatch["text_token"],
                        ebatch["audio_token"],
                        ebatch["spk_ids"],
                        ebatch["encoder_mask"],
                        ebatch["crossatt_mask"],
                        logits_mask=ebatch["logits_mask"]
                    )

            total_loss += eloss.item()
            n += 1
            eval_bar.set_postfix(loss=eloss.item())

        total_loss_t = torch.tensor(total_loss, device=accelerator.device)
        n_t = torch.tensor(n, device=accelerator.device)

        total_loss_all = accelerator.gather(total_loss_t).sum().item()
        n_all = accelerator.gather(n_t).sum().item()

        eval_loss = total_loss_all / max(n_all, 1)

        LivoTTS.train()

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            if accelerator.is_main_process:
                accelerator.print(
                    f"▶ Eval loss epoch {epoch}: {eval_loss:.4f} (new best)"
                )
        else:
            if accelerator.is_main_process:
                accelerator.print(
                    f"▶ Eval loss epoch {epoch}: {eval_loss:.4f} (no improvement, best {best_eval_loss:.4f})"
                )
                accelerator.print("▶ Early stopping triggered.")
            early_stop = True

        cfg.iteration = epoch
        accelerator.save_state()

        if accelerator.is_main_process and not early_stop:
            checkpoint_root = os.path.join(str(CKPT_DIR), "checkpoints")
            paths = glob.glob(os.path.join(checkpoint_root, "checkpoint_*"))

            paths.sort(key=lambda x: int(x.split("_")[-1]))
            while len(paths) > keep_last:
                oldest = paths.pop(0)
                if os.path.exists(oldest):
                    try:
                        shutil.rmtree(oldest)
                        accelerator.print(f"▶ Deleted old checkpoint: {os.path.basename(oldest)}")
                    except Exception as e:
                        accelerator.print(f"Error deleting {oldest}: {str(e)}")

        accelerator.wait_for_everyone()
        
        if early_stop:
            break
        
        epoch += 1

    accelerator.end_training()
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixed_precision",
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode for accelerate.Accelerator",
    )
    parser.add_argument(
        "--config",
        default=str(R(REPO_ROOT, "assets", "cfg", "config.json")),
        help="Path to config.json",
    )
    
    args = parser.parse_args()
    training_loop(args.mixed_precision, args.config)
