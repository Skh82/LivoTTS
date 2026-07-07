import os, json, glob, argparse
from pathlib import Path
from functools import reduce
from torch.utils.data import DataLoader
import math
import torch
import sentencepiece as spm
from tqdm import tqdm
import math
from safetensors.torch import load_file
import scripts.training_utils as dh
from scripts.training_utils import NavaCodesDataset, collate_codes
from src.Livmodel.Livo import LivoModel

from safetensors.torch import save_file, safe_open
REPO_ROOT = Path(__file__).resolve().parents[1]

def ist_flat_params(parameters):
    flat = []
    for layer in parameters:
        if isinstance(layer, (tuple, list)) and len(layer) == 2:
            flat.extend([layer[0], layer[1]])
        else:
            flat.append(layer)
    return flat

def resolve_cfg_path(paths_cfg, key, default_rel=None):
    raw = (paths_cfg or {}).get(key, default_rel)
    if raw is None:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / raw)
    return p.resolve()

def _strip_prefix(sd, prefix="module."):
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}

def load_model_weights(model: torch.nn.Module, model_safetensors_path: str):
    p = Path(model_safetensors_path).expanduser().resolve()
    sd = load_file(str(p), device="cpu")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded: {p}")
    if missing:
        print(f"[ckpt] missing keys: {len(missing)}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)}")

def speaker_state_dict(params):
    out = {}
    for i, layer in enumerate(params):
        if isinstance(layer, (tuple, list)) and len(layer) == 2:
            k, v = layer
            out[f"layer{i}_k"] = k.detach().cpu()
            out[f"layer{i}_v"] = v.detach().cpu()
        else:
            out[f"layer{i}"] = layer.detach().cpu()
    return out

def parse_speaker_state(path: str, device="cuda"):
    params_list = []
    with safe_open(path, framework="pt", device=device) as f:
        keys = [k for k in f.keys() if k.endswith("_k")]
        keys.sort(key=lambda x: int("".join([c for c in x if c.isdigit()])))
        for kname in keys:
            v = f.get_tensor(kname[:-2] + "_v")
            k = f.get_tensor(kname)
            params_list.append((k, v))
    return params_list

@torch.no_grad()
def clone_ist_params(params, to_cpu: bool = True):
    out = []
    for layer in params:
        if isinstance(layer, (tuple, list)) and len(layer) == 2:
            k, v = layer
            k = k.detach().clone()
            v = v.detach().clone()
            if to_cpu:
                k = k.cpu(); v = v.cpu()
            out.append((k, v))
        else:
            t = layer.detach().clone()
            if to_cpu:
                t = t.cpu()
            out.append(t)
    return out

@torch.no_grad()
def move_ist_params(params, device):
    out = []
    for layer in params:
        if isinstance(layer, (tuple, list)) and len(layer) == 2:
            out.append((layer[0].to(device), layer[1].to(device)))
        else:
            out.append(layer.to(device))
    return out

@torch.no_grad()
def mean_loss_over_dataset(model, dl_eval, ist_params, device, use_amp, amp_dtype, scale: float):
    model.eval()
    ist_params = move_ist_params(ist_params, device)

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dl_eval, desc="Eval", leave=False, dynamic_ncols=True):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        B = batch["text_token"].shape[0]

        init_state = model.attentive_rnn.get_state_from_params(ist_params, B, scale=scale)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss, *_ = model(
                batch["text_token"],
                batch["audio_token"],
                batch["spk_ids"],
                batch["encoder_mask"],
                batch["crossatt_mask"],
                logits_mask=batch.get("logits_mask"),
                init_state=init_state,
            )

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to assets/cfg/config.json")
    ap.add_argument("--ckpt", required=True, help="Checkpoint file/dir to load model weights from")
    ap.add_argument("--shards_dir", default=None, help="Override paths.shards_dir")
    ap.add_argument("--spm_model",  default=None, help="Override paths.spm_model")

    ap.add_argument("--epochs", type=int, default=1, help="Number of epochs over the dataset")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_acc", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=1, help="LoRA rank used by get_init_state_tuning_params")
    ap.add_argument("--scale", type=float, default=0.02)

    ap.add_argument("--out", required=True, help="Where to save IST params (.safetensors)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    CONFIG_JSON = Path(args.config).expanduser().resolve()
    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        CFG = json.load(f)

    P = CFG.get("paths", {})
    shards_dir = Path(args.shards_dir).expanduser().resolve() if args.shards_dir else resolve_cfg_path(P, "shards_dir")
    spm_model  = Path(args.spm_model).expanduser().resolve()  if args.spm_model  else resolve_cfg_path(P, "spm_model")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = (not args.no_amp)
    amp_dtype = torch.bfloat16
    
    dh._SP = spm.SentencePieceProcessor(model_file=str(spm_model))
    dh._SP.set_encode_extra_options("bos:eos")

    model = LivoModel.from_json(str(CONFIG_JSON)).to(device)
    load_model_weights(model, args.ckpt)

    model.attentive_rnn.to_mode("fused_recurrent")
    model.train()
    parameters = model.attentive_rnn.get_init_state_tuning_params(lora=args.rank, device=str(device))
    initial_parameters = clone_ist_params(parameters, to_cpu=True)
    
    opt_params = reduce(tuple.__add__, parameters)
    optimizer = torch.optim.Adam(opt_params, lr=args.lr)

    ds = NavaCodesDataset(root=str(shards_dir))

    def collate_ist(batch):
        out = collate_codes(batch)
        B = out["text_token"].shape[0]
        out["spk_ids"] = torch.zeros(B, dtype=torch.long)
        return out

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_ist,
    )

    steps_per_epoch = len(dl)
    total_micro_steps = steps_per_epoch * args.epochs
    total_updates = math.ceil(total_micro_steps / args.grad_acc)

    print(f"[info] batches/epoch={steps_per_epoch}, epochs={args.epochs}")
    print(f"[info] micro_steps={total_micro_steps}, grad_acc={args.grad_acc}")
    print(f"[info] optimizer_updates≈{total_updates}")

    optimizer.zero_grad(set_to_none=True)
    global_micro = 0
    global_update = 0

    pbar = tqdm(total=total_micro_steps, desc="IST tuning", dynamic_ncols=True)

    for epoch in range(args.epochs):
        for step, batch in enumerate(dl):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            init_state = model.attentive_rnn.get_state_from_params(parameters, batch["text_token"].shape[0], scale=0.02)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss, *_ = model(
                    batch["text_token"],
                    batch["audio_token"],
                    batch["spk_ids"],
                    batch["encoder_mask"],
                    batch["crossatt_mask"],
                    logits_mask=batch.get("logits_mask"),
                    init_state=init_state
                )

            (loss / args.grad_acc).backward()
            
            global_micro += 1
            pbar.update(1)
            pbar.set_postfix(epoch=epoch, loss=float(loss.item()), upd=global_update)

            if (global_micro % args.grad_acc == 0) or (step == steps_per_epoch - 1):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

    pbar.close()
    
    dl_eval = DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_ist,
    )

    init_mean = mean_loss_over_dataset(model, dl_eval, initial_parameters, device, use_amp, amp_dtype, scale=args.scale)
    final_mean = mean_loss_over_dataset(model, dl_eval, parameters,         device, use_amp, amp_dtype, scale=args.scale)

    print(f"[eval] mean loss with INITIAL IST params: {init_mean:.6f}")
    print(f"[eval] mean loss with FINAL   IST params: {final_mean:.6f}")
    print(f"[eval] delta (final - init): {final_mean - init_mean:+.6f}")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(speaker_state_dict(parameters), str(out_path))
    print(f"[save] IST params -> {out_path}")

if __name__ == "__main__":
    main()
