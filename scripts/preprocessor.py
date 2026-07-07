import sys, json, torch, torchaudio, random, argparse
import numpy as np
import pandas as pd

import torch.nn.functional as F
from types import SimpleNamespace as NS
from src.Vevmodel.codec.repcodec_model import RepCodec
from safetensors.torch import load_model as st_load_model
from tqdm.auto import tqdm
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union
from accelerate import Accelerator
from functools import partial
from tqdm.auto import tqdm as _tqdm
import torch.distributed as dist
import hashlib
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

REPO_ROOT   = Path(__file__).resolve().parents[1]
ASSETS_DIR  = REPO_ROOT / "assets"
STATS_NPZ   = str(ASSETS_DIR / "other/hubert_large_l18_mean_std.npz")
REPCODEC_SFT = str(ASSETS_DIR / "other/tokenizer.safetensors")
TARGET_SHARD_BYTES = 1_000_000_000

def R(root: Union[str, Path], *parts: str, mkdir: bool = False) -> Path:
    root = Path(root).expanduser().resolve()
    parts = [str(p).lstrip("/\\") for p in parts]
    p = root.joinpath(*parts)
    if mkdir:
        (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return p

def _build_repcodec(codebook_size=8192, hidden_size=1024, codebook_dim=8, device="cpu"):
    cfg = NS(
        codebook_size=codebook_size,
        hidden_size=hidden_size,
        codebook_dim=codebook_dim,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        num_quantizers=1,
        downsample_scale=1,
    )
    return RepCodec(cfg=cfg).to(device).eval()

class NavaTokenizer:
    def __init__(self, repcodec_safetensors: str, hubert_stats_npz: str, device: str | torch.device = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        bundle = torchaudio.pipelines.HUBERT_LARGE
        self.hubert = bundle.get_model().to(self.device).eval()

        stats = np.load(hubert_stats_npz)
        self.H_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=self.device)
        self.H_std  = torch.tensor(stats["std"],  dtype=torch.float32, device=self.device)

        self.tokenizer = _build_repcodec(device=self.device)
        st_load_model(self.tokenizer, repcodec_safetensors)
        self.tokenizer.eval()

    @torch.no_grad()
    def _wav_to_16k_mono(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.to(self.device)

    @torch.no_grad()
    def _hubert18_norm(self, wav16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats_list, feat_lens = self.hubert.extract_features(wav16, num_layers=18)
        feats = feats_list[-1]
        feats = (feats - self.H_mean.to(feats)) / self.H_std.to(feats)
        return feats, feat_lens

    @torch.no_grad()
    def quantize_path(self, wav_path: str) -> torch.Tensor:
        wav16 = self._wav_to_16k_mono(wav_path)
        feats, _ = self._hubert18_norm(wav16)
        ids, _ = self.tokenizer.quantize(feats)
        return ids[0].to("cpu")                

IDX_DTYPE = np.dtype([("offset","<u8"), ("length","<u4")])
TOK_DTYPE = np.dtype("<u2")

class MetaWriter:
    def __init__(self, meta_base: Path):
        self.path = Path(str(meta_base) + ".jsonl")
        self._fh = open(self.path, "w", encoding="utf-8", newline="")
    def write(self, d: Dict): self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    def close(self):
        try: self._fh.close()
        except Exception: pass

class ShardWriter:
    def __init__(self, shards_dir: Path, shard_id: int):
        self.shards_dir = shards_dir; shards_dir.mkdir(parents=True, exist_ok=True)
        base = shards_dir / f"shard-{shard_id:04d}"
        self.tokens_path = Path(str(base) + ".tokens.u16")
        self.idx_path    = Path(str(base) + ".tokens.idx")
        self.meta_base   = Path(str(base) + ".meta")
        self.shard_id    = shard_id

        self._idx_rows: List[Tuple[int,int]] = []
        self._token_count = 0
        self._tokens_f = open(self.tokens_path, "wb")
        self._meta = MetaWriter(self.meta_base)

    def size_bytes(self) -> int:
        return self._token_count * TOK_DTYPE.itemsize


    def add_item(self, *, wav_path: str, txt_utf8: str, speaker, split: str,
                 seconds: float, token_ids: np.ndarray):
        token_ids = np.ascontiguousarray(token_ids, dtype=TOK_DTYPE)
        length = int(token_ids.shape[0])
        if length == 0:
            return False
        offset = self._token_count
        token_ids.tofile(self._tokens_f)
        self._token_count += length

        row = len(self._idx_rows)
        self._idx_rows.append((offset, length))

        utt_id = str(Path(wav_path)).split('/')[-1]
        self._meta.write({
            "row": row,
            "utt_id": utt_id,
            "speaker": speaker,
            "split": split,
            "text": txt_utf8,
            "seconds": float(seconds),
            "tok_offset": int(offset),
            "tok_len": int(length),
            "shard": self.tokens_path.name,
        })
        return True

    def close(self):
        try:
            self._tokens_f.flush(); self._tokens_f.close()
        except Exception:
            pass
        self._meta.close()
        return {"n_items": len(self._idx_rows), "n_tokens": self._token_count,
                "tokens_path": str(self.tokens_path)}

class ShardManager:
    def __init__(self, shards_root: Path, target_bytes: int):
        self.dir = shards_root; self.dir.mkdir(parents=True, exist_ok=True)
        self.target = target_bytes
        self.manifest_rows: List[Dict] = []
        self.cur_id = 0
        self.writer = ShardWriter(self.dir, 0)
        print("[shard] starting shard 0000")

    def rotate_if_needed(self, add_frames: int):
        if self.writer.size_bytes() + add_frames * TOK_DTYPE.itemsize > self.target and self.writer.size_bytes() > 0:
            info = self.writer.close()
            print(f"[shard] closed {self.cur_id:04d}: {info['n_items']} items, {info['n_tokens']} tokens")
            self.cur_id += 1
            self.writer = ShardWriter(self.dir, self.cur_id)
            
    def add_prequantized(self, *, wav_path: str, text: str, speaker: str,
                     split: str, seconds: float, token_ids: np.ndarray):
        token_ids = np.asarray(token_ids, dtype=np.uint16)
        n_frames = int(token_ids.shape[0])
        self.rotate_if_needed(n_frames)
        ok = self.writer.add_item(
            wav_path=str(wav_path),
            txt_utf8=text,
            speaker=speaker,
            split=split,
            seconds=seconds,
            token_ids=token_ids
        )
        if ok:
            self.manifest_rows.append({
                "shard": self.writer.tokens_path.name,
                "row": len(self.writer._idx_rows) - 1,
                "speaker": speaker,
                "utt_id": Path(wav_path).stem,
                "tok_offset": self.writer._idx_rows[-1][0],
                "tok_len": self.writer._idx_rows[-1][1],
                "seconds": seconds,
        })

    def add_pair(self, wav_path: Path, text: str, speaker: str, split="train"):
        ids = tok.quantize_path(str(wav_path)).numpy().astype(np.uint16, copy=False)
        n_frames = int(ids.shape[0])
        self.rotate_if_needed(n_frames)
        try:
            si = torchaudio.info(str(wav_path))
            seconds = si.num_frames / si.sample_rate if si.sample_rate > 0 else (n_frames * 0.02)
        except Exception:
            seconds = n_frames * 0.02
        ok = self.writer.add_item(
            wav_path=str(wav_path),
            txt_utf8=text,
            speaker=speaker,
            split=split,
            seconds=seconds,
            token_ids=ids
        )
        if ok:
            self.manifest_rows.append({
                "shard": self.writer.tokens_path.name,
                "row": len(self.writer._idx_rows)-1,
                "speaker": speaker,
                "utt_id": wav_path.split('/')[-1],
                "tok_offset": self.writer._idx_rows[-1][0],
                "tok_len": self.writer._idx_rows[-1][1],
                "seconds": seconds,
            })

    def finalize(self, out_root: Path):
        info = self.writer.close()
        print(f"[shard] closed {self.cur_id:04d}: {info['n_items']} items, {info['n_tokens']} tokens")
        df = pd.DataFrame(self.manifest_rows)
        man_path = out_root/"manifest.parquet"
        df.to_parquet(man_path, index=False)
        (out_root/"README.md").write_text(
            "# Nava Codes Shards\n\n- shards/: token blobs + idx + meta\n- manifest.parquet: all rows for this run\n",
            encoding="utf-8"
        )
        print(f"[manifest] wrote {len(df)} rows -> {man_path}")

class MetaJSONLIterable(IterableDataset):
    def __init__(self, meta_jsonl: Path, *, rank: int, world: int, train_ratio: float, split_seed: int):
        self.meta_jsonl = meta_jsonl
        self.rank = rank
        self.world = world
        self.train_ratio = train_ratio
        self.split_seed = split_seed

    def _split_for_idx(self, idx: int) -> str:
        h = hashlib.md5(f"{self.split_seed}-{idx}".encode()).digest()
        x = int.from_bytes(h[:4], "little") / 2**32
        return "train" if x < self.train_ratio else "test"

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        wn  = wi.num_workers if wi else 1

        with open(self.meta_jsonl, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % self.world != self.rank:
                    continue

                local_i = i // self.world
                if (local_i % wn) != wid:
                    continue

                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)

                audio_filepath = obj["audio_filepath"]
                text = obj["text"]
                speaker = obj.get("speaker") or "unk"
                split = self._split_for_idx(i)

                yield {
                    "idx": i,
                    "wav_path": str(audio_filepath),
                    "text": text,
                    "speaker": speaker,
                    "split": split,
                }

def _load_16k_mono_cpu(wav_path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path, backend="soundfile")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).contiguous()

def collate_decode_pad(batch):
    wavs = []
    lengths = []
    seconds = []
    for r in batch:
        w = _load_16k_mono_cpu(r["wav_path"])
        wavs.append(w)
        lengths.append(int(w.numel()))
        seconds.append(float(w.numel()) / 16000.0)

    Tmax = max(lengths) if lengths else 0
    wav_bt = torch.stack([F.pad(w, (0, Tmax - w.numel())) for w in wavs], dim=0)
    lengths_t = torch.tensor(lengths, dtype=torch.long)

    return {
        "idx": [r["idx"] for r in batch],
        "wav_path": [r["wav_path"] for r in batch],
        "text": [r["text"] for r in batch],
        "speaker": [r["speaker"] for r in batch],
        "split": [r["split"] for r in batch],
        "seconds": seconds,
        "wav": wav_bt,
        "lengths": lengths_t,
    }

def worker_init_fn(_):
    torch.set_num_threads(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build shards from a JSONL meta file.")
    parser.add_argument("meta_jsonl", help="Path to shard-XXXX.meta.jsonl")
    parser.add_argument("-r", "--train-ratio", type=float, default=0.995)
    parser.add_argument(
        "-o", "--out-root",
        type=str,
        default=str(ASSETS_DIR / "shards"),
        help="Output directory for generated shards/ and manifest.parquet"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=1234)

    args = parser.parse_args()

    TRAIN_RATIO = args.train_ratio
    if not (0.0 <= TRAIN_RATIO <= 1.0):
        print("[error] --train-ratio must be between 0 and 1", file=sys.stderr); sys.exit(2)

    META_JSONL = Path(args.meta_jsonl).expanduser().resolve()
    OUT_ROOT = Path(args.out_root).expanduser().resolve()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()
    rank = accelerator.process_index
    world = accelerator.num_processes
    is_main = accelerator.is_main_process

    tok = NavaTokenizer(REPCODEC_SFT, STATS_NPZ, device=accelerator.device)
    sm = ShardManager(OUT_ROOT, TARGET_SHARD_BYTES) if is_main else None
    tqdm = partial(_tqdm, disable=not is_main, position=0, leave=True, dynamic_ncols=True)


    total = sum(1 for _ in open(META_JSONL, "r", encoding="utf-8"))
    FLUSH_EVERY = 64
    local_buf = []

    pbar = tqdm(total=total, desc="Building shards") if is_main else None

    BATCH_SIZE  = args.batch_size
    NUM_WORKERS = args.num_workers
    PREFETCH    = args.prefetch

    dataset = MetaJSONLIterable(
        META_JSONL,
        rank=rank,
        world=world,
        train_ratio=TRAIN_RATIO,
        split_seed=args.split_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_decode_pad,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None,
    )

    total = sum(1 for _ in open(META_JSONL, "r", encoding="utf-8"))
    pbar = tqdm(total=total, desc="Building shards") if is_main else None

    for batch in loader:
        wav = batch["wav"].to(accelerator.device, non_blocking=True)        
        lengths = batch["lengths"].to(accelerator.device, non_blocking=True)

        feats_list, feat_lens = tok.hubert.extract_features(wav, lengths=lengths, num_layers=18)
        feats = feats_list[-1]
        feats = (feats - tok.H_mean.to(feats)) / tok.H_std.to(feats)

        ids, _ = tok.tokenizer.quantize(feats)
        if feat_lens is None:
            feat_lens = torch.full((ids.shape[0],), ids.shape[1], device=ids.device, dtype=torch.long)

        to_send = []
        for bi in range(ids.shape[0]):
            L = int(feat_lens[bi])
            tok_ids = ids[bi, :L].detach().cpu().numpy().astype(np.uint16, copy=False)
            to_send.append({
                "idx": batch["idx"][bi],
                "wav_path": batch["wav_path"][bi],
                "text": batch["text"][bi],
                "speaker": batch["speaker"][bi],
                "split": batch["split"][bi],
                "seconds": float(batch["seconds"][bi]),
                "token_ids": tok_ids,
            })

        if dist.is_available() and dist.is_initialized():
            gathered = [None] * world if is_main else None
            dist.gather_object(to_send, gathered, dst=0)
        else:
            gathered = [to_send]

        if is_main:
            flat = []
            for chunk in gathered:
                if chunk:
                    flat.extend(chunk)
            flat.sort(key=lambda r: r["idx"])

            for rec in flat:
                rec.pop("idx", None)
                sm.add_prequantized(**rec)

            if pbar:
                pbar.update(len(flat))

    if is_main:
        if pbar: pbar.close()
        sm.finalize(OUT_ROOT)

    accelerator.wait_for_everyone()
