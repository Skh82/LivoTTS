import torch, os, io, json, glob, random
import numpy as np
import zstandard as zstd
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence

def _iter_jsonl_auto(path: str):
    """
    Iterate JSON lines from either a compressed .jsonl.zst file or a plain .jsonl file.
    Chooses the correct reader based on filename extension.
    """
    if path.endswith(".zst") or path.endswith(".jsonl.zst"):
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

class _ShardMemmaps:
    def __init__(self):
        self._maps: Dict[str, np.memmap] = {}

    def get(self, path: str) -> np.memmap:
        if path not in self._maps:
            self._maps[path] = np.memmap(path, mode="r", dtype=np.uint16)
        return self._maps[path]

def _get_worker_mmaps() -> _ShardMemmaps:
    global _worker_mmaps
    if _worker_mmaps is None:
        _worker_mmaps = _ShardMemmaps()
    return _worker_mmaps

_worker_mmaps: Optional[_ShardMemmaps] = None
_SP = None

class NavaCodesDataset(Dataset):
    def __init__(self, root: str, split: Optional[str] = "train", meta_glob: Optional[str] = None,
                 limit: Optional[int] = None, spk: Optional[str] = None):

        self._spk = spk
        self.root = root
        self.split = split
        self.items: List[Dict] = []

        if meta_glob:
            meta_paths = sorted(glob.glob(meta_glob))
        else:
            candidates = []
            candidates.extend(glob.glob(os.path.join(root, "shard-*.meta.jsonl.zst")))
            candidates.extend(glob.glob(os.path.join(root, "shard-*.meta.jsonl")))
            meta_paths = sorted(set(os.path.abspath(p) for p in candidates))
        if not meta_paths:
            patt = meta_glob or "{shard-*.meta.jsonl(.zst)}"
            raise FileNotFoundError(f"No meta files found with pattern: {patt}")

        for mp in meta_paths:
            base_dir = os.path.dirname(mp)
            for rec in _iter_jsonl_auto(mp):
                if split is None or rec.get("split") == split:   
                    if self._spk and rec.get("spk") != self._spk:
                        continue
                    shard_rel = rec["shard"]
                    tokens_path = os.path.join(base_dir, shard_rel)
                    rec["_tokens_path"] = os.path.abspath(tokens_path)
                    for k in ("tok_offset", "tok_len"):
                        if k not in rec:
                            raise ValueError(f"Missing '{k}' in record {rec.get('utt_id')}")
                    self.items.append(rec)
                    if limit and len(self.items) >= limit:
                        break
            if limit and len(self.items) >= limit:
                break

        if not self.items:
            raise RuntimeError("No records found after filtering. Check 'split' and paths.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]

        tokens_path: str = rec["_tokens_path"]
        offset: int = int(rec["tok_offset"])
        length: int = int(rec["tok_len"])

        mmap_cache = _get_worker_mmaps()
        arr = mmap_cache.get(tokens_path)
        end = offset + length
        if end > arr.shape[0]:
            raise IndexError(
                f"Slice [{offset}:{end}] out of bounds for {tokens_path} with length {arr.shape[0]}"
            )

        codes_u16 = np.asarray(arr[offset:end], dtype=np.uint16)         
        codes = torch.from_numpy(codes_u16.astype(np.int64, copy=False)) 

        sample = {
            "utt_id": rec.get("utt_id"),
            "text": rec.get("text"),
            "spk": rec.get("spk"),
            "dataset": rec.get("dataset"),
            "seconds": float(rec.get("seconds", 0.0)),
            "n_frames": int(rec.get("n_frames", length)),
            "audio_tokens": codes,
        }
        
        global _SP
        txt = rec.get("text") or ""
        text_ids = _SP.encode(txt, out_type=int)
        sample["text_ids"] = torch.tensor(text_ids, dtype=torch.long)
        sample["text_len"] = sample["text_ids"].numel()
        
        return sample


def pad_1d(seqs: List[torch.Tensor], pad_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([s.numel() for s in seqs], dtype=torch.long)
    T = int(lengths.max().item()) if len(seqs) > 0 else 0
    out = torch.full((len(seqs), T), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        t = s.numel()
        out[i, :t] = s
    return out, lengths


def delay_rvq(
    code,
    head_token: int = -2,
    tail_token: int = -3,
):
    q, _ = code.shape
    extension = torch.ones((q, q + 1)).tril() * head_token
    extension += torch.ones((q + 1, q)).tril(diagonal=-1).T * tail_token
    extension = torch.flip(extension, (1,))
    extended_code = torch.cat((code, extension), axis=1)
    for i in range(q):
        extended_code[i, :] = torch.roll(extended_code[i, :], i + 1)

    return extended_code.long()


def collate_codes(batch):
    audio_tokens, text_tokens = zip(*[(x["audio_tokens"], x["text_ids"]) for x in batch])
    
    audio_token_delayed = []
    for x in audio_tokens:
        x = x.squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = delay_rvq(x + 3, head_token=1, tail_token=2).transpose(-1, -2)
        audio_token_delayed.append(x)

    xlen, ylen = map(lambda x: [xx.shape[0] for xx in x], (text_tokens, audio_token_delayed))
    x_mask, y_mask = map(lambda x: sequence_mask(x, device="cpu"), (torch.tensor(xlen), torch.tensor(ylen)))

    audio_token, text_token = map(
        lambda seqs, pad: pad_sequence(seqs, batch_first=True, padding_value=pad),
        (audio_token_delayed, text_tokens),
        (0, 3)
    )

    encoder_mask  = (x_mask.unsqueeze(1) & x_mask.unsqueeze(2))
    crossatt_mask = (y_mask.unsqueeze(2) & x_mask.unsqueeze(1))
    crossatt_mask[:, :, 0] = True


    return {
        "text_token":    text_token,
        "audio_token":   audio_token,
        "crossatt_mask": crossatt_mask,
        "encoder_mask":  encoder_mask,
        "logits_mask":   y_mask,
        "spk": [b["spk"] for b in batch],
        "seconds": torch.tensor([b["seconds"] for b in batch], dtype=torch.float32)

    }

def sequence_mask(lengths, max_len=None, device=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask


class LengthBucketDynamicBatchSampler(BatchSampler):
    def __init__(
        self,
        lengths,
        max_frames_per_batch: Optional[int] = None,
        fixed_batch_size: Optional[int] = None,
        n_buckets: int = 12,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 1337,
        longest_first: bool = False,
        interleave: bool = True,
        bucket_temp: float = 0.7,
    ):
        if (max_frames_per_batch is None) == (fixed_batch_size is None):
            raise ValueError("Exactly one of {max_frames_per_batch, fixed_batch_size} must be set.")

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.N = len(self.lengths)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.longest_first = bool(longest_first)
        self.interleave = bool(interleave)
        self.bucket_temp = float(bucket_temp)

        self.by_frames = max_frames_per_batch is not None
        if self.by_frames:
            self.B_frames = int(max_frames_per_batch)
        else:
            self.batch_size = int(fixed_batch_size)

        qs = np.linspace(0.0, 1.0, n_buckets + 1)
        edges = np.unique(np.quantile(self.lengths, qs))
        if edges.size == 1:
            self.bucket_ids = np.zeros(self.N, dtype=np.int32)
            self.n_buckets = 1
        else:
            self.bucket_ids = np.clip(
                np.digitize(self.lengths, edges[1:-1], right=True),
                0, edges.size - 2
            ).astype(np.int32)
            self.n_buckets = edges.size - 1

        self.bucket_to_indices = [np.where(self.bucket_ids == b)[0].tolist()
                                  for b in range(self.n_buckets)]
        self.bucket_max = []
        for b in range(self.n_buckets):
            idxs = self.bucket_to_indices[b]
            self.bucket_max.append(int(self.lengths[idxs].max()) if idxs else 0)

        self._epoch = 0

    def set_epoch(self, epoch: int, *, longest_first: Optional[bool] = None):
        self._epoch = int(epoch)
        if longest_first is not None:
            self.longest_first = bool(longest_first)

    def __len__(self):
        total = 0
        if self.by_frames:
            B = self.B_frames
            for idxs in self.bucket_to_indices:
                if not idxs:
                    continue
                avgL = max(1, int(np.mean(self.lengths[idxs])))
                total += max(1, (len(idxs) * avgL + B - 1) // B)
        else:
            B = self.batch_size
            for idxs in self.bucket_to_indices:
                if not idxs:
                    continue
                total += (len(idxs) // B) if self.drop_last else ((len(idxs) + B - 1) // B)
        return total

    def __iter__(self):
        py_rng = random.Random(self.seed + self._epoch)
        np_rng = np.random.default_rng(self.seed + self._epoch)

        idxs_per_bucket = []
        for b in range(self.n_buckets):
            idxs = self.bucket_to_indices[b][:]
            if not idxs:
                idxs_per_bucket.append([])
                continue
            if self.longest_first:
                idxs.sort(key=lambda i: int(self.lengths[i]), reverse=True)
            elif self.shuffle:
                py_rng.shuffle(idxs)
            idxs_per_bucket.append(idxs)
        pos = [0] * self.n_buckets

        def remaining(bid: int) -> int:
            return len(idxs_per_bucket[bid]) - pos[bid]

        def pull_batch_from_bucket(bid: int):
            if self.by_frames:
                B = self.B_frames
                batch, cur_max = [], 0
                while pos[bid] < len(idxs_per_bucket[bid]):
                    i = idxs_per_bucket[bid][pos[bid]]
                    L = int(self.lengths[i])

                    if L > B:
                        if batch:
                            break
                        pos[bid] += 1
                        return [i]

                    new_max = L if cur_max == 0 else max(cur_max, L)
                    if new_max * (len(batch) + 1) <= B:
                        batch.append(i)
                        cur_max = new_max
                        pos[bid] += 1
                    else:
                        break
                if batch and (len(batch) > 0):
                    return batch
                return None

            else:
                B = self.batch_size
                rem = remaining(bid)
                if rem <= 0:
                    return None
                take = min(B, rem)
                if take < B and self.drop_last:
                    pos[bid] = len(idxs_per_bucket[bid])
                    return None
                start = pos[bid]
                end = start + take
                pos[bid] = end
                return idxs_per_bucket[bid][start:end]

        if not self.interleave:
            if self.longest_first:
                bucket_order = sorted(range(self.n_buckets),
                                      key=lambda b: self.bucket_max[b],
                                      reverse=True)
            else:
                bucket_order = list(range(self.n_buckets))
                if self.shuffle:
                    py_rng.shuffle(bucket_order)

            for b in bucket_order:
                while remaining(b) > 0:
                    batch = pull_batch_from_bucket(b)
                    if batch and (len(batch) > 0):
                        yield batch
                    else:
                        break
            return

        alive = [remaining(b) > 0 for b in range(self.n_buckets)]
        while any(alive):
            rem = np.array([remaining(b) for b in range(self.n_buckets)], dtype=np.float64)
            mask = rem > 0
            if not mask.any():
                break

            alpha = self.bucket_temp
            probs = rem[mask] ** max(alpha, 1e-8)
            probs = probs / probs.sum()
            candidates = np.nonzero(mask)[0]
            bid = int(np_rng.choice(candidates, p=probs))

            batch = pull_batch_from_bucket(bid)
            if batch and (len(batch) > 0):
                yield batch

            alive[bid] = remaining(bid) > 0
