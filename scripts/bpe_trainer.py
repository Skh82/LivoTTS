import argparse, io, json
from pathlib import Path
import sentencepiece as spm
import zstandard as zstd

def iter_jsonl(path: Path):
    if str(path).endswith(".zst"):
        with path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def build_train_text(inputs, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with out_path.open("w", encoding="utf-8") as out:
        for p in inputs:
            for obj in iter_jsonl(Path(p)):
                text = obj.get("text")
                if not text:
                    continue
                out.write(text.replace("\n", " ") + "\n")
                n_lines += 1
    return n_lines


def train_spm(input_txt: Path, model_prefix="fa_bpe_8k", vocab_size=8000, num_threads=8):
    spm.SentencePieceTrainer.Train(
        input=str(input_txt),
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        normalization_rule_name="identity",
        byte_fallback=True,
        input_sentence_size=20_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        hard_vocab_limit=False,
        num_threads=num_threads,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_text", default="train_text.txt")
    ap.add_argument("--model_prefix", default="fa_bpe_8k")
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--num_threads", type=int, default=8)
    args = ap.parse_args()

    out_txt = Path(args.out_text)
    n = build_train_text(args.inputs, out_txt)
    print(f"Wrote {n:,} lines to {out_txt}")

    train_spm(
        out_txt,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        num_threads=args.num_threads,
    )
    print("Training complete.")

if __name__ == "__main__":
    main()
