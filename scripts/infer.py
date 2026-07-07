import argparse, shutil, torch
import numpy as np
import soundfile as sf
import sentencepiece as spm
import numpy as np
import json

from src.Vevmodel.vevo_decoder import VevoInferencePipeline
from src.Livmodel.Livo import LivoModel
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from src.Vevmodel.vevo_decoder import load_wav
from accelerate import Accelerator
from safetensors.torch import safe_open
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
def R(root: Union[str, Path], *parts: str, mkdir: bool = False) -> Path:
    root = Path(root).expanduser().resolve()
    parts = [str(p).lstrip("/\\") for p in parts]
    p = root.joinpath(*parts)
    if mkdir:
        (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return p

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

def ensure_pretrained(amphion_repo: str = "amphion/Vevo", verbose: bool = True) -> Dict[str, Path]:
    PRE = R(REPO_ROOT, "pretrained", mkdir=True)
    cfg_dir = R(PRE, "models", "vc", "vevo", "config", mkdir=True)
    ckpt_root = R(PRE, "ckpts", "Vevo", mkdir=True)

    assets_root = R(REPO_ROOT, "assets", "cfg")
    want_cfgs = ["Vq8192ToMels.json", "Vocoder.json", "PhoneToVq8192.json"]
    cfg_paths: Dict[str, Optional[Path]] = {"Vq8192ToMels.json": None, "Vocoder.json": None, "PhoneToVq8192.json": None}
    
    for name in want_cfgs:
        src = assets_root / name
        if src.exists():
            dst = cfg_dir / name
            if verbose:
                print(f"[config] {src} -> {dst}")
            shutil.copy2(src, dst)
            cfg_paths[name] = dst
        else:
            if verbose:
                print(f"[config] missing in assets: {name}")

    #tok_snapshot = snapshot_download(
    #    repo_id=amphion_repo,
    #    repo_type="model",
    #    cache_dir=str(ckpt_root),
    #    allow_patterns=["tokenizer/vq8192/*"],
    #)
    #tokenizer_dir = Path(tok_snapshot) / "tokenizer" / "vq8192"

    #tok_snapshot = snapshot_download(
    #    repo_id=amphion_repo, repo_type="model",
    #    cache_dir=str(ckpt_root),
    #    allow_patterns=["tokenizer/vq8192/*"],
    #)
    #tokenizer_dir = Path(tok_snapshot) / "tokenizer" / "vq8192"

    #fmt_snapshot = snapshot_download(
    #    repo_id=amphion_repo, repo_type="model",
    #    cache_dir=str(ckpt_root),
    #    allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    #)
    #fmt_ckpt = Path(fmt_snapshot) / "acoustic_modeling" / "Vq8192ToMels"
    fmt_cfg  = cfg_paths["Vq8192ToMels.json"]

    #voc_snapshot = snapshot_download(
    #    repo_id=amphion_repo, repo_type="model",
    #    cache_dir=str(ckpt_root),
    #    allow_patterns=["acoustic_modeling/Vocoder/*"],
    #)
    #vocoder_ckpt = Path(voc_snapshot) / "acoustic_modeling" / "Vocoder"
    vocoder_cfg  = cfg_paths["Vocoder.json"]

    fmt_ckpt = str(REPO_ROOT) + "/pretrained/ckpts/Vevo/models--amphion--Vevo/snapshots/7edf4640c400c20542aa39c45b63f60e6c7baba0/acoustic_modeling/Vq8192ToMels/model.safetensors"
    vocoder_ckpt = str(REPO_ROOT) + "/pretrained/ckpts/Vevo/models--amphion--Vevo/snapshots/7edf4640c400c20542aa39c45b63f60e6c7baba0/acoustic_modeling/Vocoder/model.safetensors"
    tokenizer_dir = str(REPO_ROOT) + "/pretrained/ckpts/Vevo/models--amphion--Vevo/snapshots/7edf4640c400c20542aa39c45b63f60e6c7baba0/tokenizer/vq8192/model.safetensors"
    return {
        "fmt_cfg": fmt_cfg,
        "fmt_ckpt": fmt_ckpt,
        "vocoder_cfg": vocoder_cfg,
        "vocoder_ckpt": vocoder_ckpt,
        "tokenizer_dir": tokenizer_dir,
        "phone2vq_cfg": cfg_paths["PhoneToVq8192.json"],
    }

import re

_SENT_SPLIT_RE = re.compile(r"(?<=[\.!\؟\!؛…])\s+")

def split_sentences_fa(text: str):
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def as_numpy_audio(x: Union[np.ndarray, torch.Tensor], target_sr: int) -> Tuple[np.ndarray, int]:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim > 1:
        x = x.mean(axis=-1)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    return x, target_sr


def save_audio(wav: Union[np.ndarray, torch.Tensor], output_path: Union[str, Path], sr: int = 24000):
    y, sr = as_numpy_audio(wav, sr)
    sf.write(str(output_path), y, sr)


def save_audio_matched(wav: Union[np.ndarray, torch.Tensor], ref_wav_path: Union[str, Path],
                       output_path: Union[str, Path], target_sr: Optional[int] = 24000,
                       match_duration: bool = False):
    
    ref, ref_sr = sf.read(str(ref_wav_path))
    sr = target_sr or ref_sr
    y, _ = as_numpy_audio(wav, sr)
    if match_duration:
        ref_len = int(len(ref) * (sr / ref_sr)) if ref_sr != sr else len(ref)
        if len(y) < ref_len:
            y = np.pad(y, (0, ref_len - len(y)))
        else:
            y = y[:ref_len]
    sf.write(str(output_path), y, sr)


@contextmanager
def fm_cfg(pipe, cfg: float = 1.6, rescale: float = 0.75):
    orig = pipe.fmt_model.reverse_diffusion
    def patched(*args, **kwargs):
        kwargs.setdefault("cfg", cfg)
        kwargs.setdefault("rescale_cfg", rescale)
        return orig(*args, **kwargs)
    pipe.fmt_model.reverse_diffusion = patched
    try:
        yield
    finally:
        pipe.fmt_model.reverse_diffusion = orig


def build_livo_and_bank(device, checkpoint, config):
    model = LivoModel.from_json(str(config))

    accelerator = Accelerator()
    model = accelerator.prepare(model)
    model.attentive_rnn.to_mode("fused_recurrent")

    model = model.to(device).eval()
    accelerator.load_state(str(checkpoint))
    return model

def vevo_pipeline_from_paths(content_style_tokenizer_ckpt_path: Path,
                             fmt_cfg_path: Path, fmt_ckpt_path: Path,
                             vocoder_cfg_path: Path, vocoder_ckpt_path: Path,
                             device: torch.device):

    pipe = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=str(content_style_tokenizer_ckpt_path),
        fmt_cfg_path=str(fmt_cfg_path),
        fmt_ckpt_path=str(fmt_ckpt_path),
        vocoder_cfg_path=str(vocoder_cfg_path),
        vocoder_ckpt_path=str(vocoder_ckpt_path),
        device=device,
    )
    return pipe

def load_ist_params(path: str, device="cuda"):
    params = []
    with safe_open(path, framework="pt", device=device) as f:
        k_keys = [k for k in f.keys() if k.endswith("_k")]
        k_keys.sort(key=lambda s: int("".join([c for c in s if c.isdigit()])))
        for kname in k_keys:
            vname = kname[:-2] + "_v"
            params.append((f.get_tensor(kname), f.get_tensor(vname)))
    return params

def main():
    ap = argparse.ArgumentParser(description="Livo → Vevo timbre transfer inference")

    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--text", type=str, help="Text to synthesize")
    ap.add_argument("--prompt-text", type=str, help="Prompt text to attend to")
    mx.add_argument("--text-file", type=str, help="Path to UTF-8 text file")
    ap.add_argument("--prompt-wav", type=str, required=True, help="Reference prompt WAV path")
    ap.add_argument("--spk-id", type=int, default=0, help="Speaker id into IST bank cache")

    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-seqlen", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=80)
    ap.add_argument("--top-p", type=float, default=0.90)
    ap.add_argument("--temp", type=float, default=0.80)
    ap.add_argument("--first-greedy-quant", type=int, default=1)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.2)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)
    ap.add_argument("--out", type=str, default=None, help="Output WAV path (default: <root>/outputs/vevo_timbre_out.wav)")

    ap.add_argument("--fm-steps", type=int, default=24)
    ap.add_argument("--fm-cfg", type=float, default=0.85)
    ap.add_argument("--fm-rescale", type=float, default=0.10)
    ap.add_argument("--checkpoint_path", type=str, default=str(REPO_ROOT) + "/pretrained/checkpoint",
                    help="Provide the absolute path to your checkpoint, if not chosen, it will automatically pick the latest checkpoint")
    ap.add_argument(
        "--config_path",
        default=str(R(REPO_ROOT, "assets", "cfg", "config.json")),
        help="Path to config.json",
    )
    ap.add_argument(
        "--return-prompt",
        action="store_true",
        help="If set, also return the prompt portion."
    )
    ap.add_argument("--ist_params", type=str, default=None,
                help="Path to IST params .safetensors (overrides speaker_bank init_state)")

    args = ap.parse_args()
    
    CONFIG_JSON = Path(args.config_path).expanduser().resolve()

    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        CFG = json.load(f)
        P  = CFG.get("paths", {})

    SPM_MODEL  = resolve_cfg_path(P, "spm_model",  None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = ensure_pretrained()

    content_tokenizer = assets["tokenizer_dir"]
    fmt_cfg_path  = assets["fmt_cfg"]
    fmt_ckpt_path = assets["fmt_ckpt"]
    voc_cfg_path  = assets["vocoder_cfg"]
    voc_ckpt_path = assets["vocoder_ckpt"]

    pipe = vevo_pipeline_from_paths(
        content_style_tokenizer_ckpt_path=content_tokenizer,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=voc_cfg_path,
        vocoder_ckpt_path=voc_ckpt_path,
        device=device,
    )

    LivoTTS = build_livo_and_bank(
        device=device,
        checkpoint=args.checkpoint_path,
        config=args.config_path
    )

    INF_KNOBS = dict(
        batch_size=args.batch_size,
        device=device,
        max_seqlen=args.max_seqlen,
        k=args.top_k,
        top_p=args.top_p,
        temp=args.temp,
        first_greedy_quant=args.first_greedy_quant,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_seqlen=20,
    )
    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))
    sp.set_encode_extra_options("bos:eos")
    txt = args.text

    if args.prompt_text:
        #"[BOS]" + prompt + " " + txt + "[EOS]"
        full_text = f"{args.prompt_text} {txt}"
        _, _, timbre_ref_speech16k = load_wav(args.prompt_wav, device)
        timbre_ref_hubert_codecs, _ = pipe.extract_hubert_codec(
            pipe.content_style_tokenizer, timbre_ref_speech16k, duration_reduction=False)
        prompt = timbre_ref_hubert_codecs.unsqueeze(0)
    else:
        full_text = txt
        prompt = None

    ids = torch.tensor(sp.encode(full_text, out_type=int), device=device).long()
    spk_ids = torch.tensor([args.spk_id], device=device, dtype=torch.long)
    
    if args.ist_params is not None:
        ist_params = load_ist_params(args.ist_params, device=str(device))
        init_state = LivoTTS.attentive_rnn.get_state_from_params(
            ist_params,
            batch_size=args.batch_size,
            scale=0.02
        )
        LivoTTS.attentive_rnn.to_mode("fused_recurrent")
        LivoTTS = LivoTTS.eval()
    else:
        init_state = None
        LivoTTS.attentive_rnn.to_mode("fused_chunk")

    sr_target = 24000
    with torch.inference_mode():
        qs, atts, stop_tokens, cuts, next_state = LivoTTS.generate_batch(
            x=ids,
            prompt=prompt,
            init_state=init_state,
            prev_state=None,
            spk_ids=spk_ids,
            **INF_KNOBS,
        )

    rvq_slice, att_slice = cuts[0]
    codes = rvq_slice.squeeze().detach().to(device)
    codes_or_content = codes.unsqueeze(0).long()

    if not args.return_prompt and args.prompt_text:
        prompt_len = timbre_ref_hubert_codecs.size(1)
        codes_or_content = codes_or_content[:, prompt_len:]
    
    with fm_cfg(pipe, cfg=args.fm_cfg, rescale=args.fm_rescale):
        gen_audio = pipe.inference_fm(
            codes_or_content,
            args.prompt_wav,
            flow_matching_steps=args.fm_steps,
        )

    wav_np, _ = as_numpy_audio(gen_audio, sr_target)

    out_path = Path(args.out) if args.out else R(REPO_ROOT, "outputs", "utt.wav", mkdir=True)
    sf.write(str(out_path), wav_np, sr_target)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
