import torch, base64, re, io, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from difflib import SequenceMatcher

def _compute_mel_png(
    wav_path: Union[str, Path],
    sr_target: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: Optional[int] = None,
    cmap: str = "magma",
    px_per_sec: float = 60.0) -> Tuple[str, int, int, float, float]:
    
    y, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    duration = len(y) / sr
    width_px  = max(900, min(2400, int(px_per_sec * max(1.0, duration))))
    height_px = 260
    fig = plt.figure(figsize=(width_px/100, height_px/100), dpi=100)
    ax = plt.axes([0,0,1,1])
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, cmap=cmap)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return data_url, width_px, height_px, (width_px / duration if duration > 0 else px_per_sec), duration


def write_html_report_from_segments(
    full_text: str,
    wav_path: Union[str, Path],
    out_html: Union[str, Path],
    segments: list,
    step_hz: float = 50.0,
    title: str = "LivoTTS ▶",
    rtl: bool = True,
    sample_rate_label: Optional[str] = None,
    mel_cfg: Optional[dict] = None
):
    import base64, mimetypes, uuid, re
    import html as _html
    from pathlib import Path

    mel_opts = {"px_per_sec": 60.0}
    if mel_cfg:
        mel_opts.update(mel_cfg)
    mel_data_url, mel_w, mel_h, _native_px_per_sec, duration_sec = _compute_mel_png(
        wav_path, **mel_opts
    )

    word_iter = list(re.finditer(r"\S+", full_text or ""))
    idx2ms = {}
    for s_step, e_step, _wtxt, widx in (segments or []):
        s_ms = int(round(1000.0 * (s_step / step_hz)))
        e_ms = int(round(1000.0 * (e_step / step_hz)))
        if widx not in idx2ms:
            idx2ms[widx] = [s_ms, e_ms]
        else:
            idx2ms[widx][0] = min(idx2ms[widx][0], s_ms)
            idx2ms[widx][1] = max(idx2ms[widx][1], e_ms)

    pieces = []
    last_end = 0
    for wi, m in enumerate(word_iter):
        start, end = m.span()
        if start > last_end:
            pieces.append(f'<span class="gap">{_html.escape(full_text[last_end:start])}</span>')
        wtxt = full_text[start:end]
        meta = idx2ms.get(wi)
        ds = f'data-start="{meta[0]}"' if meta else ""
        de = f'data-end="{meta[1]}"'   if meta else ""
        missing = "" if meta else " missing"
        pieces.append(f'<span class="word{missing}" {ds} {de}>{_html.escape(wtxt)}</span>')
        last_end = end
    if last_end < len(full_text):
        pieces.append(f'<span class="gap">{_html.escape(full_text[last_end:])}</span>')
    transcript_html = "".join(pieces)

    mime = mimetypes.guess_type(str(wav_path))[0] or "audio/wav"
    audio_b64 = base64.b64encode(Path(wav_path).read_bytes()).decode("ascii")
    audio_src = f"data:{mime};base64,{audio_b64}"

    container_id  = f"tp-{uuid.uuid4().hex[:8]}"
    audio_id      = f"aud-{uuid.uuid4().hex[:8]}"
    transcript_id = f"tx-{uuid.uuid4().hex[:8]}"
    stage_id      = f"stage-{uuid.uuid4().hex[:8]}"
    mel_img_id    = f"mel-{uuid.uuid4().hex[:8]}"
    cursor_id     = f"cur-{uuid.uuid4().hex[:8]}"
    direction     = "rtl" if rtl else "ltr"
    n_words       = len(word_iter)
    dur_s_label   = f"{duration_sec:.1f}s" if duration_sec else "—"
    sr_label      = (sample_rate_label or "").strip()

    html_block = f"""
<div id="{container_id}" class="tp-card" dir="{direction}">
  <div class="tp-header">
    <div class="tp-title">{_html.escape(title)}</div>
    <div class="tp-meta">
      <span>کلمات: <b>{n_words}</b></span>
      <span>مدت: <b>{dur_s_label}</b></span>
      {"<span>SR: <b>"+_html.escape(sr_label)+"</b></span>" if sr_label else ""}
    </div>
  </div>

  <div class="tp-audio">
    <audio id="{audio_id}" src="{audio_src}" controls preload="auto"></audio>
  </div>

  <!-- transcript -->
  <div class="tp-transcript" id="{transcript_id}">
    {transcript_html}
  </div>

  <!-- mel BELOW transcript (responsive, no clipping) -->
  <div class="mel-wrap">
    <div class="mel-stage" id="{stage_id}" style="height:{mel_h}px">
      <img id="{mel_img_id}" class="mel-img" alt="mel"
           src="{mel_data_url}" />
      <div id="{cursor_id}" class="mel-cursor" style="height:{mel_h}px"></div>
    </div>
    <div class="mel-meta">Mel width: {mel_w}px · height: {mel_h}px</div>
  </div>
</div>

<style>
  /* ---------- theme tokens ---------- */
  #{container_id}.tp-card {{
    --bg: #0b1020;
    --panel: #0f1528;
    --ink: #e6eaf2;
    --muted: #98a2b3;
    --border: rgba(255,255,255,0.08);
    --accent: #22d3ee;
    --accent-soft: rgba(34,211,238,.18);
    --accent-strong: rgba(34,211,238,.28);
  }}
  @media (prefers-color-scheme: light) {{
    #{container_id}.tp-card {{
      --bg:#ffffff; --panel:#f8fafc; --ink:#0f172a; --muted:#475569; --border:rgba(0,0,0,0.08);
      --accent:#0ea5e9; --accent-soft:rgba(14,165,233,.15); --accent-strong:rgba(14,165,233,.22);
    }}
  }}

  #{container_id}.tp-card {{
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, "Noto Sans", "Helvetica Neue", Arial;
    background: var(--panel);
    color: var(--ink);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 10px 24px rgba(0,0,0,.25);
    line-height: 1.9;
  }}
  #{container_id} .tp-header {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; }}
  #{container_id} .tp-title {{ font-weight:800; letter-spacing:.2px; }}
  #{container_id} .tp-meta span {{ opacity:.85; margin-inline-start:12px; font-size:.9rem; }}

  #{container_id} .tp-audio {{ margin-bottom:10px; }}

  /* transcript styles */
  #{container_id} .tp-transcript {{
    background: transparent;
    border: 1px dashed var(--border);
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 12px;
  }}
  #{container_id} .tp-transcript .gap {{ white-space: pre-wrap; }}
  #{container_id} .tp-transcript .word {{
    display:inline-block;
    padding: 0 .18rem;
    margin: 0 -.04rem;
    border-radius: .35rem;
    transition: background-color .12s ease, transform .12s ease, text-shadow .12s ease;
  }}
  #{container_id} .tp-transcript .word.active {{
    background: var(--accent-soft);
    box-shadow: 0 0 0 1px var(--accent-strong) inset;
    text-shadow: 0 0 6px rgba(34,211,238,.25);
    transform: scale(1.06);
  }}
  #{container_id} .tp-transcript .word.past {{ opacity:.88; }}
  #{container_id} .tp-transcript .word.missing {{
    background: rgba(128,128,128,.12);
    border-bottom: 1px dotted rgba(128,128,128,.45);
  }}

  /* mel (responsive) */
  #{container_id} .mel-wrap {{ display:grid; gap:6px; }}
  #{container_id} .mel-meta {{ color:var(--muted); font-size:.85rem; }}
  #{container_id} .mel-stage {{
    position: relative;
    overflow: visible;          /* no clipping */
    width: 100%;                /* scale to available cell width */
    max-width: {mel_w}px;       /* don't upscale beyond native width */
    border-radius: 10px;
    border: 1px solid var(--border);
    background: #000;
  }}
  #{container_id} .mel-img {{
    display: block;
    width: 100%;                /* responsive width */
    height: {mel_h}px;          /* keep vertical size */
    image-rendering: pixelated;
  }}
  #{container_id} .mel-cursor {{
    position: absolute; top: 0; left: 0;
    width: 2px;
    background: var(--accent);
    box-shadow: 0 0 12px var(--accent);
    transform: translate3d(0,0,0);
    pointer-events: none;
  }}
</style>

<script>
(function() {{
  const audio = document.getElementById("{audio_id}");
  const tx = document.getElementById("{transcript_id}");
  const words = Array.from(tx.querySelectorAll(".word"));
  const ranges = words.map(w => [parseInt(w.dataset.start||"-1"), parseInt(w.dataset.end||"-1")]);
  let lastIdx = -1;

  function findActiveWord(ms) {{
    for (let i = 0; i < ranges.length; i++) {{
      const [s, e] = ranges[i];
      if (!Number.isFinite(s) || !Number.isFinite(e) || s < 0 || e < 0) continue;
      if (ms >= s && ms < e) return i;
    }}
    return -1;
  }}

  function updateActive(ms) {{
    const idx = findActiveWord(ms);
    if (idx === lastIdx) return;
    if (lastIdx >= 0) words[lastIdx].classList.remove("active");
    if (idx >= 0) words[idx].classList.add("active");
    if (idx > 0) words[idx - 1].classList.add("past");
    lastIdx = idx;
  }}

  // mel cursor with responsive scale + smoothing
  const stage  = document.getElementById("{stage_id}");
  const img    = document.getElementById("{mel_img_id}");
  const cursor = document.getElementById("{cursor_id}");
  let pxPerSec = 0;
  let rafId = null;
  let smoothedX = 0;

  function recomputeScale() {{
    const w = img.getBoundingClientRect().width || stage.clientWidth || {mel_w};
    const dur = Math.max(audio.duration || {duration_sec or 0}, 0.001);
    pxPerSec = w / dur;
  }}

  function tick() {{
    const t = audio.currentTime || 0;
    const ms = t * 1000.0;
    updateActive(ms);
    const targetX = t * pxPerSec;
    smoothedX = smoothedX + (targetX - smoothedX) * 0.25;  // EMA smoothing
    cursor.style.transform = `translate3d(${{smoothedX}}px,0,0)`;
    rafId = requestAnimationFrame(tick);
  }}

  function stopLoop() {{
    if (rafId !== null) cancelAnimationFrame(rafId);
    rafId = null;
  }}

  // wire events
  img.addEventListener("load", recomputeScale);
  window.addEventListener("resize", recomputeScale);
  audio.addEventListener("loadedmetadata", () => {{ smoothedX = 0; recomputeScale(); }});
  audio.addEventListener("play",  () => {{ stopLoop(); tick(); }});
  audio.addEventListener("pause", () => {{ stopLoop(); }});
  audio.addEventListener("seeking", () => {{
    // jump cursor immediately on seek
    smoothedX = audio.currentTime * pxPerSec;
    cursor.style.transform = `translate3d(${{smoothedX}}px,0,0)`;
    updateActive(audio.currentTime * 1000.0);
  }});
  audio.addEventListener("ended", () => {{
    stopLoop();
    if (lastIdx >= 0) words[lastIdx].classList.remove("active");
    lastIdx = -1;
    cursor.style.transform = "translate3d(0,0,0)";
  }});
}})();
</script>
"""
    Path(out_html).write_text(html_block, encoding="utf-8")

_WORD_RE = re.compile(r"\S+")

def _norm_fa_min(s: str) -> str:
    if not s: return ""
    s = s.replace("\u0640","").replace("\u200c","").replace("\u200f","").replace("‌","").replace("ٔ","")
    s = s.replace("\u064a","\u06cc").replace("\u0643","\u06a9")
    return s.strip()

def _sim(a: str, b: str, th: float = 0.80) -> bool:
    if not a or not b: return False
    if a == b: return True
    return SequenceMatcher(None, a, b).ratio() >= th

def _soft_lcs_map(hyp_words, ref_words):
    H, R = len(hyp_words), len(ref_words)
    eq = [[_sim(_norm_fa_min(hyp_words[i]), _norm_fa_min(ref_words[j])) for j in range(R)] for i in range(H)]
    dp = [[0]*(R+1) for _ in range(H+1)]
    for i in range(H):
        for j in range(R):
            dp[i+1][j+1] = dp[i][j]+1 if eq[i][j] else max(dp[i+1][j], dp[i][j+1])
    pairs = []
    i, j = H, R
    while i>0 and j>0:
        if eq[i-1][j-1] and dp[i][j]==dp[i-1][j-1]+1:
            pairs.append((i-1,j-1)); i-=1; j-=1
        elif dp[i-1][j] >= dp[i][j-1]: i-=1
        else: j-=1
    pairs.reverse()
    hyp2ref = [None]*H
    for hi,rj in pairs: hyp2ref[hi]=rj
    last = None
    for i in range(H):
        hyp2ref[i] = hyp2ref[i] if hyp2ref[i] is not None else last
        if hyp2ref[i] is not None: last = hyp2ref[i]
    last = None
    for i in range(H-1,-1,-1):
        hyp2ref[i] = hyp2ref[i] if hyp2ref[i] is not None else last
        if hyp2ref[i] is not None: last = hyp2ref[i]
    cur = -1
    for i in range(H):
        if hyp2ref[i] is None: hyp2ref[i] = max(cur,0)
        hyp2ref[i] = max(hyp2ref[i], cur)
        cur = hyp2ref[i]
    for i in range(H):
        if hyp2ref[i] is None: hyp2ref[i]=0
        hyp2ref[i] = max(0, min(hyp2ref[i], R-1))
    return hyp2ref

def map_segments_to_baseline_text(full_text: str, hyp_segments, min_dur_steps: int = 1):

    ref_matches = list(_WORD_RE.finditer(full_text or ""))
    ref_words   = [m.group(0) for m in ref_matches]
    R = len(ref_words)
    if R == 0:
        return []

    H = len(hyp_segments)
    if H == 0:
        lens = [max(1,len(_norm_fa_min(w))) for w in ref_words]
        tot  = sum(lens); T = R * min_dur_steps
        c=0; out=[]
        for j,w in enumerate(ref_words):
            L = max(min_dur_steps, round(T * (lens[j]/tot)))
            s=c; e=c+L; c=e
            out.append((s,e,w,j))
        return out

    hyp_words = [h[2] for h in hyp_segments]
    T = max(e for (_,e,_,_) in hyp_segments)

    hyp2ref = _soft_lcs_map(hyp_words, ref_words)

    starts_ref = [None]*R
    ends_ref   = [None]*R
    for (s,e,w,hi), rj in zip(hyp_segments, hyp2ref):
        if rj is None: continue
        starts_ref[rj] = s if starts_ref[rj] is None else min(starts_ref[rj], s)
        ends_ref[rj]   = e if ends_ref[rj]   is None else max(ends_ref[rj],   e)

    known = [(j,starts_ref[j],ends_ref[j]) for j in range(R) if starts_ref[j] is not None and ends_ref[j] is not None]
    if known:
        first_j, _, first_end = known[0]
        first_start = starts_ref[first_j]
        block = list(range(0, first_j))
        if block and first_start is not None and first_start>0:
            lens = [max(1,len(_norm_fa_min(ref_words[j]))) for j in block]
            tot = sum(lens); c = 0
            for j,L in zip(block, lens):
                span = max(min_dur_steps, round(first_start*(L/tot)))
                s=c; e=min(first_start, c+span); c=e
                starts_ref[j]=s; ends_ref[j]=e
        for (a_j, a_s, a_e), (b_j, b_s, b_e) in zip(known, known[1:]):
            mid_block = list(range(a_j+1, b_j))
            if not mid_block: continue
            gap_s = a_e; gap_e = b_s
            gap = max(0, (gap_e if gap_e is not None else gap_s) - (gap_s if gap_s is not None else gap_e))
            lens = [max(1,len(_norm_fa_min(ref_words[j]))) for j in mid_block]
            tot  = sum(lens)
            c = gap_s
            for j,L in zip(mid_block, lens):
                span = max(min_dur_steps, round((gap * L)/max(1,tot)))
                s = c; e = c + span; c = e
                starts_ref[j]=s; ends_ref[j]=e
        last_j, last_s, last_e = known[-1]
        tail_block = list(range(last_j+1, R))
        if tail_block:
            remaining = max(0, T - (last_e if last_e is not None else 0))
            lens = [max(1,len(_norm_fa_min(ref_words[j]))) for j in tail_block]
            tot  = sum(lens); c = last_e if last_e is not None else 0
            for j,L in zip(tail_block, lens):
                span = max(min_dur_steps, round((remaining * L)/max(1,tot)))
                s=c; e=c+span; c=e
                starts_ref[j]=s; ends_ref[j]=e
    else:
        lens = [max(1,len(_norm_fa_min(w))) for w in ref_words]
        tot  = sum(lens); c=0
        for j,w in enumerate(ref_words):
            span = max(min_dur_steps, round((T * lens[j])/max(1,tot)))
            s=c; e=c+span; c=e
            starts_ref[j]=s; ends_ref[j]=e
    out = []
    cur = 0
    for j,w in enumerate(ref_words):
        s = starts_ref[j] if starts_ref[j] is not None else cur
        e = ends_ref[j]   if ends_ref[j]   is not None else s + min_dur_steps
        s = max(s, cur)
        e = max(e, s + min_dur_steps)
        out.append((int(s), int(e), w, j))
        cur = e
    return out

_WORD_RE = re.compile(r"\S+")

def _norm_fa_min(s: str) -> str:
    if not s: return ""
    s = s.replace("\u0640","")
    s = s.replace("\u200c","")
    s = s.replace("\u200f","")
    s = s.replace("‌","")      
    s = s.replace("ٔ","")
    s = s.replace("\u064a", "\u06cc").replace("\u0643","\u06a9")
    return s.strip()

def _similar(a: str, b: str, th: float = 0.80) -> bool:
    if not a or not b: return False
    if a == b: return True
    return SequenceMatcher(None, a, b).ratio() >= th

def _soft_lcs_hyp2ref(hyp_words: list[str], ref_words: list[str]) -> list[int|None]:
    H, R = len(hyp_words), len(ref_words)
    dp = [[0]*(R+1) for _ in range(H+1)]
    eq = [[_similar(hyp_words[i], ref_words[j]) for j in range(R)] for i in range(H)]
    for i in range(H):
        for j in range(R):
            if eq[i][j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = dp[i+1][j] if dp[i+1][j] >= dp[i][j+1] else dp[i][j+1]
    pairs = []
    i, j = H, R
    while i > 0 and j > 0:
        if eq[i-1][j-1] and dp[i][j] == dp[i-1][j-1] + 1:
            pairs.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    hyp2ref = [None]*H
    for hi, rj in pairs:
        hyp2ref[hi] = rj
        
    last = None
    for i in range(H):
        if hyp2ref[i] is None:
            hyp2ref[i] = last
        else:
            last = hyp2ref[i]
    last = None
    for i in range(H-1, -1, -1):
        if hyp2ref[i] is None:
            hyp2ref[i] = last
        else:
            last = hyp2ref[i]
    cur = -1
    for i in range(H):
        if hyp2ref[i] is None:
            hyp2ref[i] = cur if cur >= 0 else 0
        hyp2ref[i] = max(0, min(hyp2ref[i], R-1))
        if hyp2ref[i] < cur:
            hyp2ref[i] = cur
        cur = hyp2ref[i]
    return hyp2ref

def _median_filter_int(path: list[int], k: int = 5) -> list[int]:
    if k <= 1: return path
    arr = np.asarray(path, dtype=np.int64)
    out = np.copy(arr)
    r = k//2
    for i in range(len(arr)):
        a = max(0, i-r); b = min(len(arr), i+r+1)
        out[i] = int(np.median(arr[a:b]))
    return out.tolist()

def build_segments_robust_from_att(
    text: str,
    sp,               
    pieces: list[str],
    att: torch.Tensor,
    step_hz: float = 50.0
):

    ref_matches = list(_WORD_RE.finditer(text or ""))
    ref_words = [m.group(0) for m in ref_matches]
    ref_norm  = [_norm_fa_min(w) for w in ref_words]

    word_texts, piece2word = [], []
    cur = []
    for p in pieces:
        if p in ("<s>","</s>"):
            piece2word.append(len(word_texts))
            continue
        if p.startswith("▁"):
            if cur:
                word_texts.append("".join(cur).replace("▁","").replace("␣",""))
            cur = [p]
        else:
            cur.append(p)
        piece2word.append(len(word_texts))
    if cur:
        word_texts.append("".join(cur).replace("▁","").replace("␣",""))
    hyp_words = word_texts
    hyp_norm  = [_norm_fa_min(w) for w in hyp_words]
    if not hyp_words or not ref_words:
        return []

    A = att.detach().float().cpu()
    if A.dim() == 3:
        A = A.mean(0) if A.shape[1] <= A.shape[2] else A.mean(0).t()
    elif A.dim() == 2:
        pass
    else:
        raise RuntimeError(f"Unexpected att shape: {tuple(A.shape)}")

    if A.shape[0] == len(pieces):
        A = A.t()

    A = (A + 1e-9) / (A.sum(-1, keepdims=True) + 1e-9)
    tok_idx = A.argmax(-1).tolist()
    tok_idx = _median_filter_int(tok_idx, k=5)
    S = len(pieces)
    tok_idx = [min(max(i, 0), S-1) for i in tok_idx]

    hyp_word_idx_path = [piece2word[i] for i in tok_idx]

    hyp_segments = []
    s = 0
    for t in range(1, len(hyp_word_idx_path)):
        if hyp_word_idx_path[t] != hyp_word_idx_path[t-1]:
            w_idx = hyp_word_idx_path[t-1]
            hyp_segments.append((s, t, hyp_words[w_idx], w_idx))
            s = t
    hyp_segments.append((s, len(hyp_word_idx_path), hyp_words[hyp_word_idx_path[-1]], hyp_word_idx_path[-1]))

    hyp2ref = _soft_lcs_hyp2ref(hyp_norm, ref_norm)
    segs_ref = []
    for (s, e, _w, w_idx) in hyp_segments:
        rj = hyp2ref[w_idx]
        if not (0 <= rj < len(ref_words)):
            continue
        if segs_ref and segs_ref[-1][3] == rj:
            ps, pe, pw, pj = segs_ref[-1]
            segs_ref[-1] = (ps, max(pe, e), ref_words[rj], rj)
        else:
            segs_ref.append((s, e, ref_words[rj], rj))

    matched_ref = len({r for *_, r in segs_ref})
    if matched_ref < max(1, int(0.5*len(ref_words))):
        T = len(hyp_word_idx_path)
        lens = [max(1, len(_norm_fa_min(w))) for w in ref_words]
        tot = sum(lens)
        c = 0
        segs_ref = []
        for j, w in enumerate(ref_words):
            L = int(round(T * (lens[j]/tot)))
            s = c; e = min(T, c + max(1, L))
            segs_ref.append((s, e, w, j))
            c = e
        if segs_ref: segs_ref[-1] = (segs_ref[-1][0], T, segs_ref[-1][2], segs_ref[-1][3])

    return segs_ref

def concat_with_crossfade(chunks, sr: int, crossfade_ms: float = 20.0):
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1 or crossfade_ms <= 0:
        return np.concatenate(chunks, axis=0)

    fade_samp = int(round(sr * (crossfade_ms / 1000.0)))
    fade_samp = max(1, min(fade_samp, min(len(ch) for ch in chunks) // 4 or 1))

    out = chunks[0].astype(np.float32, copy=True)
    for i in range(1, len(chunks)):
        a = out
        b = chunks[i].astype(np.float32, copy=False)

        a_keep = a[:-fade_samp] if fade_samp < len(a) else np.zeros(0, dtype=np.float32)
        a_tail = a[-fade_samp:] if fade_samp <= len(a) else a
        b_head = b[:fade_samp]
        b_rest = b[fade_samp:] if fade_samp < len(b) else np.zeros(0, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, num=len(a_tail), dtype=np.float32)
        mixed = (1.0 - ramp) * a_tail + ramp * b_head
        out = np.concatenate([a_keep, mixed, b_rest], axis=0)
    return out