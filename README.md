# LivoTTS

**LivoTTS** is a hybrid Persian TTS stack that marries a **Lina-style autoregressive (AR) text→token model** with **Vevo’s flow-matching acoustic decoder** and a neural **vocoder**.
The AR model predicts content/style tokens compatible with Vevo’s `Vq8192ToMels` FM model; the FM model then synthesizes mel-spectrograms that are rendered to waveform by the vocoder.

> TL;DR: **Lina AR front-end → Vevo FM (VQ8192→mel) → Vocoder**, with an **IST speaker bank** for timbre control and a **50 Hz AR frame rate**.

---

## 📐 Architecture

```

                               │ Text (SPM 8k)
                               ▼
                ┌────────────────────────────────────────────────────────────────────┐
                │  Autoregressive TTS (Lina-style)                                   │
                │  • TextEncoder (rotary) + Attentive RNN (fused recurrent)          │
                │  • Generates VQ8192 content/style codes @ 50 Hz                    │
                └──────────────┬─────────────────────────────────────────────────────┘
                               │ codes (VQ8192), attn, states
                               ▼
                ┌────────────────────────────────────────────────────────────────────┐
                │  Flow Matching Decoder (Vevo VC: Vq8192→Mels)                      │
                │  • CFG-enabled reverse diffusion                                   │
                │  • Produces mel-spectrograms                                       │
                └──────────────┬─────────────────────────────────────────────────────┘
                               │ mels
                               ▼
                ┌────────────────────────────────────────────────────────────────────┐
                │  Vocoder (e.g., Vocos/BigVGAN)                                     │
                └──────────────┬─────────────────────────────────────────────────────┘
                               │ waveform (24 kHz)
                               ▼
                            Audio (WAV)

```
---

### Change: Replacing WavTokenizer with VEVO in Lina Speech TTS

Lina Speech TTS originally used **WavTokenizer** to train the acoustic model. While this pipeline performs well on English, we observe **limited cross-lingual robustness**: on Persian, it yields **unacceptable intelligibility (higher WER)** and **noticeably degraded timbre fidelity**.

**Observed with WavTokenizer (in our setting)**

* Strong English reconstructions, but **poor transfer to Persian**.
* **Higher WER** on Persian test sets and **muffled/washed timbre**, especially on fricatives and high-frequency detail.
* Longer token sequences (75 Hz) leading to **more autoregressive steps**, more opportunities for exposure errors, and higher compute.

**What VEVO brings**

* **Better perceptual quality**: sharper reconstructions, clearer consonants, and **more faithful speaker timbre** across languages.
* **Shorter sequences**: VEVO runs at **50 Hz** (vs. 75 Hz), which is ~**33% fewer frames/tokens** for the same duration.
* **Fewer AR iterations** → **lower latency** and **less error accumulation** during generation.
* **Lower compute cost** end-to-end: fewer tokens to store/process, **faster training and inference**, and smaller memory footprint.
* **More stable cross-lingual behavior** in our tests, with **materially lower WER** and more consistent timbre on Persian.

**Practical impact**

* **Higher quality** with **less compute**.
* **Shorter training time** and **faster inference**.
* **Cleaner, more natural timbre** on Persian (the target application), with improved intelligibility.

---



### 🔊 Reconstruction — GT vs **VEVO (50 Hz)** vs **WavTokenizer (75 Hz)**

| Sample | GT                                      | VEVO (50 Hz)                                | WavTokenizer (75 Hz)                            |
| :----- | :-------------------------------------- | :------------------------------------------ | :---------------------------------------------- |
| s1     | 🔊 [s1_gt.wav](audio/compare/s1_gt.wav) | 🔊 [s1_vevo.wav](audio/compare/s1_vevo.wav) | 🔊 [s1_wavtok.wav](audio/compare/s1_wavtok.wav) |
| s2     | 🔊 [s2_gt.wav](audio/compare/s2_gt.wav) | 🔊 [s2_vevo.wav](audio/compare/s2_vevo.wav) | 🔊 [s2_wavtok.wav](audio/compare/s2_wavtok.wav) |
| s3     | 🔊 [s3_gt.wav](audio/compare/s3_gt.wav) | 🔊 [s3_vevo.wav](audio/compare/s3_vevo.wav) | 🔊 [s3_wavtok.wav](audio/compare/s3_wavtok.wav) |

---

### Reason 2: Zero-shot voice conversion (VC) with VEVO

A second key reason for moving to VEVO is its built-in **zero-shot voice conversion** capability.

**Why this matters**

* **Instant timbre swapping at inference**: condition on a **short reference** (seconds) and convert the source content into the target timbre—**no fine-tuning** required.
* **Privacy & safety**: easy to **avoid using a real person’s voice**; synthesize or select a neutral reference timbre to mitigate legal and security concerns.
* **Broader creative range**: quickly explore multiple voices, accents, or styles from the **same source** utterance.
* **Operational efficiency**: zero-shot conditioning simplifies deployment—**one model** handles TTS and VC, reducing model maintenance and serving complexity.
* **Rapid iteration**: great for **A/B** voice selection, content moderation pipelines, or compliant voice replacement in production.
---
### 🔊 VEVO zero-shot Voice Conversion results on Persian**

| Case | Source (content)                     | Converted timbres                                                                                                                                                                                            |
| :--- | :----------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1   | 🔊 [src.wav](audio/vc/v1_src.wav) | 🔊 [timbre-a.wav](audio/vc/v1_vevo_timbre-a.wav) • 🔊 [timbre-b.wav](audio/vc/v1_vevo_timbre-b.wav) • 🔊 [timbre-c.wav](audio/vc/v1_vevo_timbre-c.wav) |
---
### Change: Speaker-specific **initial-state training (IST)** integrated into main training (replaces post-hoc fine-tuning)

We moved Lina Speech’s “initial-state fine-tuning” into the **core training loop** and made the **per-speaker initial state learnable**. This better fits our data regime (few speakers, uneven hours) and avoids brittle, slow post-hoc fine-tunes.

* The model now learns a **speaker-conditional initial state** for the AR module (the state used to prime the Attentive RNN/GLA before decoding).
* These **IST parameters are trained jointly** with the trunk on the primary loss—no separate fine-tune stage.
* Training batches carry a `spk_id`; the model maps each `spk_id → init_state` and passes it to the AR at every forward pass.

**Why this is better for our dataset**

* With **few speakers**, global weights can learn shared acoustics while **IST captures per-speaker dynamics/timbre** without overfitting the whole trunk.
* Eliminates an extra pipeline step: **no per-voice fine-tune**, fewer checkpoints to manage, simpler resumption.
* Improves **timbre consistency** and **stability** for low-resource speakers while keeping compute low.

**How it works (conceptually)**

* A small **Speaker IST Bank** holds learnable tensors that parameterize the AR’s initial hidden/cache state.
* On each batch: `init_state = ISTBank.build_cache(attentive_rnn, spk_ids, runtime_scale)`; the AR runs from this state.
* Gradients flow into both the trunk and the IST bank; the bank can use its own LR/WD if desired.
---

### Change: Gated **SwiGLU** projection between Text Encoder and AR (decoupled dims)

We inserted a **SwiGLU-based projection** between the text encoder and the AR/cross-attention stack. Instead of feeding the encoder’s embeddings directly, we first pass them through a **gated SwiGLU down-projection** (`txt_dim → d_model`). This both **filters** and **compresses** the textual features before alignment, which matched our observation that **text encoding quality drives TTS more than the AR mechanism itself**—scaling the encoder (more heads/layers) helped, but we didn’t want to balloon the AR.

In practice this:

* **Decouples model dims**: the text encoder can use a larger `txt_dim`, while the AR runs with a leaner `d_model` (now configurable and different), reducing AR compute without sacrificing encoder capacity.
* Acts as a **learned gate** that stabilizes cross-attention and improves articulation/timbre (especially on Persian), while keeping memory and latency in check.

Net effect: we can **scale the text encoder aggressively** for better linguistic detail, and keep the AR efficient—thanks to the **SwiGLU projection** that bridges the two spaces cleanly.

---

## Installation

Please see the step-by-step **Installation**, **Training**, and **Inference** guides in the **[scripts](./scripts/)** folder.



## 🙏 Acknowledgements

* **Lina Speech** – AR TTS design and training recipes.
* **Amphion/Vevo** – VQ8192 tokenizer, Flow-Matching `Vq8192ToMels`, and vocoder components.
* SentencePiece, Accelerate, PyTorch, and the broader OSS community.


## Roadmap

* Pitch/prosody conditioning & ref-style prompts
* Streaming AR with low-latency FM refinement
* Multi-speaker finetuning & timbre reference from short clips

---
