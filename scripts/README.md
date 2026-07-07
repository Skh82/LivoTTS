## 0) Setup env and install Dependencies

```bash
cd LivoTTS
conda create -n livo python=3.11.13 -y
conda activate livo

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install wheel https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post4/causal_conv1d-1.5.0.post4+cu124torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install -r requirements.txt
ln -s src/3rdparty/flash-linear-attention/fla fla
ln -s src/3rdparty/encoder encoder
ln -s src/3rdparty/decoder decoder
```


## 1) Prepare your metadata

The preprocessor expects a **JSONL** file (one JSON object per line). Each line describes **one utterance**.
### Expected JSONL

```json
{"audio_filepath": "<absolute_path_to>/utt_0001.mp3", "text": "نمونه شماره یک", "speaker": "spk_A", "duration": 12.58}
{"audio_filepath": "<absolute_path_to>/utt_0123.mp3", "text": "نمونه شماره دو", "speaker": "spk_B", "duration": 3.21}
```

### Run the preprocessor (multi-GPU) to build shards

```bash
PYTHONPATH=. accelerate launch --num_processes 2 -m scripts.preprocessor -- \
  /path/to/your_dataset.meta.jsonl -r 0.99
```

* `--num_processes 2` → number of GPUs to use (set to your available GPU count).
* `-r 0.99` → **train split ratio** (here: 99% train / 1% eval).

This will generate the shards needed for training in the `assets/shards/` folder, which will be automatcly picked up by the trainer code.

```
assets/
└─ shards/
    ├─ shard-0000.meta.jsonl
    └─ shard-0000.tokens.u16
```


## 2) Train & use a Persian BPE tokenizer (no G2P)

We don’t use G2P. We tokenize **Persian text directly** with SentencePiece BPE.

### Train (already implemented in `scripts/bpe_trainer.py`)

Build the corpus from your JSONL metadata and train a BPE model:

```bash
PYTHONPATH=. python -m scripts.bpe_trainer --inputs \
  path-to-your-shards \
  --out_text assets/other/fa_corpus.txt \
  --model_prefix assets/other/fa_bpe \
  --vocab_size 8000 \
  --num_threads 8
```

Artifacts:

```
assets/other/fa_bpe_8k.model
assets/other/fa_bpe_8k.vocab
```
It needs to be placed under `assets/other/` so the model can use it for both inference and training. Text tokenization happens on the fly since its computationaly cheap. or you can use the already generated bpe model in the current repo.

---

## 3) Start training

```bash
PYTHONPATH=. accelerate launch --multi_gpu -m scripts.trainer --mixed_precision bf16
```

* Edit **`assets/cfg/config.json`** to change training hyperparameters and architectural values.
* Generated checkpoints will be saved to **`assets/checkpoints`**.
  
> ⚠️ **Note:** In our initial test, we observed that using fp16 would result in NaN loss over half an epoch, in order to gain the speed boost and avoid this problem, its recommended to use bf16 (provided your gpu supports it).

---

## 4) Inference

1. **Put your checkpoint** here:

```
./pretrained/model.safetensors
```

> The infer script will automatically pick the latest `.safetensors` in `./pretrained/`.

2. **Run inference** (single utterance + HTML report):

```bash
PYTHONPATH=. python -m scripts.infer \
  --prompt-wav /kaggle/input/allspks/1.abolfazlshah.wav \
  --text "این صدای مدل متن‌خوان فارسی است. با استفاده از دستورات ذکر شده، می‌توانید فایل صوتی خود را بسازید." \
  --spk-id 1 --top-k 20 --top-p 0.9 --temp 0.8 --fm-steps 16 \
  --out /kaggle/working/utt.wav \
  --checkpoint_path <absolute_path_to_the_folder_containing_safetensors>
```


### Notes

* `--prompt-wav`: short reference of the target voice (VEVO zero-shot VC).
* `--prompt-text`: the text of the prompt-wav which will be used for zero-shot TTS.
* `--spk-id`: used if your config enables the speaker IST bank; ignored otherwise.
* `--top-k / --top-p / --temp`: sampling controls (higher = more diverse).
* `--fm-steps`: flow-matching steps (quality/speed trade-off).
* `--ist_params`: Use this after IST (see next part)


## 5) Initial state finetuning
Following Lina speech, we implement IST as well, a technique to adapt the model to a single speaker or style without touching the weights of the model it self. Use the code below to train the initial state after you have fully trained the main model.


```bash
PYTHONPATH=. python -m scripts.train_ist \
  --config assets/cfg/config.json \
  --ckpt   assets/checkpoints/checkpoint_8/model.safetensors \
  --out    assets/ist/all_emotion_H.safetensors \
  --shards_dir /media/eri-4090/HDD/Saeed/LivoTTS/assets/ist_shards \
  --epochs 10 \
  --batch_size 8 \
  --grad_acc 1 \
  --lr 0.125 \
  --rank 1 \
  --scale 0.02 \
  --device cuda
```

* The code will save the initial state safetensors under the chosen dir, which will be used later during inference.
* IST shards should be generated the same way we did with our main dataset.
* Rank and scale should be left untouched, the rest of the parameters like epochs and bache_size should be further tuned for optimal results.

---

## 6) Results
https://github.com/user-attachments/assets/a77ff69c-6391-47fe-9595-b836164ab8a1
