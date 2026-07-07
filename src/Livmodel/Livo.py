import json, sys, torch
from src.Livmodel.attentive_rnn import AttentiveRNN
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import EinMix
from src.Livmodel.attentive_rnn import AttentiveRNN
from src.Livmodel.multiembed import MultiEmbedding
from torch import Tensor, nn
from typing import List, Optional
from src.Livmodel.encoder import SwiGLUProj
from typing import Optional, Any, Dict, Union, List

from src.Livmodel.encoder import SwiGLUProj, TextEncoder
from src.Livmodel.attentive_rnn import AttentiveRNN
from src.Livmodel.gla import AttentiveGLA
from src.Livmodel.multiembed import MultiEmbedding
from src.Livmodel.ist_bank import SpeakerISTBank
from src.Livmodel.tools import undelay_rvq

def exists(x):
    return x is not None
class LivoModel(nn.Module):
    def __init__(
        self,
        attentive_rnn: AttentiveRNN,
        d_model: int,
        txt_dim: int,
        n_quant: int,
        n_codebook: int,
        n_special_token_in: int,
        n_special_token_out: int,
        n_txt_vocab: int,
        tie_embed: bool = False,
        txt_encoder: Optional[nn.Module] = None,
        spk_encoder: Optional[nn.Module] = None,
        spk_bank: Optional[nn.Module] = None,
        mask_text_p: float = 0.,
    ):
        super(LivoModel, self).__init__()

        self.n_quant = n_quant
        self.n_codebook = n_codebook
        self.n_special_token_in = n_special_token_in
        self.n_special_token_out = n_special_token_out
        self.mask_text_p = mask_text_p
        self.n_txt_vocab = n_txt_vocab + int(mask_text_p > 0.)
        self.n_target_vocab = n_codebook + n_special_token_out

        self.txt_encoder = txt_encoder
        self.spk_encoder = spk_encoder
        self.spk_bank = spk_bank
        self.attentive_rnn = attentive_rnn
        
        if txt_dim != d_model:
            self.text_downproj = SwiGLUProj(in_dim=txt_dim, out_dim=d_model) # the dim of text encoder and the AR no longer needs to be the same
        else:
            self.text_downproj = None
              
        self.txt_embed = nn.Embedding(
            n_txt_vocab,
            txt_dim,
            padding_idx=0,
        )
        self.rvq_embed = MultiEmbedding(self.n_quant, n_codebook + n_special_token_in, d_model, padding_idx=0) 
        self.logits_head = EinMix(
            "b n d -> b n q l",
            weight_shape="q l d",
            q=self.n_quant,
            d=d_model,
            l=self.n_target_vocab,
        )
        if tie_embed:
            self.logits_head.weight = self.rvq_embed.weight
    def forward(self, x, y, spk_ids, encoder_mask, crossatt_mask, logits_mask=None, attention_only=False, forced_attention=None, crossatt_pos=None, init_state=None):
        if self.mask_text_p > 0.:
            mask = torch.empty(x.shape[0]).bernoulli_(self.mask_text_p)
            x[mask] = self.n_txt_vocab - 1
            
        if self.spk_bank is not None:
            init_state = self.spk_bank.build_cache(self.attentive_rnn, spk_ids, 0.02)

        x_embd = self.txt_embed(x)
        y_embd = self.rvq_embed(rearrange(y, "b n q -> q b n"))
        q, b, n, d = y_embd.shape
        y_embd = reduce(y_embd, "q b n d -> b n d", "sum", q=q)
        
        x_enc = self.txt_encoder(x_embd, mask=encoder_mask)
        if self.text_downproj is not None:
            x_enc = self.text_downproj(x_enc)

        if self.spk_encoder is not None:
            spk_embd = self.spk_encoder(y_embd)
            y_embd[:,0] = spk_embd

        y_hat, att = self.attentive_rnn(
            y_embd[:, :-1, :],
            x_enc,
            mask=crossatt_mask[:,:-1],
            forced_attention=forced_attention[:,:,:y_embd.shape[1]-1] if forced_attention is not None else None,
            attention_only=attention_only,
            init_state=init_state,
            crossatt_pos=crossatt_pos,
        )
        if attention_only:
            return att
        logits = self.logits_head(y_hat)
        if logits_mask is not None:
            masked_logits = logits[logits_mask[:, 1:], :, :]
            masked_target = y[:, 1:][logits_mask[:, 1:], :]
            flat_logits = rearrange(masked_logits, "n q l -> (n q) l")
            flat_target = rearrange(masked_target, "n q   -> (n q)")
        else:
            masked_logits = logits 
            masked_target = y[:, 1:]
            flat_logits = rearrange(masked_logits, "b n q l -> (b n q) l")
            flat_target = rearrange(masked_target, "b n q   -> (b n q)")

        loss = F.cross_entropy(flat_logits, flat_target, ignore_index=1)

        return logits, loss, att, masked_logits, masked_target

    @torch.inference_mode()
    def generate_batch(
        self,
        x: Tensor,
        batch_size: int = 3,
        prompt: Optional[Tensor] = None,
        device: str = "cpu",
        max_seqlen: int = 1500,
        k: int = 25,
        spk_ids: Optional[dict] = None,
        prev_state: Optional[dict] = None,
        init_state: Optional[dict] = None,
        first_greedy_quant: int = 1,
        temp: float = 1.0,
        force_max_seqlen: bool = False,
        top_p: float = 0.0,             
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0, 
        eos_id: int = 2,                     
        min_seqlen: int = 0,            
        no_repeat_ngram_size: int = 0
    ):

        def _filter_top_k(logits: Tensor, kth: int) -> Tensor:
            if kth is None or kth <= 0 or kth >= logits.size(-1):
                return logits
            values, _ = torch.topk(logits, kth, dim=-1)
            kth_values = values[..., -1, None]
            return torch.where(logits >= kth_values, logits, torch.full_like(logits, float('-inf')))

        def _filter_top_p(logits: Tensor, p: float) -> Tensor:
            if p is None or p <= 0.0 or p >= 1.0:
                return logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cprobs = torch.cumsum(probs, dim=-1)

            cutoff = (cprobs > p).float()
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = 0.0
            sorted_logits = torch.where(cutoff.bool(), torch.full_like(sorted_logits, float('-inf')), sorted_logits)

            unsorted = torch.full_like(sorted_logits, float('-inf'))
            unsorted.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            return unsorted

        def _apply_penalties(logits_q: Tensor,
                            token_counts_q: Optional[Tensor],
                            rep_pen: float,
                            freq_pen: float) -> Tensor:
            if token_counts_q is None:
                return logits_q
            out = logits_q
            if freq_pen and freq_pen != 0.0:
                out = out - (freq_pen * token_counts_q)

            if rep_pen and rep_pen != 1.0:
                seen = (token_counts_q > 0)
                pos = seen & (out > 0)
                neg = seen & (out <= 0)
                out = torch.where(pos, out / rep_pen, out)
                out = torch.where(neg, out * rep_pen, out)

            return out

        def _ban_ngrams(logits_q: Tensor,
                        history_q: List[List[int]],
                        n: int) -> Tensor:

            if n is None or n <= 1:
                return logits_q
            B, L = logits_q.shape
            if len(history_q) < 1:
                return logits_q
            out = logits_q.clone()
            for b in range(B):
                hist = history_q[b]
                if len(hist) < (n - 1):
                    continue
                prefix = tuple(hist[-(n - 1):])
                forb = set()
                for i in range(len(hist) - n + 1):
                    if tuple(hist[i:i + n - 1]) == prefix:
                        forb.add(hist[i + n - 1])
                if forb:
                    idx = torch.tensor(list(forb), device=out.device, dtype=torch.long)
                    out[b, idx] = float('-inf')
            return out

        x = repeat(x, "n -> b n", b=batch_size).to(device)
        stop_token = torch.ones(self.n_quant, 1, 1, device=device) * eos_id
        all_stop_token = torch.zeros(batch_size, 1, device=device).bool()
        y_start = torch.ones(self.n_quant, batch_size, 1, device=device).long()

        x_embd = self.txt_embed(x)
        y_embd = reduce(self.rvq_embed(y_start), "q b n d -> b n d", "sum")

        p_len = -1
        if exists(prompt):
            if prompt.shape[1] != batch_size:
                prompt = repeat(prompt, "q 1 n -> q b n", b=batch_size) + 3
            else:
                prompt = prompt + 3
            prompt = reduce(self.rvq_embed(prompt.to(device)), "q b n d -> b n d", "sum")
            p_len = prompt.shape[1]
            
        def _gen_steps_after_prompt(t: int) -> int:
            if not exists(prompt) or p_len < 0:
                return t + 1
            return max(0, (t + 1) - p_len)

        if self.spk_encoder is not None:
            spk_embd = self.spk_encoder(prompt)
            prompt[:, 0] = spk_embd

        x_enc = self.txt_encoder(x_embd)
        if self.text_downproj is not None:
            x_enc = self.text_downproj(x_enc)
        
        if self.spk_bank is not None:
            print('Using IST Bank')
            init_state = self.spk_bank.build_cache(
            attentive_rnn=self.attentive_rnn,
            spk_ids=spk_ids,
            runtime_scale=0.02,
            prev_state=prev_state
        )
        else: 
            init_state = init_state

        state = init_state if init_state is not None else self.attentive_rnn.init_state(
            max_seqlen=max_seqlen, batch_size=batch_size
        )

        qs, atts, stop_tokens = [], [], []
        token_counts = None
        histories = [[[] for _ in range(batch_size)] for _ in range(self.n_quant)]

        nobat=0
        for t in range(max_seqlen):
            y_embd, att, state = self.attentive_rnn.step(y_embd, x_enc, t, state)
            atts.append(att)

            logits = self.logits_head(y_embd)
            logits = rearrange(logits, "b 1 q l -> q b l")
            q_sampled = []
            
            if token_counts is None:
                L = logits.size(-1)
                token_counts = [torch.zeros(batch_size, L, device=device) for _ in range(self.n_quant)]


            for qi, q_logits in enumerate(logits):
                
                gen_steps = _gen_steps_after_prompt(t)
                if gen_steps < min_seqlen:
                    q_logits[:, eos_id] = float("-inf")
                
                q_logits = _apply_penalties(q_logits, token_counts[qi], repetition_penalty, frequency_penalty)

                if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                    q_logits = _ban_ngrams(q_logits, histories[qi], no_repeat_ngram_size)

                if temp != 1.0 and temp > 0:
                    q_logits = q_logits / temp

                q_filtered = _filter_top_k(q_logits, k if k is not None else 0)
                q_filtered = _filter_top_p(q_filtered, top_p)
                
                if qi < first_greedy_quant:
                    probs = F.softmax(q_filtered, dim=-1)
                    is_bad = torch.isinf(q_filtered).all(dim=-1)
                    if is_bad.any():
                        fallback = F.softmax(q_logits, dim=-1)
                        next_ids = torch.multinomial(fallback, num_samples=1)
                    else:
                        next_ids = torch.multinomial(probs, num_samples=1)
                else:
                    next_ids = torch.argmax(q_filtered, dim=-1, keepdim=True)
                    
                token_counts[qi].scatter_add_(1, next_ids, torch.ones_like(next_ids, dtype=token_counts[qi].dtype))
                for b in range(batch_size):
                    histories[qi][b].append(int(next_ids[b, 0].item()))

                q_sampled.append(next_ids)

            q_sampled = torch.stack(q_sampled)
            qs.append(q_sampled)

            is_stop_token = (q_sampled == stop_token).prod(dim=0)
            gen_steps = _gen_steps_after_prompt(t)
            if gen_steps < min_seqlen:
                is_stop_token = torch.zeros_like(is_stop_token)
                
            stop_tokens.append(is_stop_token)
            all_stop_token.logical_or_(is_stop_token)

            if all_stop_token.prod() and not force_max_seqlen:
                break
            
            if exists(prompt) and p_len / 4 < t < p_len:
                y_embd = prompt[:, [t]]
            else:
                y_embd = self.rvq_embed(q_sampled)
                y_embd = reduce(y_embd, "q b n d -> b n d", "sum")

        atts = torch.cat(atts, dim=2) if exists(atts[0]) else None
        qs = torch.stack(qs, dim=2).squeeze(-1)
        stop_tokens.append(torch.ones(batch_size, 1, device=device))
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze(-1)

        b, n = stop_tokens.shape
        rvq = (undelay_rvq(qs) - self.n_special_token_in).clamp_min(0)
        stop_idx = (stop_tokens * rearrange(torch.arange(n, device=stop_tokens.device), "n -> 1 n")).long()
        cuts = []
        for i in range(stop_idx.shape[0]):
            idx = torch.unique(stop_idx[i])[1]
            cuts.append((rvq[:, [i], :idx - self.n_quant], atts[i, :, :idx] if exists(atts) else None))

        return qs, atts, stop_tokens, cuts, state

    @classmethod
    def from_json(cls, cfg: Union[str, Dict[str, Any]]):
        if isinstance(cfg, str):
            with open(cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        te = TextEncoder(**cfg["text_encoder"])

        att_cfg = cfg["attentive"]
        att_type = (att_cfg.get("type") or "GLA").upper()
        if att_type == "GLA":
            ar = AttentiveGLA(**att_cfg.get("params", {}))
        elif att_type == "RNN":
            ar = AttentiveRNN(**att_cfg.get("params", {}))
        else:
            raise ValueError(f"Unknown attentive type: {att_type}")
        
        sb_cfg = cfg.get("speaker_bank", {})
        if sb_cfg.get("enabled", False):
            sb = SpeakerISTBank(
                attentive_rnn=ar,
                num_speakers=sb_cfg.get("num_speakers", 0),
                lora_rank=sb_cfg.get("lora_rank"),
                init_scale=sb_cfg.get("init_scale", 0.02)
            )
        else: 
            sb = None
            
        mcfg = cfg["model"]
        model = cls(
            attentive_rnn=ar,
            d_model=mcfg["d_model"],
            txt_dim=mcfg["txt_dim"],
            n_quant=mcfg["n_quant"],
            n_codebook=mcfg["n_codebook"],
            n_special_token_in=mcfg["n_special_token_in"],
            n_special_token_out=mcfg["n_special_token_out"],
            n_txt_vocab=mcfg["n_txt_vocab"],
            tie_embed=mcfg.get("tie_embed", False),
            txt_encoder=te,
            spk_bank=sb,
            spk_encoder=None,
            mask_text_p=mcfg.get("mask_text_p", 0.0),
        )
        if sb_cfg.get("enabled", False):
            model.attentive_rnn.to_mode("fused_recurrent")
        else:
            model.attentive_rnn.to_mode("fused_chunk")
        
        return model
