import torch
import torch.nn as nn
from einops import einsum

class _LayerIST(nn.Module):
    """
    One layer's IST for all speakers.
    Supports either direct seeds (w) or LoRA-style (k,v) factors,
    matching Lina's get_init_state_tuning_params output.
    """
    def __init__(self, template_item, num_speakers: int):
        super().__init__()
        if isinstance(template_item, tuple) and len(template_item) == 2:  # LoRA path
            k, v = template_item  # shapes: [1, R, H, K, 1], [1, R, H, 1, V]
            # Stack a speaker axis S at the front (trainable)
            self.k = nn.Parameter(k.detach().clone().expand(num_speakers, -1, -1, -1, -1).contiguous())
            self.v = nn.Parameter(v.detach().clone().expand(num_speakers, -1, -1, -1, -1).contiguous())
            self.is_lora = True
        else:  # direct path
            w = template_item if isinstance(template_item, torch.Tensor) else template_item[0]
            # shape: [1, H, K, V]  → add S
            self.w = nn.Parameter(w.detach().clone().expand(num_speakers, -1, -1, -1).contiguous())
            self.is_lora = False

    def build_state(self, spk_ids: torch.LongTensor, scale: float):
        if self.is_lora:
            # Select each example's factors; remove the leading singleton dim
            k_b = self.k[spk_ids].squeeze(1)  # [B, R, H, K, 1]
            v_b = self.v[spk_ids].squeeze(1)  # [B, R, H, 1, V]
            state = einsum(k_b, v_b, "b r h k vv, b r h kk v -> b h k v") * scale
        else:
            w_b = self.w[spk_ids]
            state = w_b.squeeze(1) if w_b.dim() == 5 else w_b  # [B, H, K, V]
        return state

class SpeakerISTBank(nn.Module):
    """
    A bank of initial states for all layers and speakers.
    """
    def __init__(self, attentive_rnn, num_speakers: int, lora_rank=None, init_scale: float=0.02, device="cuda"):
        super().__init__()
        template = attentive_rnn.get_init_state_tuning_params(lora=lora_rank, scale=init_scale, device=device)
        self.layers = nn.ModuleList([_LayerIST(item, num_speakers) for item in template])

    @torch.no_grad()
    def _empty_cache(self, attentive_rnn, batch_size: int):
        return attentive_rnn.init_state(batch_size=batch_size)

    def build_cache(self, attentive_rnn, spk_ids, runtime_scale=0.02, prev_state=None, memory_scale=0.9):
        """
        Build a per-example init_state cache for a mixed-speaker batch.
        """
        B = int(spk_ids.size(0))
        cache = self._empty_cache(attentive_rnn, batch_size=B)
        for i, layer in enumerate(self.layers):
            if prev_state == None:
                state = layer.build_state(spk_ids, scale=runtime_scale)  # [B, H, K, V]
                cache.states[i] = cache.states[i][:-1] + (state.clone(),)
            else:
                state = layer.build_state(spk_ids, scale=runtime_scale)  # [B, H, K, V]
                state = memory_scale * state + (1-memory_scale) * prev_state.states[i][-1]
                cache.states[i] = cache.states[i][:-1] + (state.clone(),)
        return cache
