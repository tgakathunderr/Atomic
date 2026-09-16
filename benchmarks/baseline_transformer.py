import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from atomic.config import AtomicConfig

class StandardMHA(nn.Module):
    """Standard Softmax Multi-Head Attention with KV Caching."""
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, H, T, d_head)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, H, T, d_head)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, H, T, d_head)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v)

        total_T = k.size(2)
        # Scaled dot product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Causal mask for q tokens against k tokens
        # q is T tokens, k is total_T tokens.
        if T > 1:
            mask = torch.tril(torch.ones(T, total_T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v) # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out), new_kv_cache


class StandardTransformerBlock(nn.Module):
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = StandardMHA(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ffn),
            nn.GELU(),
            nn.Linear(config.d_ffn, config.d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class StandardTransformerLM(nn.Module):
    """GPT-2 style standard causal Transformer for benchmarking comparisons."""
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        max_pos = max(config.max_seq_len, 16384)
        self.pos_emb = nn.Embedding(max_pos, config.d_model)
        self.blocks = nn.ModuleList([StandardTransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T = input_ids.shape
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
            pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        else:
            prev_len = kv_caches[0][0].size(2) if kv_caches[0] is not None else 0
            pos = torch.arange(prev_len, prev_len + T, device=input_ids.device).unsqueeze(0)

        x = self.embeddings(input_ids) + self.pos_emb(pos)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, new_kv = block(x, kv_cache=kv_caches[i])
            new_caches.append(new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches

    def step(
        self,
        token_id: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        logits, new_caches = self.forward(token_id.unsqueeze(1), kv_caches=kv_caches)
        return logits.squeeze(1), new_caches
