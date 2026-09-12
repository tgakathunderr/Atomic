import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from atomic.config import AtomicConfig
from atomic.ops import token_shift, recurrent_glra_step, parallel_glra_scan

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - fast on CPU, zero mean-centering overhead."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class AtomicGLRA(nn.Module):
    """
    Gated Linear Recurrent Attention (GLRA).
    Operates in O(T) parallel time for training and O(1) constant time/memory for inference.
    """
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        
        # Learnable token-shift mixing vectors initialized smoothly
        # Using sigmoid parametrization for stability in (0, 1)
        self.mu_q = nn.Parameter(torch.full((self.d_model,), 0.5))
        self.mu_k = nn.Parameter(torch.full((self.d_model,), 0.5))
        self.mu_v = nn.Parameter(torch.full((self.d_model,), 0.5))
        self.mu_g = nn.Parameter(torch.full((self.d_model,), 0.5))
        self.mu_r = nn.Parameter(torch.full((self.d_model,), 0.5))

        # Linear projections
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.w_g = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_r = nn.Linear(self.d_model, self.d_model, bias=True)
        
        # Output projection
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Per-head RMSNorm for output stabilization
        self.head_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)

        self._init_weights()

    def _init_weights(self):
        # Initialize decay bias to span diverse timescales from fast decay to long retention
        # softplus(b_g) in [-log(decay_max), -log(decay_min)]
        log_decay_min = -math.log(self.config.decay_min)
        log_decay_max = -math.log(self.config.decay_max)
        # Linear interpolation across channels for multi-scale decay horizons
        init_decay = torch.linspace(log_decay_max, log_decay_min, self.d_model)
        # inverse softplus: log(exp(y) - 1)
        inv_softplus = torch.log(torch.exp(init_decay) - 1.0)
        self.w_g.bias.data.copy_(inv_softplus)
        nn.init.zeros_(self.w_g.weight)

        # Initialize receptance bias to 0
        nn.init.zeros_(self.w_r.bias)

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Allocate initial state matrix: (B, H, dk, dv)."""
        return torch.zeros(
            batch_size, self.n_heads, self.d_head, self.d_head,
            device=device, dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        prev_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parallel sequence forward pass for training.
        
        Args:
            x: Input tensor. Shape (B, T, d_model).
            state: Optional prior state (B, H, dk, dv).
            prev_x: Optional previous token embedding (B, d_model).
            
        Returns:
            out: (B, T, d_model)
            final_state: (B, H, dk, dv)
            last_x: (B, d_model)
        """
        B, T, D = x.shape
        last_x = x[:, -1]

        # 1. Token shifting
        xq = token_shift(x, torch.sigmoid(self.mu_q), prev_x)
        xk = token_shift(x, torch.sigmoid(self.mu_k), prev_x)
        xv = token_shift(x, torch.sigmoid(self.mu_v), prev_x)
        xg = token_shift(x, torch.sigmoid(self.mu_g), prev_x)
        xr = token_shift(x, torch.sigmoid(self.mu_r), prev_x)

        # 2. Linear projections & reshape to heads
        # q, k: (B, T, H, dk), v: (B, T, H, dv)
        q = self.w_q(xq).view(B, T, self.n_heads, self.d_head)
        k = self.w_k(xk).view(B, T, self.n_heads, self.d_head)
        v = self.w_v(xv).view(B, T, self.n_heads, self.d_head)

        # Data-dependent decay: g = exp(-softplus(w_g(xg))) in (0, 1)
        gamma = F.softplus(self.w_g(xg)).view(B, T, self.n_heads, self.d_head)
        g = torch.exp(-gamma)

        # Receptance gate: r = sigmoid(w_r(xr))
        r = torch.sigmoid(self.w_r(xr))  # (B, T, D)

        # 3. Parallel associative scan
        u, final_state = parallel_glra_scan(q, k, v, g, init_state=state)  # u: (B, T, H, d_head)

        # 4. Normalize per-head and combine
        u_norm = self.head_norm(u).view(B, T, D)  # (B, T, D)

        # 5. Gating and output projection
        gated = r * u_norm
        out = self.dropout(self.w_o(gated))

        return out, final_state, last_x

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        prev_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        O(1) Recurrent step for fast streaming CPU inference.
        
        Args:
            x_t: Single token tensor (B, d_model).
            state: Prior state (B, H, dk, dv).
            prev_x: Previous token (B, d_model).
            
        Returns:
            out_t: (B, d_model)
            new_state: (B, H, dk, dv)
            curr_x: (B, d_model)
        """
        B, D = x_t.shape
        curr_x = x_t

        # 1. Token shift
        xq = token_shift(x_t, torch.sigmoid(self.mu_q), prev_x)
        xk = token_shift(x_t, torch.sigmoid(self.mu_k), prev_x)
        xv = token_shift(x_t, torch.sigmoid(self.mu_v), prev_x)
        xg = token_shift(x_t, torch.sigmoid(self.mu_g), prev_x)
        xr = token_shift(x_t, torch.sigmoid(self.mu_r), prev_x)

        # 2. Linear projections
        q = self.w_q(xq).view(B, self.n_heads, self.d_head)
        k = self.w_k(xk).view(B, self.n_heads, self.d_head)
        v = self.w_v(xv).view(B, self.n_heads, self.d_head)

        gamma = F.softplus(self.w_g(xg)).view(B, self.n_heads, self.d_head)
        g = torch.exp(-gamma)

        r = torch.sigmoid(self.w_r(xr))  # (B, D)

        # 3. Recurrent step
        u, new_state = recurrent_glra_step(q, k, v, g, state)  # u: (B, H, d_head)

        # 4. Normalize per-head
        u_norm = self.head_norm(u).view(B, D)

        # 5. Gate & project
        gated = r * u_norm
        out_t = self.dropout(self.w_o(gated))

        return out_t, new_state, curr_x


class AtomicFFN(nn.Module):
    """SwiGLU Feed-Forward Network with zero-cost Token Shifting."""
    def __init__(self, config: AtomicConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ffn = config.d_ffn

        self.mu_ffn = nn.Parameter(torch.full((self.d_model,), 0.5))
        self.w_gate = nn.Linear(self.d_model, self.d_ffn, bias=config.bias)
        self.w_up = nn.Linear(self.d_model, self.d_ffn, bias=config.bias)
        self.w_down = nn.Linear(self.d_ffn, self.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        prev_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence forward pass."""
        last_x = x[:, -1]
        x_shifted = token_shift(x, torch.sigmoid(self.mu_ffn), prev_x)
        # SwiGLU: silu(gate) * up
        gate = F.silu(self.w_gate(x_shifted))
        up = self.w_up(x_shifted)
        hidden = gate * up
        out = self.dropout(self.w_down(hidden))
        return out, last_x

    def step(
        self,
        x_t: torch.Tensor,
        prev_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent inference step."""
        curr_x = x_t
        x_shifted = token_shift(x_t, torch.sigmoid(self.mu_ffn), prev_x)
        gate = F.silu(self.w_gate(x_shifted))
        up = self.w_up(x_shifted)
        hidden = gate * up
        out = self.dropout(self.w_down(hidden))
        return out, curr_x


class AtomicBlock(nn.Module):
    """
    Atomic Transformer-Alternative Block:
    Pre-RMSNorm -> AtomicGLRA -> Residual -> Pre-RMSNorm -> AtomicFFN -> Residual.
    """
    def __init__(self, config: AtomicConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.ln_1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.glra = AtomicGLRA(config)
        
        self.ln_2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = AtomicFFN(config)

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return self.glra.init_state(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        prev_glra_x: Optional[torch.Tensor] = None,
        prev_ffn_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full sequence forward."""
        # 1. GLRA sub-layer with Pre-RMSNorm
        norm_x1 = self.ln_1(x)
        glra_out, new_state, last_glra_x = self.glra(norm_x1, state=state, prev_x=prev_glra_x)
        x = x + glra_out

        # 2. FFN sub-layer with Pre-RMSNorm
        norm_x2 = self.ln_2(x)
        ffn_out, last_ffn_x = self.ffn(norm_x2, prev_x=prev_ffn_x)
        x = x + ffn_out

        return x, new_state, last_glra_x, last_ffn_x

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        prev_glra_x: Optional[torch.Tensor] = None,
        prev_ffn_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent inference step."""
        # 1. GLRA sub-layer
        norm_x1 = self.ln_1(x_t)
        glra_out, new_state, curr_glra_x = self.glra.step(norm_x1, state=state, prev_x=prev_glra_x)
        x_t = x_t + glra_out

        # 2. FFN sub-layer
        norm_x2 = self.ln_2(x_t)
        ffn_out, curr_ffn_x = self.ffn.step(norm_x2, prev_x=prev_ffn_x)
        x_t = x_t + ffn_out

        return x_t, new_state, curr_glra_x, curr_ffn_x
