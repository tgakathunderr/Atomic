import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def token_shift(x: torch.Tensor, mu: torch.Tensor, prev_x: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Token-shift operator for zero-FLOP spatial locality injection.
    Computes \tilde{x}_t = mu * x_t + (1 - mu) * x_{t-1}
    
    Args:
        x: Input tensor. Shape (B, T, D) or (B, D).
        mu: Mixing weights in range (0, 1). Shape (D,).
        prev_x: Previous token representation. Shape (B, D).
        
    Returns:
        Shifted tensor with same shape as x.
    """
    if x.ndim == 2:
        # Recurrent single-step mode: (B, D)
        if prev_x is None:
            prev_x = torch.zeros_like(x)
        return mu * x + (1.0 - mu) * prev_x

    elif x.ndim == 3:
        # Batched sequence mode: (B, T, D)
        B, T, D = x.shape
        if prev_x is None:
            init_prev = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        else:
            init_prev = prev_x.unsqueeze(1)
        
        if T == 1:
            shifted = init_prev
        else:
            shifted = torch.cat([init_prev, x[:, :-1]], dim=1)
        return mu * x + (1.0 - mu) * shifted

    else:
        raise ValueError(f"Expected x to have 2 or 3 dims, got {x.ndim}")


def recurrent_glra_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single recurrent step of Gated Linear Recurrent Attention (GLRA).
    Operates in O(1) time and space - fits in L1 CPU cache.
    
    Args:
        q: Query tensor. Shape (B, H, dk).
        k: Key tensor. Shape (B, H, dk).
        v: Value tensor. Shape (B, H, dv).
        g: Decay gate tensor in (0, 1). Shape (B, H, dk).
        state: Prior state matrix S_{t-1}. Shape (B, H, dk, dv).
        
    Returns:
        u: Output readout tensor. Shape (B, H, dv).
        new_state: Updated state matrix S_t. Shape (B, H, dk, dv).
    """
    # S_t = diag(g_t) * S_{t-1} + k_t^T * v_t
    # g: (B, H, dk) -> (B, H, dk, 1)
    # k: (B, H, dk, 1), v: (B, H, 1, dv) -> outer product (B, H, dk, dv)
    decayed_state = g.unsqueeze(-1) * state
    kv = k.unsqueeze(-1) * v.unsqueeze(-2)
    new_state = decayed_state + kv
    
    # Readout: u = q * S_t -> (B, H, 1, dk) @ (B, H, dk, dv) -> (B, H, 1, dv) -> (B, H, dv)
    u = torch.matmul(q.unsqueeze(-2), new_state).squeeze(-2)
    return u, new_state


def parallel_glra_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked parallel scan for Gated Linear Recurrent Attention.
    Computes exact GLRA outputs in O(T) parallel time with full causal masking
    and high numerical stability using log-space intra-chunk decays and inter-chunk state passing.
    
    Args:
        q: Queries. Shape (B, T, H, dk).
        k: Keys. Shape (B, T, H, dk).
        v: Values. Shape (B, T, H, dv).
        g: Retention decay in (0, 1). Shape (B, T, H, dk).
        init_state: Optional initial state (B, H, dk, dv).
        chunk_size: Chunk size for block-parallel scanning (default 64).
        
    Returns:
        u: Output tensor. Shape (B, T, H, dv).
        final_state: State at step T. Shape (B, H, dk, dv).
    """
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    device, dtype = q.device, q.dtype

    if init_state is None:
        state = torch.zeros(B, H, dk, dv, device=device, dtype=dtype)
    else:
        state = init_state.clone()

    # For short sequences or small chunk_size, chunked computation:
    # Within each chunk, we can compute intra-chunk linear attention + inter-chunk state propagation.
    outputs = []
    
    for start_idx in range(0, T, chunk_size):
        end_idx = min(start_idx + chunk_size, T)
        L = end_idx - start_idx
        
        q_c = q[:, start_idx:end_idx]      # (B, L, H, dk)
        k_c = k[:, start_idx:end_idx]      # (B, L, H, dk)
        v_c = v[:, start_idx:end_idx]      # (B, L, H, dv)
        g_c = g[:, start_idx:end_idx]      # (B, L, H, dk)
        
        # Intra-chunk log decay
        log_g = torch.log(g_c.clamp(min=1e-8))  # (B, L, H, dk)
        cum_log_g = torch.cumsum(log_g, dim=1)  # (B, L, H, dk)
        
        # Contribution from initial state of this chunk:
        # state * exp(cum_log_g_t) -> (B, L, H, dk, dv)
        # q_t * (state * exp(cum_log_g_t)) = (q_t * exp(cum_log_g_t)) @ state
        q_decayed = q_c * torch.exp(cum_log_g)  # (B, L, H, dk)
        # (B, L, H, 1, dk) @ (B, 1, H, dk, dv) -> (B, L, H, 1, dv) -> (B, L, H, dv)
        u_prev = torch.matmul(q_decayed.unsqueeze(-2), state.unsqueeze(1)).squeeze(-2)
        
        # Intra-chunk pairwise decay matrix:
        # for j <= t: decay_factor_{t, j} = exp(cum_log_g_t - cum_log_g_j)
        # We compute this efficiently:
        # cum_log_g: (B, L, H, dk) -> transpose to (B, H, dk, L)
        cum_log_g_t = cum_log_g.permute(0, 2, 3, 1)  # (B, H, dk, L)
        # Difference matrix: (B, H, dk, L, 1) - (B, H, dk, 1, L)
        diff_log_g = cum_log_g_t.unsqueeze(-1) - cum_log_g_t.unsqueeze(-2) # (B, H, dk, L_t, L_j)
        
        # Permute q_c, k_c, v_c for batched operations:
        # q_c: (B, H, dk, L)
        q_perm = q_c.permute(0, 2, 3, 1) # (B, H, dk, L)
        k_perm = k_c.permute(0, 2, 3, 1) # (B, H, dk, L)
        
        # Causal mask: j <= t
        causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        
        # Mask out j > t in diff_log_g to avoid exp(positive large numbers)
        diff_log_g = torch.where(causal_mask, diff_log_g, torch.full_like(diff_log_g, -1e9))
        decay_weights = torch.exp(diff_log_g)  # (B, H, dk, L_t, L_j)
        
        # Attn term: sum over dk of (q_{t, d} * k_{j, d} * decay_{t, j, d})
        # q_perm.unsqueeze(-1): (B, H, dk, L_t, 1)
        # k_perm.unsqueeze(-2): (B, H, dk, 1, L_j)
        qk_decay = q_perm.unsqueeze(-1) * k_perm.unsqueeze(-2) * decay_weights # (B, H, dk, L_t, L_j)
        # Sum over dk: -> (B, H, L_t, L_j)
        attn_matrix = qk_decay.sum(dim=2) # (B, H, L_t, L_j)
        
        # Multiply with v:
        # v_c: (B, L, H, dv) -> permute to (B, H, L_j, dv)
        v_perm = v_c.permute(0, 2, 1, 3) # (B, H, L_j, dv)
        u_intra = torch.matmul(attn_matrix, v_perm) # (B, H, L_t, dv)
        u_intra = u_intra.permute(0, 2, 1, 3) # (B, L_t, H, dv)
        
        u_c = u_prev + u_intra
        outputs.append(u_c)
        
        # Update chunk state for next chunk:
        # S_{end} = state * exp(cum_log_g_last) + sum_j (k_j * exp(cum_log_g_last - cum_log_g_j))^T * v_j
        decay_to_end = torch.exp(cum_log_g[:, -1:]) # (B, 1, H, dk)
        state_decayed = state * decay_to_end.squeeze(1).unsqueeze(-1) # (B, H, dk, dv)
        
        # k contribution to final state:
        # k_j * exp(cum_log_g_last - cum_log_g_j)
        k_to_end = k_c * torch.exp(cum_log_g[:, -1:] - cum_log_g) # (B, L, H, dk)
        # Outer product sum over L:
        # k_to_end: (B, H, dk, L), v_perm: (B, H, L, dv)
        k_to_end_perm = k_to_end.permute(0, 2, 3, 1) # (B, H, dk, L)
        kv_chunk = torch.matmul(k_to_end_perm, v_perm) # (B, H, dk, dv)
        
        state = state_decayed + kv_chunk

    u_total = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
    return u_total, state
