import torch
import pytest
from atomic.ops import token_shift, recurrent_glra_step, parallel_glra_scan

def test_token_shift_equivalence():
    B, T, D = 2, 8, 16
    x = torch.randn(B, T, D)
    mu = torch.sigmoid(torch.randn(D))

    # Parallel sequence token shift
    shifted_seq = token_shift(x, mu)
    assert shifted_seq.shape == (B, T, D)

    # First token has no predecessor: x_0 = mu * x_0
    assert torch.allclose(shifted_seq[:, 0], mu * x[:, 0], atol=1e-6)

    # Step-by-step recurrent token shift
    prev = torch.zeros(B, D)
    recurrent_steps = []
    for t in range(T):
        curr_x = x[:, t]
        shifted_t = mu * curr_x + (1 - mu) * prev
        recurrent_steps.append(shifted_t)
        prev = curr_x
    recurrent_seq = torch.stack(recurrent_steps, dim=1)

    assert torch.allclose(shifted_seq, recurrent_seq, atol=1e-6)

def test_glra_parallel_vs_recurrent_equivalence():
    torch.manual_seed(42)
    B, T, H, dk, dv = 2, 16, 4, 16, 16
    q = torch.randn(B, T, H, dk)
    k = torch.randn(B, T, H, dk)
    v = torch.randn(B, T, H, dv)
    # decay g in (0.1, 0.95)
    decay_raw = torch.randn(B, T, H, dk)
    g = torch.sigmoid(decay_raw) * 0.85 + 0.1

    init_state = torch.zeros(B, H, dk, dv)

    # 1. Parallel Scan
    u_parallel, final_state_parallel = parallel_glra_scan(q, k, v, g, init_state=init_state)

    # 2. Recurrent step-by-step
    u_recurrent_list = []
    state = init_state.clone()
    for t in range(T):
        u_t, state = recurrent_glra_step(q[:, t], k[:, t], v[:, t], g[:, t], state)
        u_recurrent_list.append(u_t)
    u_recurrent = torch.stack(u_recurrent_list, dim=1)

    # Both must match with high precision
    max_diff_u = (u_parallel - u_recurrent).abs().max().item()
    max_diff_state = (final_state_parallel - state).abs().max().item()

    assert max_diff_u < 1e-4, f"Parallel and recurrent outputs differ: {max_diff_u}"
    assert max_diff_state < 1e-4, f"Parallel and recurrent states differ: {max_diff_state}"

def test_parallel_glra_scan_gradients():
    torch.manual_seed(42)
    B, T, H, dk, dv = 2, 8, 2, 8, 8
    q = torch.randn(B, T, H, dk, requires_grad=True)
    k = torch.randn(B, T, H, dk, requires_grad=True)
    v = torch.randn(B, T, H, dv, requires_grad=True)
    g_raw = torch.randn(B, T, H, dk, requires_grad=True)
    g = torch.sigmoid(g_raw)

    u, final_state = parallel_glra_scan(q, k, v, g)
    loss = u.sum() + final_state.sum()
    loss.backward()

    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()
    assert g_raw.grad is not None and not torch.isnan(g_raw.grad).any()
