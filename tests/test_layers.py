import torch
import pytest
from atomic.config import AtomicConfig
from atomic.layers import RMSNorm, AtomicGLRA, AtomicFFN, AtomicBlock

def test_rmsnorm():
    B, T, D = 2, 4, 16
    x = torch.randn(B, T, D)
    norm = RMSNorm(D)
    out = norm(x)
    assert out.shape == (B, T, D)
    # Check that root mean square of out is approximately 1.0 (with weight=1)
    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

def test_glra_layer_forward_and_step_equivalence():
    torch.manual_seed(42)
    cfg = AtomicConfig(d_model=32, n_heads=2, d_head=16, d_ffn=64, n_layers=1)
    layer = AtomicGLRA(cfg)
    layer.eval()

    B, T, D = 2, 8, cfg.d_model
    x = torch.randn(B, T, D)

    # 1. Full sequence forward pass
    with torch.no_grad():
        seq_out, final_state, last_x = layer(x)

    # 2. Step-by-step recurrent execution
    state = layer.init_state(B, device=x.device, dtype=x.dtype)
    prev_x = None
    step_outs = []
    with torch.no_grad():
        for t in range(T):
            out_t, state, prev_x = layer.step(x[:, t], state=state, prev_x=prev_x)
            step_outs.append(out_t)
    recurrent_out = torch.stack(step_outs, dim=1)

    # Verify high numerical agreement
    max_diff = (seq_out - recurrent_out).abs().max().item()
    assert max_diff < 1e-4, f"GLRA sequence vs step mismatch: {max_diff}"

def test_ffn_forward_and_step_equivalence():
    torch.manual_seed(42)
    cfg = AtomicConfig(d_model=32, n_heads=2, d_head=16, d_ffn=64, n_layers=1)
    ffn = AtomicFFN(cfg)
    ffn.eval()

    B, T, D = 2, 6, cfg.d_model
    x = torch.randn(B, T, D)

    with torch.no_grad():
        seq_out, _ = ffn(x)

    step_outs = []
    prev_x = None
    with torch.no_grad():
        for t in range(T):
            out_t, prev_x = ffn.step(x[:, t], prev_x=prev_x)
            step_outs.append(out_t)
    recurrent_out = torch.stack(step_outs, dim=1)

    max_diff = (seq_out - recurrent_out).abs().max().item()
    assert max_diff < 1e-5, f"FFN sequence vs step mismatch: {max_diff}"

def test_atomic_block_forward_and_step_equivalence():
    torch.manual_seed(42)
    cfg = AtomicConfig(d_model=32, n_heads=2, d_head=16, d_ffn=64, n_layers=1)
    block = AtomicBlock(cfg, layer_idx=0)
    block.eval()

    B, T, D = 2, 8, cfg.d_model
    x = torch.randn(B, T, D)

    # Sequence forward
    with torch.no_grad():
        seq_out, final_state, last_glra_x, last_ffn_x = block(x)

    # Step-by-step
    state = block.init_state(B, device=x.device, dtype=x.dtype)
    prev_glra_x = None
    prev_ffn_x = None
    step_outs = []
    with torch.no_grad():
        for t in range(T):
            out_t, state, prev_glra_x, prev_ffn_x = block.step(
                x[:, t], state=state, prev_glra_x=prev_glra_x, prev_ffn_x=prev_ffn_x
            )
            step_outs.append(out_t)
    recurrent_out = torch.stack(step_outs, dim=1)

    max_diff = (seq_out - recurrent_out).abs().max().item()
    assert max_diff < 1e-4, f"Block sequence vs step mismatch: {max_diff}"
