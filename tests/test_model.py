import torch
import pytest
from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM

def test_model_forward_and_loss():
    torch.manual_seed(42)
    cfg = AtomicConfig(vocab_size=100, d_model=32, n_heads=2, d_head=16, d_ffn=64, n_layers=2)
    model = AtomicForCausalLM(cfg)

    B, T = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    # Forward with targets
    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0

    # Backward pass
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing grad for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad for {name}"

def test_model_parallel_vs_recurrent_logits_equivalence():
    torch.manual_seed(42)
    cfg = AtomicConfig(vocab_size=50, d_model=32, n_heads=2, d_head=16, d_ffn=64, n_layers=2)
    model = AtomicForCausalLM(cfg)
    model.eval()

    B, T = 2, 10
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    # 1. Parallel forward
    with torch.no_grad():
        parallel_logits, _, _ = model(input_ids)

    # 2. Recurrent step-by-step
    states = model.init_states(B, device=input_ids.device)
    prev_xs = None
    step_logits = []
    with torch.no_grad():
        for t in range(T):
            token_t = input_ids[:, t]
            logits_t, states, prev_xs = model.step(token_t, states=states, prev_xs=prev_xs)
            step_logits.append(logits_t)
    recurrent_logits = torch.stack(step_logits, dim=1)

    max_diff = (parallel_logits - recurrent_logits).abs().max().item()
    assert max_diff < 1e-4, f"Parallel vs Recurrent model logits differ: {max_diff}"

def test_strict_causality_zero_future_leakage():
    torch.manual_seed(42)
    cfg = AtomicConfig(vocab_size=50, d_model=32, n_heads=2, d_head=16, d_ffn=64, n_layers=2)
    model = AtomicForCausalLM(cfg)
    model.eval()

    B, T = 1, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    # Base forward
    with torch.no_grad():
        base_logits, _, _ = model(input_ids)

    # Change only the last token at position T-1
    altered_ids = input_ids.clone()
    altered_ids[:, -1] = (altered_ids[:, -1] + 1) % cfg.vocab_size

    with torch.no_grad():
        altered_logits, _, _ = model(altered_ids)

    # Logits from position 0 to T-2 MUST NOT change at all!
    diff_prefix = (base_logits[:, :-1] - altered_logits[:, :-1]).abs().max().item()
    assert diff_prefix == 0.0, f"Future leakage detected! Diff: {diff_prefix}"

    # Logits at position T-1 SHOULD differ
    diff_last = (base_logits[:, -1] - altered_logits[:, -1]).abs().max().item()
    assert diff_last > 1e-4, "Expected last token logit to change"
