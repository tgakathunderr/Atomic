import torch
import pytest
from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM

@pytest.mark.parametrize("seq_len", [16, 64, 128])
def test_deep_parallel_recurrent_equivalence_across_lengths(seq_len):
    torch.manual_seed(1337)
    cfg = AtomicConfig(vocab_size=128, d_model=64, n_heads=4, d_head=16, d_ffn=128, n_layers=3)
    model = AtomicForCausalLM(cfg)
    model.eval()

    B = 2
    input_ids = torch.randint(0, cfg.vocab_size, (B, seq_len))

    # Parallel forward
    with torch.no_grad():
        par_logits, _, _ = model(input_ids)

    # Recurrent step
    states = model.init_states(B, device=input_ids.device)
    prev_xs = None
    rec_logits = []
    with torch.no_grad():
        for t in range(seq_len):
            tok = input_ids[:, t]
            l_t, states, prev_xs = model.step(tok, states, prev_xs)
            rec_logits.append(l_t)
    rec_logits = torch.stack(rec_logits, dim=1)

    max_err = (par_logits - rec_logits).abs().max().item()
    mean_err = (par_logits - rec_logits).abs().mean().item()

    assert max_err < 1e-4, f"Mismatch at seq_len={seq_len}: max_err={max_err}, mean_err={mean_err}"
