import torch
import pytest
from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM

def test_recurrent_state_memory_invariance():
    """Verify that ATOMIC recurrent inference state memory is strictly O(1) constant."""
    cfg = AtomicConfig.nano()
    model = AtomicForCausalLM(cfg)
    model.eval()

    states = model.init_states(batch_size=1)
    
    # Calculate bytes occupied by states at step 0
    state_bytes_step_0 = sum(s.element_size() * s.nelement() for s in states)

    prev_xs = None
    curr_token = torch.tensor([42])

    # Run for 200 steps
    for step in range(200):
        with torch.no_grad():
            logits, states, prev_xs = model.step(curr_token, states, prev_xs)
            curr_token = torch.argmax(logits, dim=-1)

        # Check that state size has NOT changed by even 1 byte
        current_state_bytes = sum(s.element_size() * s.nelement() for s in states)
        assert current_state_bytes == state_bytes_step_0, (
            f"Memory grew at step {step}! Was {state_bytes_step_0}, now {current_state_bytes}"
        )

    # In Nano config: 4 layers, 4 heads, 32x32 floats = 4 * 4 * 32 * 32 * 4 bytes = 65,536 bytes (64 KB)
    assert state_bytes_step_0 == 65536
    print(f"Total O(1) state memory across all {cfg.n_layers} layers: {state_bytes_step_0 / 1024:.2f} KB")
