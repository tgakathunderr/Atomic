import random
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM
from benchmarks.baseline_transformer import StandardTransformerLM

def generate_mqar_batch(
    batch_size: int,
    seq_len: int,
    num_kv_pairs: int,
    num_queries: int,
    vocab_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic Multi-Query Associative Recall (MQAR) sequences.
    Format:
      [k1, v1, distractor, distractor, ..., kn, vn, ..., query_k, target_v, ...]
    """
    # Key tokens in [10, 50], Value tokens in [51, 90], Distractors in [91, vocab_size-1]
    keys_pool = list(range(10, 50))
    values_pool = list(range(51, 90))
    distractors_pool = list(range(91, vocab_size - 2))

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    query_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    for b in range(batch_size):
        # Sample unique keys for this sample
        keys = random.sample(keys_pool, num_kv_pairs)
        values = [random.choice(values_pool) for _ in range(num_kv_pairs)]
        kv_map = dict(zip(keys, values))

        # Fill with random distractors first
        tokens = [random.choice(distractors_pool) for _ in range(seq_len)]

        # Place KV pairs in first 60% of the sequence
        kv_slots = sorted(random.sample(range(0, int(seq_len * 0.6) - 1, 2), num_kv_pairs))
        for i, slot in enumerate(kv_slots):
            tokens[slot] = keys[i]
            tokens[slot + 1] = values[i]

        # Place queries in last 30% of the sequence
        query_slots = sorted(random.sample(range(int(seq_len * 0.7), seq_len - 1), num_queries))
        chosen_query_keys = random.sample(keys, num_queries)
        for i, q_slot in enumerate(query_slots):
            q_key = chosen_query_keys[i]
            tokens[q_slot] = q_key
            # Target at q_slot is the corresponding value
            targets[b, q_slot] = kv_map[q_key]
            query_mask[b, q_slot] = True

        input_ids[b] = torch.tensor(tokens, dtype=torch.long)

    return input_ids, targets, query_mask


def evaluate_mqar(
    model,
    num_eval_batches: int = 10,
    seq_len: int = 128,
    num_kv: int = 4,
    num_queries: int = 2,
    vocab_size: int = 128
) -> float:
    """Evaluate exact match retrieval accuracy on MQAR."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_eval_batches):
            input_ids, targets, mask = generate_mqar_batch(
                batch_size=8, seq_len=seq_len, num_kv_pairs=num_kv,
                num_queries=num_queries, vocab_size=vocab_size
            )
            if hasattr(model, "forward"):
                out = model(input_ids)
                logits = out[0] if isinstance(out, tuple) else out
            pred = torch.argmax(logits, dim=-1)

            correct += ((pred == targets) & mask).sum().item()
            total += mask.sum().item()

    return (correct / total) if total > 0 else 0.0


def run_mqar_benchmark(
    seq_len: int = 128,
    num_kv: int = 4,
    num_queries: int = 2,
    train_steps: int = 80
) -> Dict[str, Any]:
    """Train and compare ATOMIC vs Standard Transformer on Associative Recall."""
    print(f"\n--- Benchmarking Multi-Query Associative Recall (MQAR) [Context: {seq_len}, Pairs: {num_kv}] ---")
    cfg = AtomicConfig(vocab_size=128, d_model=64, n_heads=4, d_head=16, d_ffn=128, n_layers=2)

    atomic = AtomicForCausalLM(cfg)
    trans = StandardTransformerLM(cfg)

    opt_a = torch.optim.AdamW(atomic.parameters(), lr=2e-3)
    opt_t = torch.optim.AdamW(trans.parameters(), lr=2e-3)

    # Train both models on identical batches
    for step in range(train_steps):
        inputs, targets, _ = generate_mqar_batch(batch_size=8, seq_len=seq_len, num_kv_pairs=num_kv, num_queries=num_queries)
        
        # ATOMIC step
        opt_a.zero_grad()
        _, loss_a, _ = atomic(inputs, targets=targets)
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(atomic.parameters(), 1.0)
        opt_a.step()

        # Transformer step
        opt_t.zero_grad()
        logits_t, _ = trans(inputs)
        loss_t = F.cross_entropy(logits_t.view(-1, cfg.vocab_size), targets.view(-1), ignore_index=-100)
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(trans.parameters(), 1.0)
        opt_t.step()

        if (step + 1) % 20 == 0 or step == train_steps - 1:
            print(f"Step {step+1:3d}/{train_steps} | ATOMIC Loss: {loss_a.item():.4f} | Transformer Loss: {loss_t.item():.4f}")

    acc_atomic = evaluate_mqar(atomic, seq_len=seq_len, num_kv=num_kv, num_queries=num_queries)
    acc_trans = evaluate_mqar(trans, seq_len=seq_len, num_kv=num_kv, num_queries=num_queries)

    print(f"Final MQAR Recall Accuracy: ATOMIC: {acc_atomic*100:.1f}% | Transformer: {acc_trans*100:.1f}%")
    return {"atomic_acc": acc_atomic, "transformer_acc": acc_trans}
