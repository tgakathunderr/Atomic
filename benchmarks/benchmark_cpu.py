import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, Any, List

from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM
from benchmarks.baseline_transformer import StandardTransformerLM


def benchmark_inference_latency(
    config: AtomicConfig,
    context_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048, 4096],
    gen_steps: int = 30,
    warmup: int = 5,
    num_trials: int = 3
) -> Dict[str, Any]:
    """
    Measure autoregressive generation latency (ms/token) and throughput (tok/sec)
    for ATOMIC vs Standard Transformer as prompt context length scales, averaged over multiple trials.
    """
    results = {"context_lengths": context_lengths, "atomic": [], "transformer": []}

    # Ensure max_seq_len can accommodate the longest context + gen_steps + margin
    config.max_seq_len = max(config.max_seq_len, max(context_lengths) + gen_steps + 256)

    atomic_model = AtomicForCausalLM(config)
    transformer_model = StandardTransformerLM(config)
    atomic_model.eval()
    transformer_model.eval()

    device = torch.device("cpu")
    print(f"\n--- Benchmarking CPU Inference Latency ({num_trials} trials averaged, {gen_steps} steps/trial) ---")
    print(f"Model Config: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}, vocab={config.vocab_size}")

    for ctx_len in context_lengths:
        atomic_ms_trials = []
        trans_ms_trials = []

        for trial in range(num_trials):
            prompt_ids = torch.randint(0, config.vocab_size, (1, ctx_len), device=device)

            # ----------------- Benchmark ATOMIC -----------------
            states = atomic_model.init_states(1, device=device)
            prev_xs = None
            with torch.no_grad():
                _, states, prev_xs = atomic_model.model(prompt_ids, states=states, prev_xs=prev_xs)
                curr_tok = prompt_ids[:, -1]
                for _ in range(warmup):
                    l_t, states, prev_xs = atomic_model.step(curr_tok, states, prev_xs)
                    curr_tok = torch.argmax(l_t, dim=-1)

                t0 = time.perf_counter()
                for _ in range(gen_steps):
                    l_t, states, prev_xs = atomic_model.step(curr_tok, states, prev_xs)
                    curr_tok = torch.argmax(l_t, dim=-1)
                t_atomic = time.perf_counter() - t0
                atomic_ms_trials.append((t_atomic / gen_steps) * 1000.0)

            # ----------------- Benchmark Standard Transformer -----------------
            with torch.no_grad():
                _, kv_caches = transformer_model(prompt_ids)
                curr_tok = prompt_ids[:, -1]
                for _ in range(warmup):
                    l_t, kv_caches = transformer_model.step(curr_tok, kv_caches)
                    curr_tok = torch.argmax(l_t, dim=-1)

                t0 = time.perf_counter()
                for _ in range(gen_steps):
                    l_t, kv_caches = transformer_model.step(curr_tok, kv_caches)
                    curr_tok = torch.argmax(l_t, dim=-1)
                t_trans = time.perf_counter() - t0
                trans_ms_trials.append((t_trans / gen_steps) * 1000.0)

        a_ms_mean = float(np.mean(atomic_ms_trials))
        a_ms_std = float(np.std(atomic_ms_trials))
        a_tok_mean = 1000.0 / a_ms_mean if a_ms_mean > 0 else 0.0

        t_ms_mean = float(np.mean(trans_ms_trials))
        t_ms_std = float(np.std(trans_ms_trials))
        t_tok_mean = 1000.0 / t_ms_mean if t_ms_mean > 0 else 0.0

        speedup = t_ms_mean / a_ms_mean if a_ms_mean > 0 else 1.0

        if speedup >= 1.05:
            winner = "ATOMIC"
        elif speedup <= 0.95:
            winner = "Transformer"
        else:
            winner = "Tied (~equal)"

        results["atomic"].append({
            "ctx_len": ctx_len,
            "ms_per_tok": round(a_ms_mean, 2),
            "ms_std": round(a_ms_std, 2),
            "tok_per_sec": round(a_tok_mean, 1)
        })
        results["transformer"].append({
            "ctx_len": ctx_len,
            "ms_per_tok": round(t_ms_mean, 2),
            "ms_std": round(t_ms_std, 2),
            "tok_per_sec": round(t_tok_mean, 1)
        })

        print(f"Context {ctx_len:4d} | ATOMIC: {a_ms_mean:6.2f} ms ({a_tok_mean:5.1f} tok/s) | "
              f"Trans: {t_ms_mean:6.2f} ms ({t_tok_mean:5.1f} tok/s) | Speedup: {speedup:4.2f}x ({winner})", flush=True)

    return results


def benchmark_memory_scaling(
    config: AtomicConfig,
    context_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048, 4096]
) -> Dict[str, Any]:
    """Measure inference KV cache vs recurrent state memory footprint in KB."""
    print("\n--- Benchmarking State Memory Scaling ---")
    results = {"context_lengths": context_lengths, "atomic_kb": [], "transformer_kb": []}

    # ATOMIC fixed state size:
    # n_layers * n_heads * d_head * d_head * 4 bytes
    atomic_bytes = config.n_layers * config.n_heads * config.d_head * config.d_head * 4
    atomic_kb = atomic_bytes / 1024.0

    for ctx in context_lengths:
        # Transformer KV cache size at context ctx:
        # 2 (K and V) * n_layers * n_heads * ctx * d_head * 4 bytes
        trans_bytes = 2 * config.n_layers * config.n_heads * ctx * config.d_head * 4
        trans_kb = trans_bytes / 1024.0

        results["atomic_kb"].append(round(atomic_kb, 2))
        results["transformer_kb"].append(round(trans_kb, 2))
        ratio = trans_kb / atomic_kb if atomic_kb > 0 else 1.0
        print(f"Context {ctx:5d} | ATOMIC: {atomic_kb:7.2f} KB (Fixed) | Transformer KV: {trans_kb:8.2f} KB | KV Reduction: {ratio:6.1f}x")

    return results


def benchmark_training_throughput(
    config: AtomicConfig,
    seq_lengths: List[int] = [128, 256, 512, 1024],
    batch_size: int = 2,
    num_steps: int = 10,
    num_trials: int = 3
) -> Dict[str, Any]:
    """Measure training throughput (tokens/sec forward+backward) on CPU, averaged over multiple trials."""
    print(f"\n--- Benchmarking CPU Training Throughput ({num_trials} trials averaged) ---")
    results = {"seq_lengths": seq_lengths, "atomic": [], "transformer": []}

    atomic_model = AtomicForCausalLM(config)
    transformer_model = StandardTransformerLM(config)
    opt_a = torch.optim.AdamW(atomic_model.parameters(), lr=1e-3)
    opt_t = torch.optim.AdamW(transformer_model.parameters(), lr=1e-3)

    for seq_len in seq_lengths:
        atomic_tok_s_trials = []
        trans_tok_s_trials = []

        for _ in range(num_trials):
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

            # Warmup
            atomic_model(input_ids, targets=targets)[1].backward()
            opt_a.zero_grad()
            logits, _ = transformer_model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
            loss.backward()
            opt_t.zero_grad()

            # ATOMIC training timing
            t0 = time.perf_counter()
            for _ in range(num_steps):
                opt_a.zero_grad()
                _, loss, _ = atomic_model(input_ids, targets=targets)
                loss.backward()
                opt_a.step()
            t_atomic = time.perf_counter() - t0
            atomic_tok_s_trials.append((batch_size * seq_len * num_steps) / t_atomic)

            # Transformer training timing
            t0 = time.perf_counter()
            for _ in range(num_steps):
                opt_t.zero_grad()
                logits, _ = transformer_model(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
                loss.backward()
                opt_t.step()
            t_trans = time.perf_counter() - t0
            trans_tok_s_trials.append((batch_size * seq_len * num_steps) / t_trans)

        a_tok_s_mean = float(np.mean(atomic_tok_s_trials))
        t_tok_s_mean = float(np.mean(trans_tok_s_trials))

        results["atomic"].append({"seq_len": seq_len, "tok_s": round(a_tok_s_mean, 1)})
        results["transformer"].append({"seq_len": seq_len, "tok_s": round(t_tok_s_mean, 1)})

        print(f"Train SeqLen {seq_len:4d} | ATOMIC: {a_tok_s_mean:7.1f} tok/s | Transformer: {t_tok_s_mean:7.1f} tok/s", flush=True)

    return results
