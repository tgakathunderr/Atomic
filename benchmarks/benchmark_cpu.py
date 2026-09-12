import time
import torch
import numpy as np
from typing import Dict, Any, List

from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM
from benchmarks.baseline_transformer import StandardTransformerLM

def benchmark_inference_latency(
    config: AtomicConfig,
    context_lengths: List[int] = [64, 128, 256, 512, 1024],
    gen_steps: int = 50,
    warmup: int = 5
) -> Dict[str, Any]:
    """
    Measure autoregressive generation latency (ms/token) and throughput (tok/sec)
    for ATOMIC vs Standard Transformer as prompt context length scales.
    """
    results = {"context_lengths": context_lengths, "atomic": [], "transformer": []}

    atomic_model = AtomicForCausalLM(config)
    transformer_model = StandardTransformerLM(config)
    atomic_model.eval()
    transformer_model.eval()

    device = torch.device("cpu")
    print(f"\n--- Benchmarking CPU Inference (Threads: {torch.get_num_threads()}) ---")

    for ctx_len in context_lengths:
        prompt_ids = torch.randint(0, config.vocab_size, (1, ctx_len), device=device)

        # ----------------- Benchmark ATOMIC -----------------
        # 1. Warmup & prompt processing
        states = atomic_model.init_states(1, device=device)
        prev_xs = None
        with torch.no_grad():
            _, states, prev_xs = atomic_model.model(prompt_ids, states=states, prev_xs=prev_xs)
            curr_tok = prompt_ids[:, -1]
            for _ in range(warmup):
                l_t, states, prev_xs = atomic_model.step(curr_tok, states, prev_xs)
                curr_tok = torch.argmax(l_t, dim=-1)

        # Timed generation
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(gen_steps):
                l_t, states, prev_xs = atomic_model.step(curr_tok, states, prev_xs)
                curr_tok = torch.argmax(l_t, dim=-1)
        t_atomic = time.perf_counter() - t0
        atomic_ms_per_tok = (t_atomic / gen_steps) * 1000.0
        atomic_tok_per_sec = gen_steps / t_atomic

        # ----------------- Benchmark Standard Transformer -----------------
        # 1. Warmup & prompt processing (fill KV cache)
        with torch.no_grad():
            _, kv_caches = transformer_model(prompt_ids)
            curr_tok = prompt_ids[:, -1]
            for _ in range(warmup):
                l_t, kv_caches = transformer_model.step(curr_tok, kv_caches)
                curr_tok = torch.argmax(l_t, dim=-1)

        # Timed generation
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(gen_steps):
                l_t, kv_caches = transformer_model.step(curr_tok, kv_caches)
                curr_tok = torch.argmax(l_t, dim=-1)
        t_trans = time.perf_counter() - t0
        trans_ms_per_tok = (t_trans / gen_steps) * 1000.0
        trans_tok_per_sec = gen_steps / t_trans

        speedup = trans_ms_per_tok / atomic_ms_per_tok

        results["atomic"].append({
            "ctx_len": ctx_len,
            "ms_per_tok": atomic_ms_per_tok,
            "tok_per_sec": atomic_tok_per_sec
        })
        results["transformer"].append({
            "ctx_len": ctx_len,
            "ms_per_tok": trans_ms_per_tok,
            "tok_per_sec": trans_tok_per_sec
        })

        print(f"Context {ctx_len:4d} | ATOMIC: {atomic_ms_per_tok:6.2f} ms/tok ({atomic_tok_per_sec:6.1f} tok/s) | "
              f"Transformer: {trans_ms_per_tok:6.2f} ms/tok ({trans_tok_per_sec:6.1f} tok/s) | Speedup: {speedup:4.2f}x")

    return results


def benchmark_memory_scaling(
    config: AtomicConfig,
    context_lengths: List[int] = [64, 128, 256, 512, 1024, 2048, 4096]
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

        results["atomic_kb"].append(atomic_kb)
        results["transformer_kb"].append(trans_kb)
        ratio = trans_kb / atomic_kb
        print(f"Context {ctx:5d} | ATOMIC: {atomic_kb:7.2f} KB | Transformer KV: {trans_kb:8.2f} KB | KV Reduction: {ratio:6.1f}x")

    return results


def benchmark_training_throughput(
    config: AtomicConfig,
    seq_lengths: List[int] = [128, 256, 512, 1024],
    batch_size: int = 2,
    num_steps: int = 10
) -> Dict[str, Any]:
    """Measure training throughput (tokens/sec forward+backward) on CPU."""
    print("\n--- Benchmarking CPU Training Throughput ---")
    results = {"seq_lengths": seq_lengths, "atomic": [], "transformer": []}

    atomic_model = AtomicForCausalLM(config)
    transformer_model = StandardTransformerLM(config)
    opt_a = torch.optim.AdamW(atomic_model.parameters(), lr=1e-3)
    opt_t = torch.optim.AdamW(transformer_model.parameters(), lr=1e-3)

    for seq_len in seq_lengths:
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
        atomic_tok_s = (batch_size * seq_len * num_steps) / t_atomic

        # Transformer training timing
        t0 = time.perf_counter()
        for _ in range(num_steps):
            opt_t.zero_grad()
            logits, _ = transformer_model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
            loss.backward()
            opt_t.step()
        t_trans = time.perf_counter() - t0
        trans_tok_s = (batch_size * seq_len * num_steps) / t_trans

        results["atomic"].append({"seq_len": seq_len, "tok_s": atomic_tok_s})
        results["transformer"].append({"seq_len": seq_len, "tok_s": trans_tok_s})

        print(f"Train SeqLen {seq_len:4d} | ATOMIC: {atomic_tok_s:7.1f} tok/s | Transformer: {trans_tok_s:7.1f} tok/s")

    return results
