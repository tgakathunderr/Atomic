---
language:
- en
license: mit
tags:
- cpu-optimized
- recurrent-attention
- reasoning
- linear-attention
- cot
- edge-ai
pipeline_tag: text-generation
---

# Model Card: ATOMIC-Reasoning-Prod (3.3M)

**Model Identifier**: `atomic-reasoning-prod`  
**Architecture**: ATOMIC (Adaptive Token Operator with Memory-efficient Inference on CPU)  
**Developers**: UnikAI Lab ([www.unikai.in](https://www.unikai.in))  
**Model Type**: Causal Linear Recurrent Language Model with Chain-of-Thought Scratchpads  
**Precision**: FP32 / FP16 / Exportable to TorchScript & ONNX  
**Release Date**: September 2026  

---

## 1. Model Overview

`atomic-reasoning-prod` is a lightweight, high-speed reasoning language model specifically designed to run on **any consumer CPU** without requiring a discrete GPU or specialized accelerator.

Unlike standard Transformer language models whose memory footprint grows linearly with sequence length ($\mathcal{O}(T)$ KV cache), ATOMIC maintains a **strictly constant 64.0 KB recurrent state**. This fixed footprint fits entirely within the L1/L2 data caches of consumer Intel and AMD processors, eliminating DRAM bandwidth bottlenecks and enabling sustained generation speeds of **100 to 360+ tokens per second** on laptops and desktop PCs.

The model is trained to reason in a multi-step manner using structured `<think>...</think>` scratchpad blocks before emitting its final deduction inside `<answer>...</answer>` tags.

---

## 2. Architectural Specifications

| Parameter | Value |
| :--- | :--- |
| **Total Trainable Parameters** | **3,314,048 (~3.31 Million)** |
| **Model Dimension ($d_{\text{model}}$)** | 256 |
| **Hidden Layers** | 4 stacked ATOMIC blocks |
| **Recurrent Attention Heads** | 4 ($d_k = 32, d_v = 32$) |
| **Feed-Forward Expansion** | SwiGLU with intermediate dim = 680 ($\frac{8}{3} d_{\text{model}}$) |
| **Token Shifting** | Learnable 1D spatial mixing (zero GEMM overhead) |
| **Normalization** | Pre-RMSNorm ($\epsilon = 10^{-5}$) |
| **Vocabulary Size** | 930 tokens (rich subword regex tokenizer) |
| **Inference State Size** | **64.0 KB fixed per stream** (0 bytes KV cache growth) |

---

## 3. Quickstart & Usage

### 3.1 Python In-Memory Inference

```python
import torch
from atomic.config import AtomicConfig
from atomic.model import AtomicForCausalLM
from atomic.tokenizer import AtomicTokenizer
from atomic.generate import generate, GenerationConfig

# 1. Load model and tokenizer
config = AtomicConfig.from_json("models/atomic-reasoning-prod/config.json")
tokenizer = AtomicTokenizer.from_pretrained("models/atomic-reasoning-prod/tokenizer.json")

model = AtomicForCausalLM(config)
model.load_state_dict(torch.load("models/atomic-reasoning-prod/best_model.pt", map_location="cpu"))
model.eval()

# 2. Prompt with reasoning trigger
prompt = "Question: A store had 50 apples. They got 30 new apples and sold 25. How many remain?\n<think>\n"

# 3. Generate with constant 64 KB memory
config = GenerationConfig(
    max_new_tokens=120,
    temperature=0.0,  # Greedy decoding
    repetition_penalty=1.1
)

output = generate(model, tokenizer, prompt, config)
print(output)
```

### 3.2 Real-Time Streaming Interactive CLI

Launch the terminal console with color-coded Chain-of-Thought visualization:

```bash
python demo/cli.py
```

Or pass a direct prompt:

```bash
python demo/cli.py --prompt "Calculate distance for 60 km/h over 3 hours"
```

### 3.3 Production FastAPI Server

Start the high-throughput HTTP inference server:

```bash
python demo/api.py
```

Endpoints:
- `POST /generate`: Fast generation with prompt and parameters.
- `POST /v1/chat/completions`: OpenAI-compatible completions API with streaming Server-Sent Events (`stream: true`).
- `GET /health`: Model status and hardware memory footprint inspection.

### 3.4 Exporting to ONNX and TorchScript

Export verified artifacts for C++ or mobile deployment:

```bash
python -m atomic.export --model-dir models/atomic-reasoning-prod --output-dir exported/ --format both
```

Artifacts created:
- `exported/atomic_model.pt` (TorchScript traced model)
- `exported/atomic_model.onnx` (Verified ONNX model with dynamic batch/sequence axes)

---

## 4. Empirical Performance & Benchmarks

Tested on **12th Gen Intel Core i7-12700H** (14 Cores, 20 Threads, Windows 11):

### Generation Speed vs Standard GPT-2 Transformer

| Context Length | Standard Transformer | ATOMIC (This Model) | Speedup |
| :---: | :---: | :---: | :---: |
| 128 tokens | 164.5 tok/s | **362.4 tok/s** | **2.20$\times$** |
| 256 tokens | 138.8 tok/s | **348.6 tok/s** | **2.51$\times$** |
| 512 tokens | 92.4 tok/s | **210.1 tok/s** | **2.27$\times$** |
| 1024 tokens | 49.2 tok/s | **99.4 tok/s** | **2.02$\times$** |
| 4096 tokens | 9.7 tok/s | **18.6 tok/s** | **1.92$\times$** |

### Cache Footprint Scaling

- **Transformer KV Cache**: **16.38 MB** at 4,096 tokens (exceeds CPU L2 cache, thrashing DRAM).
- **ATOMIC Recurrent State**: **64.0 KB invariant** across all sequence lengths (**256$\times$ memory reduction**).

---

## 5. Training Details

- **Dataset**: Multi-domain reasoning suite encompassing arithmetic word problems, formal deductive logic, and algorithmic execution sequences with `<think>...</think>` traces.
- **Hardware**: Single consumer CPU (Intel i7-12700H, pure CPU FP32).
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $= 0.01$).
- **Learning Rate**: Cosine Annealing from $2.5 \times 10^{-3}$ to $1.25 \times 10^{-4}$.
- **Convergence**: Training Loss $4.25 \to 0.57$, Validation Perplexity $66.35 \to \mathbf{1.77}$.
- **Training Time**: 12.6 minutes.

---

## 6. Limitations and Intended Use

- **Intended Use**: On-device edge reasoning, offline embedded reasoning agents, educational mathematics/logic tutors, low-latency CPU microservices.
- **Limitations**: As a 3.3M parameter model, broad factual world knowledge is limited compared to multi-billion parameter frontier models. The model is optimized for structural step-by-step deductive execution rather than open-ended trivia.

---

## 7. Citation

```bibtex
@article{atomic2026,
  title={ATOMIC: Adaptive Token Operator for Memory-Efficient Inference on Consumer CPUs},
  author={UnikAI Lab},
  year={2026},
  journal={UnikAI Lab Technical Reports},
  url={https://www.unikai.in}
}
```
