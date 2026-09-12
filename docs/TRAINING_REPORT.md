# ATOMIC Production Reasoning LM: Training & Empirical Evaluation Report

**Model Identifier**: `atomic-reasoning-prod`  
**Architecture**: ATOMIC (Adaptive Token Operator with Memory-efficient Inference on CPU)  
**Hardware Used for Training**: 12th Gen Intel(R) Core(TM) i7-12700H (14 Cores, 20 Threads)  
**Training Framework**: Pure CPU PyTorch with Cosine Annealing, AdamW, and FP32  
**Total Parameters**: 3,314,048 (~3.31M)  
**Inference Footprint**: Fixed **64 KB** recurrent state (0 bytes KV cache growth)  

---

## 1. Executive Summary

The `atomic-reasoning-prod` model was trained completely on consumer CPU hardware to demonstrate that the ATOMIC architecture can learn complex, multi-step Chain-of-Thought (CoT) reasoning while executing sub-10ms inference per token without GPU acceleration.

### Overall Benchmark Accuracy
- **Chain-of-Thought Structural Validity**: **60.0%** (generates well-formed `<think>...</think>` scratchpad derivations)
- **Overall Reasoning Accuracy**: **2.2%** across multi-domain reasoning tasks
- **Inference Speed**: **~100-350 tokens/second** on consumer CPU single stream

---

## 2. Benchmark Breakdown by Reasoning Domain

| Domain | Task Description | CoT Validity Rate | Accuracy |
| :--- | :--- | :--- | :--- |
| **Arithmetic Word Problems** | Multi-step reasoning | 60.0% | **0.0%** |
| **Formal Deductive Logic** | Multi-step reasoning | 20.0% | **0.0%** |
| **Algorithmic Execution** | Multi-step reasoning | 100.0% | **6.7%** |

---

## 3. Training Convergence Dynamics

The model converged smoothly under Cosine Annealing with warmup:

| Step | Epoch | Train Loss | Validation Loss | Validation Perplexity | Learning Rate | Elapsed Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|   25 | 1 | 3.7429 | **3.8718** | **48.03** | 2.50e-03 | 78.5s |
|   50 | 1 | 3.3846 | **3.0910** | **22.00** | 2.41e-03 | 144.9s |
|   75 | 1 | 2.4886 | **2.6361** | **13.96** | 2.15e-03 | 204.0s |
|  100 | 2 | 1.8810 | **1.9255** | **6.86** | 1.77e-03 | 281.9s |
|  125 | 2 | 1.3719 | **1.3240** | **3.76** | 1.31e-03 | 383.9s |
|  150 | 2 | 1.0115 | **0.9548** | **2.60** | 8.58e-04 | 484.3s |
|  175 | 3 | 0.6900 | **0.7411** | **2.10** | 4.73e-04 | 583.9s |
|  200 | 3 | 0.7030 | **0.6259** | **1.87** | 2.15e-04 | 664.6s |
|  225 | 3 | 0.5761 | **0.5699** | **1.77** | 1.25e-04 | 757.2s |

---

## 4. Qualitative Reasoning Traces (Actual Model Generations)

### Example: Arithmetic Word Problems
**Prompt**:
```text
Question: Maria started with $119. She spent $39 on groceries and $19 on gas. How much money does she have left?
<think>

```

**Model Generation with <think> trace**:
```text
Question: Maria started with $119. She spent $39 on groceries and $19 on gas. How much money does she have left?
<think>
1. The rule is P -> Q. The contrapositive is not Q -> not P.
2. We are given not Q ('the switch is on' is false).
3. By Modus Tollens, we deduce not P ('the light glows' is false).
3. By Modus Tollens, we deduce not P ('the 
```

- **Expected Answer**: `$61`
- **Predicted Answer**: ``
- **Result**: [INCORRECT]

### Example: Arithmetic Word Problems
**Prompt**:
```text
Question: Alex bought 8 notebooks at $9 each, plus a backpack for $14. What was the total cost of the purchase?
<think>

```

**Model Generation with <think> trace**:
```text
Question: Alex bought 8 notebooks at $9 each, plus a backpack for $14. What was the total cost of the purchase?
<think>
1. Calculate discount amount: 20% of $50 = $10.
2. Subtract discount from original price: $50 - $8 = $50.
</think>
<answer>$1</answer>
```

- **Expected Answer**: `$86`
- **Predicted Answer**: `$1`
- **Result**: [INCORRECT]

### Example: Formal Deductive Logic
**Prompt**:
```text
Question: All metals are conductors of electricity. Copper are metals. Are Copper a conductor of electricity?
<think>

```

**Model Generation with <think> trace**:
```text
Question: All metals are conductors of electricity. Copper are metals. Are Copper a conductor of electricity?
<think>
1. The rule is P -> Q. The contrapositive is not true that the light glows. How much money does she have left?
<think>
1. Calculate discount amount: 20% of $50 = $10.
2. We are given not Q ('the switch is on' is false).
3. By Modus Tollens, we deduce
```

- **Expected Answer**: `Yes`
- **Predicted Answer**: ``
- **Result**: [INCORRECT]

### Example: Formal Deductive Logic
**Prompt**:
```text
Question: Premise 1: Alice is taller than Emma. Premise 2: Emma is taller than Charlie. Is Alice taller than Charlie?
<think>

```

**Model Generation with <think> trace**:
```text
Question: Premise 1: Alice is taller than Emma. Premise 2: Emma is taller than Charlie. Is Alice taller than Charlie?
<think>
1. Calculate discount amount: 20% of $50 = $10.
</think>
<answer>$1</answer>
```

- **Expected Answer**: `Yes`
- **Predicted Answer**: `$1`
- **Result**: [INCORRECT]

### Example: Algorithmic Execution
**Prompt**:
```text
Question: Reverse the following ordered list: [7, 18, 2, 9, 6].
<think>

```

**Model Generation with <think> trace**:
```text
Question: Reverse the following ordered list: [7, 18, 2, 9, 6].
<think>
1. Compare each number sequentially against the current minimum.
2. The reverse ordering places 3 first and 5 last.
3. Reading backwards yields: [8, 8, 8].
</think>
<answer>$1</answer>
```

- **Expected Answer**: `[6, 9, 2, 18, 7]`
- **Predicted Answer**: `$1`
- **Result**: [INCORRECT]

### Example: Algorithmic Execution
**Prompt**:
```text
Question: Find the next number in the arithmetic progression: [8, 11, 14, 17], ?
<think>

```

**Model Generation with <think> trace**:
```text
Question: Find the next number in the arithmetic progression: [8, 11, 14, 17], ?
<think>
1. Compute common difference: 8 - 7 = 5.
2. Add the backpack cost: $50 - 10 = 3.
</think>
<answer>$1</answer>
```

- **Expected Answer**: `20`
- **Predicted Answer**: `$1`
- **Result**: [INCORRECT]

