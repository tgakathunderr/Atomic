# Architectural Design Specification: ATOMIC

**Title**: ATOMIC: Adaptive Token Operator with Memory-efficient Inference on CPU  
**Date**: 2026-09-12  
**Author**: TG Industries & DeepMind Pair  
**Status**: Approved for Implementation  

---

## 1. Executive Summary

Standard autoregressive Transformers rely on Softmax Multi-Head Attention (MHA), which incurs $O(T^2)$ time complexity and $O(T)$ Key-Value (KV) cache memory complexity. On consumer CPUs, the primary bottleneck during autoregressive generation is not peak floating-point operations (FLOPs), but **memory bandwidth**. At every generated token, the CPU must read the entire historical KV cache from DRAM through memory buses that are 20-50x narrower than GPU HBM, resulting in severe latency degradation as context lengths increase.

**ATOMIC** resolves this fundamental limitation by introducing a hybrid **Gated Linear Recurrent Attention (GLRA)** architecture augmented with zero-FLOP **Token-Shifting** and **Pre-RMSNorm SwiGLU** feed-forward layers. 

### Key Properties
1. **$O(1)$ Constant Time & Memory Inference**: Fixed-size multi-head state matrix $S_t \in \mathbb{R}^{H 	imes d_k 	imes d_v}$ that fits entirely into the CPU L1/L2 cache (e.g. 32 KB total state), eliminating DRAM roundtrips during token generation.
2. **$O(T)$ Parallel Associative Scan Training**: Full causal parallelism across the time dimension using associative prefix scan formulations, eliminating recurrent training bottlenecks.
3. **Data-Dependent Dynamic Retention Decay**: Continuous per-channel decay $g_t \in (0, 1)$ conditioned on the input token representation, solving the classic linear attention saturation issue and enabling high-fidelity associative recall.
4. **Hardware Affinity for Consumer x86/ARM CPUs**: Operations rely on cache-friendly GEMVs (matrix-vector), vectorized elementwise operations (AVX2/AVX-512/NEON), and zero dynamic allocations.

---

## 2. Mathematical Formalism

### 2.1 Zero-FLOP Token Shift Operator
Given a sequence of token representations $X = [x_1, x_2, \dots, x_T] \in \mathbb{R}^{T 	imes d}$, local temporal context is injected prior to linear projections without adding matrix multiplications:
$$\tilde{x}_t^{(\bullet)} = \mu_\bullet \odot x_t + (1 - \mu_\bullet) \odot x_{t-1}$$
where $\bullet \in \{r, k, v, g, q, ffn\}$ denotes branch-specific learnable mixing vectors initialized to smooth interpolation. At $t=1$, $x_0 = \mathbf{0}$.

### 2.2 Gated Linear Recurrent Attention (GLRA)
The projections map mixed inputs into heads $h \in \{1, \dots, H\}$:
- **Query**: $q_{t, h} = W_q^{(h)} \tilde{x}_t^{(q)} \in \mathbb{R}^{d_k}$
- **Key**: $k_{t, h} = W_k^{(h)} \tilde{x}_t^{(k)} \in \mathbb{R}^{d_k}$
- **Value**: $v_{t, h} = W_v^{(h)} \tilde{x}_t^{(v)} \in \mathbb{R}^{d_v}$
- **Gate Decay**: $\gamma_{t, h} = \text{softplus}(W_g^{(h)} \tilde{x}_t^{(g)} + b_g^{(h)}) \in \mathbb{R}_{>0}^{d_k}$
  $$g_{t, h} = \exp(-\gamma_{t, h}) \in (0, 1)^{d_k}$$
- **Receptance Output Gate**: $r_t = \sigma(W_r \tilde{x}_t^{(r)} + b_r) \in (0, 1)^{d}$

#### Recurrent State Update (Inference Mode - $O(1)$ Time & Space):
For each head $h$:
$$S_{t, h} = \text{diag}(g_{t, h}) S_{t-1, h} + k_{t, h}^\top v_{t, h}$$
Readout:
$$u_{t, h} = q_{t, h} S_{t, h} \in \mathbb{R}^{d_v}$$
Normalized by Head RMSNorm:
$$\hat{u}_{t, h} = \text{RMSNorm}(u_{t, h}) \odot \alpha_h$$
Concatenated across all $H$ heads and gated:
$$Y_t = W_o \left( r_t \odot [\hat{u}_{t, 1}, \hat{u}_{t, 2}, \dots, \hat{u}_{t, H}] \right)$$

#### Parallel Scan (Training Mode - $O(T)$ Parallel Time):
In parallel mode, unrolling the recurrence:
$$S_t = \sum_{j=1}^t \left(\prod_{m=j+1}^t \text{diag}(g_m)\right) k_j^\top v_j$$
Defining cumulative log decay $\Gamma_{t} = \sum_{m=1}^t -\gamma_m$, we have:
$$\prod_{m=j+1}^t g_{m} = \exp(\Gamma_t - \Gamma_j)$$
Hence:
$$u_t = \sum_{j=1}^t \left( (q_t \odot \exp(\Gamma_t)) \cdot (k_j \odot \exp(-\Gamma_j))^\top \right) v_j$$
This can be computed causally using chunked prefix GEMMs or associative scan in linear time without materializing full $T \times T$ attention matrices.

### 2.3 Token-Shifted SwiGLU FFN
$$\tilde{x}_t^{(ffn)} = \mu_{ffn} \odot x_t + (1 - \mu_{ffn}) \odot x_{t-1}$$
$$\text{AtomicFFN}(x_t) = W_{down} \left( \text{SiLU}(W_{gate} \tilde{x}_t^{(ffn)}) \odot (W_{up} \tilde{x}_t^{(ffn)}) \right)$$

### 2.4 Pre-RMSNorm Residual Layering
$$\bar{x}^{(l)}_t = x^{(l)}_t + \text{Dropout}\left(\text{GLRA}(\text{RMSNorm}(x^{(l)}_t))\right)$$
$$x^{(l+1)}_t = \bar{x}^{(l)}_t + \text{Dropout}\left(\text{AtomicFFN}(\text{RMSNorm}(\bar{x}^{(l)}_t))\right)$$

---

## 3. Complexity Comparison

| Feature | Standard Transformer (MHA) | Mamba-1/2 SSM | ATOMIC (GLRA) |
| :--- | :--- | :--- | :--- |
| **Training Time** | $O(T^2)$ | $O(T)$ | $O(T)$ |
| **Inference Time / Step** | $O(T)$ (KV cache scan) | $O(1)$ | $O(1)$ (GEMV) |
| **Inference Memory** | $O(T \cdot d)$ (Grows forever) | $O(d \cdot d_{state})$ | $O(H \cdot d_k \cdot d_v)$ (Fixed L1 cache) |
| **CPU Cache Thrashing** | High (DRAM bound) | Low | **Zero** (Fits in 32KB L1) |
| **Local N-Gram Induction** | Quadratic attention required | Conv1D overhead | **Zero-FLOP Token Shift** |
| **Reasoning / Recall** | Strong | Moderate to Strong | **Strong (Data-Dependent Decay)** |

---

## 4. Model Scaling Configurations

- **Atomic-Nano** (~2.5M params): $d=128$, $L=4$, $H=4$, $d_k=32$, $d_v=32$, $d_{ffn}=344$.
- **Atomic-Micro** (~12M params): $d=256$, $L=6$, $H=8$, $d_k=32$, $d_v=32$, $d_{ffn}=688$.
- **Atomic-Reasoning-Prod** (~32M params): $d=384$, $L=8$, $H=8$, $d_k=48$, $d_v=48$, $d_{ffn}=1024$.
- **Atomic-Base** (~125M params): $d=768$, $L=12$, $H=12$, $d_k=64$, $d_v=64$, $d_{ffn}=2048$.
