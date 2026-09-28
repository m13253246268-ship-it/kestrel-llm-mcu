# Kestrel-MCU MoE Framework Design (Adapting Dense → Sparse Activation)

> 中文版见 **[moe_framework.zh-CN.md](moe_framework.zh-CN.md)**。

> **Tool directory note (2026-09-28)**: The historical research scripts mentioned in this document (`train_moe.py`, `train_moe_kd.py`,
> `train_moe_qat.py`, `dense_baseline.py`, `eval_quality.py`, etc.) have been archived to
> **`tools/legacy/`**; the current mainline tools still reside in `tools/`. For a tool overview, see `tools/README.md`.

> Goal: While **keeping the dense framework**, add an MoE (Mixture of Experts) path, using "sparse activation" to break through
> the PSRAM bandwidth wall of dense decode. Follow the axiom library constraints: bit-level consistency (router determinism),
> bandwidth-limited (minimize per-token read volume), measurement beats inference.

## 1. Background and Basis for Conclusions

Hard conclusions already verified in previous rounds (see README M3-3b / M3-5 / M3-7 / M3-10 / M3-11):

- The decode bottleneck = **weight bytes read per token** (PSRAM 106 MB/s wall), not computation;
- FFN (gate/up/down) accounts for **44%** of decode weight reads (10.2M elements), and the output head accounts for 43%;
- Output head pruning (lossless upper bound + lossy ANN) is **rejected in both cases** (margin extremely small + high-dimensional curse of dimensionality);
- Weight quantization q2/q3 is **rejected** (PIE has no int4/int2 dot + requantization breaks the model);
- **MoE sparse activation is the only direction that works for "reducing per-token read volume"** (empirically demonstrated by PFor / esp32-ai).

## 2. Architecture Design

**Modification location**: Only the FFN layers (`L{i}.gate/up/down`) are replaced; attention and the output head are unchanged.

```
Dense FFN:  y = down @ (silu(gate @ h) * up @ h)
MoE  FFN:  y = Σ_{e ∈ top-k(h)} softmax(router(h))_e · Expert_e(h)
            Expert_e(h) = down_e @ (silu(gate_e @ h) * up_e @ h)
```

| Parameter | Value | Description |
|---|---|---|
| Number of experts N | **8** | Balances gains and router complexity |
| Number of activated experts top-k | **2** | Each token activates 2/8 experts, reducing FFN read volume by 75% |
| Expert dimension ffn_e | **128** (aligned to 32) | Each expert is a **complete, independent** gate/up/down triple of matrices (not a clustering fragment), trained from scratch |
| router | `[N, dim]` (312×8) | One small GEMV + top-k + softmax |

**Benefit model** (N=8, top-2, total parameters unchanged ~28M):

| Segment | Dense read volume | MoE read volume |
|---|---|---|
| attention | 3.9M | 3.9M (unchanged) |
| FFN | 10.2M | **2.55M** (2/8 × 10.2M) |
| Output head | 9.98M | 9.98M (unchanged) |
| **Total** | **23.4M** | **~16.4M (−30%)** |

decode 232 ms → theoretical **~165 ms** (pure bandwidth); router increment ~N×dim=2.5K is negligible.

## 3. KMCU Format Extension (v2)

### 3.1 Header Extension

Reuse the v1 128B header, occupying the `reserved` region (starting at offset 76):

| Offset | Field | Type |
|---|---|---|
| 76 | is_moe | u32 (0=dense, 1=MoE) |
| 80 | n_expert | u32 |
| 84 | top_k | u32 |
| 88 | ffn_expert | u32 (after alignment) |

### 3.2 Tensor Naming

- Dense path: `L{i}.gate/up/down` (unchanged, used when `is_moe=0`)
- MoE path (`is_moe=1`):
  - `L{i}.router` ([N, dim])
  - `L{i}.expert{e}.gate` / `.up` ([ffn_e, dim])
  - `L{i}.expert{e}.down` ([dim, ffn_e])

`L{i}.router` is fp32 or q4 (q4 recommended, consistent with other weights). The naming of attention/embed/norm is unchanged.

## 4. Kernel Changes

Branch in the MLP section of [km_decode_step](../main/kmcu.c):

```c
if (!a->is_moe) {
    /* Dense: original gate/up/down path, zero changes */
} else {
    /* 1) router: gemv_q8(N, dim, h) → rlogits
     * 2) top-k: select the k largest rlogits (keep the indices)
     * 3) softmax normalizes the weights w_e of the selected k
     * 4) for each selected expert e:
     *      gemv_q8(ffn_e, dim, h) → g_e
     *      gemv_q8(ffn_e, dim, h) → u_e
     *      act = silu(g_e)*u_e
     *      gemv_q8(dim, ffn_e, act) → h_e
     *      x += w_e * h_e
     */
}
```

**Reuse**: `gemv_q8` / `pie_gemv_row` are fully reused (expert FFN is just standard GEMV).
New code volume: router GEMV + top-k selection + loop scheduling, about 60 lines.

**scratch extension**: The expert output `h_e` reuses the existing `h` buffer; `w_e` and the top-k indices use small on-stack arrays
(N=8, k=2). **No additional PSRAM residency** (expert weights remain within the expanded arena, only the naming differs).

## 5. Training Scheme (Training from Scratch + Knowledge Distillation to Reduce Difficulty)

> M3-12 already falsified "adapting the dense model": MoEfication clustering split has an oracle error of 62.8%, and Sparse Upcycling
> explodes parameters 8×. Therefore we switch to **training MoE from scratch, using dense Minueza as the teacher for distillation**, to reduce training difficulty.

### 5.1 Roles

- **Teacher**: The existing dense Minueza (32M, already trained), providing soft labels (logits distribution);
- **Student**: The MoE model (N=8 experts, each expert ffn_e=128, complete gate/up/down), initialized from scratch.

The student's total parameters ≈ dense (8×128=1024 ≈ original ffn=1092), but each token activates only top-2 (256 neurons).

### 5.2 Distillation Loss

Standard soft-label distillation (Hinton KD):

```
L = (1-α)·L_CE(student_logits, hard_label)
    + α·T²·KL(softmax(student_logits/T), softmax(teacher_logits/T))
```

- `T`: temperature (e.g. 4.0), amplifies the teacher's logits distribution, letting the student learn the relative relationships between tokens;
- `α`: distillation weight (e.g. 0.9);
- `hard_label`: optional (using only the distillation loss also works).

### 5.3 Training Pipeline

1. **Data**: TinyStories (open source, runnable on a single GPU) or text from Minueza's original training domain;
2. **teacher forward**: Freeze Minueza, compute `teacher_logits` for each batch (can be precomputed and cached for speedup);
3. **student forward**: MoE (router → top-k → sparse FFN → logits);
4. **backward**: Update only the student (router + experts + attention in full);
5. **Key**: The router uses differentiable top-k (`top_k + straight-through` or noisy softmax), making the sparse selection trainable;
6. Train until the student's greedy argmax sequence approaches the teacher (target ≥ 19/20 match in cross-checking).

### 5.4 Why Distillation Reduces Difficulty

- Learning language modeling from scratch is hard (requires large amounts of data + long training + tricks); distillation only needs to "imitate the teacher's logits distribution";
- During distillation the router naturally learns "expert division of labor" (different inputs route to different experts), with no manual clustering needed;
- Experts learn through backpropagation to "condense the representation within their own input domain", avoiding the "insufficient fragment magnitude" of M3-12.

### 5.5 Quantization and Writing to Disk

The fp32 MoE weights obtained from training → reuse the q4 quantization of `convert_minueza.py` → generate v2 KMCU
(is_moe=1, containing router and expert tensors). The router and experts likewise go through q4 quantization.

### 5.6 Training Script Implementation (train_moe.py, self-developed MoE core)

`tools/legacy/train_moe.py` is a self-developed training tool (teacher=Minueza frozen, student=reusing the Mistral skeleton + self-developed `LambdaMoE`).

**Self-developed `LambdaMoE` module** (corresponding to the axiom library's MoE superposition routing axiom, see Section 6):

- The forward pass is always **hard top-k** (`R_classic`, sparse, equivalent to the inference state);
- λ only anneals to control **backward gradient softening** (straight-through): early on λ=1, all experts' soft gradients are learned together,
  later λ→0, only the selected experts have gradients, sharpening the division of labor;
- Attached lightweight load-balancing loss (Switch Transformer style), preventing hard routing from collapsing to a single expert.

**Key diagnostic conclusions** (from this round's smoke tests, with honest attribution):

1. **soft forward (λ=1, weighting all experts) makes the MoE degrade**: With symmetric initialization + uniform weighting, the 8 experts'
   gradients tend to converge, and the router stays at the uniform stationary point (true entropy constantly = log8, gradient norm ≈ 0), degrading into a "weakened dense" model;
2. **hard forward is a necessary condition for breaking symmetry**: Under straight-through (hard forward, λ-softened backward),
   the router begins to differentiate (measured: L9's expert selection frequency clearly differentiates, while L0 remains near-uniform);
3. But hard routing has a **collapse tendency** (a few experts monopolize, aux rises from the uniform value 1 to 13), requiring load-balancing loss to counteract;
4. The previous round's `train_moe_kd.py` using transformers `MixtralForCausalLM` failed (aux constantly 2.0 = uniform state);
   the root cause is precisely that its load-balancing loss pushes the router toward uniformity, and it has no λ annealing mechanism—already fixed by the self-developed `LambdaMoE`.

### 5.7 KMCU v2 Conversion Mapping (state_dict of the self-developed LambdaMoE)

The self-developed `LambdaMoE`'s experts are **per-expert independent tensors** (not the stacked 3D of transformers Mixtral), making conversion more intuitive:

| LambdaMoE state_dict key | shape | KMCU v2 target |
|---|---|---|
| `L{i}.mlp.router.weight` | (8, 312) | `L{i}.router` |
| `L{i}.mlp.experts.{e}.gate.weight` | (128, 312) | `L{i}.expert{e}.gate` |
| `L{i}.mlp.experts.{e}.up.weight` | (128, 312) | `L{i}.expert{e}.up` |
| `L{i}.mlp.experts.{e}.down.weight` | (312, 128) | `L{i}.expert{e}.down` |

(`transformers` 5.x's Mixtral uses `gate_up_proj (8,256,312)` to stack gate+up, requiring splitting by the first/second half of dim1;
the self-developed version needs no splitting, mapping one-to-one directly.)


## 6. Cross-Checking Against the Axiom Library Constraints

| Axiom | How the MoE framework satisfies it |
|---|---|
| **MoE superposition routing axiom** (`R=(1-λ)·R_classic ⊕ λ·R_new`) | `LambdaMoE` forward = `R_classic` (hard top-k), backward gradient = λ·soft + (1-λ)·hard (straight-through), λ anneals from 1 to 0 |
| **Continuous reducibility law SHS-3** | λ smoothly anneals from 1.0 to lam_min using cosine, and the router continuously converges from soft to hard sparse |
| Bit-level consistency | router = deterministic GEMV + argmax/top-k; sparse FFN = deterministic GEMV. Same input, same output, bit-level reproducible |
| Bandwidth-limited | The gain comes from "reading only the top-k experts per token", which is exactly reducing bandwidth usage |
| Measurement beats inference | Section 7 uses PC cross-checking + on-board measurement to quantify the gain, not taking the theoretical 30% as authoritative |
| irreversibility_penalty | MoE sparsity is a **model architecture** (determined at training time), not an inference-time approximation; within the same model it is equivalent to dense and reproducible |

## 7. Measurement Conclusions (M3-13 / M3-14, honest attribution)

> This section records the hard data after training, overturns two earlier misjudgments, and converges on the core bottleneck.

### 7.1 argmax match Is an Unachievable Metric (M3-13)

**Observation**: After training, the MoE student's `argmax match = 0.21` (target ≥ 0.95), but `logits cosine = 0.9899`.

**Diagnostic chain** (falsifying each misjudgment one by one):

| Hypothesis | Experiment | Conclusion |
|---|---|---|
| Insufficient training | 8000 steps vs 4000 steps | match stays at 0.21 without moving, falsified |
| Distillation hyperparameters (α/T) | α 0.9→0.3, T 4→2 | match 0.214 unchanged, falsified |
| MoE capacity gap (256 vs 1092 dims) | ffn_e 128→256 | match 0.205 unchanged, falsified |
| Autoregressive cascading amplification | Single-step logits argmax | Agreement rate **0.0000**, not a cascading issue |
| **teacher argmax itself is fragile** | Measure top-1 vs top-2 margin | **margin only 0.52**, decisive |

**Root cause**: The teacher's argmax decision margin is only 0.52 (extremely fragile in a vocab=32002 space), and the remaining 0.01 error
of the logits distribution optimized by KD distillation (cosine 0.99) falls on the 0.52 decision boundary, enough to
completely flip the 32002-dimensional argmax. **"Per-token argmax agreement" as a distillation acceptance metric is mathematically unachievable**
(unless the student is a bit-for-bit copy of the teacher).

### 7.2 Switching the Acceptance Metric: perplexity (M3-14)

Switching to the standard language modeling metric, hard data:

| Model | perplexity | vs teacher |
|---|---|---|
| teacher (dense Minueza) | **21.29** | 1.0× |
| dense FFN from scratch (control) | 77.41 | 3.6× |
| moe ffn128 | 84.18 | 4.0× |
| moe ffn256 | 84.66 | 4.0× |

**Two-level conclusion**:

1. **The MoE architecture is viable (positive)**: moe vs dense from scratch differs by only 9% (84 vs 77), showing that under the same premise of "training FFN from scratch",
   MoE sparsification incurs almost no additional quality loss. The router differentiation, λ annealing, and bandwidth gain arguments all hold.

2. **The real bottleneck is "training FFN from scratch" itself**: Whether dense or MoE, the perplexity of an FFN trained from scratch is
   3.6~4× that of the teacher, and the generated text is broken (infinite loop "Tom Tom Tom..."). KD distillation + reusing attention
   under a thousand-step-level budget **cannot transfer the language modeling knowledge the teacher hides in its FFN** (FFN accounts for 44% of decode weight reads,
   and is precisely the hardest part to learn).

### 7.3 Final State

- The MoE direction, λ-annealed routing, self-developed training tool (`train_moe.py`), and evaluation tool (`eval_quality.py`)—all verified correct and ready;
- "Training from scratch + distillation to reduce difficulty" cannot produce a usable model under a thousand-step-level budget; the bottleneck is FFN knowledge transfer;
- A larger training budget (ten-thousand to hundred-thousand step level + the complete TinyStories) or a different training paradigm is needed before perplexity can be brought closer to the teacher.

## 8. Deliverables

1. KMCU v2 format definition + parsing (`convert_minueza.py` extension + `kmcu.h/c` extension);
2. MoE kernel (router + top-k + sparse FFN, dense path preserved);
3. Training script `train_moe.py` (self-developed LambdaMoE + λ annealing + KD distillation);
4. Control script `dense_baseline.py` + evaluation script `eval_quality.py` (perplexity + generation quality);
5. Filing (the measurement conclusions in Section 7 of this document).
