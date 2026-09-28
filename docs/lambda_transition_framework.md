# λ Transition Training Framework Design (Fusing the Classic Paradigm + Axiom Library Foundation)

> 中文版见 **[lambda_transition_framework.zh-CN.md](lambda_transition_framework.zh-CN.md)**。

> **Tool directory note (2026-09-28)**: The historical research scripts mentioned in this document (`full_finetune.py`,
> `train_moe.py`, `train_moe_qat.py`, `train_lambda.py`, etc.) have been archived to
> **`tools/legacy/`**; the current mainline tools still reside in `tools/`. See `tools/README.md` for the tool overview.

> Goal: Do not diverge from the classic training paradigm; instead, **take the classic paradigm as the backbone** and use the axiom library's "superposition state + λ transition parameter"
> as an **auxiliary mechanism for structural and precision transformations**, solving the two major transformations not covered by the classic paradigm: "sparsification" and "quantization".
>
> Core principle: **Verify feasibility first**. For each technical point, first run a minimal verification experiment; hard data pass/fail; only proceed deeper once it passes.
>
> Key correction (relative to earlier versions): Do not do a λ transition at the loss level—classic pure CE is the correct answer; λ only acts on
> the two transformations not covered by the classic paradigm: structural sparsification and quantization.
>
> Key correction 2 (2026-09-26): Structural sparsification also cannot go through "dense bypass + λ_s smooth transition"—
> the dense fallback dilutes the CE differentiation pressure, causing the router to never differentiate (confirmed by four rounds of empirical tests). The correct path is
> **full training MoE with no fallback** (λ only performs the router's internal soft→hard gradient annealing, not a structural transition).

---

## 1. Re-attribution of the Failures in Previous Rounds

Hard data:

| Experiment | validation ppl | Old attribution | **New attribution (corrected)** |
|---|---|---|---|
| teacher | 21.53 | — | Baseline |
| Reuse attention + train FFN from scratch (KD α=0.9) | 77.38 | Loss fixed | **setup error: only training FFN, freezing attention** |
| Reuse attention + train FFN from scratch (pure CE α=0) | 1711.07 | Loss hard-switched | **Same as above: FFN can only memorize, overfitting** |
| MoE λ annealing router | Successfully differentiated | — | The only correct smooth transition |

**Key correction**: The failures in previous rounds were not "choosing the wrong loss function" (the CE/KD debate is a pseudo-problem), but a **training setup error**—
freezing attention and training only the FFN, causing the 10M-parameter FFN to either memorize (overfitting ppl 1711) or fail to learn anything
(insufficient capacity ppl 77).

**The correct backbone = the classic full-training paradigm**: all parameters trainable + pure CE + AdamW + cosine LR.
This is the time-tested right path, and the `full_finetune.py` currently running in the background is exactly it.

**The correct role of the λ transition**: The classic paradigm is responsible for "learning language modeling", while the λ transition is responsible for the "smooth transformation of structure and precision"
—these are the two steps that the classic paradigm naturally does not cover but that our goals (MoE sparsity + q4 quantization) must accomplish.

---

## 2. Mathematical Foundation of the Axiom Library (for Structural/Quantization Transformations, Not for the Loss)

### Pillar One: Superposition State
`S_0(1,1) = {2,3}`: An object simultaneously resides in multiple basis states, carrying multiple sets of information, without contradiction (SHS-2).

### Pillar Two: λ Transition Parameter (SHS-T + SHS-3)
`M(λ) = (1-λ)·A_classic ⊕ λ·B_new`, λ∈[0,1]. λ=0 classic basis state, λ=1 new basis state.
**SHS-3 hard constraint: the transition must be continuous; hard switching will fail.**

### Pillar Three: Bregman Divergence
CE/KL/MSE are all special cases of it. Optional, used to measure the smoothness of the transition path.

---

## 3. Overview of the Fusion Framework: Classic Paradigm Backbone + λ Transformation Auxiliary

```
Backbone (classic paradigm, responsible for "learning"):
  Full training + pure CE + AdamW + cosine LR + teacher initialization warm start
  (teacher initialization replaces KD distillation, reusing teacher knowledge, no soft-label loss needed)

λ transformation (auxiliary, responsible for "transformation", applied after the backbone):
  Structural λ_s:  FFN = (1-λ_s)·dense ⊕ λ_s·MoE sparse    λ_s: 0 → 1
  Quantization λ_q:  Weight = (1-λ_q)·fp32  ⊕ λ_q·q4-aware     λ_q: 0 → 1
```

Full training path:

```
Stage 0 (classic pretraining, pure CE, full): teacher initialization → full training until convergence
        ↑ Standard practice, responsible for truly training the model well (currently running full_finetune.py in the background)
Stage 1 (λ_s structural sparsification): classic convergence point → dense FFN smoothly transitions to MoE, router learns to divide labor
        ↑ λ transition, obtaining the bandwidth benefit of sparse activation while keeping ppl from collapsing
Stage 2 (λ_q quantization-aware): after the structure is fixed → weights smoothly transition from fp32 to q4, QAT preserves precision
        ↑ λ transition, adapting to q4 during training, avoiding PTQ precision loss
```

**Contrast with historical failures**: The previous rounds used neither the classic paradigm (only training the FFN) nor the λ transition (hard switching).
The fusion approach requires: **the classic paradigm serves as the backbone, and the λ transition only handles structural/quantization transformations**, each performing its own duty.

---

## 4. Technical Point Breakdown (with Hypothesis + Verification Plan + Pass/Fail Criteria + Implementation Description)

> Ordered by feasibility-verification priority. Technical point 0 is the master gate (classic backbone); if it does not work, the entire direction is meaningless.

### Technical Point 0: Classic Full Pretraining Baseline [Priority: Highest / Master Gate]

**Hypothesis**: Full training (teacher initialization + pure CE) can break through the ppl 77 ceiling and approach the teacher's 21.
This is the classic paradigm and should theoretically always work; if it does not, it means there is a fundamental flaw in the data/model scale—stop the loss.

**Verification plan**: `full_finetune.py` (running in the background)—teacher initialization, fully trainable, pure CE,
AdamW + cosine, 1 epoch, measure validation ppl every 5000 steps.

**Pass/fail criteria**:
- ✅ PASS: validation ppl clearly below 77, with a continued downward trend;
- ❌ FAIL: ppl stalls at 77+, indicating that "training a 32M model from scratch/fine-tuning" has a hard constraint on the current data—stop the loss.

**Implementation description**: Already completed (`full_finetune.py`), no new work.

---

### Technical Point 1: λ_s Structural Sparsification (Dense FFN → MoE) [Priority: High / Already FAILED, Stopped]

**Hypothesis**: After classic pretraining converges, use a λ_s smooth transition to sparsify the dense FFN into MoE, so the router can
learn to divide labor under the premise of "not collapsing ppl" (this is a transformation not covered by the classic paradigm, and it must use a λ transition rather than a hard switch).

**Verification plan**: Superimpose structural annealing on the converged model from technical point 0:
- Per-layer FFN: `out = (1-λ_s)·dense_ffn(h) ⊕ λ_s·moe_ffn(h)`;
- λ_s smoothly anneals from 0 to 1, observing the joint change of router entropy (differentiation) and validation ppl;
- Control group: direct hard switch (λ_s instantaneously 0→1), comparing the degree of ppl collapse.

**Pass/fail criteria**:
- ✅ PASS: when λ_s→1, router entropy < 1.8 (differentiated) and ppl rises < 20% relative to the dense baseline;
- ❌ FAIL: router does not differentiate (entropy ≈ log8) or ppl collapses—stop the loss.

**Empirical result (2026-09-26)**: ❌ FAIL, stopped.

All four rounds of corrections failed; hard data:

| Version | Correction | ppl at λ_s→1 | Increase | L9 entropy | Verdict |
|---|---|---|---|---|---|
| v1 | λ_s 0→1 (direction reversed) | 59.95 | +587% | 2.013 | ❌ |
| v2 | + λ_s floor + independent lr | 21.36 | +145% | 1.989 | ❌ |
| v4 | + aux_w=0 + 5000 steps | 15.84 | +81.5% | 1.965 | ❌ |
| ffn_e=256 | + capacity doubled (0.47) | 13.67 | +56.7% | 1.950 | ❌ |

**Root cause (confirmed by hard data)**: Structural distillation has an **inherent deadlock**—the `dense bypass fallback` (its output dominates when λ_s<1)
dilutes the CE gradient to the MoE down to the level of a "fine-tuning residual" → the router has no differentiation pressure → the 8 experts see the same tokens,
learn the same residuals, and become increasingly homogeneous → the router has even less reason to differentiate (a negative feedback loop).

In all four rounds, the router entropy was **always** stuck at 1.95~1.97 (log8=2.079 uniform state), **completely unrelated** to capacity (ffn_e 128/256),
the aux switch, or the λ_s direction. This proves that "smooth transition (dense fallback keeps it from collapsing)" and "router differentiation"
are **mutually exclusive** under this architecture—the smoother the fallback, the weaker the differentiation pressure.

**Pivot after stopping**: Switch to **full training MoE** (`train_moe.py`), with no dense fallback, where pure CE directly drives
router differentiation. Empirically, the router entropy successfully dropped to 1.80, with argmax match 0.607 (close to the dense teacher),
proving that full MoE is the right path. See technical point 1-b for details.

---

### Technical Point 1-b: Full Training MoE (No Dense Fallback) [Replaces Technical Point 1, Verified Feasible]

**Hypothesis**: Without introducing a dense bypass, directly replace each layer's FFN with `LambdaMoE` (router + 8 experts),
reuse the teacher's attention/embed/lm_head weights + MoE bypass from scratch, full training + pure CE.
With no fallback, CE directly forces router differentiation.

**Empirical result (2026-09-26)**: ✅ Feasible.

- Config: ffn_e=256, top_k=2, n_expert=8, pure CE, 5000 steps;
- router entropy: 1.968 → 1.80 (genuine differentiation, versus structural distillation's constant 1.95~1.97);
- final argmax match = **0.607** (dense teacher level);
- trainable parameters 41.77M, bandwidth benefit 2.13x, capacity ratio 0.47.

**ffn_e trade-off sweep (2026-09-26)**:

| ffn_e | Capacity ratio | Bandwidth benefit | Parameters | router entropy | argmax match |
|---|---|---|---|---|---|
| **128** | 0.23 | **4.27x** | 32.18M | 1.80 | **0.641** |
| 256 | 0.47 | 2.13x | 41.77M | 1.80 | 0.607 |
| 384 | 0.70 | 1.42x | 51.35M | 1.80 | 0.581 |

**Conclusion**: ffn_e=128 is the MCU deployment sweet spot—simultaneously achieving the highest quality (match 0.641) and the highest bandwidth benefit
(4.27x). argmax match actually decreases as ffn_e increases, which is a mild case of overfitting from excess capacity on the simple TinyStories data.
All three configurations successfully differentiated the router (entropy 1.80), and the full MoE path is stable.

**Implementation description**: `train_moe.py`, changed from KD distillation to pure CE (`--loss ce`), and fixed the
StopIteration single-epoch exit + flush + periodic checkpoint.

---

### Technical Point 2: λ_q Quantization Aware (fp32 → q4) [Priority: Medium / Rejected, PTQ Is Sufficient]

**Hypothesis**: Post-training quantization (PTQ) has a large precision loss, and quantization-aware training (QAT) with a λ_q smooth transition lets the model
adapt to q4 during training, resulting in a smaller precision loss.

**Verification plan**: After technical point 1 passes, introduce QAT at the end of training:
- Weights `W = (1-λ_q)·W_fp32 ⊕ λ_q·round(W_fp32/scale)·scale` (straight-through);
- λ_q smoothly anneals from 0 to 1, comparing the ppl loss against PTQ (direct quantization after training).

**Pass/fail criteria**:
- ✅ PASS: the ppl of the QAT q4 model is clearly better than PTQ;
- ❌ FAIL: QAT has no gain or training is unstable—fall back to PTQ + a more conservative bit width.

**Empirical result (2026-09-26)**: ❌ FAIL, QAT has no gain.

| Approach | ppl | Loss relative to fp32 |
|---|---|---|
| fp32 baseline | 12.73 | — |
| **PTQ (direct quantization)** | **12.84** | **+0.9%** |
| QAT (λ_q annealing 2000 steps) | 12.90 | +1.4% |

**Conclusion**: QAT (+1.4%) is not better than PTQ (+0.9%), and is in fact slightly worse. PTQ is only +0.9%, nearly lossless, indicating that
the ffn_e=128 MoE model is naturally robust to q4_0. QAT's straight-through gradient noise + the perturbation of an already-converged model
outweighs the gains. **Technical point 2 is rejected; just use PTQ**, saving the complexity of the QAT mechanism.

**Implementation description**: `train_moe_qat.py` (the fake-quant is bit-level identical to convert_minueza's q4_0 rule,
implementing straight-through with `torch.nn.utils.parametrize`). The script is kept as a tool, but is no longer used in the official pipeline.

---

### Technical Point 3: Equivalence Verification of Teacher Initialization vs KD Distillation [Priority: Low / Optional]

**Hypothesis**: Teacher weight initialization (warm start) is already sufficient to reuse the teacher's knowledge, without needing an additional KD soft-label
loss. That is, "initialization" is equivalent to "distillation", and is simpler.

**Verification plan**: Building on technical point 0, compare "teacher initialization + pure CE" vs "random initialization + KD distillation",
to see whether the former's ppl is already better than the latter's.

**Pass/fail criteria**:
- ✅ PASS: teacher initialization + pure CE is already better than KD distillation, confirming that removing the KD loss is correct;
- ❌ FAIL: KD distillation has an additional gain—reintroduce it.

**Implementation description**: Only requires a control training script, no core changes.

---

## 5. Step-by-Step Implementation Plan (Strictly in Order, Each Step Has a Gate)

| Step | Content | Gate (pass to proceed to the next step) | Status |
|---|---|---|---|
| 0 | Classic full pretraining baseline (`full_finetune.py`) | ppl < 77 | ✅ Done (ppl 8.73) |
| 1 | Technical point 1: λ_s structural sparsification minimal verification | router differentiates + ppl does not collapse | ❌ FAIL, stopped |
| 1-b | Technical point 1-b: full training MoE | router differentiates + approaches dense | ✅ Feasible (match 0.607) |
| 1-c | Sweep ffn_e ∈ {128,256,384} trade-off | Determine the bandwidth benefit upper bound | ✅ Done (ffn_e=128 sweet spot, 4.27x) |
| 2 | Technical point 2: λ_q quantization aware minimal verification | QAT better than PTQ | ❌ Rejected (PTQ +0.9% is sufficient) |
| 3 | (Optional) Technical point 3: initialization vs distillation | Confirm KD can be removed | ✅ KD already falsified by 1-b |
| 4 | Integration: fusion training tool | End-to-end produce a usable MoE model | Depends on prior steps |

**Stop-loss principle**: If any step FAILs, stop immediately, return to that technical point's hypothesis and re-analyze, and do not force it forward.

---

## 6. Cross-Check Against Axiom Library Constraints

| Axiom | How the fusion framework satisfies it |
|---|---|
| SHS-3 continuous reducibility law | Structural and quantization transformations go through a λ smooth transition; hard switching is forbidden (the classic paradigm backbone itself is unaffected) |
| SHS-2 superposition non-contradiction law | The transition state `(1-λ)·A ⊕ λ·B` simultaneously carries both sets of information, dense/sparse, without forcing a binary choice |
| SHS-T universal extension template | Every technical point is a λ instance of "classic basis state → new basis state" |
| Classic paradigm compatibility | Backbone = pure CE + AdamW + full training; λ only superimposes and does not replace, without diverging from the classic paradigm |
| Empirical over inferential | Every technical point mandates a minimal verification experiment, with hard data pass/fail |
| Bit-level consistency | λ_q quantization aware ensures determinism of the q4 weights, reproducible on the MCU side |
| Bandwidth-constrained | The goal of λ_s sparsification is "read only the top-k experts per token", reducing bandwidth |

---

## 7. Deliverables

1. This design document (fusing the classic paradigm + λ transformation auxiliary);
2. `full_finetune.py` (classic full pretraining baseline, step 0);
3. `train_lambda.py` (unified training tool for λ_s structural sparsification + λ_q quantization aware, output of steps 1~2);
4. Verification reports for each technical point (hard data comparing ppl / router entropy / QAT);
5. Documentation (README + moe_framework.md section 7 update).
