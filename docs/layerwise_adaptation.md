# Layer-wise Adaptation Optimization —— Future Research Direction

> 中文版见 **[layerwise_adaptation.zh-CN.md](layerwise_adaptation.zh-CN.md)**。

> **Tool directory note (2026-09-28)**: The historical research scripts mentioned in this
> document (`train_moe.py`, `train_moe_qat.py`, etc.) have been archived to **`tools/legacy/`**;
> the current mainline tools remain in `tools/`.
> For an overview of the tools, see `tools/README.md`.

> Positioning: This document records "layer-wise adaptation optimization" as a **future research direction** of the kestrel_mcu training framework,
> which has not yet entered formal technical-point implementation. Core question: how are large-parameter models trained under limited compute/memory?
> The answer is layer-wise differentiated adaptation, rather than a global uniform strategy.
>
> This document organizes the mathematical foundations in the axiom library that support "layer-wise adaptation", annotates the confidence of each, and gives a fusion scheme and an implementation path.

---

## 1. Problem Definition

"The key to running large-parameter models" is essentially three types of hard constraints:

| Constraint | Manifestation | Layer-wise adaptation's response |
|---|---|---|
| Memory | Activations + optimizer states exceed memory | Layer-wise freezing / gradient checkpointing (recompute activations) |
| Bandwidth | Per-layer weight read/write is limited | Layer-wise quantization (sensitive layers keep precision, insensitive layers compress aggressively) |
| Convergence | Deep/shallow layers have different learning rates | Layer-wise learning rate decay (LLRD) |

The current framework (`train_moe.py` / `train_moe_qat.py`) uses **globally uniform** λ annealing and quantization,
and has not yet done layer-wise differentiation. Layer-wise adaptation generalizes it into "an independent λ_m per layer".

---

## 2. Mathematical Foundations of the Axiom Library (sorted by confidence, honestly annotated)

### Foundation 1: λ Additivity Layer-wise Decomposition Law [confidence 0.72, consistent]

Location: `axiom_registry.json` L2812 (`shs_5faxiom_5fstatement_00552e8a`).

> For any λ1, λ2 ∈ [0,1] with λ1+λ2 ≤ 1, the two-step composition `result_λ2(result_λ1(x,y))` equals
> the single step `result_{λ1+λ2}(x,y)`.

**Meaning**: The global λ transition can be arbitrarily split into multiple segments and advanced layer by layer, with mathematical equivalence.

**This is the hardest legitimacy foundation for "layer-wise adaptation"** —— splitting the global λ annealing into one λ_m per layer, each independently
traversing the transition interval, does not violate the SHS system. Logical validation: consistent.

### Foundation 2: SHS-T Universal Extension Template [confidence 0.99, most reliable]

Location: `axiom_registry.json` L2795.

> `M(λ) = (1-λ)·classical ground state A ⊕ λ·new ground state B`, a four-step extension: ① A (λ=0) ② B (λ=1)
> ③ transition state M (0<λ<1) ④ λ=0 automatically maps to the classical.

**Meaning**: The adaptation of each layer (quantization/sparsification/freezing) is an instance `M_m(λ_m)` of this template.
Together with the SHS-3 continuous reducibility law (confidence 0.98) —— the layer-wise transition must also be continuous, and hard switching is prohibited.

### Foundation 3: Bregman Divergence Optimal Scheduling [confidence 0.65]

Location: `axiom_registry.json` L2536 (`shs_axiom_bregman_divergence_v1`).

> "minimum divergence path corresponds to optimal scheduling"
> (minimum divergence path = optimal scheduling).

**Meaning**: The **order** of layer-wise adaptation is not arbitrary; it should advance along the path of "minimum cumulative Bregman divergence" ——
at each step, select the layer with the smallest divergence increment to transition first. Engineering correspondence: perform layer-wise quantization/freezing ordered by layer sensitivity.

### Foundation 4: tensor graph decomposition [confidence 0.72]

Location: `axiom_registry.json` L2492.

> "Graph decomposition quality determines communication cost"
> (graph decomposition quality determines communication cost).

**Meaning**: The communication cost of layer-wise/tile-wise partitioning is determined by decomposition quality, providing "communication cost" as an optimization dimension for layer-wise splitting.

### Foundation 5: Depth Phase-Density Axiom [confidence 0.464, ⚠️ refuted]

Location: `axiom_registry.json` L9953 (`shs_5faxiom_5fstatement_6405ed15`).

> `λ_m = min(1, (min(π_m, π*) · d_m^α_m) · λ₁)`
> where π_m = number of phase-state channels of layer m (width), d_m = effective depth, α_m ∈ [0,1] = depth saturation exponent.

**Meaning**: Each layer's λ_m is uniquely determined by "that layer's width × depth" —— the bottom layers (shallow) and top layers (deep) proceed at different rates.

⚠️ **Honest annotation**: `symbolic_validator: refuted` (symbolic validation refuted), confidence only 0.464.
It can only serve as a **heuristic form** of "the layer-wise λ should be determined by layer attributes", and cannot be used directly as a reliable formula.

---

## 3. Fusion Scheme

Combine Foundations 1 and 3 into an implementable "layer-wise adaptation" framework (without relying on the refuted Foundation 5):

```
Global:      M(λ)   = (1-λ)·classical ⊕ λ·new                  (SHS-T, 0.99)
Per-layer:   M_m(λ_m) = (1-λ_m)·classical_m ⊕ λ_m·new_m        (λ additivity, 0.72)
Scheduling:  order = argmin cumulative Bregman divergence increment   (Bregman optimal scheduling, 0.65)
Constraint:  λ_m continuous transition, hard switching prohibited     (SHS-3, 0.98)
```

Key conclusion: **layer-wise adaptation = the "layer-wise instantiation" of global λ annealing**, both its legitimacy and its optimal order are supported by high-confidence
foundations; what is missing is only "the specific assignment rule for λ_m" (Foundation 5 is a refuted candidate, and a new one must be established separately).

---

## 4. Implementation Path (by priority)

| Priority | Direction | Description | Dependent Foundations |
|---|---|---|---|
| High | Layer-wise quantization | Different layers have different sensitivity to q4 (embed/router/expert/attention all differ); allocate bit width or λ_q transition rate by sensitivity | 1, 3 |
| High | Layer-wise learning rate decay (LLRD) | Bottom layers use a small lr (general features stable), top layers use a large lr (task specialization), a classical practice | 1 |
| Medium | Progressive freezing | Gradually freeze the bottom layers and train only the top layers in the later stage of training, saving memory | 1, 3 |
| Medium | Gradient checkpointing | Recompute activations layer by layer in exchange for memory, a standard practice for large-model training | 4 |
| Low | Layer-wise sparsification | Each layer independently transitions to MoE with λ_s, differentiated by layer sensitivity | 1, 3 |

## 5. Connection with Current Technical Points

- Technical point 2 (λ_q quantization awareness) currently uses a **whole-model uniform λ_q**; after layer-wise transformation it becomes "layer-wise quantization" ——
  this is the most natural first implementation step; it does not require a new axiom, and only needs to change `train_moe_qat.py`'s
  `set_lam_q(qparams, lam_q)` to `set_lam_q_per_layer(lam_q_dict)`.
- Layer-wise sparsification can reuse the λ annealing of technical point 1-b (full MoE), changing it to independent annealing per layer.

## 6. To-do / Open Questions

1. **New axiom draft**: based on Foundation 1 (λ additivity) + Foundation 3 (Bregman scheduling), derive a
   "layer-wise adaptation axiom" to replace the refuted Depth Phase-Density, annotate it as pending verification,
   and write it into `axiom_registry.json`.
2. **Layer sensitivity measurement**: use Bregman divergence to quantify the ppl loss increment of q4/sparsification per layer, as the
   sorting basis for layer-wise scheduling.
3. **Minimal verification**: on an MoE model with ffn_e=128, compare "layer-wise quantization vs whole-model quantization"
   to verify whether layer-wise transformation yields gains.

---

## 7. Conclusion

The axiom library **directionally supports** layer-wise adaptation: the λ additivity layer-wise decomposition law (0.72) is the hardest legitimacy foundation,
Bregman optimal scheduling (0.65) gives the order, and SHS-T/SHS-3 (0.99/0.98) give the form and constraints.
What is missing is a dedicated axiom for "layer-wise adaptation" —— the off-the-shelf Depth Phase-Density (0.464) is in a refuted state,
and a new one must be established based on high-confidence foundations. This is a complete topic worthy of being a future research direction.
