# Content-Level Agent Router — Formal Training and Capability Profile

> 中文版见 **[agent_loop_router.zh-CN.md](agent_loop_router.zh-CN.md)**。

> Positioning: This document records the complete pipeline of kestrel_mcu's first **"usable-grade" agent loop model**: task definition,
> data construction, multi-task training, held-out evaluation, capability probing (including causal ablation), PC and on-device deployment verification, and known limitations.
>
> Criteria for "usable-grade" (aligned with the user): ① genuine held-out evaluation (document-level split + per-class P/R/F1 + confusion matrix);
> ② preserved generation capability (multi-task LM anchoring + ppl comparison); ③ multi-step agent loop closed; ④ on-device deployment verification.
>
> One-sentence conclusion: a **reliable 4-class content router** (held-out 0.9986 / natural distribution 0.9981), whose core abilities are
> **in-window number detection (causal, position-independent)** and **question-sentence semantic recognition**; the error is compressed onto the
> **irreducible boundary** of "label ambiguity at the full 64-token window"; it has no instruction-following or reasoning capability.

---

## 1. Problem and Motivation

Prior work has already validated "decision head vs generation head": on an MCU, **decision tasks** should go through "a single forward pass + a small decision head",
rather than "full lm_head sweep + token-by-token generation" — a 4-orders-of-magnitude difference in read volume (cls_head 0.31KB vs lm_head 4.99MB).

But the first version of the action task had a **fatal methodological flaw**: labels were determined by the **last-byte punctuation** (`?`→route / `!`→ask_human /
`.`→stop / out-of-window→generate). Hard data falsified it: **freezing the pretrained core + training only a 1248-parameter linear head → 500 steps,
train acc 0.969**, showing that the task is separable at the surface level and solvable by a single linear probe, which **does not constitute "usable"**.

Accordingly, it was upgraded to a **content-level task**: labels must depend on the **entire window content**, and rules 3/4 require scanning all tokens in the window,
**not just reading the last byte**.

---

## 2. Task Definition

The action space is aligned with `KM_ACT_*` in `kmcu.h`:

| id | action | semantics | trigger (by priority) |
|---|---|---|---|
| 0 | `route` | needs solving/retrieval → route to QA | window contains `?` |
| 1 | `generate` | text incomplete → continue generating | window is truncated / no sentence-ending punctuation |
| 2 | `stop` | complete statement → no action needed | all other cases |
| 3 | `ask_human` | quantified content → escalate to human/tool verification | window contains digits |

**Label priority rules** (matched in order, one by one, `SL=64`):

1. token count > 64 → `generate` (window truncated, text incomplete)
2. end is not `.` `!` `?` → `generate` (a fragment without sentence-ending punctuation, treated as incomplete)
3. contains `?` → `route`
4. contains digits → `ask_human`
5. otherwise → `stop`

Rules 1/2 depend on length and sentence-ending punctuation; rules 3/4 require **scanning the whole window**. This is the essential difference from the old version (the last-byte shortcut).

---

## 3. Data Construction (`_prep_agent_data.py`)

Source: FineWeb-Edu English text (`fineweb_edu_train.txt`).

| set | size | sampling method |
|---|---|---|
| train | 800,000 (200k per class) | balanced |
| val | 40,000 (10k per class) | balanced |
| test | 40,000 (10k per class) | balanced |
| **nat (natural distribution)** | 40,000 | **no sampling**, reflects the true prior |

- **Document-level split**: bucketed by `doc_index % 10` (0–7 train / 8 val / 9 test) to avoid same-document leakage.
- **Natural distribution prior**: `stop` 71.1% / `ask_human` 22.0% / `route` 3.3% / `generate` 3.6%
  → **majority-class baseline 0.710**, the passing reference for natural acc.
- padding: `pad_id = 2` (`</s>`). Window fixed at 64 token, padded if shorter.

### Pitfalls Encountered

1. **Only encode the first 65 token**: labels depend only on "whether it exceeds the window" and sentence-ending punctuation, so no full-segment tokenize is needed.
   The first version fully encoded the whole segment (up to 8000 token), making preprocessing too slow to use; after switching to `truncation=True,
   max_length=SL+1`, it dropped to the minute level.
2. **Bucketing bug**: the first version bucketed by "the doc number at the time of the whole-batch flush", while one batch contains about 200 documents →
   **about 90% of sentences were assigned to the wrong bucket** (val/test contaminated by train). Only after switching to precise bucketing by **the doc a sentence belongs to** was true isolation achieved.

---

## 4. Model

Authoritative source for the architecture: `kestrel_mcu/tools/mistral_ple.py` (`MistralPLE`).

| component | configuration |
|---|---|
| backbone | Mistral: GQA attention (n_heads 12 / n_kv_heads 4 / head_dim 26), RMSNorm, RoPE(θ=1e4), SwiGLU |
| MoE | per-layer router + 8 experts, top-2, expert hidden `ffn_e=128` |
| PLE | per-layer embedding lookup: `ple_model_proj` + `ple_table` (vocab × n_layers×ple_dim), **ternary quantization STE** (1.6 bit/element) |
| head | `tied` (lm_head reuses embed_tokens) + `cls_head` (dim→4, mean pooling) |
| scale | vocab 32002 / dim 312 / 10 layers; **64.36M** parameters |

Decision-head forward: `transformer → final_norm → mean pooling → cls_head → [4]` (`forward_features` / `cls_forward`).
Initialized from `moe_ple_fw` (FineWeb-Edu pretrained, not random).

---

## 5. Training (`_train_agent_model.py`)

**Multi-task objective** (key design): `loss = CE(cls) + λ·CE(lm)`, λ=1.0.
LM data is drawn from **random 64-token windows** of `fw_train_big` (`bs_lm=16`). Its purpose is to **anchor generation capability** —
the agent's `generate` action relies on the same core to continue writing, and training end-to-end on cls alone would degrade generation.

| hyperparameter | value |
|---|---|
| two param groups lr | core `5e-5` / cls_head `1e-3` |
| schedule | cosine + warmup 200 |
| optimizer | AdamW, wd 0.01, grad clip 1.0 |
| bs | cls 32 / lm 16 |
| budget | target 38000 steps, **5.5h wall-clock cap** → actual run **33086 steps / 330 min** |

**Safety valve**: save a checkpoint every 5k steps and evaluate val+ppl; **auto stop-loss when ppl drift > 10%** (not triggered this time).

---

## 6. Results (all from the held-out set, not the training batch)

| metric | value |
|---|---|
| test balanced acc / macroF1 | **0.9986 / 0.9986** |
| test natural distribution acc | **0.9981** (majority-class baseline 0.710) |
| test natural distribution macroF1 | 0.9930 |
| per-class F1 (balanced test) | route 0.9995 / generate 0.9973 / stop **0.9996** / ask_human 0.9978 |
| **ppl comparison** | 48.01 → **44.86 (−6.6%)** |

> Note on the ppl definition: computed teacher-forced over 512 64-token windows (about 32k token) on `fw_val_big`,
> with the baseline being the **un-finetuned moe_ple_fw**, using the same code and the same data. This absolute value **cannot be directly compared** with other historical ppl numbers;
> only the A/B within this run is valid.

### 6.1 ppl Drift **Is Not Monotonically Worsening**

| step | 5k | 10k | 15k | 20k | 30k |
|---|---|---|---|---|---|
| ppl drift | +5.7% | +3.0% | +0.4% | −2.4% | **−6.0%** |

`CE(lm)` anchoring not only prevents degradation but also lets generation quality **gradually overtake the baseline** during training. This also falsified the early-training concern.

### 6.2 Generation Diversity A/B (`_verify_gen_ab.py`)

In-distribution real text (FineWeb-Edu validation-set prefixes), 3 prompts × 32 token greedy:

| model | distinct-1 | distinct-2 | longest repeated run |
|---|---|---|---|
| base moe_ple_fw | 0.365 | 0.568 | 2 |
| **final agent_model** | **0.438** | **0.589** | **1** |

### 6.3 A Self-Correction of One Misjudgment

The report at the end of training once showed: ppl improved by −6.6%, but the **greedy sample repetition worsened** (`28774` repeated consecutively),
based on which I once judged "generation degraded". After doing an A/B with in-distribution prompts, **that conclusion was falsified**:
diversity was actually better. The source of the misjudgment was a **single garbled OOD prompt** (`[1,450,2217,4996]`, with tokens falling in the
byte-fallback high-id region), whose output was half-noise for both models and does not constitute evidence.

---

## 7. Capability Probing (`_probe_agent_caps.py`)

"Overall acc" alone is insufficient to explain the capability boundary. The following are the per-item probing results.

### 7.1 Structural Attribution of the Error

In the confusion matrix, the errors are **almost all "X → generate"** (test: route→gen 9 / stop→gen 6 / ask_human→gen 40),
while `generate` has recall **1.000** and precision only 0.9945 → **it is the "attractor class"**.

Bucketed by the window's **true length** (not the pad token count):

| true length | n | acc | share truly generate in this bucket |
|---|---|---|---|
| 0–31 | 24260 | 0.9999 | 5.2% |
| 32–47 | 5316 | 0.9998 | 3.7% |
| 48–55 | 1171 | **1.0000** | 3.9% |
| 56–59 | 409 | **1.0000** | 4.2% |
| **60–63** | 321 | **0.9751** | 7.8% |
| **64 (full window)** | 8523 | **0.9945** | 99.3% |

The entire test set has about 56 errors, **about 84% of which fall in the 64-token full-window bucket**, and the error rate of the 60–63 bucket is about 8 times that of the other buckets.

**The root cause is the ambiguity of the label definition itself**: a **complete sentence of exactly 64 token** and "truncated by the window" are
**information-theoretically indistinguishable** within a 64 window. → This is an **irreducible error**, not the model being weak. To eliminate it, one must change the task definition
(add an explicit EOS / continuation marker, or extend the window), rather than continue training.

### 7.2 Causal Ablation / Counterfactual Edit (the most valuable finding)

| id | operation | result | meaning |
|---|---|---|---|
| B1 | ask_human **erase the digit-character token** | **98.6% flip to stop** | digit detection is **truly causal** |
| B2 | route **erase `?`** (→`.`) | only **1.3%** flip to stop | question judgment **does not rely on punctuation** |
| B3 | stop **append `?` at the end** | only **29.4%** flip to route | same as above |
| B4 | stop **append a digit at the end** | **99.8%** flip to ask_human | digit detection is causal |
| B5 | stop **overwrite one token in the middle with a digit** | **99.9%** flip to ask_human | **position-independent** (true in-window needle) |

**Conclusion**:
- For **digits**, the model detects them by literal token, **causally and position-independently** — a real capability rather than a correlation.
- For **questions**, the model uses **semantic/lexical cues** rather than the sentence-final `?`. This is a **generalization advantage** (it still reads the interrogative mood after `?` is removed),
  but it is **inconsistent with the training label rules** (the rules are defined by the character `?`) → on such counterfactual inputs it "violates the rules".

### 7.3 The "Visible Evidence Amount → Decision Usability" Curve

Giving only the first k real tokens (the rest padded), content three-class only (`generate` is excluded since it is defined by length):

| k | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| acc | 0.0014 | 0.0467 | 0.1735 | 0.3051 | 0.4304 | 0.6328 | 0.7823 | 0.9426 | **0.9981** |

Monotonically increasing, isomorphic to the "observation feedback" experiment. Note when reading: **labels are defined by the whole window**, so the curve contains a k/64 coverage effect —
it measures "visible evidence amount → decision usability", not exactly "how much context is needed".

### 7.4 OOD Robustness: 9/9 Correct

| input | expected (by the rules) | prediction |
|---|---|---|
| `The theory was developed over many decades.` | stop | ✅ stop |
| `The sample contained 42 participants in total.` | ask_human | ✅ ask_human |
| `What is the purpose of this experiment?` | route | ✅ route |
| `这个实验的目的是什么?` | route | ✅ route |
| `该实验共有42名参与者。` (`。` ≠ `.`) | generate | ✅ generate |
| `123 456 789 0 12 34` | generate | ✅ generate |
| `x = 3 + 5 ; print(x)` | generate | ✅ generate |
| `Hi` | generate | ✅ generate |
| `.` | stop | ✅ stop |

Cross-language / code / pure digits / extremely short all hit according to the rules.

> **Self-correction**: the first version of the probe marked 4 of these cases as "model errors", but in fact **I missed rule 2 when assigning the expected values**
> (any fragment not ending in `.!?` is always `generate`). The model's judgment of `generate` is correct; **it was a bug in my script**.

> **Probing pitfall**: this vocab encodes digits as **two tokens** — `28705` (empty string, shared by all digits) +
> a digit-character token (`28734/28740/28750/28770/28774/28781/28782/28783/28784/28787`).
> The first version replaced only the shared `28705` → the ablation was a **false negative** (0.46% flip). One must take `encode(c)[-1]`.

### 7.5 Generation Capability

8 in-distribution prompts × 32 token greedy: **pathological repetition (maxrun ≥ 4) 0/8**, distinct-1 0.316.
Samples:

```
'ifies the most vulnerable populations. The report also shows that the '
'inity, and temperature. The temperature of the air is 100°C (100°F) an'
```

It is **fluent English educational-text continuation** (the semantics will drift). Only OOD garbled prompts cause pathological repetition.

### 7.6 What It Explicitly **Cannot** Do

```
"Please summarize the following text:"  -> '- 1. The following text is from the text of the text: -'
"Q: What is 2+2? A:"                    -> 'What is 2+2? A: What is 2+2?'
```

**Not an instruct model**: it does not follow instructions or perform question-answering reasoning, it only **continues writing**.
It is a "content router + weak continuer", not a general-purpose assistant.

---

## 8. Deployment Verification

### 8.1 PC Full-Pipeline A/B Comparison

`convert_minueza.py` → **20.77 MB** KMCU (346 tensors, n_cls=4, tied, MoE 8 experts top-2, ple_dim 128).

`host_verify` (C, x86) vs `ref_forward.py` (Python):

| item | result |
|---|---|
| greedy sequence (20 token) | **token-identical** |
| cls logits (8-token) | diff ~4.8e-4, argmax **both id=2** |
| agent loop 8 steps | all action=1, gen token identical |

The magnitude of the difference comes from q4 (core) + ternary (PLE table) quantization, as expected.

### 8.2 On-Device

**Obstacle**: the formal 64M model is **20.77 MB > Flash `model` partition 13.87 MB** (16MB on-chip),
so it must go through the SD card; but the SD card slot had a hardware failure at the time (see 8.3 for details).

**Solution**: added a `FLASH_AGENT` deployment channel (`main.c`) — the model is read from the Flash `model` partition →
`km_fast_build_ex` expands it into PSRAM → reuse the same set of three modes `run_generate / run_dual_head / run_agent`
(serial port `0/1/2/q` dynamic switching). Verified with a
17.42M small model with **the same task, the same n_cls=4, the same code path** (`dim128/6层/ffn_e64/n_expert4/ple_dim64`, **5.13 MB** after convert, fits in Flash):

| item | on-device (RISC-V) | PC reference |
|---|---|---|
| 20-token sequence | **token-identical** | — |
| single-token top8 | same order, logit diff ~1.7e-5 | — |
| mode 1 decision head | `action=1 logits: -4.4643 4.7823 -2.4585 -1.9285` | agent s0 `-4.464250 4.782335 -2.458621 -1.928520` |
| mode 2 agent 8 steps | all action=1, gen token all equal | same |
| decode speed | **19.5 tok/s**, TTFT(prefill 4)=199 ms | — |

**Code changes** (authoritative source `kestrel_mcu/main/main.c`): relaxed 3 guards by adding `||FLASH_AGENT`
(SD includes block / SD feature block / three-mode block) + forward declarations of `run_*` + eliminated the duplicate definition of `k_prompt`
(the top guard changed from `!SD_STREAM && !SD_RESIDENT` to `!SD_STREAM`).

### 8.3 SD Card Failure, Hardening, and Recovery

**Phenomenon**: `sdmmc_init_ocr ... 0x107` / deep-sector read `0x107`; in one round all 6 combinations failed, and after reseating and cleaning
**all 8 combinations still failed**, and even 400kHz failed in `sd_host_slot_clock_update_command` (during the CMD-send phase)
→ communication was completely dead, judged to be a **card-socket/power-circuit or card itself failure** (not software).

**Hardening** (written into `main.c`; the two paths `sd_mount_stream` and `probe_sdcard` share the same table):

1. Added **4 MHz / 400 kHz** low-speed gears under `width=1` to the retry table.
2. Added a **deep-sector read verification** after mounting (sector 34957 + the capacity midpoint), accepting only configurations that **truly can read** —
   otherwise there would be a **false positive** where "init passes, mount succeeds, but data reads fail".
3. Changed `format_if_mount_failed` from `true` to **`false`**: the original value would **automatically format the card** on mount failure,
   with a real risk of losing `model.kmcu`.

**Recovery (re-verified on 2026-09-28)**: detection passed after the user reseated the card —

```
[SD] try pwr=0 width=1 4000kHz ... -> ESP_OK
[SD] read-verify @34957 -> ESP_OK , @30533632 -> ESP_OK
Name: TW000  Type: SDHC  Speed: 4.00 MHz  Size: 29818MB  (bus_width=1)
[SD] model.kmcu first block read 512 bytes (magic=KMCU)
```

**The winning combination was exactly the newly added 4 MHz gear**: logs show that at 20 MHz/1-bit, "@34957 readable, @capacity-midpoint timeout"
would be judged a failure, while the old table **had no low-speed gear at all**, so it would directly misreport "all failed".

**Known gap**: the **protocols are out of sync** between `tools/upload_uart.py` (plug-free UART upload) and the firmware `uart_upload_model` —
the script waits for `READY(0xAA)` and the status bytes `0x55/0xEE/0xE3` and stays at 115200 throughout,
while the firmware prints "switching to 921600 after 3 seconds" and sends neither `0xAA` nor status bytes; moreover the firmware enters upload mode only when there is **no**
`model.kmcu` on the card. So currently only "PC card-reader writing" works.

### 8.4 Three Pitfalls in the ESP-IDF Build Environment (unrelated to the model, all exposed only when building on the PC side)

| phenomenon | root cause | solution |
|---|---|---|
| `git submodule init failed for components/esp_wifi/lib` (gitee 423) | submodule repository is blocked | `IDF_SKIP_CHECK_SUBMODULES=1` (this project has WiFi/BT disabled, safe) |
| `std::filesystem ... Illegal byte sequence` | **ccache on a non-ASCII path** (`E:\项目\...`) | `idf.py --no-ccache` + put the build in a pure-ASCII directory (`C:\kmcu`) |
| `fatal error: bits/error_constants.h: No such file` | **MAX_PATH exceeded**: the header's full path is 259 characters (`c++config.h` in the same directory at 251 characters passes), the prefix `C:\Users\Administrator\.espressif\...` is too long | create a directory junction `C:\et` → `.espressif`, set `IDF_TOOLS_PATH=C:\et` (shortens by 28 characters, **changes no system settings**) |

> Lesson: **ESP-IDF builds on Windows are extremely sensitive to "non-ASCII path + long path + submodule network"**, and
> any one of them can make the on-device build non-reproducible. After fixing these three points, the build passes stably.

---

## 9. Conclusion and Limitations

### Capability Profile

> A **reliable 4-class content router** (held-out 0.9986 / natural distribution 0.9981), whose core abilities are
> **in-window number detection (causal, position-independent)** and **question-sentence semantic recognition**; `generate` is the
> **fallback class** for "low-confidence/needs continuation"; the error is compressed onto the irreducible boundary of **label ambiguity at the full 64-token window**;
> it **has no instruction-following or reasoning capability**.

### Engineering Implications

- Downstream should treat `generate` as **low-confidence/needs continuation**, not as a hard classification.
- To squeeze out that 0.14% error, one should **change the task definition** (add EOS / continuation markers, or extend the window), rather than continue training.
- If concerned about the consistency of `?` judgment, note that the model actually uses **semantic cues**: on counterfactual inputs such as "a declarative sentence appended with `?`",
  it violates the rules (B3 flips to route only 29.4%).

### Limitations (An Honest List)

1. Although the task is content-level, it **saturates very quickly** (0.995 reached by step 4000) — it is a reliable **engineering-grade router**,
   not a hard problem requiring deep reasoning.
2. **The formal 64M model was not run on-device**: 20.77 MB exceeds the Flash partition, and although the SD path has recovered, the on-device three-mode A/B of the
   formal model has not yet been completed (current on-device verification uses the same-task small model).
3. There is a systematic deviation between the `route` judgment and the label rules (semantics vs punctuation), so the evaluation metrics **slightly overestimate**
   the strictness of "executing by the rules".
4. ppl uses a custom definition and is not comparable across runs.

---

## 10. Appendix: Compatibility with the vLLM-Kestrel Engine (`GitHupSRC/`)

**Conclusion: cannot run directly** (purely static assessment, no code changed; archived). Four hard obstacles:

| # | obstacle | evidence |
|---|---|---|
| 1 | **Geometry (decisive)**: the engine blocks by integer `cols/32`, so for `dim=312` (312/32=9) the trailing 24 elements are silently dropped; the MoE sparse kernel has `if (d & 31) return;` → **the whole-layer FFN is a no-op**; `head_dim=26` triggers `hd/32=0` | `vllm_safetensors.c:12940,797,2059,1238` |
| 2 | **No corresponding operator for PLE** (the engine has no `ple_` identifier at all) | `vqf_format.h:47-52` |
| 3 | **No qtype for ternary** (only F32/Q8_0/Q4_0/Q4_8X8L/F16) | same as above |
| 4 | **No exit for the decision head** (`STModelWeights` has no cls_head, only logits+sampling) | `vllm_safetensors.h:123-225,504` |

Two other hidden pitfalls: config key names mismatch (this model writes `n_expert` vs the engine reads `num_experts`) → MoE is treated as dense;
tensor names mismatch (`mlp.router.weight` vs the engine's `mlp.gate.weight`).

**Compliance point**: H1 bit-level consistency only covers paths **within** the engine, and **cross-engine bit-for-bit consistency with this project's `kmcu.c` does not hold**
(the engine's prefill/q8 **quantizes activations to int8 for integer dot products**); using the `VLLM_ACTQ` approximate track can only count as the FAST grade.
**In the engine's context, the PLE table is a resident-memory problem rather than a bandwidth problem** (only 1 row is read per token, but the table itself is
f32 ≈ 156 MB / q8 ≈ 39 MB / ternary ≈ 8 MB).

**Key judgment**: the crux is that the `dim=312 / head_dim=26` geometry was hand-picked for the ESP32-P4, and **it inherently violates the engine's
32-alignment red line**. So "porting the existing 64M model over" has extremely low cost-effectiveness; **the correct approach is to design the model in the opposite direction, according to the engine's geometry**
(take dim as a multiple of 32 such as 320, head_dim=32); the MoE skeleton can reuse the existing kernel, and only PLE and the decision head need a project-level decision.

---

## 11. Artifacts and Reproduction

### Artifacts

| path | description |
|---|---|
| `E:\models\agent_model` | formal model HF directory (64.36M) |
| `E:\models\agent_model.kmcu` | deployment image (20.77 MB, n_cls=4) |
| `E:\models\agent_model_report.json` | evaluation report (per-class metrics + confusion matrix + ppl + generation) |
| `E:\models\agent_model.log` | training log (including val every 2k steps, ppl checkpoint every 5k steps) |
| `E:\models\agent_model_flash_step{5000,10000,15000}` | small-model checkpoints for on-device verification |
| `E:\models\agent_flash.kmcu` | small-model deployment image (5.13 MB, fits in Flash) |
| `E:\models\agent_x.bin` / `agent_y.bin` etc. | data (four sets: train/val/test/nat) |

### Reproduction Commands

```powershell
# 1) data (content-level task + document-level split)
python E:\models\_prep_agent_data.py

# 2) formal training (5.5h wall clock)
python E:\models\_train_agent_model.py --steps 38000 --time_budget_h 5.5 --out E:\models\agent_model

# 3) convert + PC full-pipeline A/B comparison
python kestrel_mcu\tools\convert_minueza.py --src E:\models\agent_model --out E:\models\agent_model.kmcu
python kestrel_mcu\tools\ref_forward.py --km E:\models\agent_model.kmcu
gcc -O2 -std=c11 -I main -o host_verify host_verify.c main/kmcu.c -lm   # under kestrel_mcu\
.\host_verify.exe ..\agent_model.kmcu

# 4) capability probing / generation A/B
python E:\models\_probe_agent_caps.py
python E:\models\_verify_gen_ab.py

# 5) on-device (FLASH_AGENT small-model path)
#    main.c: SD_PROBE=0 / SD_RESIDENT=0 / FLASH_AGENT=1
idf.py --no-ccache build ; idf.py -p COM5 flash
python -m esptool --chip esp32p4 -p COM5 -b 460800 write-flash 0x210000 agent_flash.kmcu
#    serial port 0/1/2/q switches generate / dual-head / agent
```

> Before building on Windows / ESP-IDF, set: `$env:IDF_TOOLS_PATH="C:\et"` (the junction shortens the toolchain path),
> `$env:IDF_SKIP_CHECK_SUBMODULES=1`, and use `idf.py --no-ccache`.
