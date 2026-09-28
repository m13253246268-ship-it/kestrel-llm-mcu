# Kestrel-MCU — Tools

Host-side toolchain for training, quantizing, and verifying the models that run on the
ESP32-P4. Everything here is Python 3 (plus one C reference in `../host_verify.c`).

> 中文版见 **[README.zh-CN.md](README.zh-CN.md)**。

---

## Pipeline overview

```
                         tools/train_moe_ple.py
                                   │
                                   ▼
                     HF model dir  (config.json + model.safetensors)
                                   │
                         tools/convert_minueza.py
                                   │
                                   ▼
                             model.kmcu  ──── flash / SD card ───►  ESP32-P4
                                   │                              (main/kmcu.c)
                                   │
                      tools/ref_forward.py  (NumPy fp32 reference)
                                   ▲
                                   │   token-by-token A/B
                             host_verify.c  (same kmcu.c source, compiled for PC)
```

`mistral_ple.py` is the **single authoritative architecture source**: the trainer,
`convert_minueza`, `ref_forward`, and `main/kmcu.c` all follow the tensor naming and
numerical conventions defined there.

---

## Requirements

| Package | Used by | Notes |
|---|---|---|
| `numpy` | convert, ref_forward | required |
| `torch` | train, mistral_ple | required for training only |
| `transformers` | train (tokenizer) | required for training only |
| `safetensors` | convert (read), train (save/load) | required |
| `pyserial` | watch_boot, upload_uart | required for board tools |
| `scipy` | ref_forward (`erf`) | **optional** — falls back to `math.erf` |

```bash
pip install numpy torch transformers safetensors pyserial
# optional: pip install scipy
```

For the on-device build you additionally need **ESP-IDF 5.3+** with target `esp32p4`.
See the project [README](../README.md) for the Windows-specific build caveats
(non-ASCII paths, `MAX_PATH`, ccache, submodules).

---

## Tool index

| Tool | Role | Typical use |
|---|---|---|
| [`mistral_ple.py`](#1-mistral_plepy--architecture-source) | Architecture definition (library) | imported by trainer / converter |
| [`convert_minueza.py`](#2-convert_minuezapy--hf--kmcu) | HF safetensors → KMCU quantized image | deploy |
| [`ref_forward.py`](#3-ref_forwardpy--numpy-reference-forward) | NumPy fp32 reference forward | correctness A/B |
| [`train_moe_ple.py`](#4-train_moe_plepy--training) | Training (Mistral + MoE + PLE) | train |
| [`watch_boot.py`](#5-watch_bootpy--boot-monitor) | Reset board + stream boot log | board diagnostics |
| [`upload_uart.py`](#6-upload_uartpy--uart-upload-experimental) | Push `model.kmcu` to the TF card over UART | ⚠️ experimental |
| [`legacy/`](#legacy--archived-research-scripts) | Archived research scripts | reproduce past pass/fail results |

---

## 1. `mistral_ple.py` — architecture source

Library (no CLI). Defines `MistralPLEConfig` and `MistralPLE`.

### Architecture

Mistral backbone (GQA + RMSNorm + RoPE + SwiGLU) with three additions:

* **MoE FFN** — per layer: `router` + `n_expert` experts (`gate`/`up`/`down`),
  `softmax → top_k → renormalize → weighted sum`;
* **PLE (Per-Layer Embeddings)** — a per-layer lookup table, ternarized
  (`{-1,0,+1} × absmean`, straight-through estimator) during training;
* **heads** — `tied` (lm_head reuses `embed_tokens`) and an optional
  `cls_head` (`n_cls > 0`), used by the agent-loop classifier.

PLE numerics (copied formula-for-formula from the reference implementation):

```
x0    = embed_tokens(idx)
ple   = ple_model_proj(x0) * dim^-0.5                  # [seq, L*ple_dim]
ple   = RMSNorm(ple.view(seq, L, ple_dim))
table = ple_table(idx)                                 # [seq, L*ple_dim]  ternary
ple   = (ple + table * ple_dim^0.5) * 2^-0.5
# per layer i, after attention + FFN:
g = gelu(ple_gate_i(x))
x = x + RMSNorm(ple_proj_i(g * ple[:, :, i]))          # ple_proj: ple_dim -> dim
```

### `MistralPLEConfig` fields

| Field | Default | Meaning |
|---|---|---|
| `vocab_size` | 32002 | tokenizer vocab (Mistral) |
| `dim` | 312 | model width |
| `n_layers` | 10 | depth |
| `n_heads` / `n_kv_heads` | 12 / 4 | GQA; `head_dim = dim // n_heads` |
| `ffn_e` | 128 | **expert** hidden size (`intermediate_size` in HF config) |
| `n_expert` / `top_k` | 8 / 2 | MoE geometry (`n_expert=0` → dense) |
| `ple_dim` | 128 | per-layer lookup width (`table_width = n_layers * ple_dim`) |
| `seq_len` | 512 | RoPE table length |
| `rope_theta`, `norm_eps` | 10000.0, 1e-6 | |
| `tied` | True | lm_head reuses `embed_tokens` |
| `has_ple` | True | enable the PLE branch |
| `n_cls` | 0 | decision-head classes (`0` = no head, keeps generation path untouched) |
| `ternary_table` | True | ternarize `ple_table` with STE |

### Key methods

| Method | Returns | Notes |
|---|---|---|
| `forward(idx, targets=None)` | `(logits, loss)` | language-model path; `loss` is CE with `ignore_index=-1` |
| `forward_features(idx)` | `[B, T, dim]` | last hidden states (decision-head branch) |
| `cls_forward(idx)` | `[B, n_cls]` | `mean pool → cls_head` |
| `param_budget()` | `{core, stream, table, total}` | parameter accounting by access pattern |
| `export_state_dict()` | `dict` | HF-style keys; matches what `convert_minueza` reads |
| `save_pretrained(dir)` | — | writes `model.safetensors` + `config.json` |

```python
import sys; sys.path.insert(0, "tools")
from mistral_ple import MistralPLE, MistralPLEConfig

cfg = MistralPLEConfig(vocab_size=32002, dim=312, n_layers=10, n_heads=12,
                       n_kv_heads=4, ffn_e=128, n_expert=8, top_k=2,
                       ple_dim=128, seq_len=64, n_cls=4)
model = MistralPLE(cfg)
print(model.param_budget())
```

---

## 2. `convert_minueza.py` — HF → KMCU

Converts a HuggingFace model directory into a **KMCU v1** quantized image that
`main/kmcu.c` can mmap/stream on the device.

```bash
python convert_minueza.py --src <hf_dir> --out <file.kmcu> [--top_k 2]
```

| Argument | Default | Meaning |
|---|---|---|
| `--src` | `model_dir` (env `KMCU_SRC`) | source dir containing `config.json` + `model.safetensors` |
| `--out` | `model.kmcu` (env `KMCU_OUT`) | output image |
| `--top_k` | 2 | MoE top-k (ignored for dense; config `top_k` wins if present) |
| `--force-tie` | off | force `lm_head` to reuse `tok_embed` (drops the trained output head). **Lossy** — reproduces the historical M1–M3 image (Minueza: 22.81M / 12.58 MB, fits the Flash `model` partition) |

> Defaults are relative to the current directory. Override them with `--src`/`--out`
> or the `KMCU_SRC`/`KMCU_OUT` environment variables.
>
> `tied_embed` is taken from `config.json` (`tie_word_embeddings`) unless `--force-tie` is passed.
> The cost of forcing the tie is **not** just the 9.98M saved parameters: on Minueza the two images
> produce **completely different** greedy text — forced-tie yields word salad, the untied default
> yields fluent English. See the article series, `02-模型量化与KMCU格式`.

The converter also writes `<src>/model.kmcu.manifest.txt` (human-readable header +
per-tensor table) for cross-checking.

### Auto-detection rules

| Condition | Effect |
|---|---|
| `model.layers.0.mlp.router.weight` present | **MoE**: store `router` (f32) + per-expert `gate`/`up`/`down` (q4); header `ffn_dim` = expert hidden |
| config `ple_dim > 0` | **PLE**: export `ple_model_proj`, `ple_proj_norm`, per-layer `ple_gate`/`ple_proj`/`ple_norm`, plus `ple_table` (ternary) |
| `cls_head.weight` present | export `cls_head` (f32), header `n_cls` |
| `tie_word_embeddings` | `tied=1` → no `lm_head` stored; otherwise `lm_head` (q4) is stored if present |

### File layout (little-endian)

```
[0,128)                                  Header
[128, 128 + n_tensors*48)                Directory (48 B per entry)
[data_offset, ...)                       Data area, each tensor 64 B aligned
```

**Header**

| Offset | Type | Field |
|---|---|---|
| 0 | `char[4]` | magic `"KMCU"` |
| 4 | u32 | version = 1 |
| 8 / 12 / 16 / 20 / 24 | u32 | `n_layers` / `dim` / `n_heads` / `n_kv_heads` / `head_dim` |
| 28 / 32 / 36 | u32 | `ffn_dim` (expert hidden for MoE) / `vocab_size` / `max_seq` |
| 40 / 44 | u32 | `bos_id` / `eos_id` |
| 48 | u32 | `tied_embed` (1 = reuse `embed_tokens`) |
| 52 / 56 / 60 | u32 | `n_tensors` / `dir_offset` / `data_offset` |
| 64 / 68 | f32 | `rope_theta` / `norm_eps` |
| 72 | u32 | `total_params` |
| 76 | u32 | `n_expert` (0 = dense) |
| 80 | u32 | `top_k` |
| 84 | u32 | `ple_dim` (0 = no PLE) |
| 88 | f32 | `ple_gamma` (ternary table scale) |
| 92 | u32 | `n_cls` (0 = no decision head) |

**Directory entry (48 B)**

| Offset | Type | Field |
|---|---|---|
| 0 | `char[24]` | name (ASCII, NUL-padded) |
| 24 | u8 | dtype; 25..27 pad |
| 28 / 32 | u32 | `n_rows` / `n_cols` (element counts; **original** columns) |
| 36 | u32 | absolute file offset |
| 40 | u32 | stored bytes |
| 44 | u32 | reserved |

**Dtypes**

| Value | Name | Encoding |
|---|---|---|
| 1 | `F32` | raw little-endian f32 (norms, router, `cls_head`) |
| 2 | `Q4_0` | 18 B / 32 elements: `[0,2)` fp16 scale `d`, `[2,18)` 16 B low/high nibbles. `d = max_abs/8`, `q = round(w/d)` clamped to `[-8,7]`, `w = (nibble-8)*d`. **Each row is zero-padded to a multiple of 32 so blocks never straddle rows.** |
| 3 | `TERNARY` | 5 trits / byte, base-3 (`byte = Σ trit_i·3^i`), row-aligned to `ceil(cols/5)` bytes. `code = clip(round(w/gamma),-1,1)`, `trit = code+1 ∈ {0,1,2}`, `gamma = mean(|w|)`. |

The `n_cols` field always records the **original** column count; the device derives
`nblk = align32(n_cols)` itself.

```bash
# dense Minueza
python convert_minueza.py --src E:/models/minueza --out E:/models/model.kmcu
# MoE + PLE + decision head
python convert_minueza.py --src E:/models/agent_model --out E:/models/agent_model.kmcu
```

Expected output looks like:

```
[moe] n_expert=8 top_k=2 ffn_e=128
[cls] n_cls=4
n_tensors=346  params=64.36M  q4=23.36M
file size = 20.77 MB   (tied_embed=1, moe=True n_expert=8 top_k=2 ple_dim=128 ple_gamma=0.028555)
written: agent_model.kmcu
```

---

## 3. `ref_forward.py` — NumPy reference forward

Reads a KMCU image and runs an **fp32 reference forward** in NumPy to produce the
greedy token sequence — the ground truth for A/B against the device.

The reference dequantizes the **same** quantized weights the device uses, so any
discrepancy should come only from floating-point summation order/rounding, not
from the weights themselves.

```bash
python ref_forward.py --km <file.kmcu> [--ref_out ref_out.txt]
```

Prints: header dict, tensor count, prompt, single-token top-8 (a bisection anchor:
`pos=0` ⇒ RoPE identity and degenerate attention), top-8 at first generation,
the 16-token greedy sequence, and (when `n_cls > 0`) the decision-head logits.

### Use as a library

```python
import sys; sys.path.insert(0, "tools")
import ref_forward as rf

arch, T = rf.load_kmcu("E:/models/agent_model.kmcu")
out  = rf.generate(arch, T, [1, 450, 2217, 4996], 16)          # greedy
out2 = rf.generate(arch, T, [1, 450, 2217, 4996], 16, rep_penalty=1.2)
cl   = rf.cls_forward(arch, T, [1, 450, 2217, 4996, 522, 28723, 415, 907])
```

| Function | Signature | Notes |
|---|---|---|
| `load_kmcu(path)` | `→ (arch, tensors)` | parses header + directory |
| `forward(arch, T, tokens)` | `→ logits [T, vocab]` | full sequence |
| `generate(arch, T, ids, n, rep_penalty=1.0)` | `→ list[int]` | greedy; `rep_penalty` matches `kmcu.c` semantics (positive logits divided, negative multiplied, reset at `pos==0`) |
| `cls_forward(arch, T, tokens)` | `→ [n_cls]` | transformer → `final_norm` → mean pool → `cls_head` |

The A/B counterpart on the PC is `../host_verify.c`, which compiles the **same**
`main/kmcu.c` used on the device:

```bash
gcc -O2 -std=c11 -I main -o host_verify host_verify.c main/kmcu.c -lm
./host_verify <file.kmcu>
```

---

## 4. `train_moe_ple.py` — training

Trains `MistralPLE` on a pre-tokenized flat token stream. Recipe:
**full training + pure CE + AdamW + cosine**, no freezing, no KD, no dense fallback.

```bash
python train_moe_ple.py --out_dir moe_ple_fw --max_steps 30000
python train_moe_ple.py --smoke          # 1 fwd+bwd, checks the setup runs
```

Data: flat `uint16` token arrays (Mistral vocab 32002), e.g. `fw_train.bin` / `fw_val.bin`.

| Argument | Default | Meaning |
|---|---|---|
| `--out_dir` | `moe_ple_fw` (env `KMCU_OUT_DIR`) | checkpoint output |
| `--max_steps` | 30000 | optimizer steps |
| `--batch_size` / `--seq_len` | 32 / 512 | |
| `--lr` / `--warmup` | 3e-4 / 500 | cosine decays to `0.1 × peak` |
| `--eval_every` / `--eval_iters` | 250 / 40 | validation cadence |
| `--ckpt_every` | 5000 | also saves `ckpt_stepN/`; 0 disables |
| `--dim` / `--n_layers` / `--n_heads` / `--n_kv_heads` | 312 / 10 / 12 / 4 | |
| `--n_expert` / `--top_k` / `--ffn_e` / `--ple_dim` | 8 / 2 / 128 / 128 | |
| `--no_ternary` | off | keep `ple_table` in fp32 instead of ternary STE |
| `--seed` | 0 | |
| `--train_bin` / `--val_bin` | `fw_train.bin` / `fw_val.bin` (env `KMCU_TRAIN_BIN`/`KMCU_VAL_BIN`) | **override these** |

Optimizer details: AdamW `betas=(0.9, 0.95)`, `weight_decay=0.1` for `ndim ≥ 2`
weights (excluding `table`/`embed_tokens`), `0.0` otherwise; grad-norm clip 1.0.

Per-step log line:

```
step   2500/30000 train 2.9134 val 2.7411 ppl   15.51 lr 2.81e-04 (412s)
```

> Note: this script prints `val` loss and `ppl`. It does **not** compute task metrics
> (decision-head accuracy etc.) — those live in the task-specific scripts.

---

## 5. `watch_boot.py` — boot monitor

Resets the board and streams everything it prints. Handy when the firmware crashes
before it can open a console.

```bash
python watch_boot.py 30            # 30 s on COM5
python watch_boot.py 60 COM7       # 60 s on COM7
KMCU_PORT=COM7 python watch_boot.py 60
```

| Argument | Default | Meaning |
|---|---|---|
| `argv[1]` | 30 | capture duration (seconds) |
| `argv[2]` / `KMCU_PORT` | `COM5` | serial port |

Reset is done with `RTS → EN` (active low) while `DTR` keeps GPIO0 high (normal boot).
This is a **USB-UART (DTR/RTS)** reset; on boards wired only to the native
USB-Serial-JTAG this may not reset the chip — use `idf.py monitor` there.

---

## 6. `upload_uart.py` — UART upload (experimental)

Sends `model.kmcu` to the on-board TF card over the serial link, without removing the card.

```bash
python upload_uart.py                 # COM5 + default file
python upload_uart.py COM6 <file.kmcu>
```

Protocol: `"KMUP" + u32(len)` header, then 8192-byte chunks, one `ACK(0x06)` byte
per chunk for flow control. The firmware side is `uart_upload_model()` in `main/main.c`
(console `UART_NUM_0`, 921600 baud after a 3 s handshake window).

> ⚠️ **Known issue — this tool does not currently match the firmware.**
> The script waits for a `READY(0xAA)` byte and a `fopen` status byte
> (`0x55`/`0xEE`/`0xE3`) and keeps 115200 baud throughout, while the firmware prints a
> "switching to 921600" notice and sends neither. Additionally, the firmware only enters
> upload mode when `/sdcard/model.kmcu` is **absent**.
>
> **Until this is resynchronized, write the card with a PC card reader instead.**
> The default file path in the script also points at an unrelated project — always pass
> the file explicitly.

---

## `legacy/` — archived research scripts

These are the **early research branches**, kept so the negative/positive results
documented in `docs/` and `README.md` remain reproducible. They are not part of the
current pipeline.

| Script | Purpose | Documented outcome |
|---|---|---|
| `train_moe.py` | Self-contained MoE trainer (`LambdaMoE`, λ-annealed routing, KD or pure CE) | ✅ produced `moe_3ep` (argmax match 0.744) |
| `train_moe_kd.py` | From-scratch MoE distilled from a dense teacher via `MixtralForCausalLM` | ❌ failed (aux loss stuck at 2.0 = uniform router) |
| `train_lambda.py` | λ_s structural sparsification (dense → MoE structural distillation) | ❌ rejected (dense fallback dilutes CE gradient) |
| `full_finetune.py` | Classic full-fine-tune baseline (teacher init, all params, pure CE) | ✅ baseline (ppl 8.73) |
| `train_moe_qat.py` | QAT vs PTQ (`λ_q`), fake-quant bit-identical to `convert_minueza` q4_0 | ❌ rejected (QAT +1.4% vs PTQ +0.9%) |
| `dense_baseline.py` | Dense-FFN control (teacher weights except FFN, trained from scratch) | control |
| `eval_quality.py` | Perplexity + generation-quality evaluation of MoE student vs dense teacher | tooling |
| `verify_ann_head.py` | Output-head ANN (K-means + cluster select) pruning feasibility | ❌ rejected |
| `verify_moe.py` | Oracle router — upper bound of dense → MoE conversion | ❌ rejected |
| `verify_q2.py` | Lossy q3/q2 re-quantization boundary | ❌ rejected |

Cross-directory note: `verify_q2.py`, `verify_moe.py`, and `verify_ann_head.py` import
`ref_forward` from the parent `tools/` directory. They bootstrap `sys.path` with both
their own directory and the parent, so they run from anywhere; the remaining scripts
only depend on modules inside `legacy/`.

---

## Porting notes / known limitations

1. **Legacy scripts still carry author-local Windows defaults.** Scripts under
   `legacy/` default to paths such as `E:\models\...`; always pass their path arguments
   explicitly. Mainline tools default to relative paths and can be redirected with the
   `KMCU_*` environment variables (`KMCU_SRC`, `KMCU_OUT`, `KMCU_KM`, `KMCU_REF_OUT`,
   `KMCU_OUT_DIR`, `KMCU_TRAIN_BIN`, `KMCU_VAL_BIN`, `KMCU_FILE`, `KMCU_PORT`).
2. **Vocab is fixed at 32002** (Mistral tokenizer) in the training scripts; the
   converter itself reads `vocab_size` from `config.json`.
3. **KMCU is little-endian** and its layout is co-designed with `main/kmcu.c`. Any change
   to the format must be mirrored in `kmcu.h`/`kmcu.c` and in `ref_forward.py`.
4. **`n_cols` in the directory is the original (unpadded) column count** — the device
   recomputes `nblk = align32(n_cols)`. Do not "helpfully" pad it during conversion.
5. **Reproducibility.** `ref_forward` and `kmcu.c` are expected to agree token-by-token,
   but not bit-for-bit: the device accumulates in a different float order and uses
   q4-dequantized weights, so small logit deltas (~1e-4) are expected.

---

## License

Apache License 2.0 — see [`../LICENSE`](../LICENSE) and [`../NOTICE`](../NOTICE).
Third-party components (ESP-IDF, PyTorch, transformers, safetensors, NumPy, SciPy,
pyserial) are **not** vendored here and remain under their own licenses. No model
weights, tokenizer files, or `*.kmcu` images are distributed with this repository.
