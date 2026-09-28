# Kestrel-MCU

A **pure-scalar C11 inference kernel** for tiny LLMs on MCU-class RISC-V devices —
plus the whole host-side toolchain behind it: training → quantization → deployment →
verification.

Target board: **Waveshare ESP32-P4-WIFI6-DEV-KIT (SKU 32054)**.

> 中文版见 **[README.zh-CN.md](README.zh-CN.md)**。

---

## What it is

Run a small language model **on an MCU**, with no NEON/AVX, no OpenMP, no Linux
syscalls — just scalar C11 plus the chip's own PIE (SIMD) instructions.

| | |
|---|---|
| **Kernel** | Scalar C11 transformer decode: GQA attention, RoPE, RMSNorm, SwiGLU; MoE sparse FFN (`router` + top-k experts); PLE per-layer lookup tables; tied output head; optional decision head |
| **Weight formats** | `q4_0` for the core (18 B / 32 elements), **ternary (1.6 bit)** for PLE tables, f32 for norms |
| **Storage paths** | Fully PSRAM-resident, streamed from Flash, or streamed from a TF card (SD) |
| **Acceleration** | Fixed-point GEMV via ESP32-P4 PIE dot-product (hand-written `pie_dotprod.S`) |
| **Toolchain** | PyTorch training → KMCU quantized image → on-device inference → NumPy + C reference for A/B |
| **Determinism** | An **independent determinism line**: not bit-identical to other engines, and it deliberately does **not** reuse their kernels |

---

## Highlights (measured)

| Result | Value |
|---|---|
| PIE fixed-point GEMV vs scalar dot-product | **21×** on the primitive → **2.11×** end-to-end |
| Minueza-32M (0.032B), fully PSRAM-resident | **0.229 s/token (4.37 tok/s)** — **7.1×** vs the scalar starting point |
| int16 fixed-point weight scales | **+1.5%** (the only lossless-ish approximation that paid off) |
| Agent-loop router (17.4M, Flash-resident) | **19.5 tok/s**, output **token-identical** to the PC reference |
| Agent-loop router (64.4M, q4 + ternary PLE) | held-out routing accuracy **0.9986** (natural-distribution 0.9981) |
| PSRAM bandwidth (16 MB sequential) | ~**84 MB/s** write / ~**107 MB/s** read → **the wall is bandwidth, not compute** |

Full benchmark tables, rejected directions and pitfalls: **[docs/milestones.md](docs/milestones.md)**.

---

## Sample outputs

Everything below is **actually produced by the trained 64.4M agent-loop model**
(`agent_model.kmcu`, q4 core + ternary PLE), greedy decoding, through the NumPy
reference (`ref_forward.py`). Raw evidence: [docs/agent_loop_router.md](docs/agent_loop_router.md).

**1) Content-level routing** — read a text window, pick one of
`route` / `generate` / `stop` / `ask_human`:

| Input window | Decision |
|---|---|
| `The theory was developed over many decades.` | `stop` |
| `The sample contained 42 participants in total.` | `ask_human` |
| `What is the purpose of this experiment?` | `route` |
| `这个实验的目的是什么?` | `route` |
| `该实验共有42名参与者。` | `generate` |
| `x = 3 + 5 ; print(x)` | `generate` |
| `Hi` | `generate` |

**2) Text continuation** — 8-token real FineWeb-Edu prompt → 32-token greedy continuation:

| Prompt | Continuation |
|---|---|
| `a 21st century viewpoint` | `. The first two-dimensional view of the universe is the first to be seen in the 19th century. The first view of the universe is the` |
| `metabolic syndrome examines its effects` | `on the heart, heart, and kidneys. The results of the study are published in the journal Pediatrics. The results of the study were published in` |
| `2.5 percent of the population ident` | `ifies as a "flying-and-flying-and-flying-and-flying-and-flying-and-flying-` |

Locally fluent, semantically drifting — and the third row shows the failure mode openly.

**3) What it cannot do** — this is a **router + weak continuer**, not an instruct model:

```text
"Please summarize the following text:"  ->  "- 1. The following text is from the text of the text: -"
"Q: What is 2+2? A:"                    ->  "What is 2+2? A: What is 2+2?"
```

---

## Target hardware

| Item | Value |
|---|---|
| SoC | ESP32-P4 (RISC-V 32-bit, dual-core HP up to 400 MHz + LP core) |
| Coprocessor | ESP32-C6 (Wi-Fi 6 / BT 5) — **not used** by this project |
| Memory | 768 KB L2 + **32 MB PSRAM** (hex mode) + **16 MB Flash** |
| Toolchain | ESP-IDF 6.x, target `esp32p4` |
| Debug | USB Type-C (flash + serial console) |

---

## Repository layout

```
kestrel_mcu/
├── main/                 firmware
│   ├── kmcu.c / kmcu.h   inference kernel + KMCU format parsing
│   ├── pie_dotprod.S     PIE (SIMD) fixed-point GEMV kernel
│   └── main.c            entry point: self-tests, probes, model load, run modes
├── tools/                host-side toolchain — see tools/README.md
│   ├── mistral_ple.py    authoritative architecture source
│   ├── convert_minueza.py  HF safetensors → KMCU
│   ├── ref_forward.py    NumPy fp32 reference forward
│   ├── train_moe_ple.py  training
│   ├── watch_boot.py     reset + boot-log monitor
│   ├── upload_uart.py    UART upload (experimental)
│   └── legacy/           archived research scripts (reproduce past pass/fail results)
├── docs/                 design docs + development log (see index below)
├── host_verify.c         PC-side A/B harness — compiles the *same* kmcu.c
├── partitions.csv        Flash layout (app 2 MB / model 13.87 MB)
├── sdkconfig.defaults    ESP-IDF defaults (PSRAM, newlib, RTTI, -O2, no WiFi/BT)
└── LICENSE / NOTICE      Apache-2.0 + third-party notices
```

---

## Quick start

### 1. Build the firmware (Windows, verified)

Three environment pitfalls must be handled up front — all three are real and were hit
during development:

```powershell
# (a) Short toolchain path: the real path is deep enough to break Windows MAX_PATH(260),
#     which surfaces as: fatal error: bits/error_constants.h: No such file or directory
New-Item -ItemType Junction -Path C:\et -Target "$env:USERPROFILE\.espressif"
$env:IDF_TOOLS_PATH = "C:\et"

# (b) Skip the git-submodule check: components/esp_wifi/lib is blocked upstream (HTTP 423),
#     and this project disables WiFi/BT anyway.
$env:IDF_SKIP_CHECK_SUBMODULES = "1"

. E:\esp-idf\export.ps1                 # activate ESP-IDF

# (c) Build from an ASCII-only directory, and disable ccache:
#     ccache aborts with "std::filesystem ... Illegal byte sequence" on non-ASCII paths.
#     Copy main/, CMakeLists.txt, partitions.csv and sdkconfig.defaults there.
Set-Location C:\kmcu
idf.py --no-ccache build
```

### 2. Flash

```powershell
idf.py -p COM5 flash
```

### 3. Model pipeline

```bash
# train (needs a pre-tokenized uint16 token stream)
python tools/train_moe_ple.py --out_dir <out_dir> --max_steps 30000 \
       --train_bin <train.bin> --val_bin <val.bin>

# quantize → KMCU image
python tools/convert_minueza.py --src <out_dir> --out model.kmcu

# PC reference (NumPy fp32) — the ground truth for on-device A/B
python tools/ref_forward.py --km model.kmcu
```

### 4. Deploy

```powershell
# Fit the image into the `model` partition (offset/size in partitions.csv)
python -m esptool --chip esp32p4 -p COM5 -b 460800 write-flash 0x210000 model.kmcu
```

> **Size limit:** the `model` partition is **13.87 MB**. Larger images (e.g. the 64.4M
> PLE model at 20.77 MB) must be read from a **TF card** instead of Flash — the firmware
> has a dedicated SD-resident path for that.

### 5. Verify against the reference

```bash
gcc -O2 -std=c11 -I main -o host_verify host_verify.c main/kmcu.c -lm
./host_verify model.kmcu
```

The device and the reference use the **same quantized weights**, so expectations are:

* greedy token sequence — **token-identical**;
* logits — agree within ~**1e-4** (different float accumulation order), argmax identical.

The same source (`main/kmcu.c`) runs on both sides, which is what makes this comparison
meaningful. Run modes (pure generation / dual-head / agent loop) are switched over the
serial console with `0` / `1` / `2` / `q`.

---

## Documentation index

| Document | Contents |
|---|---|
| [docs/milestones.md](docs/milestones.md) | **development log**: M1 capability probes, M2 end-to-end inference, M3-x optimizations — full hard-data tables, rejected directions, pitfalls |
| [docs/moe_framework.md](docs/moe_framework.md) | MoE design: dense → sparse activation |
| [docs/lambda_transition_framework.md](docs/lambda_transition_framework.md) | λ-transition training framework (classic paradigm + axiom-library base) |
| [docs/layerwise_adaptation.md](docs/layerwise_adaptation.md) | layer-wise adaptation (future direction) |
| [docs/agent_loop_router.md](docs/agent_loop_router.md) | content-level agent-loop router: task definition, data, training, capability probe, deployment |
| [tools/README.md](tools/README.md) / [tools/README.zh-CN.md](tools/README.zh-CN.md) | toolchain usage (EN / 中文) |

---

## Design constraints (self-imposed)

* **Pure scalar C11** — no NEON/AVX/OpenMP/Linux syscalls; the only ISA-specific code is
  the PIE dot-product kernel, and a scalar fallback is always available.
* **Determinism is per-engine.** Agreement with other engines (e.g. an aarch64/x86-64
  large-model engine) is **not** claimed, and no cross-engine performance equivalence is
  asserted.
* **No silent lossy shortcuts.** Any lossy approximation (extra quantization, pruning)
  requires separate authorization plus a quality gate before it lands.
* **Measurement over inference.** Optimization claims in the docs are backed by
  before/after measurements on the device, with the rejection cases recorded too.

---

## License

**Apache License 2.0** — see [LICENSE](LICENSE).

* Commercial use, modification and redistribution are permitted; keep the copyright and
  license notices and **state significant changes** (§4b).
* Includes an express **patent grant** with defensive termination.
* Third-party components and their licenses are listed in [NOTICE](NOTICE); they are
  **not** vendored here.
* **No model weights, tokenizer files or `*.kmcu` images are distributed with this
  repository.** Such artifacts may carry their own upstream terms (e.g. the base model
  fine-tuned, or the tokenizer used for training), which are independent of this
  project's license — users are responsible for complying with them.
