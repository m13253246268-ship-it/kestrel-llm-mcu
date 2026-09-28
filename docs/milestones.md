# Kestrel-MCU Development Log (milestones and measured ledger)

> 中文版见 **[milestones.zh-CN.md](milestones.zh-CN.md)**。

> This document is the **development log** of kestrel_mcu: it records, milestone by
> milestone, all the hard data and pitfall conclusions for capability probes,
> end-to-end inference, and performance optimization (M1 / M2 / M3-x). For the project
> overview and quick start see
> **[../README.md](../README.md)** (English) / **[../README.zh-CN.md](../README.zh-CN.md)** (Chinese);
> for the toolchain documentation see **[../tools/README.md](../tools/README.md)**.
>
> License see **[../LICENSE](../LICENSE)** and **[../NOTICE](../NOTICE)**.

## Positioning

A **pure-scalar C11** tiny inference kernel for **MCU-class RISC-V devices**
(first target: Waveshare ESP32-P4-WIFI6-DEV-KIT, SKU 32054).

It is **completely independent** of `GitHupSRC/` (vLLM-Kestrel, an aarch64/x86-64 large-model engine):
- Does not reuse its NEON dotprod/fp16 or AVX2 kernels
- Does not reuse its Linux runtime (mmap//proc//sys/OpenMP/dlopen)
- Does not load GB-scale weights such as Qwen3-VL (physically impossible on an MCU)
- Is an **independent determinism line**, with no guaranteed bit-level agreement with x86/arm

## Target hardware (SKU 32054)

| Item | Value |
|---|---|
| SoC | ESP32-P4 (RISC-V 32-bit, dual-core HP 360MHz + single-core LP 40MHz) |
| Coprocessor | ESP32-C6 (Wi-Fi 6 / BT 5) |
| Memory | 768KB L2 + 32MB PSRAM + 16MB Flash |
| Toolchain | ESP-IDF 5.3+ (target = `esp32p4`) |
| Debug | Type-C UART (flashing + serial readback) |

## First milestone

Implement and verify the basic operators `S_0` / `UPA` of the SHS axiom library
(`axiom_registry.json`) on pure-scalar RISC-V, outputting self-test results over
UART. **Without introducing any NEON/AVX assumptions**, paving the way for the
later tiny networks.

## Build / flash (verified on Windows)

### Environment already installed on this machine

| Component | Version / location |
|---|---|
| ESP-IDF | v6.0.2 → `E:\esp-idf` (Gitee mirror clone) |
| Toolchain | riscv32-esp-elf 15.2.0 → `%USERPROFILE%\.espressif` |
| Python | 3.12.9 (IDF-dedicated venv) |
| Serial port | COM5 (Type-C USB serial device, i.e. this development board) |

### Build commands (PowerShell)

```powershell
# 0) Short drive-letter alias: the real toolchain path is too deep and triggers the Windows MAX_PATH(260) limit
#    After mapping, the C++ standard header path drops from 261 characters to ~229. Must be re-run after a reboot.
subst X: "%USERPROFILE%\.espressif"

$env:IDF_TOOLS_PATH = "X:\"
$env:IDF_SKIP_CHECK_SUBMODULES = "1"   # the esp32-wifi-lib submodule is blocked on Gitee; this project has WiFi/BT disabled

. E:\esp-idf\export.ps1                 # activate the IDF environment

# This project's build/ was configured with an X:\-form venv; the X: venv must be put back at the
# front of PATH, otherwise idf.py reports "python.exe is currently active in the environment while the
# project was configured with 'X:\python_env\...'" and exits immediately.
$env:PATH = "X:\python_env\idf6.0_py3.12_env\Scripts;" + $env:PATH

Set-Location E:\kestrel_mcu             # must be a pure ASCII path, see "Known constraints" ① below
idf.py set-target esp32p4
idf.py build
```

> After modifying `sdkconfig.defaults` you must **also update `sdkconfig`** (or delete it
> and regenerate), otherwise IDF will not re-run Kconfig. Regeneration triggers the
> git submodule check ⇒ you must first set `IDF_SKIP_CHECK_SUBMODULES=1`, otherwise it
> hangs on esp32-wifi-lib's HTTP 423.

### Flashing

```powershell
# Close any other serial monitor before flashing (to avoid port contention)
idf.py -p COM5 flash monitor
```

### Known constraints (all measured pitfalls; do not change them casually)

1. **The project path must be pure ASCII**. The original project path contains Chinese
   (`E:\项目\...`); when the riscv toolchain reads the `-specs` file it corrupts
   non-ASCII characters into `\x0a`, reporting
   `cannot read spec file 'E:/\x0a/...'`. Therefore the build copy is placed at `E:\kestrel_mcu`.
2. **A too-deep toolchain path triggers Windows MAX_PATH(260)**. The symptom is
   `fatal error: bits/c++config.h: No such file or directory`.
   Solution: `subst` a short drive letter (see above). `CONFIG_COMPILER_CXX_RTTI=y` in
   `sdkconfig.defaults` is a belt-and-braces safeguard (it makes the C++ headers fall
   back to the shorter `ilp32f` directory).
3. **The submodule check must be skipped**. `components/esp_wifi/lib` is blocked on the
   Gitee mirror (HTTP 423), and this project already has `CONFIG_ESP_WIFI_ENABLED=n` /
   `CONFIG_BT_ENABLED=n`, so it is not actually needed.
   Hence set `IDF_SKIP_CHECK_SUBMODULES=1`.
4. **The default picolibc has compatibility issues**; `sdkconfig.defaults` has already
   switched to `CONFIG_LIBC_NEWLIB=y`.
5. **The ESP32-P4's PSRAM is in octal (hex) mode**, not quad; the config symbol is
   `CONFIG_SPIRAM_MODE_HEX`.

### Build artifacts (measured)

| File | Size |
|---|---|
| `build/kestrel_mcu.bin` | 208.6 KB (app partition 1 MB, 80% headroom) |
| `build/bootloader/bootloader.bin` | 23.3 KB |
| `build/partition_table/partition-table.bin` | 3.0 KB |

ELF check: `Machine: RISC-V`, `Class: ELF32`, `Flags: RVC, single-float ABI`.

## M1 capability probe results (2026-09-26, on-device)

Firmware: `main/main.c` (SHS self-test + PSRAM bandwidth + TF-card diagnostics).

| Item | Measured | Reading |
|---|---|---|
| Chip | ESP32-P4 rev v3.1, dual-core + LP core, **400 MHz** | Consistent with spec |
| PSRAM capacity | **32768 KB** (32 MB, hex mode @200 MHz) | Weight-resident budget usable |
| PSRAM write bandwidth | **83 MB/s** (16 MB, u32 sequential) | — |
| PSRAM read bandwidth | **84 MB/s** (u32 sequential) / **90 MB/s** (u32 ×4 unrolled) | 4-way unroll gains only +7% ⇒ **bandwidth-bound**, not latency-bound |
| TF card (SDIO) | **Completely unresponsive**: all 6 combinations report `ESP_ERR_TIMEOUT` @ `send_op_cond(CMD1)` | See below |

### TF card diagnostic details

**First detection (M1)** —— exhaustively tested (all failed, with the identical error of no CMD1 response):

| GPIO45 power polarity | Bus width | Clock | Result |
|---|---|---|---|
| High | 4-bit | 20 MHz | ESP_ERR_TIMEOUT |
| High | 1-bit | 20 MHz | ESP_ERR_TIMEOUT |
| High | 4-bit | 40 MHz | ESP_ERR_TIMEOUT |
| Low | 4-bit | 20 MHz | ESP_ERR_TIMEOUT |
| Low | 1-bit | 20 MHz | ESP_ERR_TIMEOUT |
| Low | 4-bit | 40 MHz | ESP_ERR_TIMEOUT |

`send_op_cond` is the **earliest** step of card initialization (before the file system):
the card does not respond at all, indicating the problem is at the **electrical layer**
rather than the protocol/format layer.

**Second detection (re-test, authoritative conclusion)** —— the nature of the error changed **fundamentally**:

| Item | First | Second |
|---|---|---|
| Failure point | `send_op_cond(CMD1)` | `esp_vfs_fat_sdmmc_mount` |
| Error code | `ESP_ERR_TIMEOUT` (no response) | `ESP_FAIL` / `failed to mount card (13)` |
| FatFS meaning | — | `FR_NO_FILESYSTEM` |
| Reading | Card completely unresponsive at the electrical layer | **Card's electrical link is now fine**, only the FAT filesystem is missing |

⇒ Corrected conclusion: **the card is good, and the CMD/DAT link is good too** (valid
combination: `GPIO45 = 0`, i.e. that load switch is **active-low**). The only thing
missing is that there is **no FAT partition table** on the card. Formatting (which wipes
the card's data) would make it usable for the later 0.1B layer-resident work, and requires
separate authorization before doing so.

> Note: the **0.05B resident milestone does not depend on the TF card** —— 0.05B@INT4 ≈ 25 MB
> fits directly into the 32 MB PSRAM. The TF card is only necessary in the later 0.1B
> layer-resident stage.

### Pitfall log (measured in this session)

- When `esp_vfs_fat_sdmmc_mount` fails it **already performs host deinit internally**; if the
  caller then calls `sdmmc_host_deinit()` it causes a **double deinit → Instruction access fault crash**.
- The default `SDMMC_SLOT_CONFIG_DEFAULT()` `width` allows 8-bit, which would seize
  **GPIO45 as D4** (and that is exactly this board's TF-card power switch); you must
  explicitly set `slot.width = 4`.

## M2 end-to-end inference measurements (2026-09-26, on-device)

**Conclusion: the 0.032B model has run end-to-end, fully PSRAM-resident, on the ESP32-P4, with output token-identical to the PC reference.**

### Model and format

| Item | Value |
|---|---|
| Model | `Felladrin/Minueza-32M-Base` (Mistral family: RMSNorm+RoPE+GQA+SwiGLU) |
| Architecture | layers=10, dim=312, heads=12, kv_heads=4, head_dim=26, ffn=1092, vocab=32002 |
| Parameter count | 22.81M (**weight tying forcibly enabled**; the original model had `tie_word_embeddings=false`, a lossy change) |
| Quantization | Q4_0 flat blocks (32 weights/block, fp16 scale) = 4.5 bit/weight |
| Footprint | **12.26 MB** → placed in the `model` partition of the 16MB NOR Flash (offset `0x210000`) |

### Verification (sole criterion: A/B comparison against an fp32 NumPy reference on the same quantized weights)

| Item | MCU | PC reference | Diff |
|---|---|---|---|
| single token[1] top1 | id=684, logit=5.229595 | id=684, logit=5.229599 | 4e-6 |
| 4-token prompt top1 | id=23624, logit=5.184734 | id=23624, logit=5.184731 | 3e-6 |
| **20-token greedy sequence** | see below | same | **exactly identical** |

```
1,450,2217,4996,23624,1199,8752,23624,1199,20925,3161,4865,442,5102,2672,3161,13040,17574,28394,1628
```

> Note: this is "numerical equivalence between MCU and PC under the same quantized weights",
> **not** bit-level agreement across architectures and engines (the latter is explicitly not
> required, and must not be claimed, per H1).
>
> ⚠️ The numbers and sequence in the table above belong to the **M2 point in time** (unaligned
> q4 layout). After M3-2 introduced row alignment, the quantization block boundaries change, so
> **that sequence is void**; for the post-M3 baseline sequence see the M3-2 section.

### Performance and memory

| Item | Measured |
|---|---|
| decode | **1.63 s/token ≈ 0.61 tok/s** (19 steps 31.0 s) |
| Flash→PSRAM load | 12.26 MB / 1.61 s = **7.8 MB/s** (DIO 80MHz limited) |
| PSRAM usage | weights 12.26 MB + KV 2×520 KB + directory table |
| Internal RAM | scratch 18 KB |
| Compute utilization | 22.8M MAC / 1.63 s ≈ **14 MMAC/s** (a **~100×** margin relative to PIE potential) |

**Main bottlenecks**: scalar per-element q4 dequantization (one nibble extraction + fp16
conversion per weight), and the output head's streaming argmax over
`32002×312 = 10M` elements (44% of total work).
Optimization directions: PIE 128-bit vectorized GEMV, block-level batched dequantization,
output-head chunked pruning.

### Pitfall log (measured in this session)

- **Test-harness bug (not a model bug)**: the greedy loop once wrote the generated result
  back into `seq[0..PROMPT_LEN-1]`, overwriting the prompt elements, so the MCU never
  processed the full prompt and the sequence could not be compared against the reference.
  After fixing it to "first preprocess the entire prompt, then autoregressively generate",
  20/20 matched exactly.
- Compute-type tasks (a single step of 1.6 s without feeding the watchdog) require disabling
  the task watchdog `CONFIG_ESP_TASK_WDT_EN=n`, otherwise it prints a register dump every step.
- On the riscv32 toolchain `uint32_t` is `unsigned long`, so printf must use `PRIu32`/cast explicitly to `unsigned`.

## M3 performance optimization (in progress, 2026-09-26)

### M3-1 landed: q4 → int8 load-time expansion + separate fp16 scale array (PSRAM)

Motivation: the original implementation did "nibble extraction + shift/mask/branch + fp16
conversion" for every weight element, which was the dominant decode cost. After expansion,
the inner loop degrades to a compact `(float)q * x` accumulation.

| Item | Before | After |
|---|---|---|
| Speed | 1.63 s/token (0.61 tok/s) | **1.11 s/token (0.90 tok/s)** |
| Correctness | 20/20 match | **20/20 match (preserved)** |
| logits deviation | ~3e-6 | ~2e-6 |
| PSRAM usage | weights 12.26 MB | after expansion **23.1 MB** (arena) + 7.6 MB left |
| Startup overhead | 1.61 s (whole-image load) | 2.48 s (chunked expansion) |

Implementation points:
- To save PSRAM, **no longer load the whole 12.26 MB image**, but instead
  **expand from flash in chunks** via the `km_read_fn` callback
  (`km_fast_build_ex`); the F32 norm weights are also folded into the same arena.
- The inner loop adds `#pragma GCC unroll 8` (0.86 → 0.90 tok/s).
- The fast path **introduces no new quantization approximation**: `q` and `d` come entirely
  from the original q4 data; only the floating-point association order changes (`d` is
  hoisted out of the block).

### M3-2 landed: row alignment to 32 (converter + layout + kernel synchronized)

**Change**: q4 weights are **padded to a multiple of 32 per row** before quantization
(zero-padding at row ends), so that each 32-element block belongs to only one row and
blocks do not cross rows. After expansion
`q[r][c] = q[r*align32(n_cols)+c]`, and the block index is `= r*nblk + b`.
The GEMV inner loop therefore goes from "variable-length `cnt` + phase drift + branches"
to a **fixed-length 32 fully-unrolled loop**.

**Cost (must be stated explicitly)**: this is a **lossy change**. The block boundary positions
change ⇒ each block's `max_abs`/scale changes ⇒ the dequantized weights are no longer
bit-identical to M2/M3-1 (after alignment each block covers only the same row, which usually
gives slightly better quantization fidelity, but **equivalence no longer holds**). As a result,
the greedy sequence of the PC reference itself also changes, and the 20-token sequence recorded
at M2 is void.

| Item | M3-1 | M3-2 @ `-Og` | **M3-2 @ `-O2`** |
|---|---|---|---|
| Speed | 1.11 s/token (0.90 tok/s) | 1.24 s/token (**0.81** tok/s) | **0.511 s/token (1.96 tok/s)** |
| Correctness | 20/20 | 20/20 | **20/20 (preserved)** |
| logits deviation | ~2e-6 | ~1e-6 | ~1e-6 |
| Model image | 12.26 MB | 13.19 MB | 13.19 MB |
| arena | 23.1 MB | 23.7 MB | 23.7 MB |
| Cycles/element | 19.5 | 21.8 | **8.7** |

**Key finding (a correction in understanding)**: under `-Og`, M3-2 was **instead slower**
(0.90 → 0.81). The root cause is not row alignment but that **the entire firmware had until
now been built at `-Og`** (the ESP-IDF default `CONFIG_COMPILER_OPTIMIZATION_DEBUG=y`). `-Og`
sacrifices loop optimization for debuggability, making `#pragma GCC unroll` essentially
ineffective —— which also explains why unroll bought only 4% in M3-1. After switching to `-O2`
(`CONFIG_COMPILER_OPTIMIZATION_PERF=y`):

- decode: 1.11 → **0.511 s/token** (**2.17×** vs M3-1, **3.19×** vs M2)
- PSRAM read bandwidth probe: 84 → **106 MB/s** (same probe code) ⇒
  **all previous performance readings under `-Og` were systematically depressed**, including M1/M2/M3-1.

**Attribution honesty**: the gains of `-O2` and "row alignment" **have not yet been decoupled
by measurement**. What is certain is only that M3-2's kernel change is −10% under `-Og`; its
standalone contribution under `-O2` would require reverting to the M3-1 layout and testing
`-O2` again.

### M3-3a landed: fp16 scale → fp32 load-time pre-expansion (lossless)

**Change**: `km_ft_t.d` changed from `uint16_t*` (fp16) to `float*` (fp32). At load time,
`km_fast_build_ex` converts each block's fp16 scale **once** to fp32 via `km_f16_to_f32`, and
the hot loops (`gemv_q8` / `q8_row_get` / output head) degrade from "per-block branch
conversion" to a direct `acc += dr[b] * s`. fp16→fp32 is an exact conversion, **changing no
values**.

| Item | M3-2 @ `-O2` | **M3-3a @ `-O2`** |
|---|---|---|
| Speed | 0.511 s/token (1.96 tok/s) | **0.489 s/token (2.04 tok/s)** |
| Correctness | 20/20 | **20/20 (preserved)** |
| logits | ~1e-6 | **bit-identical** (5.350636 / 4.978979) |
| arena | 23.7 MB | **25.7 MB** (d array 2B→4B, 5.6 MB left) |
| Gain | — | **+4.3%** |

**Attribution honesty**: under `-O2` the compiler had already fully inlined `km_f16_to_f32`, so
removing that conversion bought only +4%. This confirms a more important judgment: **the scalar
path is already near its ceiling of gains**, and the remaining bulk must come from vectorization
(PIE) or **lossy** pruning.

### M3-3b conclusion (important): lossless output-head pruning is infeasible under q4_0

The originally planned "two-stage argmax (chunked upper-bound pruning)", if it is to be
**lossless** (preserving 20/20 and bit-identical agreement with the reference), was shown by
branch-and-bound analysis to have a **pruning rate approaching 0** under q4_0 quantization:

- A strict upper bound requires constructing the worst-case sign over `q∈[-8,7]`
  (`M_b = Σ 7·h⁺ − 8·h⁻`), with the remaining-dimension upper bound `U_v = d_max_v · Σ M_b`.
- The quantization scale `d` itself is a "worst-case" amplification (`d = max_abs/8`), making
  `U_v ≈ 19` far larger than the logit's own magnitude (~5) ⇒ the criterion
  `partial_v + U_v < B` almost never holds, pruning off barely any v.
- The Cauchy-Schwarz bound (`‖q‖·‖h‖`) is likewise too loose.

**Conclusion**: substantial speedup of the output head leaves only two paths ——
1. **PIE vectorization** (M3-4, lossless, expected 5–10×);
2. **Lossy two-stage pruning** (coarse top-K screening over part of the dimensions, accepting a
   very-low-probability decision difference), which would break the verification constraint of
   "MCU vs ref per-token A/B comparison", and is an **H4-unauthorized lossy approximation**
   that **requires a separate project authorization** before it can be implemented.

### M3-4 investigation conclusion (PIE feasibility, 2026-09-26)

Under the axiom library's constraint of "measurement over inference", the PIE path was verified
with hard data. **Conclusion: feasible but with a high bar, and the gain is constrained by the
bandwidth wall —— not a naive 5–10×.**

**Measured chain of facts**:

| # | Fact | Basis |
|---|---|---|
| 1 | CPU = RISC-V dual-core 400 MHz + single-precision FPU + **PIE** (proprietary 128-bit SIMD) | official datasheet |
| 2 | The compiler knows `xespv`: `__riscv_xespv = 2001000`, and march already includes `_xesploop_xespv` | `gcc -dM -E` measured |
| 3 | The toolchain contains two generations of assembler/disassembler: **xespv2p1 / xespv2p2** | Glob toolchain bin |
| 4 | **No public C intrinsics header** (no `esp_pie.h`/`xespv.h`), and no builtins | Glob/Grep toolchain |
| 5 | **GCC does not auto-vectorize to xespv**: disassembling `kmcu.c.o` gives 4136 instructions, all standard RV32I/F scalar (`lw/sw/flw/fcvt.s.w/fmadd.s`…), **zero PIE, zero standard RVV instructions** | `objdump -d` measured |
| 6 | The PIE ISA manual exists in the official **TRM Chapter 4 "Processor Instruction Extensions (PIE)", status Published** (latest v0.7) | Espressif TRM |

**Key distinction**: PIE (`xespv`, Espressif proprietary) ≠ standard RVV (the `v` extension). The
toolchain's `riscv_vector.h` corresponds to standard RVV, but this march **has no `v`**, so all
standard RVV intrinsics are unavailable. Online articles that misstate "the ESP32-P4 uses RVV 1.0"
contradict the datasheet and cannot be trusted.

**Feasible path (the only one)**: hand-written **RISC-V inline assembly**, with instruction
encodings/semantics taken from TRM Chapter 4.
- Losslessness: int8×fp32 multiply-accumulate + fp32 accumulation is a deterministic operation;
  SIMD parallel accumulation changes the floating-point association order (~1e-6 level), the same
  nature as the existing M3 changes, **not a quantization approximation**, so argmax 20/20 should
  be preserved.
- Risks: ① compatibility of the two PIE generations (v2p1/v2p2); ② the debugging cost of inline
  assembly; ③ the TRM is a PDF requiring manual study.

**Bandwidth wall (the axiom library's "bandwidth-bound" constraint)**: currently 8.4
cycles/element, with weight+activation reads in PSRAM (~84–106 MB/s). If PIE compresses compute to
~1 cycle/element, decode will hit the bandwidth wall; the theoretical 5–10× **upper bound must be
defined by measurement**, and will very likely converge to 2–4×.

### M3-4 decisive conclusion (2026-09-26, Espressif source-code proof)

> **The "losslessness" judgment above is overturned.** After digging further into PIE, we obtained
> Espressif's official source-code-level proof:
> **PIE SIMD only supports integer datatypes; there is no fp32 vector instruction.**

**Evidence**: the source comment of ESP-DL's `dl_esp32p4_dotprod_f32` (fp32 dot product), verbatim:
> *"The PIE SIMD extension only supports integer datatypes, so the float dot product
> is implemented with the RISC-V single-precision FPU."*

That is: for an fp32 dot product, Espressif itself can only fall back to the **scalar FPU** (4-way
unroll + 4 independent accumulators to hide `fmadd.s` latency). This corroborates my disassembly of
`kmcu.c.o` being all-scalar (`fcvt.s.w`+`fmadd.s`).

**Therefore**: the current GEMV is `int8 weights × fp32 activations`, and **cannot be losslessly
PIE-accelerated**. To reap PIE's benefits, the **only path is fixed-pointizing the activations**
(the user has already authorized the approximate track):

| Ready-made PIE dot products (ESP-DL `dl_esp32p4_dotprod*.S`) | Data format | Precision |
|---|---|---|
| `dl_esp32p4_dotprod_i16k8o16` | **int8 weights × int16 activations** (`w8a16`) | Higher (best match for the existing int8 weights) |
| `dl_esp32p4_dotprod_i16k16o16` | int16 × int16 | High |
| `dl_esp32p4_dotprod_i8k8o16` | int8 × int8 | Lowest |

Core PIE instructions (confirmed from source): `esp.vldext.s8.ip` (int8 load + sign extension),
`esp.vmulas.s8/s16.xacc(.ld.ip)` (vector multiply-accumulate into `xacc`), `esp.zero.xacc`
(zero-clear), `esp.srs.s.xacc` (take the accumulated result + saturating right shift). 128-bit
registers `q0..q7`, scalars `a0..a7`/`x*`.

**Incidental lossless gain**: `dl_esp32p4_dotprod_f32` reveals the optimal way to write a scalar
fp32 dot product (4-way unroll + 4 accumulators), which can be losslessly applied to the current
`gemv_q8` —— a free step before fixed-pointization.

**Scope of fixed-pointization (engineering effort tiers)**:
- Minimal: fixed-pointize only GEMV (int16 activation quantization + PIE dot product + block scale
  restoration); the rest of RMSNorm/attention/SwiGLU still uses fp32. Activations are
  quantized/dequantized per layer, and the error accumulates layer by layer.
- Full: fixed-pointize the entire forward pass (including RMSNorm, which already has
  `dl_esp32p4_rms_normalization.S`); the effort = rewriting an int8 fixed-point Transformer kernel.

### M3-4 stage 0 measurement: PIE fixed-point GEMV single operator (2026-09-26)

Ported Espressif's `dl_esp32p4_dotprod_i16k8o16` (signature changed to return int32 without
saturation) as [pie_dotprod.S](../main/pie_dotprod.S),
measured on board (`pie_bench`, N=4096, 200 iterations):

```
[PIE] dotprod N=4096  scalar=19713  pie=19713  match=1
[PIE] 200 iters: scalar=16396 us  pie=782 us  speedup=20.97x
```

| Item | Result |
|---|---|
| Correctness | **match=1** (PIE and scalar C bit-identical, int8×int16→int32) |
| Pure compute speedup | **20.97×** (data in internal RAM, no PSRAM bandwidth wall) |
| decode impact | 20/20 preserved, 0.489 s/token unchanged (only bench added, not wired into forward) |

**Verdict**: PIE fixed-point GEMV is **feasible and correct**, with a pure-compute ceiling of
**~21×** (16 int8 multiply-accumulates per instruction). This is the first bite at "digging into
PIE": instruction syntax, register constraints, and correctness all cleared.

**Actual gain to be delivered to decode**: 21× is the **upper bound of the compute speedup**; the
real GEMV weights live in PSRAM (~84–106 MB/s), so after wiring it in it will hit the bandwidth
wall, and the actual convergence range remains to be measured (estimated 2–4×, following the
axiom library's "define gains by measurement" —— the next step is to wire it into `gemv_q8` and
measure).

**Integration cost (next step)**: to change `gemv_q8` from `int8×fp32` to `int8×int16`
fixed-point, the fp32 activations must be quantized to int16 before each layer's GEMV
(`x_q16 = round(x * scale)`), and after GEMV the PIE dot product + fp32 block scale restoration
are used. Activation quantization is lossy (the user has authorized the approximate track); the
remaining RMSNorm/attention/SwiGLU paths stay fp32 for now.

### M3-4 stage 1 landed: PIE fixed-point GEMV wired into forward (2026-09-26)

`gemv_q8` and the output head have been changed to a **w8a16 fixed-point PIE path** (`pie_gemv_row`),
with per-vector int16 activation quantization (`xq = round(x·S)`, `S = 32767/max_abs`),
PIE chunked dot product + fp32 block scale restoration, and a final multiply by `1/S`.

**Final measurements (`-O2`)**:

| Item | M3-3a (scalar fp32) | **M3-4 stage 1 (PIE fixed-point)** |
|---|---|---|
| decode | 0.489 s/token (2.04 tok/s) | **0.232 s/token (4.31 tok/s)** |
| Speedup | — | **2.11×** |
| Correctness | 20/20 | **20/20 (sequence exactly identical to fp32)** |
| single-token | argmax=684 | **argmax=684 (unchanged)** |

```
seq:1,450,2217,4996,23624,1199,8752,23624,1199,1199,8752,23624,1199,10436,23624,1199,3065,3161,4865,442
```

**Key conclusion**: int16 activation quantization (per-vector) has **zero impact** on this model's
argmax (20/20 exactly identical to fp32; PC simulation has shown a quantization error of ~3e-4).
The actual gain is **2.11×**, landing within the bandwidth-wall convergence range (2–4×) —— the
pure-compute 21× is capped by the PSRAM bandwidth (~84–106 MB/s).

**Two key pitfalls (hard PIE constraints)**:
1. **Scalar-operand register constraints**: the **address/scalar operands of PIE instructions**
   (`esp.vld*/vldext*/vmulas*/srs`) **cannot use `t0-t2` (x5-x7)**, which reports
   `illegal operands`; you must use `a0-a7` (x10-x17) or `t3+` (x28+). The `srs` destination using
   `t3` (x28) is legal, while using `t2` (x7) is illegal.
2. **16-byte alignment**: `esp.vld.128`/`vldext.s8` require a 16-byte-aligned address. If the int16
   quantization buffer xq in scratch uses `heap_caps_malloc` (which only guarantees 4-byte
   alignment) it reads the wrong data (manifesting as wrong large dot values and argmax all
   changing). You must use `heap_caps_aligned_alloc(16, ...)`.

### M3-5 layer-resident verification (rejected, 2026-09-26)

**Problem**: the 0.1B model (q4_0 ≈ 56 MB) exceeds the 32 MB PSRAM; can "weight layer residency"
(read each layer from Flash when computing it, free it after) be used to run it? Benchmarking
against GitHupSRC's `VLLM_VQF_STREAM=1`.

**Test artifact**: `LAYER_RESIDENT_TEST` (main.c, off by default) —— autoregressive decode
re-reads the entire weight set from Flash and re-expands it for every token, timing "re-read" and
"compute" separately.

**On-device measurement** (0.032B, 12.58 MB weights):

| Item | Resident | Layer-resident (re-read per token) |
|---|---|---|
| read (Flash read + expand) | 0 | **2343 ms** |
| calc (decode) | 232 ms | 232 ms |
| **per token** | **232 ms** | **2575.8 ms** |
| Slower | — | **11.1×** |

**Conclusion (hard data)**:
- Layer-resident read accounts for 91% and calc for 9%; **the Flash bandwidth wall (~5.4 MB/s read+expand) is the absolute bottleneck**.
- Extrapolating to 0.1B (56 MB ≈ 12.58 MB × 4.5): read ≈ 2343×4.5 ≈ **10.5 s/token**, unusable.
- Correctness preserved 20/20 (the re-read weights are bit-identical, only slower).

**Decision**: for 0.1B on the ESP32-P4, **layer residency is not used** as a production path, but
**the layer-resident code is kept**, not for inference but to **verify technical feasibility** ——
using measurement to quantify the hard boundary of the "storage bandwidth wall".
**Next step**: run a layer-resident measurement on the **0.1B model** to verify whether the
~10.5 s/token extrapolated from 0.032B holds.
The viable production path remains more aggressive quantization (q2 ≈ 31 MB) compressing the whole
thing into 32 MB PSRAM, or TF-card layering (~2.8 s/token, still marginal). The test artifact is
off by default (`LAYER_RESIDENT_TEST=1` to enable).

### M3-7 Flash per-layer streaming kernel verification (2026-09-26, on-device)

**Background**: 0.1B layer residency requires a **truly per-layer streaming kernel** —— the
whole-image expansion (`km_fast_build_ex`) expands SmolLM-135M to ~150MB, far exceeding the 32MB
PSRAM. The TF card's CMD1 is currently not working (electrical layer), so we first verify the
streaming kernel's **numerical correctness** + **bandwidth cost** on Flash with Minueza-32M.

**Implementation**: `km_decode_step_stream` (kmcu.c) —— embed read row-by-row streamingly, each
layer's 9 tensors "read → expand → compute → free" (reusing a single-tensor arena, only 383 KB),
output head streaming argmax row by row. The `FLASH_STREAM` path uses `esp_partition_read` as
read_fn. See the streaming comments in `kmcu.c`.

**On-device measurement** (Minueza-32M, Flash source):

| Item | Whole-image resident (M3-4) | Per-layer streaming (M3-7) |
|---|---|---|
| per token | **232 ms** | **3409 ms** |
| read volume per token | 0 (resident) | **12.57 MB** |
| sequence | 20/20 | **20/20 (token-identical to whole-image)** |

**Conclusion (hard data)**:
- **The streaming kernel is numerically correct**: top8 and the full sequence are exactly identical
  to whole-image decode, verifying that embed row-wise reads, per-layer expansion, and output-head
  streaming argmax are **numerically equivalent** to whole-image expansion.
- **Per-layer streaming is 14.7× slower**: 12.57 MB read per token (≈ the whole weight set, with
  each layer + the output head streaming-reading embed), and **the storage bandwidth wall is the
  absolute bottleneck**, consistent with M3-5's "Flash bandwidth wall" conclusion.
- 0.1B extrapolation: SmolLM-135M per-layer streaming read ≈ 72 MB/token, unusable at Flash/SD-card bandwidth.

**Decision**: 0.1B layer residency on the ESP32-P4 is **rejected** (the storage bandwidth wall
makes it unusable, regardless of whether the TF card is fixed). The streaming kernel is kept as a
technical feasibility verification artifact (`FLASH_STREAM=1`).

### M3-8 weight scale down to fp16 (rejected, 2026-09-26, on-device)

**Motivation**: whole-image resident decode's PSRAM read = weights `q` (1B/element) + scale `d`
(4B/block = 0.125B/element), with `d` accounting for 11% of the weight read. `d` is already fp16 in
the file and was converted to fp32 at expansion time; if it were kept fp16 during expansion, the
`d` read volume could be halved, theoretically saving ~5.5% bandwidth.

**Test artifact**: `d` retains the fp16 bit pattern during expansion, and `pie_gemv_row` uses a
software fast path (12 integer instructions) to convert to fp32. A pre-check confirmed that all
730,820 `d` values are normal numbers (no subnormals/special values), so the fast path is
**lossless**.

**On-device measurement** (Minueza-32M, whole-image resident):

| Item | fp32 `d` (baseline) | fp16 `d` (scheme) |
|---|---|---|
| per token | **230–234 ms** | **242–246 ms** |
| total 19 steps | **4431 ms** | **4663 ms** |
| sequence | 20/20 | 20/20 (A/B `diff=0`) |

**Conclusion (hard data)**: **negative gain of −4.3%**. The ESP32-P4 has no Zfh (half-precision FPU)
extension, so `fp16→fp32` can only be done in software (12 integer instructions/block, ~8M total);
the conversion issue overhead of ~23 ms is **greater** than the bandwidth saved (halving the `d`
read volume ≈ 13.8 ms), for a net slowdown of ~9 ms.

**Decision**: **rejected, reverted**. On an MCU without hardware fp16, "reducing the scale bit width
to save bandwidth" is backfired on by the software conversion overhead. The only **lossless**
bandwidth optimization point is thus disproven; the remaining gains can only come from lossy paths
(q4→q2, H4 authorization).

### M3-9 weight scale changed to int16 fixed-point (✅ landed, 2026-09-26, on-device)

**Background**: M3-8 disproved the fp16 scale, but the root cause was "12 software conversion
instructions", not "reducing the bit width". The ESP32-P4 has a single-precision FPU, and
`int16→fp32` is the hardware `fcvt.s.w` (1 instruction), so switch to an **int16 fixed-point scale
(per-tensor normalization)**: `d_fp32 = d_int16 × d_scale`.

**Implementation**: during expansion, first scan once to find `max_d`, then quantize
`d_int16 = round(d_fp32 / d_scale)` (two passes of reading, one-time); the hot loop `pie_gemv_row`
reads d with `lh + fcvt.s.w`, and `d_scale` is hoisted out of the accumulation as an outer
multiply. Pre-check: the per-tensor d dynamic range is at most **32.6×** (`L0.down`), far smaller
than the 3.3e4× representable by int16.

**On-device measurement** (Minueza-32M, whole-image resident):

| Item | fp32 d | fp16 d (M3-8) | **int16 fixed-point d (this)** |
|---|---|---|---|
| per token | 230–234 ms | 242–246 ms | **226–231 ms** |
| total 19 steps | 4431 ms | 4663 ms | **4365 ms** |
| sequence | 20/20 | 20/20 | **20/20** |

**Conclusion (hard data)**: **positive gain of +1.5%** (+0.06 tok/s). The int16 fixed-point precision
(Q15 relative ~3e-5) is higher than fp16 (~1e-3), and argmax is stable. The gain is smaller than the
theoretical 5.5% because `fcvt.s.w` is a floating-point instruction that **competes** with PIE's
`fmadd.s` for FPU issue slots, partially offsetting it —— proving that the decode bottleneck is a
**PSRAM bandwidth + FPU issue mix**, not pure bandwidth.

**Decision**: **keep and land** (the only approximately-lossless direction that delivers a positive gain).

### M3-10 weight quantization direction (q2/q3) rejected after H4 authorization (2026-09-26, PC side)

**Background**: after H4 authorization, verify the feasibility of "extreme quantization reducing
weight read bandwidth". Both lines are disproven simultaneously:

**① Bandwidth line: q4→q2 gain = 0 (correcting an earlier misjudgment)**

The previously recorded "q4→q2 halves the weight read volume, gain ~1.9×" is a **misjudgment** ——
mistaking "reduced file/Flash storage" for "reduced decode bandwidth". The truth (already recorded
in M3-4): the minimum weight bit width for a PIE dot is **int8**
(`dl_esp32p4_dotprod_i16k8o16`, no int4/int2 variant). After expansion both q2 and q4 are **int8
(1 B/element)** entering the PIE dot, so decode reads exactly the same number of weight bytes ⇒
**bandwidth gain = 0**; q2 only saves Flash storage.

**② Precision line: re-quantizing to q2/q3 directly collapses the model (hard data)**

`tools/legacy/verify_q2.py` re-quantizes the q4 dequantized weights to q3 (7 values) / q2 (3 values),
PC reference forward:

```
q3: match 4/20  seq=[...,0,0,0,0,...]  top3 logit=0.0000 (rms_norm overflow)
q2: match 4/20  seq=[...,0,0,0,0,...]  top3 logit=0.0000
```

Of the 4/20 matches, 4 are the prompt itself; the remaining 16 generated tokens are all 0 and the
logits are all zero ⇒ **the model is completely broken**, not a precision drop.

**Conclusion**: the weight quantization direction is **doubly sealed off** ——
1. Bandwidth: PIE has no int4/int2 dot; after expansion to int8 the read volume is unchanged, gain 0;
2. Precision: ternary/7-value quantization cannot "re-quantize" an existing model — it collapses
   directly (confirming that BitNet is a "ternary architecture trained from scratch", not a
   post-training quantization).

**Decision**: **rejected**. The MCU's decode bandwidth wall is **the PIE hardware instruction set
(int8-weight dot only) + the PSRAM physical boundary of 106 MB/s**, which even H4 authorization
cannot break through. Breaking through would require changing hardware (PIE supporting an int4 dot /
high-bandwidth storage) or changing the model (MoE sparsity / BitNet trained from scratch), both
beyond the scope of the current kernel.

### M3-11 output-head ANN sparse lookup (PLE idea) rejected (2026-09-26, PC side)

**Background**: borrowing the "sparse lookup" idea of esp32-ai's PLE, change the tied lm_head (a
32002-row embedding table) from "dense argmax" to "K-means clustering + cluster selection + exact
computation of candidates", to verify the recall rate and pruning rate.

**Key diagnostics** (`tools/legacy/verify_ann_head.py`):
- embed norm 0.309–0.627 (std 10%), h norm stable at ~46.6;
- **the true top-1 is always within the cosine top-16** (M=16 gives 16/16 recall);
- but pure-cosine top-1 matches only 8/16 —— the other half are flipped by "±10% norm", showing that
  the top-1/top-2 logit margin is extremely small.

**Hard data (spherical K-means; after normalization Euclidean = cosine)**:

| C | K | Recall | Pruning rate |
|---|---|---|---|
| 16 | 4 | 8/16 | 78% |
| 16 | 8 | 11/16 | 55% |
| 16 | 16 | 16/16 | **0%** |
| 32 | 32 | 16/16 | **0%** |

To achieve 100% recall you must select nearly all clusters, and the pruning rate approaches 0.

**Conclusion**: output-head ANN pruning is **rejected**. Using a different method from M3-3b's
"upper-bound pruning" (clustering vs upper bound), it reaches the same conclusion —— the output
head's 43% bandwidth is a hard wall. Root causes:
1. Margin too small: the top-1/top-2 logit gap is small enough that ±10% norm can flip it; the argmax
   information hides in tiny differences of direction and norm;
2. High-dimensional curse of dimensionality: in 312 dimensions, the neighborhood structure cannot be
   captured by clustering/upper bounds, and guaranteeing recall requires near-full computation.

**Decision**: **rejected**. At this point both dimensions —— "reducing bytes read per token"
(row chunking / bit-width reduction / weight quantization) and "reducing the number of elements"
(output-head pruning) —— are sealed off by the PIE int8 instruction set + the information-theoretic
lower bound of high-dimensional argmax; the only way out is changing the architecture (PLE/MoE
self-trained models), which is beyond the kernel's scope.

### M3-12 MoE-converting the existing dense model rejected (2026-09-26, PC side)

**Background**: under H4 authorization, split Minueza's dense FFN (ffn=1092) into N=8 experts + top-k
activation using MoEfication clustering, to verify whether "converting an existing model" can capture
the bandwidth benefit of MoE sparse activation.

**Oracle verification** (`tools/legacy/verify_moe.py`, oracle selects the optimal expert combination,
giving a precision upper bound):

| top-k | Activated neurons | Oracle relative error |
|---|---|---|
| 2 | 25% | **62.8%** |
| 4 | 50% | 42.0% |
| 6 | 75% | 25.4% |
| 8 | 100% | 0.0% |

Two facts: ① top-8 = dense (error 0; clustering does not destroy information); ② top-2 (25% of
neurons) error 62.8% —— even if the oracle picks the optimal experts every time, it still loses 62.8%
of the representational capacity.

**Root cause**: Minueza's FFN has no redundancy (ffn=1092 / dim=312 = 3.5×; neurons cooperate
closely), and sparse activation (top-2 = 25%) structurally loses 75% of the representational
capacity, which training cannot recover. Both conversion paths are blocked:
1. MoEfication (cluster splitting): experts are 1/N neuron fragments, with an output magnitude of only
   k/N of the dense one;
2. Sparse Upcycling (full expert copies): parameters explode N-fold (10.2M→81.6M), far exceeding the
   PSRAM 28M.

**Decision**: **rejected**. Same pattern as M3-10 (q2 ternary collapses the model): **architectural
features such as sparsity/ternary/expert division of labor must be trained from scratch and cannot be
retrofit from a dense model.** The MoE bandwidth benefit can only come via the "train complete small
experts from scratch" route (the esp32-ai/PFor approach), pivoting to a distillation training plan
(see docs/moe_framework.md).

### Remaining bottlenecks and follow-up optimization directions (ordered by expected gain)

1. **PIE fixed-point GEMV (M3-4 stage 1)**: ✅ landed, 2.11× (see above).
2. **Scalar fp32 dot product 4-way unroll**: the `dl_esp32p4_dotprod_f32` style, lossless, can continue
   (but most of the gain is already covered by PIE).
3. ~~Row chunking (compute 4 rows at a time, amortizing x loads)~~: disproven —— activations are in
   internal RAM (not PSRAM), so re-read is not a bottleneck, gain ≈ 0.
4. **Weight scale int16 fixed-point**: ✅ landed, +1.5% (see M3-9); the fp16 version is disproven (M3-8).
5. ~~Weight quantization q2/q3~~: rejected (M3-10) —— PIE has no int4/int2 dot (bandwidth 0) +
   re-quantization collapses the model.
6. ~~Output-head pruning (lossless upper bound + lossy ANN)~~: rejected (M3-3b / M3-11) —— extremely
   small margin + high-dimensional curse of dimensionality, pruning rate approaches 0.
7. **MoE architecture (training from scratch + distillation)**: the kernel's router/top-k/sparse FFN are
   ready; the model follows the "complete small experts + distillation teacher" route (M3-12 conclusion),
   see docs/moe_framework.md.

> Currently 0.229 s/token (4.37 tok/s), a cumulative **7.1×** over the M2 scalar starting point (1.63 s).
> The largest remaining bottleneck is PSRAM bandwidth (GEMV weight+activation reads) + FPU issue, not
> pure compute; PIE's pure compute 21× has been pressed down to 2.11× by the bandwidth wall. Further
> gains require reducing bandwidth usage or full fixed-pointization (including RMSNorm/attention/SwiGLU,
> effort = rewriting the int8 kernel).


## Boundary constraints (self-stated)

- Pure-scalar C11; introducing NEON/AVX/OpenMP/Linux syscalls is forbidden.
- Bit-level agreement with vLLM-Kestrel **does not hold**; no performance-equivalence comparison is made
  with it.
- Any lossy approximation (quantization/pruning) requires separate project authorization and is outside
  the scope of this milestone.

## License

Apache License 2.0 —— see **[../LICENSE](../LICENSE)** and **[../NOTICE](../NOTICE)**.
(The **sole authoritative source** for the license and third-party component notices is the root README;
it is not repeated here to avoid drift.)
