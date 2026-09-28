# Kestrel-MCU 开发日志（milestones 与实测台账）

> English version: **[milestones.md](milestones.md)**

> 本文档是 kestrel_mcu 的**开发日志**：按里程碑记录能力探针、端到端推理、性能优化的
> 全部硬数据与踩坑结论（M1 / M2 / M3-x）。项目概览与快速开始见
> **[../README.md](../README.md)**（英文）/ **[../README.zh-CN.md](../README.zh-CN.md)**（中文）；
> 工具链说明见 **[../tools/README.zh-CN.md](../tools/README.zh-CN.md)**。
>
> 许可证见 **[../LICENSE](../LICENSE)** 与 **[../NOTICE](../NOTICE)**。

## 定位

面向 **MCU 级 RISC-V 设备**（首目标：Waveshare ESP32-P4-WIFI6-DEV-KIT，SKU 32054）
的**纯标量 C11** 微型推理内核。

它与 `GitHupSRC/`（vLLM-Kestrel，aarch64/x86-64 大模型引擎）**完全独立**：
- 不复用其 NEON dotprod/fp16、AVX2 内核
- 不复用其 Linux 运行时（mmap//proc//sys/OpenMP/dlopen）
- 不加载 Qwen3-VL 等 GB 级权重（MCU 物理上不可能）
- 是一条**独立的 determinism line**，与 x86/arm 不保证位级一致

## 目标硬件（SKU 32054）

| 项 | 值 |
|---|---|
| SoC | ESP32-P4（RISC-V 32 位，双核 HP 360MHz + 单核 LP 40MHz） |
| 协处理器 | ESP32-C6（Wi-Fi 6 / BT 5） |
| 内存 | 768KB L2 + 32MB PSRAM + 16MB Flash |
| 工具链 | ESP-IDF 5.3+（target = `esp32p4`） |
| 调试 | Type-C UART（烧录 + 串口回读） |

## 首里程碑

在纯标量 RISC-V 上实现并验证 SHS 公理库（`axiom_registry.json`）中的
基础算子 `S_0` / `UPA`，通过 UART 输出自检结果。**不引入任何 NEON/AVX
假设**，为后续微型网络铺路。

## 构建 / 烧录（Windows 实测通过）

### 本机已装环境

| 组件 | 版本 / 位置 |
|---|---|
| ESP-IDF | v6.0.2 → `E:\esp-idf`（Gitee 镜像克隆） |
| 工具链 | riscv32-esp-elf 15.2.0 → `%USERPROFILE%\.espressif` |
| Python | 3.12.9（IDF 专用 venv） |
| 串口 | COM5（Type-C USB 串行设备，即本开发板） |

### 构建命令（PowerShell）

```powershell
# 0) 短盘符别名：工具链真实路径过深，会触发 Windows MAX_PATH(260) 限制
#    映射后 C++ 标准头路径从 261 字符降到 ~229。重开机后需重新执行。
subst X: "%USERPROFILE%\.espressif"

$env:IDF_TOOLS_PATH = "X:\"
$env:IDF_SKIP_CHECK_SUBMODULES = "1"   # esp32-wifi-lib 子模块在 Gitee 被封锁，本项目已禁用 WiFi/BT

. E:\esp-idf\export.ps1                 # 激活 IDF 环境

# 本项目 build/ 是用 X:\ 形态的 venv 配置的，必须把 X: 的 venv 放回 PATH 最前，
# 否则 idf.py 会报 "python.exe is currently active in the environment while the
# project was configured with 'X:\python_env\...'" 并直接退出。
$env:PATH = "X:\python_env\idf6.0_py3.12_env\Scripts;" + $env:PATH

Set-Location E:\kestrel_mcu             # 必须纯 ASCII 路径，见下方「已知约束」①
idf.py set-target esp32p4
idf.py build
```

> 改动 `sdkconfig.defaults` 后需**同步改 `sdkconfig`**（或删除它重新生成），
> 否则 IDF 不会重跑 Kconfig。重新生成会触发 git 子模块检查 ⇒
> 必须先设 `IDF_SKIP_CHECK_SUBMODULES=1`，否则卡在 esp32-wifi-lib 的 HTTP 423。

### 烧录

```powershell
# 烧录前关闭其它串口监视器（避免端口占用）
idf.py -p COM5 flash monitor
```

### 已知约束（均为实测踩坑，勿随意改动）

1. **项目路径必须纯 ASCII**。原项目路径含中文（`E:\项目\...`），riscv 工具链读
   `-specs` 文件时会把非 ASCII 字符破坏成 `\x0a`，报
   `cannot read spec file 'E:/\x0a/...'`。因此构建副本置于 `E:\kestrel_mcu`。
2. **工具链路径过深会触发 Windows MAX_PATH(260)**。表现为
   `fatal error: bits/c++config.h: No such file or directory`。
   解决：`subst` 短盘符（见上）。`sdkconfig.defaults` 中的
   `CONFIG_COMPILER_CXX_RTTI=y` 是叠加保险（使 C++ 头退回较短的 `ilp32f` 目录）。
3. **子模块检查需跳过**。`components/esp_wifi/lib` 在 Gitee 镜像被封锁（HTTP 423），
   而本项目已 `CONFIG_ESP_WIFI_ENABLED=n` / `CONFIG_BT_ENABLED=n`，实际不需要。
   故设 `IDF_SKIP_CHECK_SUBMODULES=1`。
4. **默认 picolibc 有兼容问题**，`sdkconfig.defaults` 已切 `CONFIG_LIBC_NEWLIB=y`。
5. **ESP32-P4 的 PSRAM 是八线（hex）模式**，不是 quad；配置符号为 `CONFIG_SPIRAM_MODE_HEX`。

### 构建产物（实测）

| 文件 | 大小 |
|---|---|
| `build/kestrel_mcu.bin` | 208.6 KB（app 分区 1 MB，余量 80%） |
| `build/bootloader/bootloader.bin` | 23.3 KB |
| `build/partition_table/partition-table.bin` | 3.0 KB |

ELF 校验：`Machine: RISC-V`，`Class: ELF32`，`Flags: RVC, single-float ABI`。

## M1 能力探针实测结果（2026-09-26，板端）

固件：`main/main.c`（SHS 自检 + PSRAM 带宽 + TF 卡诊断）。

| 项目 | 实测值 | 判读 |
|---|---|---|
| 芯片 | ESP32-P4 rev v3.1，双核 + LP 核，**400 MHz** | 与规格一致 |
| PSRAM 容量 | **32768 KB**（32 MB，hex 模式 @200 MHz） | 权重常驻预算可用 |
| PSRAM 写带宽 | **83 MB/s**（16 MB，u32 顺序） | — |
| PSRAM 读带宽 | **84 MB/s**（u32 顺序）/ **90 MB/s**（u32 ×4 展开） | 4 路展开仅 +7% ⇒ **带宽受限**，非延迟受限 |
| TF 卡（SDIO） | **完全无响应**：6 种组合全报 `ESP_ERR_TIMEOUT` @ `send_op_cond(CMD1)` | 见下 |

### TF 卡诊断详情

**第一次检测（M1）** —— 已穷举测试（均失败，错误一致为 CMD1 无应答）：

| GPIO45 供电极性 | 总线宽度 | 时钟 | 结果 |
|---|---|---|---|
| 高 | 4-bit | 20 MHz | ESP_ERR_TIMEOUT |
| 高 | 1-bit | 20 MHz | ESP_ERR_TIMEOUT |
| 高 | 4-bit | 40 MHz | ESP_ERR_TIMEOUT |
| 低 | 4-bit | 20 MHz | ESP_ERR_TIMEOUT |
| 低 | 1-bit | 20 MHz | ESP_ERR_TIMEOUT |
| 低 | 4-bit | 40 MHz | ESP_ERR_TIMEOUT |

`send_op_cond` 是卡初始化**最早**的一步（在文件系统之前）：卡完全不回话，
说明问题在**电气层**而非协议/格式层。

**第二次检测（复测，权威结论）** —— 错误性质发生**根本变化**：

| 项 | 第一次 | 第二次 |
|---|---|---|
| 失败点 | `send_op_cond(CMD1)` | `esp_vfs_fat_sdmmc_mount` |
| 错误码 | `ESP_ERR_TIMEOUT`（无应答） | `ESP_FAIL` / `failed to mount card (13)` |
| FatFS 含义 | — | `FR_NO_FILESYSTEM` |
| 判读 | 卡在电气层完全不回话 | **卡电气链路已正常**，仅缺 FAT 文件系统 |

⇒ 结论修正：**卡是好的，CMD/DAT 链路也是好的**（有效组合：`GPIO45 = 0`，
该负载开关**低有效**）。唯一缺的是卡上**没有 FAT 分区表**。
格式化（会清空卡内数据）即可用于后续 0.1B 分层驻留，需单独授权后执行。

> 注意：**0.05B 常驻里程碑不依赖 TF 卡** —— 0.05B@INT4 ≈ 25 MB，可直接放入
> 32 MB PSRAM。TF 卡只在后续 0.1B 分层驻留阶段才是必需。

### 踩坑记录（本会话实测）

- `esp_vfs_fat_sdmmc_mount` 失败时**内部已自行 host deinit**；调用方再调
  `sdmmc_host_deinit()` 会**双重 deinit → Instruction access fault 崩溃**。
- 默认 `SDMMC_SLOT_CONFIG_DEFAULT()` 的 `width` 允许 8-bit，会把 **GPIO45 当 D4
  抢走**（而它正是本板 TF 卡电源开关）；必须显式设 `slot.width = 4`。

## M2 端到端推理实测（2026-09-26，板端）

**结论：0.032B 模型已在 ESP32-P4 上全量 PSRAM 常驻跑通，输出与 PC 参考逐一 token 一致。**

### 模型与格式

| 项 | 值 |
|---|---|
| 模型 | `Felladrin/Minueza-32M-Base`（Mistral 家族：RMSNorm+RoPE+GQA+SwiGLU） |
| 架构 | layers=10, dim=312, heads=12, kv_heads=4, head_dim=26, ffn=1092, vocab=32002 |
| 参数量 | 22.81M（**已强制权重共享**，原模型 `tie_word_embeddings=false`，属有损变更） |
| 量化 | Q4_0 扁平块（32 权重/块，fp16 scale）= 4.5 bit/权重 |
| 体积 | **12.26 MB** → 放入 16MB NOR Flash 的 `model` 分区（偏移 `0x210000`） |

### 验证（唯一判据：与同量化权重的 fp32 NumPy 参考对拍）

| 项 | MCU | PC 参考 | 差 |
|---|---|---|---|
| 单 token[1] top1 | id=684, logit=5.229595 | id=684, logit=5.229599 | 4e-6 |
| 4-token prompt top1 | id=23624, logit=5.184734 | id=23624, logit=5.184731 | 3e-6 |
| **20 token 贪心序列** | 见下 | 同 | **完全一致** |

```
1,450,2217,4996,23624,1199,8752,23624,1199,20925,3161,4865,442,5102,2672,3161,13040,17574,28394,1628
```

> 注意：这是「同量化权重下 MCU vs PC 数值等价」，**不是**跨架构与引擎的位级一致
> （后者按 H1 明确不要求、也不得声称）。
>
> ⚠️ 上表数字与序列属于 **M2 时点**（未行对齐的 q4 布局）。M3-2 引入行对齐后
> 量化块边界改变，**该序列已作废**；M3 之后的基准序列见 M3-2 一节。

### 性能与内存

| 项 | 实测 |
|---|---|
| decode | **1.63 s/token ≈ 0.61 tok/s**（19 步 31.0 s） |
| Flash→PSRAM 装载 | 12.26 MB / 1.61 s = **7.8 MB/s**（DIO 80MHz 受限） |
| PSRAM 占用 | 权重 12.26 MB + KV 2×520 KB + 目录表 |
| 内部 RAM | scratch 18 KB |
| 算力利用率 | 22.8M MAC / 1.63 s ≈ **14 MMAC/s**（相对 PIE 潜力有 **~100×** 余量） |

**主要瓶颈**：标量逐元素 q4 反量化（每权重一次半字节提取 + fp16 转换），
以及输出头对 `32002×312 = 10M` 元素的流式 argmax（占总工作量 44%）。
优化方向：PIE 128-bit 向量化 GEMV、块级批量反量化、输出头分块剪枝。

### 踩坑记录（本会话实测）

- **测试脚手架 bug（非模型 bug）**：贪心循环曾把生成结果写回 `seq[0..PROMPT_LEN-1]`，
  覆盖了 prompt 元素，导致 MCU 从未处理完整 prompt，序列与参考无法对照。
  修正为「先预处理整条 prompt，再自回归生成」后 20/20 完全一致。
- 计算型任务（单步 1.6 s 不喂狗）需关闭任务看门狗 `CONFIG_ESP_TASK_WDT_EN=n`，
  否则每步都打印寄存器转储。
- riscv32 工具链上 `uint32_t` 是 `unsigned long`，printf 必须用 `PRIu32`/显式转 `unsigned`。

## M3 性能优化（进行中，2026-09-26）

### M3-1 已落地：q4 → int8 装载期展开 + 独立 fp16 scale 数组（PSRAM）

动机：原实现对每个权重元素做「半字节提取 + 移位/掩码/分支 + fp16 转换」，
是 decode 的主导开销。展开后内循环退化为紧凑的 `(float)q * x` 累加。

| 项 | 改动前 | 改动后 |
|---|---|---|
| 速度 | 1.63 s/token（0.61 tok/s） | **1.11 s/token（0.90 tok/s）** |
| 正确性 | 20/20 一致 | **20/20 一致（保持）** |
| logits 偏差 | ~3e-6 | ~2e-6 |
| PSRAM 占用 | 权重 12.26 MB | 展开后 **23.1 MB**（arena）+ 余 7.6 MB |
| 启动开销 | 1.61 s（整图载入） | 2.48 s（分块展开） |

实现要点：
- 为省 PSRAM，**不再整份载入 12.26 MB 镜像**，改为经 `km_read_fn` 回调
  **从 flash 分块展开**（`km_fast_build_ex`）；F32 范数权重也并入同一 arena。
- 内循环加 `#pragma GCC unroll 8`（0.86 → 0.90 tok/s）。
- 快路径**不引入新的量化近似**：`q` 与 `d` 完全来自原 q4 数据；
  仅浮点结合顺序变化（`d` 提到块外）。

### M3-2 已落地：行对齐到 32（converter + 布局 + 内核 三方同步）

**改动**：q4 权重在量化前**按行补齐到 32 的整数倍**（行尾零填充），
使每个 32 元素块只属于一行、块不跨行。展开后
`q[r][c] = q[r*align32(n_cols)+c]`，块号 `= r*nblk + b`。
GEMV 内层因此从「变长 `cnt` + 相位漂移 + 分支」变为**定长 32 的全展开循环**。

**代价（须明确）**：这是**有损变更**。块边界位置改变 ⇒ 每块的 `max_abs`/scale
改变 ⇒ 反量化权重与 M2/M3-1 不再逐位相同（对齐后每块只覆盖同一行，
通常量化保真度略优，但**等价性不再成立**）。因此 PC 参考的贪心序列本身
也随之改变，M2 记录的那条 20-token 序列已作废。

| 项 | M3-1 | M3-2 @ `-Og` | **M3-2 @ `-O2`** |
|---|---|---|---|
| 速度 | 1.11 s/token（0.90 tok/s） | 1.24 s/token（**0.81** tok/s） | **0.511 s/token（1.96 tok/s）** |
| 正确性 | 20/20 | 20/20 | **20/20（保持）** |
| logits 偏差 | ~2e-6 | ~1e-6 | ~1e-6 |
| 模型镜像 | 12.26 MB | 13.19 MB | 13.19 MB |
| arena | 23.1 MB | 23.7 MB | 23.7 MB |
| 周期/元素 | 19.5 | 21.8 | **8.7** |

**关键发现（认知修正）**：M3-2 在 `-Og` 下**反而变慢**（0.90 → 0.81）。
根因不是行对齐，而是**整个固件此前一直构建在 `-Og`**（ESP-IDF 默认
`CONFIG_COMPILER_OPTIMIZATION_DEBUG=y`）。`-Og` 为可调试性牺牲循环优化，
使 `#pragma GCC unroll` 基本失效 —— 这也解释了 M3-1 里 unroll 只换来 4%。
切到 `-O2`（`CONFIG_COMPILER_OPTIMIZATION_PERF=y`）后：

- decode：1.11 → **0.511 s/token**（较 M3-1 **2.17×**，较 M2 **3.19×**）
- PSRAM 读带宽探针：84 → **106 MB/s**（同一份探针代码）⇒
  **此前所有 `-Og` 下的性能读数都被系统性压低**，含 M1/M2/M3-1。

**归因诚实性**：`-O2` 与「行对齐」的收益**尚未解耦实测**。
可以确定的只有：M3-2 的内核改动在 `-Og` 下是 −10%；在 `-O2` 下的单独贡献
需要回退到 M3-1 布局再测一次 `-O2` 才能给出。

### M3-3a 已落地：fp16 scale → fp32 装载期预展开（无损）

**改动**：`km_ft_t.d` 从 `uint16_t*`（fp16）改为 `float*`（fp32）。装载期
`km_fast_build_ex` 里对每块的 fp16 scale **一次性** `km_f16_to_f32` 转成 fp32，
热循环（`gemv_q8` / `q8_row_get` / 输出头）内层从「逐块分支转换」退化为
直接 `acc += dr[b] * s`。fp16→fp32 是精确转换，**不改变任何数值**。

| 项 | M3-2 @ `-O2` | **M3-3a @ `-O2`** |
|---|---|---|
| 速度 | 0.511 s/token（1.96 tok/s） | **0.489 s/token（2.04 tok/s）** |
| 正确性 | 20/20 | **20/20（保持）** |
| logits | ~1e-6 | **逐位不变**（5.350636 / 4.978979） |
| arena | 23.7 MB | **25.7 MB**（d 数组 2B→4B，余 5.6 MB） |
| 收益 | — | **+4.3%** |

**归因诚实性**：`-O2` 下编译器已把 `km_f16_to_f32` 充分内联，故移除该转换
只换来 +4%。这印证了一个更重要的判断：**标量内已接近收益天花板**，
剩余大头必须靠向量化（PIE）或**有损**剪枝。

### M3-3b 结论（重要）：无损输出头剪枝在 q4_0 下不可行

原计划的「两阶段 argmax（分块上界剪枝）」若要**无损**（保持 20/20 与参考
逐位一致），经分支定界分析在 q4_0 量化下**剪枝率趋近于 0**：

- 严格上界需用 `q∈[-8,7]` 的最坏符号构造（`M_b = Σ 7·h⁺ − 8·h⁻`），
  剩余维度上界 `U_v = d_max_v · Σ M_b`。
- 量化 scale `d` 本身即「最坏情况」放大（`d = max_abs/8`），
  使 `U_v ≈ 19` 远大于 logit 本身量级（~5）⇒ 判据
  `partial_v + U_v < B` 几乎永不成立，剪不掉几个 v。
- Cauchy-Schwarz 界（`‖q‖·‖h‖`）同样太松。

**结论**：输出头的实质加速只剩两条路——
1. **PIE 向量化**（M3-4，无损，预期 5–10×）；
2. **有损两阶段剪枝**（部分维度粗筛 top-K，接受极低概率的决策差异），
   会破坏「MCU 与 ref 逐 token 对拍」这一验证约束，属 **H4 未授权有损近似**，
   **须另行立项授权**后方可实施。

### M3-4 调研结论（PIE 可行性，2026-09-26）

在公理库「实测胜于推论」约束下，用硬数据核实了 PIE 路径。**结论：可行但门槛高，
且收益受带宽墙约束，并非无脑 5–10×。**

**实测事实链**：

| # | 事实 | 依据 |
|---|---|---|
| 1 | CPU = RISC-V 双核 400 MHz + 单精度 FPU + **PIE**（私有 128-bit SIMD） | 官方 datasheet |
| 2 | 编译器认识 `xespv`：`__riscv_xespv = 2001000`，march 已含 `_xesploop_xespv` | `gcc -dM -E` 实测 |
| 3 | 工具链含 **xespv2p1 / xespv2p2** 两代汇编器/反汇编器 | Glob 工具链 bin |
| 4 | **无公开 C intrinsics 头文件**（无 `esp_pie.h`/`xespv.h`），无 builtins | Glob/Grep 工具链 |
| 5 | **GCC 不自动向量化到 xespv**：反汇编 `kmcu.c.o` 共 4136 条指令，全为标准 RV32I/F 标量（`lw/sw/flw/fcvt.s.w/fmadd.s`…），**零 PIE、零标准 RVV 指令** | `objdump -d` 实测 |
| 6 | PIE ISA 手册存在于官方 **TRM 第 4 章「Processor Instruction Extensions (PIE)」，状态 Published**（最新 v0.7） | Espressif TRM |

**关键区分**：PIE（`xespv`，乐鑫私有）≠ 标准 RVV（`v` 扩展）。工具链里的
`riscv_vector.h` 对应标准 RVV，而本 march **没有 `v`**，故标准 RVV intrinsics
全部不可用。网络上有文章误称「ESP32-P4 用 RVV 1.0」，与 datasheet 矛盾，不可采信。

**可行路径（唯一）**：手写 **RISC-V 内联汇编**，指令编码/语义取自 TRM 第 4 章。
- 无损性：int8×fp32 乘加 + fp32 累加是确定性运算；SIMD 并行累加会改变浮点
  结合顺序（~1e-6 级），与 M3 既有改动同性质，**非量化近似**，argmax 20/20 应保持。
- 风险：① 两代 PIE（v2p1/v2p2）兼容；② 内联汇编调试成本；③ TRM 为 PDF 需人工研读。

**带宽墙（公理库「带宽受限」约束）**：当前 8.4 周期/元素，权重+激活读在 PSRAM
（~84–106 MB/s）。若 PIE 把计算压到 ~1 周期/元素，decode 将撞带宽墙，
理论 5–10× 的**上限需实测界定**，很可能收敛到 2–4×。

### M3-4 决定性结论（2026-09-26，乐鑫源码铁证）

> **上述「无损性」判断被推翻。** 继续啃 PIE 后拿到乐鑫官方源码级铁证：
> **PIE SIMD 只支持整数类型，没有 fp32 向量指令。**

**证据**：ESP-DL 的 `dl_esp32p4_dotprod_f32`（fp32 点积）源码注释原文：
> *"The PIE SIMD extension only supports integer datatypes, so the float dot product
> is implemented with the RISC-V single-precision FPU."*

即：fp32 点积乐鑫自己也只能退回**标量 FPU**（4 路展开 + 4 个独立累加器隐藏
`fmadd.s` 延迟）。这与我反汇编 `kmcu.c.o` 全标量（`fcvt.s.w`+`fmadd.s`）互为印证。

**因此**：当前 GEMV 是 `int8 权重 × fp32 激活`，**无法无损 PIE 加速**。
要吃 PIE 红利，**唯一路径是激活定点化**（用户已授权近似轨）：

| 现成 PIE 点积（ESP-DL `dl_esp32p4_dotprod*.S`） | 数据格式 | 精度 |
|---|---|---|
| `dl_esp32p4_dotprod_i16k8o16` | **int8 权重 × int16 激活**（`w8a16`） | 较高（最匹配现有 int8 权重） |
| `dl_esp32p4_dotprod_i16k16o16` | int16 × int16 | 高 |
| `dl_esp32p4_dotprod_i8k8o16` | int8 × int8 | 最低 |

核心 PIE 指令（已从源码确认）：`esp.vldext.s8.ip`（int8 加载+符号扩展）、
`esp.vmulas.s8/s16.xacc(.ld.ip)`（向量乘加累加到 `xacc`）、`esp.zero.xacc`（清零）、
`esp.srs.s.xacc`（取累加结果+饱和右移）。128-bit 寄存器 `q0..q7`，标量 `a0..a7`/`x*`。

**附带无损收益**：`dl_esp32p4_dotprod_f32` 揭示标量 fp32 点积的最优写法
（4 路展开 + 4 累加器），可无损套用到当前 `gemv_q8`，是定点化前的免费一步。

**定点化范围（工程量分层）**：
- 最小：只定点化 GEMV（激活 int16 量化 + PIE 点积 + 块 scale 还原），其余
  RMSNorm/attention/SwiGLU 仍走 fp32。激活每层量化/反量化，误差逐层累积。
- 全量：整个前向定点化（含 RMSNorm 已有 `dl_esp32p4_rms_normalization.S`），
  工作量 = 重写一个 int8 定点 Transformer 内核。

### M3-4 阶段 0 实测：PIE 定点 GEMV 单算子（2026-09-26）

移植乐鑫 `dl_esp32p4_dotprod_i16k8o16`（改签名返回 int32 不饱和）为
[pie_dotprod.S](../main/pie_dotprod.S)，
板上实测（`pie_bench`，N=4096，200 次）：

```
[PIE] dotprod N=4096  scalar=19713  pie=19713  match=1
[PIE] 200 iters: scalar=16396 us  pie=782 us  speedup=20.97x
```

| 项 | 结果 |
|---|---|
| 正确性 | **match=1**（PIE 与标量 C 逐位一致，int8×int16→int32） |
| 纯计算加速 | **20.97×**（数据在内部 RAM，无 PSRAM 带宽墙） |
| decode 影响 | 20/20 保持，0.489 s/token 不变（仅加 bench，未接入 forward） |

**判定**：PIE 定点 GEMV **可行且正确**，纯计算上限 **~21×**（16 个 int8 乘加/指令）。
这是「啃 PIE」的第一口：指令语法、寄存器约束、正确性全部打通。

**兑现到 decode 的实际收益**：21× 是**计算加速比上限**，真实 GEMV 权重在
PSRAM（~84–106 MB/s），接入后会撞带宽墙，实际收敛区间待测（预估 2–4×，
遵循公理库「实测界定收益」——下一步就是接入 `gemv_q8` 实测）。

**接入成本（下一步）**：把 `gemv_q8` 的 `int8×fp32` 改为 `int8×int16` 定点，
需在每层 GEMV 前把 fp32 激活量化成 int16（`x_q16 = round(x * scale)`），
GEMV 后用 PIE 点积 + fp32 块 scale 还原。激活量化有损（用户已授权近似轨），
RMSNorm/attention/SwiGLU 其余路径暂保持 fp32。

### M3-4 阶段 1 落地：PIE 定点 GEMV 接入 forward（2026-09-26）

`gemv_q8` 与输出头已改为 **w8a16 定点 PIE 路径**（`pie_gemv_row`），
激活 per-vector int16 量化（`xq = round(x·S)`，`S = 32767/max_abs`），
PIE 分块点积 + fp32 块 scale 还原，最后乘 `1/S`。

**最终实测（`-O2`）**：

| 项 | M3-3a（标量 fp32） | **M3-4 阶段 1（PIE 定点）** |
|---|---|---|
| decode | 0.489 s/token（2.04 tok/s） | **0.232 s/token（4.31 tok/s）** |
| 加速 | — | **2.11×** |
| 正确性 | 20/20 | **20/20（序列与 fp32 完全一致）** |
| single-token | argmax=684 | **argmax=684（不变）** |

```
seq:1,450,2217,4996,23624,1199,8752,23624,1199,1199,8752,23624,1199,10436,23624,1199,3065,3161,4865,442
```

**关键结论**：int16 激活量化（per-vector）对该模型 argmax **零影响**（20/20
与 fp32 完全一致，PC 模拟已证量化误差 ~3e-4）。实际收益 **2.11×**，落在
带宽墙收敛区间（2–4×）内——纯计算 21× 被 PSRAM 带宽（~84–106 MB/s）封顶。

**两个关键踩坑（PIE 硬约束）**：
1. **标量操作数寄存器约束**：PIE 指令（`esp.vld*/vldext*/vmulas*/srs`）的
   **地址/标量操作数不能用 `t0-t2`（x5-x7）**，报 `illegal operands`；
   须用 `a0-a7`（x10-x17）或 `t3+`（x28+）。`srs` 目标用 `t3`（x28）合法，
   用 `t2`（x7）非法。
2. **16 字节对齐**：`esp.vld.128`/`vldext.s8` 要求 16 字节对齐地址。scratch 里
   int16 量化缓冲 xq 若用 `heap_caps_malloc`（只保证 4 字节对齐）会读错数据
   （表现为 dot 大值出错、argmax 全变）。须用 `heap_caps_aligned_alloc(16, ...)`。

### M3-5 分层驻留验证（判否，2026-09-26）

**问题**：0.1B 模型（q4_0 ≈ 56 MB）超过 32 MB PSRAM，能否用「权重逐层驻留」
（算哪层从 Flash 读哪层、算完释放）来跑？对标 GitHupSRC 的 `VLLM_VQF_STREAM=1`。

**实测件**：`LAYER_RESIDENT_TEST`（main.c，默认关）——自回归 decode 每 token 从
Flash 重读整份权重并重新展开，把「重读」与「计算」分开计时。

**板端实测**（0.032B，12.58 MB 权重）：

| 项 | 常驻 | 分层驻留（每 token 重读） |
|---|---|---|
| read（Flash 读+展开） | 0 | **2343 ms** |
| calc（decode） | 232 ms | 232 ms |
| **每 token** | **232 ms** | **2575.8 ms** |
| 慢 | — | **11.1×** |

**结论（硬数据）**：
- 分层驻留 read 占 91%、calc 占 9%，**Flash 带宽墙（~5.4 MB/s 读+展开）是绝对瓶颈**。
- 外推 0.1B（56 MB ≈ 12.58 MB × 4.5）：read ≈ 2343×4.5 ≈ **10.5 s/token**，不可用。
- 正确性保持 20/20（重读的权重逐位相同，仅慢）。

**决策**：0.1B 在 ESP32-P4 上**不走分层驻留**作为生产路径，但**保留分层驻留代码**，
目的不是用于推理，而是**验证技术可行性**——用实测量化「存储带宽墙」的硬边界。
**下一步**：对 **0.1B 模型**做分层驻留实测，验证 0.032B 外推的 ~10.5 s/token 是否成立。
生产可行路径仍是更激进量化（q2 ≈ 31 MB）整份压进 32 MB PSRAM，或 TF 卡分层
（~2.8 s/token，仍边缘）。验证件默认关（`LAYER_RESIDENT_TEST=1` 开启）。

### M3-7 Flash 逐层流式内核验证（2026-09-26，板端）

**背景**：0.1B 分层驻留需要**真正的逐层流式内核**——整份展开（`km_fast_build_ex`）
把 SmolLM-135M 展开到 ~150MB，远超 32MB PSRAM。TF 卡当前 CMD1 不通（电气层），
故先在 Flash 上用 Minueza-32M 验证流式内核**数值正确性** + **带宽代价**。

**实现**：`km_decode_step_stream`（kmcu.c）——embed 按行流式读、每层 9 个张量
「读入→展开→计算→释放」（复用单张量 arena，仅 383 KB）、输出头按行流式 argmax。
`FLASH_STREAM` 路径用 `esp_partition_read` 作 read_fn。详见 `kmcu.c` 流式注释。

**板端实测**（Minueza-32M，Flash 源）：

| 项 | 整份常驻（M3-4） | 逐层流式（M3-7） |
|---|---|---|
| 每 token | **232 ms** | **3409 ms** |
| 每 token read 量 | 0（常驻） | **12.57 MB** |
| 序列 | 20/20 | **20/20（与整份逐 token 一致）** |

**结论（硬数据）**：
- **流式内核数值正确**：top8 与完整序列和整份 decode 完全一致，验证了 embed 按行读、
  逐层展开、输出头流式 argmax 三者与整份展开**数值等价**。
- **逐层流式慢 14.7×**：每 token read 12.57 MB（≈ 整份权重，各层 + 输出头流式读 embed），
  **存储带宽墙是绝对瓶颈**，与 M3-5 的「Flash 带宽墙」结论一致。
- 0.1B 外推：SmolLM-135M 逐层流式 read ≈ 72 MB/token，Flash/SD 卡带宽下不可用。

**决策**：0.1B 分层驻留在 ESP32-P4 上**判否**（存储带宽墙使其不可用，与 TF 卡是否修好无关）。
流式内核作为技术可行性验证件保留（`FLASH_STREAM=1`）。

### M3-8 权重 scale 降 fp16（判否，2026-09-26，板端）

**动机**：整份常驻 decode 的 PSRAM 读 = 权重 `q`（1B/元素）+ scale `d`（4B/块 = 0.125B/元素），
`d` 占权重读的 11%。`d` 在文件里本就是 fp16，展开时转成了 fp32；若展开期保持 fp16，
可把 `d` 读量减半，理论省 ~5.5% 带宽。

**实测件**：`d` 展开期保留 fp16 位模式，`pie_gemv_row` 内用软件快路径（12 条整数指令）转 fp32。
预检确认 730,820 个 `d` 全为正常数（无次正规/特殊值），快路径**无损**。

**板端实测**（Minueza-32M，整份常驻）：

| 项 | fp32 `d`（基准） | fp16 `d`（方案） |
|---|---|---|
| 每 token | **230–234 ms** | **242–246 ms** |
| total 19 steps | **4431 ms** | **4663 ms** |
| 序列 | 20/20 | 20/20（对拍 `diff=0`） |

**结论（硬数据）**：**负收益 −4.3%**。ESP32-P4 无 Zfh（半精度 FPU）扩展，`fp16→fp32`
只能软件转换（12 条整数指令/块，共 ~8M 条），转换发射开销 ~23 ms **大于**省下的带宽
（`d` 读量减半 ≈ 13.8 ms），净慢 ~9 ms。

**决策**：**判否，已回退**。无硬件 fp16 的 MCU 上，「降低 scale 位宽省带宽」被软件转换
开销反噬。唯一**无损**带宽优化点就此证伪；剩余收益只能走有损路径（q4→q2，H4 授权）。

### M3-9 权重 scale 改 int16 定点（✅ 落地，2026-09-26，板端）

**背景**：M3-8 证伪 fp16 scale，但根因是「软件转换 12 条指令」，不是「降位宽」。
ESP32-P4 有单精度 FPU，`int16→fp32` 是硬件 `fcvt.s.w`（1 条），故改用
**int16 定点 scale（per-tensor 归一化）**：`d_fp32 = d_int16 × d_scale`。

**实现**：展开期先扫一遍求 `max_d`，再量化 `d_int16 = round(d_fp32 / d_scale)`（两遍读，
一次性）；热循环 `pie_gemv_row` 用 `lh + fcvt.s.w` 读 d，`d_scale` 提到累加外乘。
预检：per-tensor d 动态范围最大 **32.6 倍**（`L0.down`），远小于 int16 可表示 3.3e4 倍。

**板端实测**（Minueza-32M，整份常驻）：

| 项 | fp32 d | fp16 d（M3-8） | **int16 定点 d（本）** |
|---|---|---|---|
| 每 token | 230–234 ms | 242–246 ms | **226–231 ms** |
| total 19 steps | 4431 ms | 4663 ms | **4365 ms** |
| 序列 | 20/20 | 20/20 | **20/20** |

**结论（硬数据）**：**正收益 +1.5%**（+0.06 tok/s）。int16 定点精度（Q15 相对 ~3e-5）
高于 fp16（~1e-3），argmax 稳定。收益小于理论 5.5%，因 `fcvt.s.w` 是浮点指令、与 PIE 的
`fmadd.s` 竞争 FPU 发射槽，部分抵消——证明 decode 瓶颈是 **PSRAM 带宽 + FPU 发射混合**，
非纯带宽。

**决策**：**保留落地**（唯一兑现正收益的近似无损方向）。

### M3-10 权重量化方向（q2/q3）H4 授权后判否（2026-09-26，PC 端）

**背景**：H4 授权后，验证「极端量化减少权重读带宽」的可行性。两条线同时证伪：

**① 带宽线：q4→q2 收益 = 0（纠正此前的误判）**

此前记录的「q4→q2 权重读量减半、收益 ~1.9×」是**误判**——把「文件/Flash 存储减少」误当成
「decode 带宽减少」。真相（M3-4 已记录）：PIE dot 最小权重位宽是 **int8**
（`dl_esp32p4_dotprod_i16k8o16`，无 int4/int2 变体）。q2 与 q4 展开后**都是 int8（1 B/元素）**
进 PIE dot，decode 读的权重字节数完全相同 ⇒ **带宽收益 = 0**，q2 只省 Flash 存储。

**② 精度线：q2/q3 再量化直接崩模型（硬数据）**

`tools/legacy/verify_q2.py` 在 q4 反量化权重上再量化到 q3（7 值）/ q2（3 值），PC 参考前向：

```
q3: match 4/20  seq=[...,0,0,0,0,...]  top3 logit=0.0000（rms_norm overflow）
q2: match 4/20  seq=[...,0,0,0,0,...]  top3 logit=0.0000
```

match 4/20 中 4 个是 prompt 本身，后 16 个生成 token 全 0、logit 全零 ⇒ **模型完全失效**，
不是精度下降。

**结论**：权重量化方向被**双重封死**——
1. 带宽：PIE 无 int4/int2 dot，展开成 int8 后读量不变，收益 0；
2. 精度：三值/七值量化无法「再量化」现有模型，直接崩（印证 BitNet 是「从头训练的三值架构」，
   不是量化后处理）。

**决策**：**判否**。MCU 的 decode 带宽墙是 **PIE 硬件指令集（仅 int8 权重 dot）+ PSRAM
106 MB/s 的物理边界**，H4 授权也无法穿透。要突破只能换硬件（PIE 支持 int4 dot / 高带宽存储）
或换模型（MoE 稀疏 / BitNet 从头训练），均超出当前内核范畴。

### M3-11 输出头 ANN 稀疏查找（PLE 思想）判否（2026-09-26，PC 端）

**背景**：借鉴 esp32-ai 的 PLE「稀疏查表」思想，把 tied lm_head（32002 行嵌入表）从
「密集 argmax」改成「K-means 聚类 + 选簇 + 精确算候选」，验证召回率与剪枝率。

**关键诊断**（`tools/legacy/verify_ann_head.py`）：
- embed 模长 0.309–0.627（std 10%），h 模长稳定 ~46.6；
- **真 top-1 在余弦排序里始终在 top-16 内**（M=16 即 16/16 召回）；
- 但纯余弦 top-1 只匹配 8/16——另一半被「模长 ±10%」翻转，说明 top-1/top-2 logit margin 极小。

**硬数据（球形 K-means，归一化后欧氏=余弦）**：

| C | K | 召回 | 剪枝率 |
|---|---|---|---|
| 16 | 4 | 8/16 | 78% |
| 16 | 8 | 11/16 | 55% |
| 16 | 16 | 16/16 | **0%** |
| 32 | 32 | 16/16 | **0%** |

要 100% 召回必须选几乎全部簇，剪枝率趋近 0。

**结论**：输出头 ANN 剪枝**判否**。与 M3-3b「上界剪枝」用不同方法（聚类 vs 上界）得到同一
结论——输出头 43% 带宽是硬墙。根因：
1. margin 太小：top-1/top-2 logit 差距小到 ±10% 模长即可翻转，argmax 信息藏在方向与模长的微小差异里；
2. 高维维度灾难：312 维下近邻关系无法用聚类/上界捕获，要保证召回就须近乎全量计算。

**决策**：**判否**。至此「减少每 token 读字节」（行分块/降位宽/权重量化）与「减少元素数」
（输出头剪枝）两个维度均被 PIE int8 指令集 + 高维 argmax 信息论下界封死；唯一出路是换架构
（PLE/MoE 自训模型），超出内核范畴。

### M3-12 MoE 改造现有稠密模型判否（2026-09-26，PC 端）

**背景**：H4 授权下，把 Minueza 的稠密 FFN（ffn=1092）用 MoEfication 聚类拆成 N=8 专家 +
top-k 激活，验证「改造现有模型」能否拿到 MoE 稀疏激活的带宽收益。

**oracle 验证**（`tools/legacy/verify_moe.py`，oracle 选最优专家组合，给出精度上限）：

| top-k | 激活神经元 | oracle 相对误差 |
|---|---|---|
| 2 | 25% | **62.8%** |
| 4 | 50% | 42.0% |
| 6 | 75% | 25.4% |
| 8 | 100% | 0.0% |

两条事实：① top-8=稠密（误差 0，聚类不破坏信息）；② top-2（25% 神经元）误差 62.8%——
即使 oracle 每次选最优专家也丢 62.8% 表达能力。

**根因**：Minueza 的 FFN 无冗余（ffn=1092 / dim=312 = 3.5×，神经元紧密协作），稀疏激活
（top-2=25%）结构性损失 75% 表达能力，非训练可补。两条改造路均堵死：
1. MoEfication（聚类拆分）：专家是 1/N 神经元碎片，输出量级只有稠密 k/N；
2. Sparse Upcycling（完整专家副本）：参数 N 倍爆炸（10.2M→81.6M），远超 PSRAM 28M。

**决策**：**判否**。与 M3-10（q2 三值崩模型）同一规律：**稀疏/三值/专家分工等架构特性必须
从头训练，不能从稠密模型事后改造。** MoE 带宽收益只能走「从头训练完整小专家」路线
（esp32-ai/PFor 做法），转向蒸馏训练方案（见 docs/moe_framework.md）。

### 剩余瓶颈与后续优化方向（按预期收益排序）

1. **PIE 定点化 GEMV（M3-4 阶段 1）**：✅ 已落地，2.11×（见上）。
2. **标量 fp32 点积 4 路展开**：`dl_esp32p4_dotprod_f32` 写法，无损，可继续（但已被 PIE 覆盖大部分收益）。
3. ~~行分块（一次算 4 行，摊薄 x 载入）~~：已证伪——激活在内部 RAM（非 PSRAM），重读非瓶颈，收益 ≈ 0。
4. **权重 scale int16 定点**：✅ 已落地，+1.5%（见 M3-9）；fp16 版已证伪（M3-8）。
5. ~~权重量化 q2/q3~~：已判否（M3-10）——PIE 无 int4/int2 dot（带宽 0）+ 再量化崩模型。
6. ~~输出头剪枝（无损上界 + 有损 ANN）~~：已判否（M3-3b / M3-11）——margin 极小 + 高维维度灾难，
   剪枝率趋近 0。
7. **MoE 架构（从头训练 + 蒸馏）**：内核 router/top-k/稀疏 FFN 已就绪；模型走「完整小专家 +
   蒸馏 teacher」路线（M3-12 结论），见 docs/moe_framework.md。

> 当前 0.229 s/token（4.37 tok/s），较 M2 标量起点（1.63 s）累计 **7.1×**。
> 剩余最大瓶颈是 PSRAM 带宽（GEMV 权重+激活读）+ FPU 发射，非纯计算；PIE 纯计算 21×
> 已被带宽墙压到 2.11×。进一步收益需减少带宽占用或全量定点化（含 RMSNorm/attention/SwiGLU，
> 工作量 = 重写 int8 内核）。


## 边界约束（自述）

- 纯标量 C11；禁止引入 NEON/AVX/OpenMP/Linux 系统调用。
- 与 vLLM-Kestrel 位级一致**不成立**，不与其做性能等价比较。
- 任何有损近似（量化/剪枝）须另行立项授权，不在本里程碑范围。

## 许可证

Apache License 2.0 —— 见 **[../LICENSE](../LICENSE)** 与 **[../NOTICE](../NOTICE)**。
（许可证与第三方组件声明的**唯一权威出处是根目录 README**，此处不重复，以免漂移。）

