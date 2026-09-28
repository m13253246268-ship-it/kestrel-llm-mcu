# Kestrel-MCU

面向 **MCU 级 RISC-V 设备**的**纯标量 C11 微型 LLM 推理内核**，以及它背后完整的宿主
工具链：训练 → 量化 → 部署 → 对拍验证。

目标开发板：**Waveshare ESP32-P4-WIFI6-DEV-KIT（SKU 32054）**。

> English version: **[README.md](README.md)**

---

## 这是什么

在 **MCU 上跑小语言模型** —— 不用 NEON/AVX、不用 OpenMP、不用 Linux 系统调用，
只有标量 C11 加芯片自带的 PIE（SIMD）指令。

| | |
|---|---|
| **内核** | 标量 C11 transformer decode：GQA 注意力、RoPE、RMSNorm、SwiGLU；MoE 稀疏 FFN（`router` + top-k 专家）；PLE 逐层查表；tied 输出头；可选决策头 |
| **权重格式** | core 用 `q4_0`（18 字节 / 32 元素），PLE 表用**三值（1.6 bit）**，norm 类用 f32 |
| **存储路径** | 全量 PSRAM 常驻 / 从 Flash 流式 / 从 TF 卡（SD）流式 |
| **加速** | 经 ESP32-P4 PIE 点积的定点 GEMV（手写 `pie_dotprod.S`） |
| **工具链** | PyTorch 训练 → KMCU 量化镜像 → 板端推理 → NumPy + C 参考对拍 |
| **确定性** | 一条**独立的 determinism line**：与其他引擎不保证位级一致，且**刻意不复用**它们的任何内核 |

---

## 亮点（均为实测）

| 结果 | 数值 |
|---|---|
| PIE 定点 GEMV vs 标量点积 | 单算子 **21×** → 端到端 **2.11×** |
| Minueza-32M（0.032B）全量 PSRAM 常驻 | **0.229 s/token（4.37 tok/s）** —— 相对标量起点 **7.1×** |
| 权重 scale 改 int16 定点 | **+1.5%**（唯一兑现正收益的近似无损方向） |
| agent 循环路由器（17.4M，Flash 常驻） | **19.5 tok/s**，输出与 PC 参考**逐 token 一致** |
| agent 循环路由器（64.4M，q4 + 三值 PLE） | 留出集路由准确率 **0.9986**（自然分布 0.9981） |
| PSRAM 带宽（16 MB 顺序） | 写 ~**84 MB/s** / 读 ~**107 MB/s** → **瓶颈是带宽而非算力** |

完整基准表、已判否的方向与踩坑记录：**[docs/milestones.zh-CN.md](docs/milestones.zh-CN.md)**。

---

## 实际模型输出

以下全部是**训练完成的 64.4M agent 循环模型**（`agent_model.kmcu`，q4 core + 三值 PLE）
的真实产物：贪心解码、经 NumPy 参考实现（`ref_forward.py`）跑出。原始证据见
[docs/agent_loop_router.zh-CN.md](docs/agent_loop_router.zh-CN.md)。

**1）内容级路由** —— 读一段文本窗口，四选一：`route` / `generate` / `stop` / `ask_human`：

| 输入窗口 | 决策 |
|---|---|
| `The theory was developed over many decades.` | `stop` |
| `The sample contained 42 participants in total.` | `ask_human` |
| `What is the purpose of this experiment?` | `route` |
| `这个实验的目的是什么?` | `route` |
| `该实验共有42名参与者。` | `generate` |
| `x = 3 + 5 ; print(x)` | `generate` |
| `Hi` | `generate` |

**2）文本续写** —— 真实 FineWeb-Edu 的 8-token 前缀 → 32-token 贪心续写：

| 前缀（prompt） | 续写（continuation） |
|---|---|
| `a 21st century viewpoint` | `. The first two-dimensional view of the universe is the first to be seen in the 19th century. The first view of the universe is the` |
| `metabolic syndrome examines its effects` | `on the heart, heart, and kidneys. The results of the study are published in the journal Pediatrics. The results of the study were published in` |
| `2.5 percent of the population ident` | `ifies as a "flying-and-flying-and-flying-and-flying-and-flying-and-flying-` |

局部通顺、语义漂移 —— 第三行把失败模式也原样放出来。

**3）它做不到什么** —— 它是**路由器 + 弱续写器**，不是 instruct 模型：

```text
"Please summarize the following text:"  ->  "- 1. The following text is from the text of the text: -"
"Q: What is 2+2? A:"                    ->  "What is 2+2? A: What is 2+2?"
```

---

## 目标硬件

| 项 | 值 |
|---|---|
| SoC | ESP32-P4（RISC-V 32 位，双核 HP 最高 400 MHz + LP 核） |
| 协处理器 | ESP32-C6（Wi-Fi 6 / BT 5）—— 本项目**不使用** |
| 内存 | 768 KB L2 + **32 MB PSRAM**（hex 模式）+ **16 MB Flash** |
| 工具链 | ESP-IDF 6.x，target `esp32p4` |
| 调试 | USB Type-C（烧录 + 串口控制台） |

---

## 仓库结构

```
kestrel_mcu/
├── main/                 固件
│   ├── kmcu.c / kmcu.h   推理内核 + KMCU 格式解析
│   ├── pie_dotprod.S     PIE（SIMD）定点 GEMV 内核
│   └── main.c            入口：自检、探针、模型加载、运行模式
├── tools/                宿主工具链 —— 见 tools/README.zh-CN.md
│   ├── mistral_ple.py    架构唯一权威源
│   ├── convert_minueza.py  HF safetensors → KMCU
│   ├── ref_forward.py    NumPy fp32 参考前向
│   ├── train_moe_ple.py  训练
│   ├── watch_boot.py     复位 + boot 日志监视
│   ├── upload_uart.py    UART 上传（实验性）
│   └── legacy/           归档研究脚本（复现历史判否/基线结论）
├── docs/                 设计文档 + 开发日志（索引见下）
├── host_verify.c         PC 端对拍程序 —— 编译的是**同一份** kmcu.c
├── partitions.csv        Flash 布局（app 2 MB / model 13.87 MB）
├── sdkconfig.defaults    ESP-IDF 默认配置（PSRAM、newlib、RTTI、-O2、禁 WiFi/BT）
└── LICENSE / NOTICE      Apache-2.0 + 第三方组件声明
```

---

## 快速开始

### 1. 构建固件（Windows，实测通过）

有三个环境坑必须先处理——它们都是真实踩过的：

```powershell
# (a) 缩短工具链路径：真实路径过深会触发 Windows MAX_PATH(260)，
#     表现为 fatal error: bits/error_constants.h: No such file or directory
New-Item -ItemType Junction -Path C:\et -Target "$env:USERPROFILE\.espressif"
$env:IDF_TOOLS_PATH = "C:\et"

# (b) 跳过 git 子模块检查：components/esp_wifi/lib 被上游封锁（HTTP 423），
#     而本项目本就禁用 WiFi/BT
$env:IDF_SKIP_CHECK_SUBMODULES = "1"

. E:\esp-idf\export.ps1                 # 激活 ESP-IDF

# (c) 在纯 ASCII 目录下构建，并关闭 ccache：
#     ccache 在非 ASCII 路径下会抛 "std::filesystem ... Illegal byte sequence"
#     把 main/、CMakeLists.txt、partitions.csv、sdkconfig.defaults 拷到该目录
Set-Location C:\kmcu
idf.py --no-ccache build
```

### 2. 烧录

```powershell
idf.py -p COM5 flash
```

### 3. 模型流水线

```bash
# 训练（需要预先 tokenize 的 uint16 token 流）
python tools/train_moe_ple.py --out_dir <out_dir> --max_steps 30000 \
       --train_bin <train.bin> --val_bin <val.bin>

# 量化 → KMCU 镜像
python tools/convert_minueza.py --src <out_dir> --out model.kmcu

# PC 参考（NumPy fp32）—— 板端对拍的基准
python tools/ref_forward.py --km model.kmcu
```

### 4. 部署

```powershell
# 写入 `model` 分区（偏移/大小见 partitions.csv）
python -m esptool --chip esp32p4 -p COM5 -b 460800 write-flash 0x210000 model.kmcu
```

> **尺寸上限**：`model` 分区为 **13.87 MB**。更大的镜像（例如 64.4M PLE 模型的
> 20.77 MB）必须改从 **TF 卡**读取——固件为此有专门的 SD 常驻路径。

### 5. 与参考对拍

```bash
gcc -O2 -std=c11 -I main -o host_verify host_verify.c main/kmcu.c -lm
./host_verify model.kmcu
```

设备与参考使用**同一份量化权重**，故预期是：

* 贪心 token 序列 —— **逐 token 一致**；
* logits —— 差异在 ~**1e-4** 内（浮点累加顺序不同），argmax 一致。

两侧跑的是**同一份源码**（`main/kmcu.c`），这正是该对拍有意义的前提。运行模式
（纯生成 / 双头 / agent 循环）通过串口 `0` / `1` / `2` / `q` 动态切换。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/milestones.zh-CN.md](docs/milestones.zh-CN.md) | **开发日志**：M1 能力探针、M2 端到端推理、M3-x 性能优化 —— 全部硬数据表、已判否方向、踩坑记录 |
| [docs/moe_framework.zh-CN.md](docs/moe_framework.zh-CN.md) | MoE 框架设计：稠密 → 稀疏激活 |
| [docs/lambda_transition_framework.zh-CN.md](docs/lambda_transition_framework.zh-CN.md) | λ 过渡训练框架（经典范式 + 公理库底座） |
| [docs/layerwise_adaptation.zh-CN.md](docs/layerwise_adaptation.zh-CN.md) | 逐层适配优化（后续方向） |
| [docs/agent_loop_router.zh-CN.md](docs/agent_loop_router.zh-CN.md) | 内容级 agent 循环路由器：任务定义、数据、训练、能力探测、部署 |
| [tools/README.md](tools/README.md) / [tools/README.zh-CN.md](tools/README.zh-CN.md) | 工具链使用说明（英 / 中） |

---

## 设计约束（自述）

* **纯标量 C11** —— 不引入 NEON/AVX/OpenMP/Linux 系统调用；唯一的架构相关代码是 PIE
  点积内核，且始终保留标量回退路径。
* **确定性是「逐引擎」的。** 与其他引擎（例如面向 aarch64/x86-64 的大模型引擎）**不**
  声明位级一致，也不做跨引擎的性能等价比较。
* **不做静默的有损捷径。** 任何有损近似（额外量化、剪枝）都需另行授权并通过质量门后才落地。
* **实测胜于推论。** 文档里的优化结论都有板端前后对照数据支撑，**被证伪的方向同样记录在案**。

---

## 许可证

**Apache License 2.0** —— 全文见 [LICENSE](LICENSE)。

* 允许商用、修改、再分发；须保留版权与许可声明，**并声明你所做的重大修改**（§4b）。
* 含**明确的专利授权**与防御性终止条款。
* 第三方组件及其许可清单见 [NOTICE](NOTICE)；这些组件**不随本仓库分发**。
* **本仓库不分发任何模型权重、tokenizer 文件或 `*.kmcu` 镜像。** 这类产物可能有各自的
  上游许可（例如微调所用的基座模型、训练所用的 tokenizer），该许可**独立于本项目的
  LICENSE**，由使用者自行遵守。
