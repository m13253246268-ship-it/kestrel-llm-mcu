# Kestrel-MCU — 工具集（tools）

面向 ESP32-P4 的宿主机侧工具链：训练、量化、以及对拍验证。全部为 Python 3
（另有一个 C 参考程序 `../host_verify.c`）。

> English version: **[README.md](README.md)**

---

## 流水线总览

```
                         tools/train_moe_ple.py
                                   │
                                   ▼
                     HF 模型目录  (config.json + model.safetensors)
                                   │
                         tools/convert_minueza.py
                                   │
                                   ▼
                             model.kmcu  ──── Flash / SD 卡 ───►  ESP32-P4
                                   │                              (main/kmcu.c)
                                   │
                      tools/ref_forward.py  (NumPy fp32 参考前向)
                                   ▲
                                   │   逐 token 对拍
                             host_verify.c  (同一份 kmcu.c 源码，PC 编译)
```

`mistral_ple.py` 是**唯一权威架构源**：训练、`convert_minueza`、`ref_forward`、
`main/kmcu.c` 全部以它定义的张量命名与数值口径为准。

---

## 环境依赖

| 包 | 被谁使用 | 说明 |
|---|---|---|
| `numpy` | convert、ref_forward | 必需 |
| `torch` | 训练、mistral_ple | 仅训练需要 |
| `transformers` | 训练（tokenizer） | 仅训练需要 |
| `safetensors` | convert（读）、训练（存/读） | 必需 |
| `pyserial` | watch_boot、upload_uart | 板端工具需要 |
| `scipy` | ref_forward（`erf`） | **可选**，缺失时自动回退 `math.erf` |

```bash
pip install numpy torch transformers safetensors pyserial
# 可选: pip install scipy
```

板端构建另需 **ESP-IDF 5.3+**，target = `esp32p4`。Windows 下的构建坑
（非 ASCII 路径、`MAX_PATH`、ccache、子模块）见项目 [README](../README.md)。

---

## 工具索引

| 工具 | 作用 | 典型用法 |
|---|---|---|
| [`mistral_ple.py`](#1-mistral_plepy--架构源) | 架构定义（库） | 被训练/转换脚本 import |
| [`convert_minueza.py`](#2-convert_minuezapy--hf--kmcu) | HF safetensors → KMCU 量化镜像 | 部署 |
| [`ref_forward.py`](#3-ref_forwardpy--numpy-参考前向) | NumPy fp32 参考前向 | 正确性对拍 |
| [`train_moe_ple.py`](#4-train_moe_plepy--训练) | 训练（Mistral + MoE + PLE） | 训练 |
| [`watch_boot.py`](#5-watch_bootpy--boot-监视) | 复位板子 + 持续打印 boot 日志 | 板端诊断 |
| [`upload_uart.py`](#6-upload_uartpy--uart-上传实验性) | 经 UART 把 `model.kmcu` 推到 TF 卡 | ⚠️ 实验性 |
| [`legacy/`](#legacy--归档的研究脚本) | 归档研究脚本 | 复现已得出的判否/基线结论 |

---

## 1. `mistral_ple.py` — 架构源

纯库，无 CLI。定义 `MistralPLEConfig` 与 `MistralPLE`。

### 架构

Mistral 主干（GQA + RMSNorm + RoPE + SwiGLU），加三项扩展：

* **MoE FFN** —— 每层 `router` + `n_expert` 个专家（`gate`/`up`/`down`），
  `softmax → top_k → 重新归一化 → 加权求和`；
* **PLE（Per-Layer Embeddings）** —— 逐层查表，训练时**三值化**
  （`{-1,0,+1} × absmean`，STE 直通）；
* **heads** —— `tied`（lm_head 复用 `embed_tokens`）与可选 `cls_head`
  （`n_cls > 0` 时挂，供 agent 循环做分类）。

PLE 数值口径（逐式与参考实现一致）：

```
x0    = embed_tokens(idx)
ple   = ple_model_proj(x0) * dim^-0.5                  # [seq, L*ple_dim]
ple   = RMSNorm(ple.view(seq, L, ple_dim))
table = ple_table(idx)                                 # [seq, L*ple_dim]  三值
ple   = (ple + table * ple_dim^0.5) * 2^-0.5
# 每层 i，在 attention + FFN 之后：
g = gelu(ple_gate_i(x))
x = x + RMSNorm(ple_proj_i(g * ple[:, :, i]))          # ple_proj: ple_dim -> dim
```

### `MistralPLEConfig` 字段

| 字段 | 默认 | 含义 |
|---|---|---|
| `vocab_size` | 32002 | tokenizer 词表（Mistral） |
| `dim` | 312 | 模型宽度 |
| `n_layers` | 10 | 层数 |
| `n_heads` / `n_kv_heads` | 12 / 4 | GQA；`head_dim = dim // n_heads` |
| `ffn_e` | 128 | **专家** hidden（对应 HF 的 `intermediate_size`） |
| `n_expert` / `top_k` | 8 / 2 | MoE 几何（`n_expert=0` → 稠密） |
| `ple_dim` | 128 | 每层查表宽度（`table_width = n_layers * ple_dim`） |
| `seq_len` | 512 | RoPE 表长度 |
| `rope_theta`、`norm_eps` | 10000.0、1e-6 | |
| `tied` | True | lm_head 复用 `embed_tokens` |
| `has_ple` | True | 启用 PLE 分支 |
| `n_cls` | 0 | 决策头类别数（`0` = 无决策头，不破坏生成路径） |
| `ternary_table` | True | 对 `ple_table` 做三值 STE |

### 关键方法

| 方法 | 返回 | 说明 |
|---|---|---|
| `forward(idx, targets=None)` | `(logits, loss)` | 语言模型路径；`loss` 为 CE，`ignore_index=-1` |
| `forward_features(idx)` | `[B, T, dim]` | last hidden（决策头分支） |
| `cls_forward(idx)` | `[B, n_cls]` | `mean pool → cls_head` |
| `param_budget()` | `{core, stream, table, total}` | 按访问模式做参数记账 |
| `export_state_dict()` | `dict` | HF 风格键名，与 `convert_minueza` 读取口径对齐 |
| `save_pretrained(dir)` | — | 写 `model.safetensors` + `config.json` |

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

把 HuggingFace 模型目录转成 **KMCU v1** 量化镜像，供 `main/kmcu.c` 在设备侧
mmap / 流式读取。

```bash
python convert_minueza.py --src <hf_dir> --out <file.kmcu> [--top_k 2]
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `--src` | `model_dir`（环境变量 `KMCU_SRC`） | 源目录（含 `config.json` + `model.safetensors`） |
| `--out` | `model.kmcu`（环境变量 `KMCU_OUT`） | 输出镜像 |
| `--top_k` | 2 | MoE top-k（稠密忽略；config 里有 `top_k` 时以 config 为准） |
| `--force-tie` | 关 | 强制 `lm_head` 复用 `tok_embed`（丢弃已训练的输出头）。**有损**，用于复现历史 M1–M3 镜像（Minueza：22.81M / 12.58 MB，装得进 Flash `model` 分区） |

> 默认值为当前目录下的相对路径；可用 `--src`/`--out` 或环境变量
> `KMCU_SRC`/`KMCU_OUT` 覆盖。
>
> `tied_embed` 默认取自 `config.json` 的 `tie_word_embeddings`，只有传 `--force-tie` 才强制共享。
> 强制共享的代价**不只是省下的 9.98M 参数**：在 Minueza 上两种镜像的贪心输出**完全不同**——
> 强制共享是词沙拉，尊重配置（默认）是通顺英文。详见文章系列 `02-模型量化与KMCU格式`。

转换同时会写 `<src>/model.kmcu.manifest.txt`（可读的 header + 逐张量表），便于人工核对。

### 自动检测规则

| 条件 | 效果 |
|---|---|
| 存在 `model.layers.0.mlp.router.weight` | **MoE**：存 `router`(f32) + 每专家 `gate`/`up`/`down`(q4)；header 的 `ffn_dim` 记为专家 hidden |
| config `ple_dim > 0` | **PLE**：导出 `ple_model_proj`、`ple_proj_norm`、逐层 `ple_gate`/`ple_proj`/`ple_norm`，以及 `ple_table`（三值） |
| 存在 `cls_head.weight` | 导出 `cls_head`(f32)，header `n_cls` |
| `tie_word_embeddings` | `tied=1` → 不存 `lm_head`；否则存在时以 q4 存 `lm_head` |

### 文件布局（小端）

```
[0,128)                                  Header
[128, 128 + n_tensors*48)                目录（每项 48B）
[data_offset, ...)                       数据区，每张量 64B 对齐
```

**Header**

| 偏移 | 类型 | 字段 |
|---|---|---|
| 0 | `char[4]` | magic `"KMCU"` |
| 4 | u32 | version = 1 |
| 8 / 12 / 16 / 20 / 24 | u32 | `n_layers` / `dim` / `n_heads` / `n_kv_heads` / `head_dim` |
| 28 / 32 / 36 | u32 | `ffn_dim`（MoE 时为专家 hidden）/ `vocab_size` / `max_seq` |
| 40 / 44 | u32 | `bos_id` / `eos_id` |
| 48 | u32 | `tied_embed`（1 = 复用 `embed_tokens`） |
| 52 / 56 / 60 | u32 | `n_tensors` / `dir_offset` / `data_offset` |
| 64 / 68 | f32 | `rope_theta` / `norm_eps` |
| 72 | u32 | `total_params` |
| 76 | u32 | `n_expert`（0 = 稠密） |
| 80 | u32 | `top_k` |
| 84 | u32 | `ple_dim`（0 = 无 PLE） |
| 88 | f32 | `ple_gamma`（三值表 scale） |
| 92 | u32 | `n_cls`（0 = 无决策头） |

**目录项（48B）**

| 偏移 | 类型 | 字段 |
|---|---|---|
| 0 | `char[24]` | 名称（ASCII，NUL 填充） |
| 24 | u8 | dtype；25..27 填充 |
| 28 / 32 | u32 | `n_rows` / `n_cols`（元素数；**原始**列数） |
| 36 | u32 | 绝对文件偏移 |
| 40 | u32 | 存储字节数 |
| 44 | u32 | 保留 |

**dtype**

| 值 | 名称 | 编码 |
|---|---|---|
| 1 | `F32` | 原始小端 f32（norm、router、`cls_head`） |
| 2 | `Q4_0` | 18B / 32 元素：`[0,2)` fp16 scale `d`，`[2,18)` 16B 低/高半字节。`d = max_abs/8`，`q = round(w/d)` 截断到 `[-8,7]`，`w = (nibble-8)*d`。**每行零填充到 32 的倍数，使块不跨行。** |
| 3 | `TERNARY` | 5 trit/字节，base-3（`byte = Σ trit_i·3^i`），行对齐到 `ceil(cols/5)` 字节。`code = clip(round(w/gamma),-1,1)`，`trit = code+1 ∈ {0,1,2}`，`gamma = mean(|w|)`。 |

`n_cols` 始终记录**原始**列数；设备侧自行推导 `nblk = align32(n_cols)`。

```bash
# 稠密 Minueza
python convert_minueza.py --src E:/models/minueza --out E:/models/model.kmcu
# MoE + PLE + 决策头
python convert_minueza.py --src E:/models/agent_model --out E:/models/agent_model.kmcu
```

典型输出：

```
[moe] n_expert=8 top_k=2 ffn_e=128
[cls] n_cls=4
n_tensors=346  params=64.36M  q4=23.36M
file size = 20.77 MB   (tied_embed=1, moe=True n_expert=8 top_k=2 ple_dim=128 ple_gamma=0.028555)
written: agent_model.kmcu
```

---

## 3. `ref_forward.py` — NumPy 参考前向

读取 KMCU 镜像，用 NumPy 做 **fp32 参考前向**，产出贪心 token 序列——设备侧对拍的基准。

参考实现反量化的是**与设备同一份量化权重**，因此二者的差异只应来自浮点累加顺序/舍入，
不应来自权重本身。

```bash
python ref_forward.py --km <file.kmcu> [--ref_out ref_out.txt]
```

打印内容：header 字典、张量数、prompt、单 token top-8（二分定位锚点：`pos=0` ⇒
RoPE 恒等、注意力退化）、首个生成位的 top-8、16 token 贪心序列，以及
（`n_cls > 0` 时）决策头 logits。

### 作为库调用

```python
import sys; sys.path.insert(0, "tools")
import ref_forward as rf

arch, T = rf.load_kmcu("E:/models/agent_model.kmcu")
out  = rf.generate(arch, T, [1, 450, 2217, 4996], 16)          # 贪心
out2 = rf.generate(arch, T, [1, 450, 2217, 4996], 16, rep_penalty=1.2)
cl   = rf.cls_forward(arch, T, [1, 450, 2217, 4996, 522, 28723, 415, 907])
```

| 函数 | 签名 | 说明 |
|---|---|---|
| `load_kmcu(path)` | `→ (arch, tensors)` | 解析 header + 目录 |
| `forward(arch, T, tokens)` | `→ logits [T, vocab]` | 整序列前向 |
| `generate(arch, T, ids, n, rep_penalty=1.0)` | `→ list[int]` | 贪心；`rep_penalty` 与 `kmcu.c` 语义一致（正 logit 相除、负 logit 相乘，`pos==0` 清零） |
| `cls_forward(arch, T, tokens)` | `→ [n_cls]` | transformer → `final_norm` → mean pool → `cls_head` |

PC 侧的对拍程序是 `../host_verify.c`，它编译的是**设备上同一份** `main/kmcu.c`：

```bash
gcc -O2 -std=c11 -I main -o host_verify host_verify.c main/kmcu.c -lm
./host_verify <file.kmcu>
```

---

## 4. `train_moe_ple.py` — 训练

在预 tokenize 的扁平 token 流上训练 `MistralPLE`。范式：
**全量训练 + 纯 CE + AdamW + cosine**，无冻结、无 KD、无 dense 兜底。

```bash
python train_moe_ple.py --out_dir moe_ple_fw --max_steps 30000
python train_moe_ple.py --smoke          # 1 步 fwd+bwd，验证配置可跑
```

数据：扁平的 `uint16` token 数组（Mistral vocab 32002），例如 `fw_train.bin` / `fw_val.bin`。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--out_dir` | `moe_ple_fw`（环境变量 `KMCU_OUT_DIR`） | checkpoint 输出目录 |
| `--max_steps` | 30000 | 优化步数 |
| `--batch_size` / `--seq_len` | 32 / 512 | |
| `--lr` / `--warmup` | 3e-4 / 500 | cosine 衰减到 `0.1 × 峰值` |
| `--eval_every` / `--eval_iters` | 250 / 40 | 验证频率 |
| `--ckpt_every` | 5000 | 另存 `ckpt_stepN/`；0 关闭 |
| `--dim` / `--n_layers` / `--n_heads` / `--n_kv_heads` | 312 / 10 / 12 / 4 | |
| `--n_expert` / `--top_k` / `--ffn_e` / `--ple_dim` | 8 / 2 / 128 / 128 | |
| `--no_ternary` | 关 | `ple_table` 保持 fp32（不做三值 STE） |
| `--seed` | 0 | |
| `--train_bin` / `--val_bin` | `fw_train.bin` / `fw_val.bin`（环境变量 `KMCU_TRAIN_BIN`/`KMCU_VAL_BIN`） | **请覆盖** |

优化器细节：AdamW `betas=(0.9, 0.95)`；`ndim ≥ 2` 的权重（排除 `table`/`embed_tokens`）
`weight_decay=0.1`，其余 `0.0`；梯度范数裁剪 1.0。

每步日志：

```
step   2500/30000 train 2.9134 val 2.7411 ppl   15.51 lr 2.81e-04 (412s)
```

> 注意：本脚本只打印 `val` loss 与 `ppl`，**不**计算任务指标（决策头准确率等）
> ——那些在任务专用脚本里。

---

## 5. `watch_boot.py` — boot 监视

复位板子并持续打印其输出。当固件在打开控制台之前就崩溃时特别有用。

```bash
python watch_boot.py 30            # COM5 上抓 30 秒
python watch_boot.py 60 COM7       # COM7 上抓 60 秒
KMCU_PORT=COM7 python watch_boot.py 60
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `argv[1]` | 30 | 抓取时长（秒） |
| `argv[2]` / `KMCU_PORT` | `COM5` | 串口 |

复位方式：`RTS → EN`（低有效复位），`DTR` 保持 GPIO0 高电平（正常运行）。
这是 **USB-UART（DTR/RTS）** 复位；若板子只接了原生 USB-Serial-JTAG，
可能无法复位芯片——那种情况下请用 `idf.py monitor`。

---

## 6. `upload_uart.py` — UART 上传（实验性）

经串口把 `model.kmcu` 写入板载 TF 卡，无需拔出读卡器。

```bash
python upload_uart.py                 # COM5 + 默认文件
python upload_uart.py COM6 <file.kmcu>
```

协议：`"KMUP" + u32(长度)` 头，随后按 8192 字节分块，每块等 1 字节 `ACK(0x06)` 作流控。
固件侧为 `main/main.c` 的 `uart_upload_model()`（console `UART_NUM_0`，
3 秒握手窗口后切 921600）。

> ⚠️ **已知问题——本工具当前与固件不匹配。**
> 脚本等待 `READY(0xAA)` 与 `fopen` 状态字节（`0x55`/`0xEE`/`0xE3`）且全程 115200；
> 而固件打印的是「3 秒后切 921600」，且既不发 `0xAA` 也不发状态字节。
> 此外，固件只在 `/sdcard/model.kmcu` **不存在**时才进入上传模式。
>
> **在协议重新对齐之前，请改用 PC 读卡器写卡。**
> 脚本内的默认文件路径也指向另一个无关项目——请始终显式传入文件。

---

## `legacy/` — 归档的研究脚本

这些是**早期研究分支**，保留的目的是让 `docs/` 与 `README.md` 中记录的
判否/基线结论仍可复现。它们不属于当前流水线。

| 脚本 | 用途 | 已记录的结论 |
|---|---|---|
| `train_moe.py` | 自研 MoE 训练（`LambdaMoE`、λ 退火路由、KD 或纯 CE） | ✅ 产出 `moe_3ep`（argmax match 0.744） |
| `train_moe_kd.py` | 用 `MixtralForCausalLM` 从稠密 teacher 蒸馏 MoE | ❌ 失败（aux 恒 2.0 = router 均匀态） |
| `train_lambda.py` | λ_s 结构稀疏化（稠密 → MoE 结构蒸馏） | ❌ 判否（dense 兜底稀释 CE 梯度） |
| `full_finetune.py` | 经典全量微调基线（teacher 初始化、全参数、纯 CE） | ✅ 基线（ppl 8.73） |
| `train_moe_qat.py` | QAT vs PTQ（`λ_q`），fake-quant 与 `convert_minueza` 的 q4_0 位级一致 | ❌ 判否（QAT +1.4% vs PTQ +0.9%） |
| `dense_baseline.py` | 稠密 FFN 对照（除 FFN 外用 teacher 权重，FFN 从零训） | 对照 |
| `eval_quality.py` | MoE student vs 稠密 teacher 的 ppl + 生成质量评估 | 工具 |
| `verify_ann_head.py` | 输出头 ANN（K-means 聚类 + 选簇）剪枝可行性 | ❌ 判否 |
| `verify_moe.py` | oracle router——稠密 → MoE 改造的精度上限 | ❌ 判否 |
| `verify_q2.py` | 有损 q3/q2 再量化的精度边界 | ❌ 判否 |

跨目录说明：`verify_q2.py`、`verify_moe.py`、`verify_ann_head.py` 需要从父目录
`tools/` import `ref_forward`。它们已在 `sys.path` 中同时加入自身目录与父目录，
因此在任何工作目录下都能运行；其余脚本只依赖 `legacy/` 内的模块。

---

## 移植注意 / 已知限制

1. **`legacy/` 下的历史脚本仍带作者本地 Windows 默认路径**（如 `E:\models\...`），
   使用这些脚本时请显式传路径参数。主线工具已改为相对路径，并可用 `KMCU_*`
   环境变量重定向（`KMCU_SRC`、`KMCU_OUT`、`KMCU_KM`、`KMCU_REF_OUT`、
   `KMCU_OUT_DIR`、`KMCU_TRAIN_BIN`、`KMCU_VAL_BIN`、`KMCU_FILE`、`KMCU_PORT`）。
2. **词表固定 32002**（Mistral tokenizer）——仅限训练脚本；转换脚本本身从
   `config.json` 读 `vocab_size`。
3. **KMCU 为小端**，其布局与 `main/kmcu.c` 协同设计。任何格式改动都必须同步到
   `kmcu.h`/`kmcu.c` 与 `ref_forward.py`。
4. **目录里的 `n_cols` 是原始（未填充）列数**——设备侧自行重算
   `nblk = align32(n_cols)`。不要在转换时「顺手」把它填成对齐值。
5. **可复现性。** `ref_forward` 与 `kmcu.c` 预期**逐 token 一致，但不是逐位一致**：
   设备侧浮点累加顺序不同且使用 q4 反量化权重，因此出现 ~1e-4 量级的 logit 差异属正常。

---

## 许可证

Apache License 2.0 —— 见 [`../LICENSE`](../LICENSE) 与 [`../NOTICE`](../NOTICE)。
第三方组件（ESP-IDF、PyTorch、transformers、safetensors、NumPy、SciPy、pyserial）
**不**随本目录分发，各自保留其原有许可。本仓库不分发任何模型权重、tokenizer 文件或
`*.kmcu` 镜像。
