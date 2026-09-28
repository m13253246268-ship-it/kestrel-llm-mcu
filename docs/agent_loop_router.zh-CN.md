# 内容级 Agent 循环模型（Content-Level Agent Router）—— 正式训练与能力画像

> English version: **[agent_loop_router.md](agent_loop_router.md)**

> 定位：本文档记录 kestrel_mcu 第一个**「可用级」agent 循环模型**的完整链路：任务定义、
> 数据构建、多任务训练、留出集评估、能力探测（含因果消融）、PC 与板端部署验证、已知局限。
>
> 「可用级」的判定标准（与用户对齐）：① 真正的留出集评估（文档级划分 + 逐类 P/R/F1 + 混淆矩阵）；
> ② 保留生成能力（多任务 LM 锚定 + ppl 对照）；③ 多步 agent 循环闭环；④ 板端部署验证。
>
> 一句话结论：**可靠的 4 类内容路由器**（留出 0.9986 / 自然分布 0.9981），核心能力是
> **窗内数字检测（因果、位置无关）** 与 **问句语义识别**；误差被压缩到「满窗 64-token 的标签歧义」
> 这一**不可约边界**上；不具备指令跟随与推理能力。

---

## 1. 问题与动机

前序工作已验证「决策头 vs 生成头」：MCU 上做**决策任务**时应走「单次前向 + 小决策头」，
而非「lm_head 全扫 + 逐 token 生成」——读量差 4 个量级（cls_head 0.31KB vs lm_head 4.99MB）。

但首版动作任务存在**致命的方法论缺陷**：标签由**末字节标点**决定（`?`→route / `!`→ask_human /
`.`→stop / 越窗→generate）。硬数据证伪：**冻结预训练 core + 只训 1248 参数线性头 → 500 步
train acc 0.969**，说明该任务表层可分、一个线性探针即可解决，**不构成「可用」**。

据此升级为**内容级任务**：标签必须依赖**整窗内容**，规则的第 3/4 条要求扫描全窗 token，
**不能只读最后一个字节**。

---

## 2. 任务定义

动作空间与 `kmcu.h` 的 `KM_ACT_*` 对齐：

| id | 动作 | 语义 | 触发（按优先级） |
|---|---|---|---|
| 0 | `route` | 需求解/检索 → 路由到 QA | 窗口内含 `?` |
| 1 | `generate` | 文本未完 → 继续生成 | 窗口被截断 / 无句末标点 |
| 2 | `stop` | 完整陈述 → 无需动作 | 其余情况 |
| 3 | `ask_human` | 量化内容 → 升级人工/工具核验 | 窗口内含数字 |

**标签优先级规则**（逐条按序命中，`SL=64`）：

1. token 数 > 64 → `generate`（窗口截断，文本未完）
2. 末尾不是 `.` `!` `?` → `generate`（无句末标点的残片，视为未完）
3. 含 `?` → `route`
4. 含数字 → `ask_human`
5. 否则 → `stop`

规则 1/2 依赖长度与句末标点；规则 3/4 要求**扫全窗**。这是与旧版（末字节捷径）的本质区别。

---

## 3. 数据构建（`_prep_agent_data.py`）

来源 FineWeb-Edu 英文文本（`fineweb_edu_train.txt`）。

| 集合 | 规模 | 采样方式 |
|---|---|---|
| train | 800,000（各类 20 万） | 均衡 |
| val | 40,000（各类 1 万） | 均衡 |
| test | 40,000（各类 1 万） | 均衡 |
| **nat（自然分布）** | 40,000 | **不采样**，反映真实先验 |

- **文档级划分**：按 `doc_index % 10` 分桶（0–7 train / 8 val / 9 test），避免同文档泄漏。
- **自然分布先验**：`stop` 71.1% / `ask_human` 22.0% / `route` 3.3% / `generate` 3.6%
  → **多数类基线 0.710**，是自然 acc 的及格参照。
- padding：`pad_id = 2`（`</s>`）。窗口固定 64 token，不足补 pad。

### 踩过的坑

1. **只编码前 65 token**：标签只依赖「是否越窗」与句末标点，无需整段 tokenize。
   初版对整段（可达 8000 token）全量编码，预处理慢到不可用；改为 `truncation=True,
   max_length=SL+1` 后降到分钟级。
2. **分桶 bug**：初版按「整批 flush 时的 doc 号」分桶，而一批含约 200 篇文档 →
   **约 90% 的句子被分到错误的桶**（val/test 被 train 污染）。改为按**句子所属 doc**
   精确分桶后才真正隔离。

---

## 4. 模型

架构权威源 `kestrel_mcu/tools/mistral_ple.py`（`MistralPLE`）。

| 组件 | 配置 |
|---|---|
| 主干 | Mistral：GQA 注意力（n_heads 12 / n_kv_heads 4 / head_dim 26）、RMSNorm、RoPE(θ=1e4)、SwiGLU |
| MoE | 每层 router + 8 专家，top-2，专家 hidden `ffn_e=128` |
| PLE | 逐层嵌入查表：`ple_model_proj` + `ple_table`（vocab × n_layers×ple_dim），**三值量化 STE**（1.6 bit/元素） |
| head | `tied`（lm_head 复用 embed_tokens）+ `cls_head`（dim→4，mean pooling） |
| 规模 | vocab 32002 / dim 312 / 10 层；**64.36M** 参数 |

决策头前向：`transformer → final_norm → mean pooling → cls_head → [4]`（`forward_features` / `cls_forward`）。
初始化自 `moe_ple_fw`（FineWeb-Edu 预训练，非随机）。

---

## 5. 训练（`_train_agent_model.py`）

**多任务目标**（关键设计）：`loss = CE(cls) + λ·CE(lm)`，λ=1.0。
LM 数据取 `fw_train_big` 的**随机 64-token 窗口**（`bs_lm=16`）。作用是**锚定生成能力**——
agent 的 `generate` 动作要靠同一个 core 续写，端到端只训 cls 会让生成退化。

| 超参 | 值 |
|---|---|
| 双参数组 lr | core `5e-5` / cls_head `1e-3` |
| 调度 | cosine + warmup 200 |
| 优化器 | AdamW，wd 0.01，grad clip 1.0 |
| bs | cls 32 / lm 16 |
| 预算 | 目标 38000 步，**5.5h 墙钟上限** → 实跑 **33086 步 / 330 min** |

**安全阀**：每 5k 步存 checkpoint 并评 val+ppl；**ppl 漂移 > 10% 自动止损**（本次未触发）。

---

## 6. 结果（全部为留出集，非训练批）

| 指标 | 值 |
|---|---|
| test 均衡 acc / macroF1 | **0.9986 / 0.9986** |
| test 自然分布 acc | **0.9981**（多数类基线 0.710） |
| test 自然分布 macroF1 | 0.9930 |
| 逐类 F1（均衡 test） | route 0.9995 / generate 0.9973 / stop **0.9996** / ask_human 0.9978 |
| **ppl 对照** | 48.01 → **44.86（−6.6%）** |

> ppl 口径说明：在 `fw_val_big` 上取 512 个 64-token 窗口（约 3.2 万 token）teacher-forced 计算，
> 基线为**未微调的 moe_ple_fw**、同一份代码同一份数据。此绝对值与其他历史 ppl 数字**不可直接比较**，
> 只有本次运行内的 A/B 有效。

### 6.1 ppl 漂移**不是单调恶化**

| step | 5k | 10k | 15k | 20k | 30k |
|---|---|---|---|---|---|
| ppl 漂移 | +5.7% | +3.0% | +0.4% | −2.4% | **−6.0%** |

`CE(lm)` 锚定不仅防止退化，还让生成质量在训练中**逐步反超基线**。这也证伪了训练早期的担忧。

### 6.2 生成多样性 A/B（`_verify_gen_ab.py`）

分布内真实文本（FineWeb-Edu 验证集前缀），3 prompt × 32 token 贪心：

| 模型 | distinct-1 | distinct-2 | 最长重复连段 |
|---|---|---|---|
| base moe_ple_fw | 0.365 | 0.568 | 2 |
| **final agent_model** | **0.438** | **0.589** | **1** |

### 6.3 一次误判的自我更正

训练结束时的报告曾显示：ppl 改善 −6.6%，但**贪心样例重复变重**（`28774` 连出），
据此我一度判断「生成退化」。用分布内 prompt 做 A/B 后**该结论被证伪**：
多样性反而更好。误判来源是**单个乱码 OOD prompt**（`[1,450,2217,4996]`，token 落在
byte-fallback 高 id 区），其输出对两个模型都是半噪声，不构成证据。

---

## 7. 能力探测（`_probe_agent_caps.py`）

仅有「总 acc」不足以说明能力边界。以下为分项探测结果。

### 7.1 误差的结构性归因

混淆矩阵中错误**几乎全是「X → generate」**（test: route→gen 9 / stop→gen 6 / ask_human→gen 40），
而 `generate` 召回率 **1.000**、精度仅 0.9945 → **它是「吸引子类」**。

按窗口**真实长度**（非 pad token 数）分桶：

| 真实长度 | n | acc | 该桶真为 generate 占比 |
|---|---|---|---|
| 0–31 | 24260 | 0.9999 | 5.2% |
| 32–47 | 5316 | 0.9998 | 3.7% |
| 48–55 | 1171 | **1.0000** | 3.9% |
| 56–59 | 409 | **1.0000** | 4.2% |
| **60–63** | 321 | **0.9751** | 7.8% |
| **64（满窗）** | 8523 | **0.9945** | 99.3% |

全测试集约 56 个错，**约 84% 落在 64-token 满窗桶**，且 60–63 桶错误率是其余桶的约 8 倍。

**根因是标签定义本身的歧义**：一个**恰好 64 token 的完整句**与「被窗口截断」在 64 窗内
**在信息上不可区分**。→ 属**不可约误差**，不是模型弱。要消除必须改任务定义
（加显式 EOS / 续写标记，或延长窗口），而非继续训练。

### 7.2 因果消融 / 反事实编辑（最有价值的发现）

| 编号 | 操作 | 结果 | 含义 |
|---|---|---|---|
| B1 | ask_human **抹掉数字字符 token** | **98.6% 翻成 stop** | 数字检测**真实因果** |
| B2 | route **抹掉 `?`**（→`.`） | 仅 **1.3%** 翻 stop | 问句判定**不靠标点** |
| B3 | stop **末尾追加 `?`** | 仅 **29.4%** 翻 route | 同上 |
| B4 | stop **末尾追加数字** | **99.8%** 翻 ask_human | 数字检测因果 |
| B5 | stop **中间覆盖一个 token 为数字** | **99.9%** 翻 ask_human | **位置无关**（真·窗内 needle） |

**结论**：
- 对**数字**，模型是按字面 token 检测，**因果且位置无关**——是真实能力而非相关性。
- 对**问句**，模型走的是**语义/词法线索**而非句末 `?`。这是**泛化优点**（删掉 `?` 仍读出疑问语气），
  但**与训练标签规则不一致**（规则按字符 `?` 定义）→ 在这类反事实输入上它会「违规则」。

### 7.3 「可见证据量 → 决策可用性」曲线

只给前 k 个真实 token（其余 pad），仅内容三类（`generate` 由长度定义故排除）：

| k | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| acc | 0.0014 | 0.0467 | 0.1735 | 0.3051 | 0.4304 | 0.6328 | 0.7823 | 0.9426 | **0.9981** |

单调上升，与「观察回接」实验同构。读数需注意：**标签按整窗定义**，故曲线含 k/64 覆盖效应——
它度量的是「可见证据量 → 决策可用性」，不完全是「需要多长上下文」。

### 7.4 OOD 鲁棒性：9/9 正确

| 输入 | 期望（按规则） | 预测 |
|---|---|---|
| `The theory was developed over many decades.` | stop | ✅ stop |
| `The sample contained 42 participants in total.` | ask_human | ✅ ask_human |
| `What is the purpose of this experiment?` | route | ✅ route |
| `这个实验的目的是什么?` | route | ✅ route |
| `该实验共有42名参与者。`（`。` ≠ `.`） | generate | ✅ generate |
| `123 456 789 0 12 34` | generate | ✅ generate |
| `x = 3 + 5 ; print(x)` | generate | ✅ generate |
| `Hi` | generate | ✅ generate |
| `.` | stop | ✅ stop |

跨语言 / 代码 / 纯数字 / 极短全部按规则命中。

> **自我更正**：初版探测把其中 4 例标为「模型错」，实为**我给定期望值时漏了规则第 2 条**
> （不以 `.!?` 结尾的残片一律 `generate`）。模型判 `generate` 是正确的，**属我的脚本 bug**。

> **探测踩坑**：该 vocab 把数字编码为**两个 token**——`28705`（空串，所有数字共用）+
> 数字字符 token（`28734/28740/28750/28770/28774/28781/28782/28783/28784/28787`）。
> 初版只替换了共用的 `28705` → 消融**假阴性**（0.46% 翻转）。必须取 `encode(c)[-1]`。

### 7.5 生成能力

8 个分布内 prompt × 32 token 贪心：**病态重复（maxrun ≥ 4）0/8**，distinct-1 0.316。
样例：

```
'ifies the most vulnerable populations. The report also shows that the '
'inity, and temperature. The temperature of the air is 100°C (100°F) an'
```

是**通顺的英文教育文本续写**（语义会漂）。只有 OOD 乱码 prompt 才会病态重复。

### 7.6 它明确**不能**做

```
"Please summarize the following text:"  -> '- 1. The following text is from the text of the text: -'
"Q: What is 2+2? A:"                    -> 'What is 2+2? A: What is 2+2?'
```

**非 instruct 模型**：不遵循指令、不做问答推理，只会**续写**。
它是「内容路由器 + 弱续写器」，不是通用助手。

---

## 8. 部署验证

### 8.1 PC 全链路对拍

`convert_minueza.py` → **20.77 MB** KMCU（346 tensors，n_cls=4，tied，MoE 8 专家 top-2，ple_dim 128）。

`host_verify`(C, x86) vs `ref_forward.py`(Python)：

| 项 | 结果 |
|---|---|
| 贪心序列（20 token） | **逐 token 完全一致** |
| cls logits（8-token） | 差 ~4.8e-4，argmax **同为 id=2** |
| agent 循环 8 步 | 全 action=1，gen token 一致 |

差异量级来自 q4（core）+ 三值（PLE 表）量化，符合预期。

### 8.2 板端

**障碍**：正式 64M 模型 **20.77 MB > Flash `model` 分区 13.87 MB**（16MB 片内），
必须走 SD 卡；而 SD 卡槽当时出现硬件故障（详见 8.3）。

**解法**：新增 `FLASH_AGENT` 部署通道（`main.c`）——模型从 Flash `model` 分区读入 →
`km_fast_build_ex` 展开 PSRAM → 复用同一套 `run_generate / run_dual_head / run_agent`
三模式（串口 `0/1/2/q` 动态切换）。用**同任务、同 n_cls=4、同代码路径**的
17.42M 小模型（`dim128/6层/ffn_e64/n_expert4/ple_dim64`，convert 后 **5.13 MB**，装得下 Flash）验证：

| 项 | 板端（RISC-V） | PC 参考 |
|---|---|---|
| 20-token 序列 | **逐 token 完全一致** | — |
| single-token top8 | 顺序一致，logit 差 ~1.7e-5 | — |
| 模式 1 决策头 | `action=1 logits: -4.4643 4.7823 -2.4585 -1.9285` | agent s0 `-4.464250 4.782335 -2.458621 -1.928520` |
| 模式 2 agent 8 步 | 全 action=1，gen token 全等 | 同 |
| 解码速度 | **19.5 tok/s**，TTFT(prefill 4)=199 ms | — |

**代码改动**（权威源 `kestrel_mcu/main/main.c`）：放开 3 处 guard 加 `||FLASH_AGENT`
（SD includes 块 / SD 功能块 / 三模式块）+ 前置声明 `run_*` + 消除 `k_prompt` 重复定义
（顶部 guard 由 `!SD_STREAM && !SD_RESIDENT` 改为 `!SD_STREAM`）。

### 8.3 SD 卡故障、加固与恢复

**现象**：`sdmmc_init_ocr ... 0x107` / 深位读 `0x107`；某一轮 6 组合全败，重插清洁后
**8 组合仍全败**，连 400kHz 都在 `sd_host_slot_clock_update_command`（发 CMD 阶段）失败
→ 通信彻底不通，判定为**卡座/供电电路或卡本身故障**（非软件）。

**加固**（写入 `main.c`，两条路径 `sd_mount_stream` 与 `probe_sdcard` 共用同一张表）：

1. 重试表加入 `width=1` 下的 **4 MHz / 400 kHz** 低速档。
2. 挂载后增加**深位读校验**（sector 34957 + 容量中点），只接受**真能读**的配置——
   否则会出现「初始化可过、挂载成功、但数据读失败」的**假阳性**。
3. `format_if_mount_failed` 由 `true` 改 **`false`**：原值在挂载失败时会**自动格式化卡**，
   有丢失 `model.kmcu` 的真实风险。

**恢复（2026-09-28 复核）**：用户重插后检测通过 ——

```
[SD] try pwr=0 width=1 4000kHz ... -> ESP_OK
[SD] read-verify @34957 -> ESP_OK , @30533632 -> ESP_OK
Name: TW000  Type: SDHC  Speed: 4.00 MHz  Size: 29818MB  (bus_width=1)
[SD] model.kmcu 首块读 512 字节 (magic=KMCU)
```

**赢的组合正是新加的 4 MHz 档**：日志显示 20 MHz/1-bit 时「@34957 能读、@容量中点超时」
会被判失败，而旧表**根本没有低速档**，会直接误报「全败」。

**已知缺口**：`tools/upload_uart.py`（免插拔 UART 上传）与固件 `uart_upload_model` 的
**协议已不同步**——脚本等 `READY(0xAA)` 与状态字节 `0x55/0xEE/0xE3` 且全程 115200，
而固件打印「3 秒后切 921600」、既不发 `0xAA` 也不发状态字节；且固件只在卡上**无**
`model.kmcu` 时才进入上传模式。故当前只能走「PC 读卡器写卡」。

### 8.4 ESP-IDF 构建环境三坑（与模型无关，均为 PC 侧构建才暴露）

| 现象 | 根因 | 解法 |
|---|---|---|
| `git submodule init failed for components/esp_wifi/lib`（gitee 423） | 子模块仓库被屏蔽 | `IDF_SKIP_CHECK_SUBMODULES=1`（本项目已禁 WiFi/BT，安全） |
| `std::filesystem ... Illegal byte sequence` | **ccache 在非 ASCII 路径**（`E:\项目\...`） | `idf.py --no-ccache` + 构建放纯 ASCII 目录（`C:\kmcu`） |
| `fatal error: bits/error_constants.h: No such file` | **MAX_PATH 越界**：该头文件完整路径 259 字符（同目录 `c++config.h` 251 字符能过），前缀 `C:\Users\Administrator\.espressif\...` 过长 | 建目录 junction `C:\et` → `.espressif`，设 `IDF_TOOLS_PATH=C:\et`（缩短 28 字符，**不改任何系统设置**） |

> 教训：**Windows 上 ESP-IDF 构建对「非 ASCII 路径 + 长路径 + 子模块网络」三者极敏感**，
> 任一项都能让板端构建不可复现。修完这三点后构建稳定通过。

---

## 9. 结论与局限

### 能力画像

> **可靠的 4 类内容路由器**（留出 0.9986 / 自然分布 0.9981），核心能力是
> **窗内数字检测（因果、位置无关）** 与 **问句语义识别**；`generate` 是
> 「低置信/需续写」的**兜底类**；误差被压缩到**满窗 64-token 的标签歧义**这一不可约边界上；
> **不具备指令跟随与推理能力**。

### 工程含义

- 下游应把 `generate` 当作**低置信度/需续写**处理，而非硬分类。
- 想压掉那 0.14% 误差，应**改任务定义**（加 EOS / 续写标记，或延长窗口），而非继续训练。
- 若关心 `?` 的判定一致性，需注意模型实际走**语义线索**：对「陈述句被追加 `?`」这类
  反事实输入它会违规则（B3 仅 29.4% 翻 route）。

### 局限（诚实清单）

1. 任务虽为内容级，但**饱和很快**（step 4000 已达 0.995）——它是可靠的**工程级路由器**，
   不是需要深度推理的难题。
2. **正式 64M 模型未在板端运行**：20.77 MB 超 Flash 分区，SD 通路虽已恢复但尚未完成
   正式模型的板端三模式对拍（当前板端验证用的是同任务小模型）。
3. `route` 判定与标签规则存在系统性偏差（语义 vs 标点），评估指标因此**略微高估**了
   「按规则执行」的严格性。
4. ppl 为自定口径，跨运行不可比。

---

## 10. 附录：与 vLLM-Kestrel 引擎（`GitHupSRC/`）的兼容性

**结论：不可直接跑**（纯静态评估，未改任何代码；已归档）。四项硬障碍：

| # | 障碍 | 证据 |
|---|---|---|
| 1 | **几何（决定性）**：引擎按 `cols/32` 整数分块，`dim=312`（312/32=9）尾部 24 元素被静默丢弃；MoE 稀疏核有 `if (d & 31) return;` → **整层 FFN no-op**；`head_dim=26` 触发 `hd/32=0` | `vllm_safetensors.c:12940,797,2059,1238` |
| 2 | **PLE 无对应算子**（引擎全量无 `ple_` 标识符） | `vqf_format.h:47-52` |
| 3 | **三值无 qtype**（仅 F32/Q8_0/Q4_0/Q4_8X8L/F16） | 同上 |
| 4 | **决策头无出口**（`STModelWeights` 无 cls_head，只有 logits+采样） | `vllm_safetensors.h:123-225,504` |

另两个隐性坑：config 键名不匹配（本模型写 `n_expert` vs 引擎读 `num_experts`）→ MoE 被当稠密；
张量名不匹配（`mlp.router.weight` vs 引擎 `mlp.gate.weight`）。

**合规要点**：H1 位级一致只管引擎**内部**跨路径，**跨引擎与本项目 `kmcu.c` 逐位一致不成立**
（引擎 prefill/q8 会把**激活量化成 int8 做整数点积**）；用 `VLLM_ACTQ` 近似轨只能算 FAST 档。
**PLE 表在引擎语境是常驻内存问题而非带宽问题**（每 token 只读 1 行，但表本身
f32 ≈ 156 MB / q8 ≈ 39 MB / 三值 ≈ 8 MB）。

**关键判断**：症结是 `dim=312 / head_dim=26` 这套几何是为 ESP32-P4 手选的，**天生违反引擎
32 对齐红线**。故「把已有 64M 模型搬上去」性价比极低；**正确做法是反方向按引擎几何设计模型**
（dim 取 32 倍数如 320、head_dim=32），MoE 骨架可复用现成核，仅 PLE 与决策头需立项决策。

---

## 11. 产物与复现

### 产物

| 路径 | 说明 |
|---|---|
| `E:\models\agent_model` | 正式模型 HF 目录（64.36M） |
| `E:\models\agent_model.kmcu` | 部署镜像（20.77 MB，n_cls=4） |
| `E:\models\agent_model_report.json` | 评估报告（逐类指标 + 混淆矩阵 + ppl + 生成） |
| `E:\models\agent_model.log` | 训练日志（含每 2k 步 val、每 5k 步 ppl checkpoint） |
| `E:\models\agent_model_flash_step{5000,10000,15000}` | 板端验证用小模型 checkpoint |
| `E:\models\agent_flash.kmcu` | 小模型部署镜像（5.13 MB，装得下 Flash） |
| `E:\models\agent_x.bin` / `agent_y.bin` 等 | 数据（train/val/test/nat 四套） |

### 复现命令

```powershell
# 1) 数据（内容级任务 + 文档级划分）
python E:\models\_prep_agent_data.py

# 2) 正式训练（5.5h 墙钟）
python E:\models\_train_agent_model.py --steps 38000 --time_budget_h 5.5 --out E:\models\agent_model

# 3) 转换 + PC 全链路对拍
python kestrel_mcu\tools\convert_minueza.py --src E:\models\agent_model --out E:\models\agent_model.kmcu
python kestrel_mcu\tools\ref_forward.py --km E:\models\agent_model.kmcu
gcc -O2 -std=c11 -I main -o host_verify host_verify.c main/kmcu.c -lm   # 在 kestrel_mcu\ 下
.\host_verify.exe ..\agent_model.kmcu

# 4) 能力探测 / 生成 A/B
python E:\models\_probe_agent_caps.py
python E:\models\_verify_gen_ab.py

# 5) 板端（FLASH_AGENT 小模型路径）
#    main.c: SD_PROBE=0 / SD_RESIDENT=0 / FLASH_AGENT=1
idf.py --no-ccache build ; idf.py -p COM5 flash
python -m esptool --chip esp32p4 -p COM5 -b 460800 write-flash 0x210000 agent_flash.kmcu
#    串口 0/1/2/q 切换 generate / dual-head / agent
```

> Windows / ESP-IDF 构建前需设：`$env:IDF_TOOLS_PATH="C:\et"`（junction 缩短工具链路径）、
> `$env:IDF_SKIP_CHECK_SUBMODULES=1`，并用 `idf.py --no-ccache`。
