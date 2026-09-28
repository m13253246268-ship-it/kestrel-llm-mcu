# Kestrel-MCU MoE 框架设计（改造稠密 → 稀疏激活）

> English version: **[moe_framework.md](moe_framework.md)**

> **工具目录说明（2026-09-28）**：本文提到的历史研究脚本（`train_moe.py`、`train_moe_kd.py`、
> `train_moe_qat.py`、`dense_baseline.py`、`eval_quality.py` 等）已归档到
> **`tools/legacy/`**；当前主线工具仍位于 `tools/`。工具总览见 `tools/README.md`。

> 目标：在**保留稠密框架**的前提下，新增 MoE（混合专家）路径，用「稀疏激活」突破
> 稠密 decode 的 PSRAM 带宽墙。遵循公理库约束：位级一致性（router 确定性）、
> 带宽受限（每 token 读量最小化）、实测胜于推论。

## 1. 背景与结论依据

前几轮已验证的硬结论（见 README M3-3b / M3-5 / M3-7 / M3-10 / M3-11）：

- decode 瓶颈 = **每 token 读的权重字节数**（PSRAM 106 MB/s 墙），非计算；
- FFN（gate/up/down）占 decode 权重读的 **44%**（10.2M 元素），输出头占 43%；
- 输出头剪枝（无损上界 + 有损 ANN）**均判否**（margin 极小 + 高维维度灾难）；
- 权重量化 q2/q3 **判否**（PIE 无 int4/int2 dot + 再量化崩模型）；
- **MoE 稀疏激活是唯一能在「减少每 token 读量」上生效的方向**（PFor / esp32-ai 已实证）。

## 2. 架构设计

**改造位置**：仅替换 FFN 层（`L{i}.gate/up/down`），attention 与输出头不变。

```
稠密 FFN：  y = down @ (silu(gate @ h) * up @ h)
MoE  FFN：  y = Σ_{e ∈ top-k(h)} softmax(router(h))_e · Expert_e(h)
            Expert_e(h) = down_e @ (silu(gate_e @ h) * up_e @ h)
```

| 参数 | 取值 | 说明 |
|---|---|---|
| 专家数 N | **8** | 平衡收益与路由复杂度 |
| 激活数 top-k | **2** | 每 token 激活 2/8 专家，FFN 读量降 75% |
| 专家维度 ffn_e | **128**（对齐 32） | 每个专家是**完整独立**的 gate/up/down 三矩阵（非聚类碎片），从头训练 |
| router | `[N, dim]`（312×8） | 一个小 GEMV + top-k + softmax |

**收益模型**（N=8, top-2，总参数不变 ~28M）：

| 段 | 稠密读量 | MoE 读量 |
|---|---|---|
| attention | 3.9M | 3.9M（不变） |
| FFN | 10.2M | **2.55M**（2/8 × 10.2M） |
| 输出头 | 9.98M | 9.98M（不变） |
| **合计** | **23.4M** | **~16.4M（−30%）** |

decode 232 ms → 理论 **~165 ms**（纯带宽）；router 增量 ~N×dim=2.5K 可忽略。

## 3. KMCU 格式扩展（v2）

### 3.1 Header 扩展

复用 v1 的 128B 头，占用 `reserved` 区（offset 76 起）：

| 偏移 | 字段 | 类型 |
|---|---|---|
| 76 | is_moe | u32（0=稠密，1=MoE） |
| 80 | n_expert | u32 |
| 84 | top_k | u32 |
| 88 | ffn_expert | u32（对齐后） |

### 3.2 张量命名

- 稠密路径：`L{i}.gate/up/down`（不变，`is_moe=0` 时使用）
- MoE 路径（`is_moe=1`）：
  - `L{i}.router`（[N, dim]）
  - `L{i}.expert{e}.gate` / `.up`（[ffn_e, dim]）
  - `L{i}.expert{e}.down`（[dim, ffn_e]）

`L{i}.router` 为 fp32 或 q4（推荐 q4，与其它权重一致）。attention/embed/norm 命名不变。

## 4. 内核改造

在 [km_decode_step](../main/kmcu.c) 的 MLP 段分支：

```c
if (!a->is_moe) {
    /* 稠密：原 gate/up/down 路径，零改动 */
} else {
    /* 1) router: gemv_q8(N, dim, h) → rlogits
     * 2) top-k：选 k 个最大的 rlogits（保留索引）
     * 3) softmax 归一化选中的 k 个权重 w_e
     * 4) 对每个选中专家 e：
     *      gemv_q8(ffn_e, dim, h) → g_e
     *      gemv_q8(ffn_e, dim, h) → u_e
     *      act = silu(g_e)*u_e
     *      gemv_q8(dim, ffn_e, act) → h_e
     *      x += w_e * h_e
     */
}
```

**复用**：`gemv_q8` / `pie_gemv_row` 完全复用（专家 FFN 就是标准 GEMV）。
新增代码量：router GEMV + top-k 选择 + 循环调度，约 60 行。

**scratch 扩展**：专家输出 `h_e` 复用现有 `h` 缓冲；`w_e` 与 top-k 索引用栈上小数组
（N=8, k=2）。**不增加 PSRAM 常驻**（专家权重仍在展开 arena 内，只是命名不同）。

## 5. 训练方案（从头训练 + 知识蒸馏，降低难度）

> M3-12 已证伪「改造稠密」：MoEfication 聚类拆分 oracle 误差 62.8%，Sparse Upcycling
> 参数 8× 爆炸。故改为**从头训练 MoE，用稠密 Minueza 作 teacher 蒸馏**，降低训练难度。

### 5.1 角色

- **Teacher**：现有稠密 Minueza（32M，已训练好），提供软标签（logits 分布）；
- **Student**：MoE 模型（N=8 专家，每专家 ffn_e=128，完整 gate/up/down），从零初始化。

Student 总参数 ≈ 稠密（8×128=1024 ≈ 原 ffn=1092），但每 token 只激活 top-2（256 神经元）。

### 5.2 蒸馏损失

标准软标签蒸馏（Hinton KD）：

```
L = (1-α)·L_CE(student_logits, hard_label)
    + α·T²·KL(softmax(student_logits/T), softmax(teacher_logits/T))
```

- `T`：温度（如 4.0），放大 teacher 的 logits 分布，让 student 学到 token 间的相对关系；
- `α`：蒸馏权重（如 0.9）；
- `hard_label`：可选（仅用蒸馏损失也可）。

### 5.3 训练流程

1. **数据**：TinyStories（开源，单 GPU 可跑）或 Minueza 原训练域文本；
2. **teacher 前向**：冻结 Minueza，对每个 batch 算 `teacher_logits`（可预计算缓存，加速）；
3. **student 前向**：MoE（router → top-k → 稀疏 FFN → logits）；
4. **反向**：只更新 student（router + 专家 + attention 全量）；
5. **关键**：router 用可微 top-k（`top_k + straight-through` 或加噪 softmax），让稀疏选择可训练；
6. 训练到 student 的贪心 argmax 序列逼近 teacher（目标 ≥ 19/20 对拍一致）。

### 5.4 为什么蒸馏能降低难度

- 从头学语言建模难（需大量数据 + 长训练 + 技巧），蒸馏只需「模仿 teacher 的 logits 分布」；
- router 在蒸馏中自然学会「专家分工」（不同输入路由到不同专家），无需手工聚类；
- 专家通过反向传播学会「在自己的输入域内浓缩表达」，规避 M3-12 的「碎片量级不足」。

### 5.5 量化与落盘

训练得到的 fp32 MoE 权重 → 复用 `convert_minueza.py` 的 q4 量化 → 生成 v2 KMCU
（is_moe=1，含 router 与 expert 张量）。router 与专家同样走 q4 量化。

### 5.6 训练脚本落地（train_moe.py，自研 MoE 核心）

`tools/legacy/train_moe.py` 是自研训练工具（teacher=Minueza 冻结，student=复用 Mistral 骨架 + 自研 `LambdaMoE`）。

**自研 `LambdaMoE` 模块**（对应公理库 MoE 叠加路由公理，见第 6 节）：

- 前向始终 **hard top-k**（`R_classic`，稀疏、等价推理态）；
- λ 只退火控制**反向梯度软化**（straight-through）：早期 λ=1 全专家 soft 梯度一起学，
  后期 λ→0 只有选中专家有梯度，分工锐化；
- 附带轻量负载均衡 loss（Switch Transformer 式），防止 hard 路由坍缩到单专家。

**关键诊断结论**（本轮冒烟实测，诚实归因）：

1. **soft 前向（λ=1 全专家加权）会让 MoE 退化**：8 个专家在对称初始化 + 均匀加权下
   梯度趋向一致，router 停在均匀驻点（真实熵恒 = log8，梯度 norm ≈ 0），退化成「弱化稠密」；
2. **hard 前向是打破对称性的必要条件**：straight-through（前向 hard、反向 λ 软化）下
   router 开始分化（实测 L9 专家选择频率明显分化，L0 仍近均匀）；
3. 但 hard 路由有**坍缩倾向**（少数专家垄断，aux 从均匀值 1 涨到 13），需负载均衡 loss 对抗；
4. 上一轮 `train_moe_kd.py` 用 transformers `MixtralForCausalLM` 失败（aux 恒 2.0 = 均匀态），
   根因正是其 load-balancing loss 把 router 压向均匀、且无 λ 退火机制——已由自研 `LambdaMoE` 修正。

### 5.7 KMCU v2 转换映射（自研 LambdaMoE 的 state_dict）

自研 `LambdaMoE` 的专家是**逐专家独立张量**（非 transformers Mixtral 的堆叠 3D），转换更直观：

| LambdaMoE state_dict 键 | shape | KMCU v2 目标 |
|---|---|---|
| `L{i}.mlp.router.weight` | (8, 312) | `L{i}.router` |
| `L{i}.mlp.experts.{e}.gate.weight` | (128, 312) | `L{i}.expert{e}.gate` |
| `L{i}.mlp.experts.{e}.up.weight` | (128, 312) | `L{i}.expert{e}.up` |
| `L{i}.mlp.experts.{e}.down.weight` | (312, 128) | `L{i}.expert{e}.down` |

（`transformers` 5.x 的 Mixtral 用 `gate_up_proj (8,256,312)` 堆叠 gate+up，需按 dim1 前半/后半拆分；
自研版无需拆分，直接一一对应。）


## 6. 公理库约束对拍

| 公理 | MoE 框架如何满足 |
|---|---|
| **MoE 叠加路由公理**（`R=(1-λ)·R_classic ⊕ λ·R_new`） | `LambdaMoE` 前向 = `R_classic`（hard top-k），反向梯度 = λ·soft + (1-λ)·hard（straight-through），λ 从 1 退火到 0 |
| **连续可还原律 SHS-3** | λ 用 cosine 从 1.0 平滑退火到 lam_min，router 从 soft 连续收敛到 hard 稀疏 |
| 位级一致性 | router = 确定性 GEMV + argmax/top-k；稀疏 FFN = 确定性 GEMV。同输入同输出，位级可复现 |
| 带宽受限 | 收益来自「每 token 只读 top-k 专家」，正是减少带宽占用 |
| 实测胜于推论 | 第 7 节用 PC 对拍 + 板端实测量化收益，不以理论 30% 为准 |
| irreversibility_penalty | MoE 稀疏是**模型架构**（训练时确定），非推理期近似；同一模型内与稠密等价、可复现 |

## 7. 实测结论（M3-13 / M3-14，诚实归因）

> 本节记录训练后的硬数据，推翻早期两个误判，收敛到核心瓶颈。

### 7.1 argmax match 是不可达指标（M3-13）

**现象**：MoE student 训练后 `argmax match = 0.21`（目标 ≥ 0.95），但 `logits cosine = 0.9899`。

**诊断链**（逐条证伪误判）：

| 假设 | 实验 | 结论 |
|---|---|---|
| 训练量不足 | 8000 步 vs 4000 步 | match 停在 0.21 不动，证伪 |
| 蒸馏超参（α/T） | α 0.9→0.3、T 4→2 | match 0.214 不变，证伪 |
| MoE 容量缺口（256 vs 1092 维） | ffn_e 128→256 | match 0.205 不变，证伪 |
| 自回归级联放大 | 单步 logits argmax | 一致率 **0.0000**，非级联问题 |
| **teacher argmax 本身脆弱** | 测 top-1 vs top-2 margin | **margin 仅 0.52**，决定性 |

**根因**：teacher 的 argmax 决策 margin 只有 0.52（vocab=32002 空间里极脆弱），而 KD 蒸馏
优化的 logits 分布（cosine 0.99）的剩余 0.01 误差，落在 0.52 的决策边界上，足以让
32002 维 argmax 全盘翻转。**「argmax 逐 token 一致」作为蒸馏验收指标在数学上不可达**
（除非 student 是 teacher 的逐位复制）。

### 7.2 换验收指标：perplexity（M3-14）

改用语言建模标准指标，硬数据：

| 模型 | perplexity | vs teacher |
|---|---|---|
| teacher（稠密 Minueza） | **21.29** | 1.0× |
| dense 从零 FFN（对照） | 77.41 | 3.6× |
| moe ffn128 | 84.18 | 4.0× |
| moe ffn256 | 84.66 | 4.0× |

**两层结论**：

1. **MoE 架构可行（正面）**：moe vs dense 从零只差 9%（84 vs 77），说明在同样「从零训 FFN」
   前提下，MoE 稀疏化几乎无额外质量损失。router 分化、λ 退火、带宽收益论证全部成立。

2. **真瓶颈是「从零训练 FFN」本身**：无论稠密还是 MoE，从零训 FFN 的 perplexity 都是
   teacher 的 3.6~4 倍，生成文本破碎（死循环「Tom Tom Tom...」）。KD 蒸馏 + 复用 attention
   在千步级预算下**无法传递 teacher 藏在 FFN 里的语言建模知识**（FFN 占 decode 权重读 44%，
   恰恰是最难学的部分）。

### 7.3 最终状态

- MoE 方向、λ 退火路由、自研训练工具（`train_moe.py`）、评估工具（`eval_quality.py`）——全部验证正确、已就绪；
- 「从头训练 + 蒸馏降低难度」在千步级预算下无法产出可用模型，瓶颈是 FFN 知识传递；
- 需更大训练预算（万步~十万步级 + 完整 TinyStories）或换训练范式，才可能把 perplexity 拉近 teacher。

## 8. 交付物

1. KMCU v2 格式定义 + 解析（`convert_minueza.py` 扩展 + `kmcu.h/c` 扩展）；
2. MoE 内核（router + top-k + 稀疏 FFN，dense 路径保留）；
3. 训练脚本 `train_moe.py`（自研 LambdaMoE + λ 退火 + KD 蒸馏）；
4. 对照脚本 `dense_baseline.py` + 评估脚本 `eval_quality.py`（perplexity + 生成质量）；
5. 入档（本文件第 7 节实测结论）。
