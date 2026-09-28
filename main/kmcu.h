/* ================================================================
 * kmcu.h - KMCU 量化权重格式读取 + 微型 Transformer 前向（标量 C11）
 *
 * 格式定义见 tools/convert_minueza.py 顶部注释（KMCU v1）。
 * 本项目与 GitHupSRC/ 引擎零耦合：仅按格式规格独立实现，不引用引擎源码。
 * 声明：独立 determinism line，与引擎不保证位级一致。
 * ================================================================ */
#ifndef KMCU_H
#define KMCU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KM_MAGIC "KMCU"
#define KM_DT_F32 1
#define KM_DT_Q4  2
#define KM_DT_TERNARY 3
#define KM_DIR_ENTRY_BYTES 48

/* 模型架构（对应 KMCU 头部） */
typedef struct {
    uint32_t version;
    uint32_t n_layers;
    uint32_t dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t ffn_dim;
    uint32_t vocab_size;
    uint32_t max_seq;
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t tied_embed;
    uint32_t n_tensors;
    uint32_t dir_offset;
    uint32_t data_offset;
    float    rope_theta;
    float    norm_eps;
    uint32_t total_params;
    uint32_t n_expert;   /* 0=稠密，>0=MoE */
    uint32_t top_k;      /* MoE top-k */
    uint32_t ple_dim;    /* PLE 每层查表宽度（0=无 PLE） */
    float    ple_gamma;  /* PLE 三值表 scale（code × gamma） */
    uint32_t n_cls;      /* 决策头类别数（0=无决策头） */
} km_arch_t;

/* 单个张量的目录项（解析后的视图，指向 base 内的字节） */
typedef struct {
    char     name[25];
    uint8_t  dtype;
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t offset;   /* 相对 base */
    uint32_t nbytes;
} km_tensor_t;

/* 只读模型视图：base 指向整份 KMCU 镜像（PSRAM 或 flash mmap） */
typedef struct {
    const uint8_t *base;
    km_arch_t      arch;
    km_tensor_t   *tensors;   /* 调用方分配的 n_tensors 数组 */
} km_model_t;

/* ---- 快速权重（装载时 q4 → 展开 int8 + 独立 fp16 scale） ----
 *
 * 动机：q4 逐元素反量化含半字节提取/移位/掩码/分支，是 decode 的主导开销
 * （实测 22.8M 元素 × ~10 周期 = 1.42s / 1.63s）。展开为 int8 后内循环退化为
 * 紧凑的 (float)q * x 累加。
 *
 * 数值口径：每 32 元素块先求 Σ(q*x)，再乘该块的 fp16 scale（d 提出块外），
 * 与参考的逐元素 (q*d*x) 累加存在浮点结合顺序差异，但**不引入新的量化近似**
 * （q 与 d 完全来自原 q4 数据）。
 *
 * 行对齐（M3-2）：文件里的 q4 已按行补齐到 32 的整数倍（行尾零填充），
 * 因此每块只属于一行、块不跨行。展开后布局为
 *   q[r][c] = q[r*align32(n_cols) + c]，d 的块号 = r*nblk + b（nblk=align32/32）。
 * GEMV 内层因此是定长 32 的可完全展开循环，无相位漂移、无分支。 */
typedef struct {
    const int8_t   *q;   /* 展开后的 int8（行对齐，行尾含零填充） */
    const int16_t  *d;   /* 每 32 元素一块的 int16 定点 scale（块号 = r*nblk + b；
                          真实 fp32 scale = d_int16 * d_scale，per-tensor 归一化） */
    float           d_scale; /* per-tensor scale：d_fp32 = d_int16 * d_scale */
    const float    *f32; /* 非 q4 张量（norm）的 fp32 副本 */
    uint32_t        n;   /* Q4: n_rows*align32(n_cols)；F32: n_rows*n_cols */
    uint8_t         is_q4;
} km_ft_t;

typedef struct {
    km_ft_t *t;          /* n_tensors 项，下标与 km_model_t.tensors 对齐 */
} km_fast_t;

/* 展开所需 arena 字节数（含 64B 对齐填充） */
size_t km_fast_bytes(const km_model_t *m);

/* 源数据读取回调：从模型镜像 off 处读 len 字节到 dst，返回 0 成功 */
typedef int (*km_read_fn)(void *ctx, uint32_t off, void *dst, uint32_t len);

/* 展开全部张量到 arena（F32 张量复制，q4 张量展开为 int8+scale）。
 * read_fn == NULL 时从 m->base 直读；否则经 read_fn 分块读取（stage 为暂存区），
 * 后者可让 flash 原图不必整份载入 PSRAM。 */
void km_fast_build_ex(const km_model_t *m, km_fast_t *f,
                      uint8_t *arena, size_t cap,
                      km_read_fn read_fn, void *ctx,
                      uint8_t *stage, size_t stage_cap);

/* 便捷封装：从 m->base 直读 */
void km_fast_build(const km_model_t *m, km_fast_t *f, uint8_t *arena, size_t cap);

typedef struct {
    char buf[256];
} km_err_t;

/* 打开模型：校验 magic/version，填充 arch 与 tensors[0..n_tensors)。
 * 返回 0 成功；非 0 失败并把原因写入 err。 */
int km_open(km_model_t *m, const uint8_t *base, km_err_t *err);

/* 按名字找张量（大小写敏感）；未找到返回 -1。 */
int km_find(const km_model_t *m, const char *name);

/* fp16 -> fp32 */
float km_f16_to_f32(uint16_t h);

/* ---- 前向（decode 单步） ---- */
typedef struct {
    uint32_t n_ctx;       /* KV cache 容量（token 数） */
    float   *k_cache;     /* [n_layers][n_ctx][n_kv_heads*head_dim] */
    float   *v_cache;
    uint32_t pos;         /* 下一个写入位置 */
    float   *scratch;     /* 工作区，见 km_state_bytes() */
} km_state_t;

/* 计算 km_state 所需的外部缓冲字节数（不含模型权重） */
size_t km_state_bytes(const km_arch_t *a, uint32_t n_ctx);

/* scratch（工作区）字节数。GEMV 的输入向量按 align32 读取尾部填充，
 * 故各向量缓冲已按 32 对齐加长；padding 区在 km_state_init 中清零后不再被写。 */
size_t km_scratch_bytes(const km_arch_t *a);

/* 绑定外部缓冲并清零 KV cache */
void km_state_init(km_state_t *st, const km_arch_t *a, uint32_t n_ctx,
                   float *k_cache, float *v_cache, float *scratch);

/* 单步 decode：输入 token id，返回下一个 token（贪心 argmax）。
 * f 为展开后的快速权重（下标与 m->tensors 对齐）。
 * topk>0 时把 top-k logits 写入 out_ids/out_vals（降序），用于与 PC 参考对拍；
 * out_logit 可 NULL。
 *
 * MoE 模型（n_expert>0）下 embed/lm_head 不展开进 PSRAM，需经 read_fn 流式按行读；
 * read_fn 传 NULL 时按稠密常驻路径处理（embed/lm_head 必须已在 f 中展开）。
 * stage/stage_cap 供流式读单行 embed / 单行 lm_head 的反量化暂存。 */
int km_decode_step(const km_model_t *m, const km_fast_t *f, km_state_t *st,
                   int token, float *out_logit,
                   int *out_ids, float *out_vals, int topk,
                   km_read_fn read_fn, void *ctx,
                   uint8_t *stage, size_t stage_cap);

/* 决策头前向：对整段 token 序列跑 transformer，mean pooling last hidden 后接
 * cls_head（f32）做分类。单次前向，不碰 lm_head。返回 argmax 类别 id。 */
int km_cls_forward(const km_model_t *m, const km_fast_t *f, km_state_t *st,
                   const int *tokens, int n_tokens, int n_cls,
                   int *out_id, float *out_logits,
                   km_read_fn read_fn, void *ctx, uint8_t *stage, size_t stage_cap);

/* ---- Agent 循环（观察回接）骨架 ----
 * 动作空间（与训练侧 n_cls 对齐）：
 *   0 = route      路由到工具/模块
 *   1 = generate   生成（走 lm_head 自回归）
 *   2 = stop       结束循环
 *   3 = ask_human  求助人工
 * 骨架阶段用「全量重算」保证正确性（增量 KV 复用留后续优化）。
 * 循环：决策（km_cls_forward）→ 动作分派 → 观察回接（把生成/观察 token 拼回
 * context）→ 再决策，直到 stop 或 max_steps 或 context 满。 */
#define KM_ACT_ROUTE      0
#define KM_ACT_GENERATE   1
#define KM_ACT_STOP       2
#define KM_ACT_ASK_HUMAN  3
#define KM_ACT_N          4

/* 返回实际步数；out_actions[step]、out_gen[step]（generate 生成的 token，
 * 非 generate 为 -1）、out_logits[step*n_cls + c] 记录每步决策。 */
int km_agent_loop(const km_model_t *m, const km_fast_t *f, km_state_t *st,
                  const int *init_tokens, int n_init, int n_cls, int max_steps,
                  int max_ctx, int *out_actions, int *out_gen, float *out_logits,
                  km_read_fn read_fn, void *ctx, uint8_t *stage, size_t stage_cap);

/* ---- 逐层流式 decode（0.1B 分层驻留，权重不常驻 PSRAM） ----
 * 整份展开（km_fast_build_ex）需要把全部权重展开进 PSRAM；对 0.1B 模型
 * 展开后 ~150MB，远超 32MB PSRAM。流式路径把内存峰值压到「单层 + KV cache」：
 *   - embed / 输出头按行从 read_fn 读取，不整份展开
 *   - 每层权重读入 → 展开到 layer_arena → 计算 → 下一层复用同一 arena
 * read_fn 语义同 km_fast_build_ex（从外部存储按偏移读取）。 */

/* 单层（一层全部张量）流式展开所需的最大字节数（含 64B 对齐）。 */
size_t km_layer_bytes(const km_model_t *m);

/* 流式展开单个张量（tensors[idx]）到 arena。返回对齐后占用字节数（>0 成功，0 失败）。 */
size_t km_build_tensor(const km_model_t *m, uint32_t idx, km_ft_t *ft,
                       uint8_t *arena, size_t cap,
                       km_read_fn read_fn, void *ctx,
                       uint8_t *stage, size_t stage_cap);

/* 流式读 embed 张量第 row 行（q4 反量化到 fp32，out 长度 dim）。
 * 不整份展开 embed；返回 0 成功。 */
int km_embed_row(const km_model_t *m, const km_tensor_t *emb, int row, int dim,
                 float *out, km_read_fn read_fn, void *ctx,
                 uint8_t *stage, size_t stage_cap);

/* 逐层流式单步 decode：语义同 km_decode_step，但权重经 read_fn 逐层流式加载。 */
int km_decode_step_stream(const km_model_t *m, km_state_t *st,
                          int token, float *out_logit,
                          int *out_ids, float *out_vals, int topk,
                          km_read_fn read_fn, void *ctx,
                          uint8_t *layer_arena, size_t layer_cap,
                          uint8_t *stage, size_t stage_cap);

#ifdef __cplusplus
}
#endif

#endif /* KMCU_H */
