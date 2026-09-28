/* ================================================================
 * kmcu.c - KMCU 权重格式读取 + 标量 Transformer decode 前向
 *
 * 两条权重路径：
 *   A) 慢路径：直接从 q4 原图逐元素反量化（保留为正确性基准）
 *   B) 快路径：装载时把 q4 展开为 int8 + 独立 fp16 scale 数组（默认使用）
 * 全程 fp32 累加；快路径不引入新的量化近似（q/d 完全来自原 q4 数据）。
 *
 * 与 tools/ref_forward.py 使用同一份量化权重，二者差异应仅来自浮点累加顺序。
 * ================================================================ */
#include "kmcu.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* PIE（xespv）定点内核，见 pie_dotprod.S */
extern float pie_gemv_row(const int8_t *q, const int16_t *x, const int16_t *d, int nblk, int shift);

/* ---------------- 重复惩罚（repetition penalty） ----------------
 * 可选开关，默认关闭。对已出现在上下文中的 token 打压 logit，抑制
 * 「big, strong lion / the cat and the little cat」类重复。
 * 语义与 tools/ref_forward.py 的 generate(rep_penalty=...) 严格一致：
 *   正 logit 除以系数，负 logit 乘以系数（HuggingFace repetition_penalty 口径）。
 * 单线程嵌入式固件：用静态 seen 位图，st->pos==0（新会话首步）时自动清零。
 * ---------------------------------------------------------------- */
#define REP_PENALTY 0          /* 0=关；非 0 开启 */
#if REP_PENALTY
#define REP_PENALTY_COEF 1.2f
#define KM_REP_VOCAB_MAX 32768
static uint8_t s_rep_seen[KM_REP_VOCAB_MAX >> 3];

static void rep_clear(void) { memset(s_rep_seen, 0, sizeof(s_rep_seen)); }
static void rep_mark(int tok) {
    if (tok >= 0 && tok < KM_REP_VOCAB_MAX)
        s_rep_seen[tok >> 3] |= (uint8_t)(1u << (tok & 7));
}
static float rep_apply(float acc, int v) {
    if (v < 0 || v >= KM_REP_VOCAB_MAX) return acc;
    if (s_rep_seen[v >> 3] & (uint8_t)(1u << (v & 7)))
        return acc >= 0.0f ? acc / REP_PENALTY_COEF : acc * REP_PENALTY_COEF;
    return acc;
}
#endif

/* ---------------- 基础工具 ---------------- */

float km_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t man  = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            int e = 0;
            while (!(man & 0x400u)) { man <<= 1; e++; }
            man &= 0x3FFu;
            f = (sign << 31) | ((uint32_t)(127 - 14 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static inline uint32_t rd_u32(const uint8_t *p, size_t off) {
    uint32_t v;
    memcpy(&v, p + off, 4);
    return v;
}
static inline float rd_f32(const uint8_t *p, size_t off) {
    float v;
    memcpy(&v, p + off, 4);
    return v;
}
static inline size_t align_up(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

/* 行对齐到 32：q4 权重每行补齐成整数个 32 元素块 */
#define KM_ALIGN32(v) (((v) + 31u) & ~31u)
#define KM_NBLK(cols) (KM_ALIGN32(cols) / 32u)

/* ---------------- 打开模型 ---------------- */

int km_open(km_model_t *m, const uint8_t *base, km_err_t *err) {
    if (!m || !base) return -1;
    if (memcmp(base, KM_MAGIC, 4) != 0) {
        if (err) snprintf(err->buf, sizeof(err->buf), "bad magic (expect KMCU)");
        return -1;
    }
    km_arch_t *a = &m->arch;
    a->version      = rd_u32(base, 4);
    a->n_layers     = rd_u32(base, 8);
    a->dim          = rd_u32(base, 12);
    a->n_heads      = rd_u32(base, 16);
    a->n_kv_heads   = rd_u32(base, 20);
    a->head_dim     = rd_u32(base, 24);
    a->ffn_dim      = rd_u32(base, 28);
    a->vocab_size   = rd_u32(base, 32);
    a->max_seq      = rd_u32(base, 36);
    a->bos_id       = rd_u32(base, 40);
    a->eos_id       = rd_u32(base, 44);
    a->tied_embed   = rd_u32(base, 48);
    a->n_tensors    = rd_u32(base, 52);
    a->dir_offset   = rd_u32(base, 56);
    a->data_offset  = rd_u32(base, 60);
    a->rope_theta   = rd_f32(base, 64);
    a->norm_eps     = rd_f32(base, 68);
    a->total_params = rd_u32(base, 72);
    a->n_expert     = rd_u32(base, 76);
    a->top_k        = rd_u32(base, 80);
    a->ple_dim      = rd_u32(base, 84);
    a->ple_gamma    = rd_f32(base, 88);
    a->n_cls        = rd_u32(base, 92);

    if (a->version != 1) {
        if (err) snprintf(err->buf, sizeof(err->buf), "bad version %u",
                          (unsigned)a->version);
        return -1;
    }
    if (a->n_heads == 0 || a->n_kv_heads == 0 || a->head_dim == 0 ||
        a->n_heads % a->n_kv_heads != 0) {
        if (err) snprintf(err->buf, sizeof(err->buf), "bad head config %u/%u/%u",
                          (unsigned)a->n_heads, (unsigned)a->n_kv_heads,
                          (unsigned)a->head_dim);
        return -1;
    }

    m->base = base;
    for (uint32_t i = 0; i < a->n_tensors; i++) {
        size_t b = a->dir_offset + (size_t)i * KM_DIR_ENTRY_BYTES;
        km_tensor_t *t = &m->tensors[i];
        memcpy(t->name, base + b, 24);
        t->name[24] = '\0';
        t->dtype   = base[b + 24];
        t->n_rows  = rd_u32(base, b + 28);
        t->n_cols  = rd_u32(base, b + 32);
        t->offset  = rd_u32(base, b + 36);
        t->nbytes  = rd_u32(base, b + 40);
    }
    return 0;
}

int km_find(const km_model_t *m, const char *name) {
    for (uint32_t i = 0; i < m->arch.n_tensors; i++) {
        if (strcmp(m->tensors[i].name, name) == 0) return (int)i;
    }
    return -1;
}

/* ---------------- 快路径：装载期展开 ----------------
 * arena 同时承载 q4 展开结果与 F32（norm）副本，因此展开完成后
 * 调用方即可释放 flash 原图（12.26MB），避免 PSRAM 超预算。 */

size_t km_fast_bytes(const km_model_t *m) {
    size_t off = 0;
    for (uint32_t i = 0; i < m->arch.n_tensors; i++) {
        const km_tensor_t *t = &m->tensors[i];
        /* MoE 模型下 tok_embed 不展开进 PSRAM（每 token 只读一行，流式读即可）；
         * 但 tied_embed=1 时 tok_embed 同时是输出头（全 vocab 扫描），必须展开 */
        if (m->arch.n_expert > 0 && !m->arch.tied_embed && strcmp(t->name, "tok_embed") == 0)
            continue;
        /* 三值查表常驻 flash，逐行随机读，不展开进 PSRAM */
        if (t->dtype == KM_DT_TERNARY)
            continue;
        off = align_up(off, 64);
        if (t->dtype == KM_DT_Q4) {
            uint32_t nb = t->n_rows * KM_NBLK(t->n_cols);        /* 行对齐块数 */
            uint32_t n  = nb * 32u;                              /* 展开后 int8 数 */
            off += align_up(n, 64);                              /* int8 q */
            off = align_up(off, 64);
            off += align_up((size_t)nb * 2, 64);                 /* int16 d */
        } else {
            off += align_up((size_t)t->n_rows * t->n_cols * 4, 64);  /* f32 副本 */
        }
    }
    return off;
}

/* m->base 直读的默认读取器 */
typedef struct { const uint8_t *base; } km_mem_ctx_t;
static int km_mem_read(void *ctx, uint32_t off, void *dst, uint32_t len) {
    memcpy(dst, ((km_mem_ctx_t *)ctx)->base + off, len);
    return 0;
}

void km_fast_build_ex(const km_model_t *m, km_fast_t *f,
                      uint8_t *arena, size_t cap,
                      km_read_fn read_fn, void *ctx,
                      uint8_t *stage, size_t stage_cap) {
    km_mem_ctx_t memctx = { m->base };
    if (!read_fn) { read_fn = km_mem_read; ctx = &memctx; }

    size_t off = 0;
    for (uint32_t i = 0; i < m->arch.n_tensors; i++) {
        const km_tensor_t *t = &m->tensors[i];
        f->t[i].q = NULL;
        f->t[i].d = NULL;
        f->t[i].f32 = NULL;
        f->t[i].n = 0;
        f->t[i].is_q4 = 0;

        /* MoE 模型下 tok_embed 不展开进 PSRAM，decode 时流式按行读；
         * 但 tied_embed=1 时 tok_embed 同时是输出头，必须展开 */
        if (m->arch.n_expert > 0 && !m->arch.tied_embed && strcmp(t->name, "tok_embed") == 0)
            continue;
        /* 三值查表常驻 flash，逐行随机读，不展开 */
        if (t->dtype == KM_DT_TERNARY)
            continue;

        if (t->dtype != KM_DT_Q4) {
            uint32_t n = t->n_rows * t->n_cols;
            off = align_up(off, 64);
            float *dst = (float *)(arena + off);
            off += align_up((size_t)n * 4, 64);
            if (off > cap) return;
            if (read_fn(ctx, t->offset, dst, (uint32_t)((size_t)n * 4)) != 0) return;
            f->t[i].f32 = dst;
            f->t[i].n = n;
            continue;
        }

        uint32_t nb = t->n_rows * KM_NBLK(t->n_cols);
        uint32_t n  = nb * 32u;
        off = align_up(off, 64);
        int8_t *q = (int8_t *)(arena + off);
        off += align_up(n, 64);
        off = align_up(off, 64);
        int16_t *d = (int16_t *)(arena + off);
        off += align_up((size_t)nb * 2, 64);
        if (off > cap) return;

        /* int16 定点 scale（per-tensor 归一化）：d_fp32 = d_int16 * d_scale。
         * 先求本张量 max_d，再量化；热循环用硬件 fcvt.s.w 转 fp32（1 条），
         * 免去 fp16 的 12 条软件转换。 */
        uint32_t bpc = (uint32_t)(stage_cap / 18u);
        if (bpc == 0) return;
        float max_d = 0.0f;
        for (uint32_t b0 = 0; b0 < nb; b0 += bpc) {
            uint32_t cnt = nb - b0;
            if (cnt > bpc) cnt = bpc;
            if (read_fn(ctx, t->offset + (size_t)b0 * 18u, stage, cnt * 18u) != 0) return;
            for (uint32_t k = 0; k < cnt; k++) {
                const uint8_t *blk = stage + (size_t)k * 18u;
                float df = km_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
                if (df > max_d) max_d = df;
            }
        }
        float d_scale = max_d / 32767.0f;
        if (d_scale < 1e-30f) d_scale = 1.0f;

        for (uint32_t b0 = 0; b0 < nb; b0 += bpc) {
            uint32_t cnt = nb - b0;
            if (cnt > bpc) cnt = bpc;
            if (read_fn(ctx, t->offset + (size_t)b0 * 18u, stage, cnt * 18u) != 0) return;
            for (uint32_t k = 0; k < cnt; k++) {
                uint32_t b = b0 + k;
                const uint8_t *blk = stage + (size_t)k * 18u;
                float df = km_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
                d[b] = (int16_t)(df / d_scale + 0.5f);
                int8_t *dst = q + (size_t)b * 32u;
                for (uint32_t kk = 0; kk < 32u; kk++) {
                    uint8_t byte = blk[2 + (kk >> 1)];
                    int nib = (kk & 1) ? (byte >> 4) : (byte & 0x0F);
                    dst[kk] = (int8_t)(nib - 8);
                }
            }
        }
        f->t[i].q = q;
        f->t[i].d = d;
        f->t[i].d_scale = d_scale;
        f->t[i].n = n;
        f->t[i].is_q4 = 1;
    }
}

void km_fast_build(const km_model_t *m, km_fast_t *f, uint8_t *arena, size_t cap) {
    static uint8_t stage[4096];
    km_fast_build_ex(m, f, arena, cap, NULL, NULL, stage, sizeof(stage));
}

/* ---------------- 算子 ---------------- */

/* 取快路径张量第 row 行的 n_cols 个元素，反量化为 fp32。
 * 行对齐后每块属于同一行；尾部填充块写 0（保证 x 的 padding 区恒为 0）。 */
static void q8_row_get(const km_ft_t *ft, int n_cols, int row, float *out) {
    const int nblk = (n_cols + 31) >> 5;
    const int8_t   *qr = ft->q + (size_t)row * (size_t)(nblk << 5);
    const int16_t  *dr = ft->d + (size_t)row * nblk;
    for (int b = 0; b < nblk; b++) {
        float dsc = (float)dr[b] * ft->d_scale;
        const int8_t *qq = qr + (b << 5);
        float *oo = out + (b << 5);
        int cnt = n_cols - (b << 5);
        if (cnt > 32) cnt = 32;
        for (int j = 0; j < cnt; j++) oo[j] = (float)qq[j] * dsc;
        for (int j = cnt; j < 32; j++) oo[j] = 0.0f;
    }
}

/* 三值查表行解码（base-3 打包，5 trit/字节）已内联进 PLE 底部合并循环，
 * 避免暂存 nl*pd 个 float（可能超过 dim_p 的 scratch 缓冲）。 */

/* y[r] = dot(W[r,:], x)，W 为展开后的 int8 + int16 定点 scale（d_int16 * d_scale）。
 * PIE 定点路径（w8a16）：先把 fp32 激活 x 量化成 int16（per-vector scale S），
 * 每行用 PIE 分块点积 + int16 scale 累加，最后乘 1/S * d_scale 还原。
 * xq 为调用方提供的 int16 缓冲（长度 ≥ align32(n_cols)，16 字节对齐）。 */
static void gemv_q8(const km_ft_t *ft, int n_rows, int n_cols,
                    const float *x, int16_t *xq, float *y) {
    const int nblk  = (n_cols + 31) >> 5;
    const int align = nblk << 5;
    const int8_t   *q = ft->q;
    const int16_t  *d = ft->d;

    /* 量化 x → xq（per-vector scale，round-half-away-from-zero） */
    float maxa = 0.0f;
    for (int j = 0; j < n_cols; j++) {
        float ax = x[j] < 0.0f ? -x[j] : x[j];
        if (ax > maxa) maxa = ax;
    }
    float S = maxa > 1e-6f ? 32767.0f / maxa : 1.0f;
    for (int j = 0; j < n_cols; j++) {
        xq[j] = (int16_t)(x[j] * S + (x[j] >= 0.0f ? 0.5f : -0.5f));
    }
    for (int j = n_cols; j < align; j++) xq[j] = 0;

    const float coeff = (1.0f / S) * ft->d_scale;
    for (int r = 0; r < n_rows; r++) {
        y[r] = pie_gemv_row(q + (size_t)r * align, xq,
                            d + (size_t)r * nblk, nblk, 0) * coeff;
    }
}

static void rms_norm(const float *x, const float *w, int n, float eps, float *out) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss /= (float)n;
    float inv = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

/* 与 HF 一致：out1 = x1*cos - x2*sin ; out2 = x2*cos + x1*sin */
static void rope_apply(float *vec, int n_heads, int hd, int pos, float theta) {
    int half = hd / 2;
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + (size_t)h * hd;
        for (int i = 0; i < half; i++) {
            float inv = 1.0f / powf(theta, (float)(2 * i) / (float)hd);
            float ang = (float)pos * inv;
            float cv = cosf(ang), sv = sinf(ang);
            float a = v[i], b = v[i + half];
            v[i]        = a * cv - b * sv;
            v[i + half] = b * cv + a * sv;
        }
    }
}

/* ---------------- state ---------------- */

/* scratch 布局：
 *   [float 区] x[dim_p] h[dim_p] q[dim_p] kk[kvw] vv[kvw] ao[dim_p] g[ffn] u[ffn] act[ffn_p]
 *   [int16 区] xq[align32(max(dim,ffn))]  ← PIE GEMV 的激活量化缓冲（16 字节对齐）
 * 其中 dim_p/ffn_p 为 align32，供 GEMV 读取尾部 padding（权重侧为 0）。 */
static size_t scratch_floats(const km_arch_t *a) {
    uint32_t dim_p = KM_ALIGN32(a->dim);
    uint32_t ffn_p = KM_ALIGN32(a->ffn_dim);
    uint32_t kvw   = a->n_kv_heads * a->head_dim;
    uint32_t pd    = a->ple_dim;
    uint32_t pd_p  = KM_ALIGN32(pd > 0 ? pd : 1u);
    size_t ple_n   = (size_t)a->n_layers * pd;
    /* 追加 PLE 分支缓冲：ple[nl*pd] + pleg[pd_p] + pleo[dim_p]（pd=0 时退化为常量小缓冲） */
    return (size_t)dim_p * 4 + (size_t)kvw * 2 + (size_t)a->ffn_dim * 2 + ffn_p
         + ple_n + (size_t)pd_p + (size_t)dim_p;
}

size_t km_scratch_bytes(const km_arch_t *a) {
    size_t fbytes = scratch_floats(a) * sizeof(float);
    uint32_t xq_n = KM_ALIGN32(a->ffn_dim > a->dim ? a->ffn_dim : a->dim);
    fbytes = (fbytes + 15u) & ~(size_t)15u;          /* xq 需 16 字节对齐 */
    return fbytes + (size_t)xq_n * sizeof(int16_t);
}

size_t km_state_bytes(const km_arch_t *a, uint32_t n_ctx) {
    size_t kv = (size_t)a->n_layers * n_ctx * a->n_kv_heads * a->head_dim;
    return kv * 2 * sizeof(float) + km_scratch_bytes(a);
}

void km_state_init(km_state_t *st, const km_arch_t *a, uint32_t n_ctx,
                   float *k_cache, float *v_cache, float *scratch) {
    st->n_ctx = n_ctx;
    st->k_cache = k_cache;
    st->v_cache = v_cache;
    st->scratch = scratch;
    st->pos = 0;
    size_t kv = (size_t)a->n_layers * n_ctx * a->n_kv_heads * a->head_dim;
    memset(k_cache, 0, kv * sizeof(float));
    memset(v_cache, 0, kv * sizeof(float));
    /* 清零工作区：其中对齐 padding 区此后不会被写入（rms_norm/gemv 只写逻辑长度），
     * 保持为 0 才能让 GEMV 的尾部整块读取保持数值正确。 */
    memset(scratch, 0, km_scratch_bytes(a));
}

/* ---------------- decode 单步 ---------------- */

int km_decode_step(const km_model_t *m, const km_fast_t *f, km_state_t *st,
                   int token, float *out_logit,
                   int *out_ids, float *out_vals, int topk,
                   km_read_fn read_fn, void *ctx,
                   uint8_t *stage, size_t stage_cap) {
    const km_arch_t *a = &m->arch;
    const int dim = (int)a->dim;
    const int nh = (int)a->n_heads;
    const int nkv = (int)a->n_kv_heads;
    const int hd = (int)a->head_dim;
    const int ffn = (int)a->ffn_dim;
    const int kvw = nkv * hd;
    const int kv_rep = nh / nkv;
    const uint32_t pos = st->pos;

#if REP_PENALTY
    if (pos == 0) rep_clear();
    rep_mark(token);
#endif

    const int dim_p = (dim + 31) & ~31;   /* 与 km_scratch_bytes 布局一致 */
    uint8_t *sb = (uint8_t *)st->scratch;

    float *x  = (float *)sb;
    float *h  = x + dim_p;
    float *q  = h + dim_p;
    float *kk = q + dim_p;
    float *vv = kk + kvw;
    float *ao = vv + kvw;
    float *g  = ao + dim_p;
    float *u  = g + ffn;
    float *act= u + ffn;                  /* 尾部长度 = align32(ffn)，由 scratch 分配保证 */

    const uint32_t ffn_p = KM_ALIGN32(ffn);
    const uint32_t pd    = a->ple_dim;
    const uint32_t pd_p  = KM_ALIGN32(pd > 0 ? pd : 1u);
    float *ple  = act + ffn_p;                            /* [nl*pd] 每 token PLE 输入 */
    float *pleg = ple + (size_t)a->n_layers * pd;         /* [pd_p] gelu/乘积缓冲 */
    float *pleo = pleg + pd_p;                            /* [dim_p] ple_proj 输出 */

    size_t fbytes = scratch_floats(a) * sizeof(float);
    int16_t *xq = (int16_t *)(sb + ((fbytes + 15u) & ~(size_t)15u));

    int idx = km_find(m, "tok_embed");
    if (idx < 0) return -1;
    const km_ft_t *emb = &f->t[idx];
    char nm[32];

    /* 输入嵌入：MoE 模型 embed 不展开（q==NULL），流式按行读；稠密则从展开权重读 */
    if (emb->q != NULL) {
        q8_row_get(emb, dim, token, x);
    } else {
        if (!read_fn) return -1;
        if (km_embed_row(m, &m->tensors[idx], token, dim, x,
                         read_fn, ctx, stage, stage_cap) != 0) return -1;
    }

    /* --- PLE 底部：ple = (RMSNorm(ple_model_proj(x)·dim^-0.5) + table·pd^0.5) · 2^-0.5 --- */
    if (pd > 0) {
        const int ple_rows = (int)a->n_layers * (int)pd;
        snprintf(nm, sizeof(nm), "ple_model_proj");
        idx = km_find(m, nm);
        if (idx < 0) return -1;
        gemv_q8(&f->t[idx], ple_rows, dim, x, xq, ple);

        const float inv_dim = 1.0f / sqrtf((float)dim);
        for (int i = 0; i < ple_rows; i++) ple[i] *= inv_dim;

        snprintf(nm, sizeof(nm), "ple_proj_norm");
        idx = km_find(m, nm);
        if (idx < 0 || !f->t[idx].f32) return -1;
        const float *pn = f->t[idx].f32;
        for (int l = 0; l < (int)a->n_layers; l++)
            rms_norm(ple + (size_t)l * pd, pn, (int)pd, a->norm_eps, ple + (size_t)l * pd);

        snprintf(nm, sizeof(nm), "ple_table");
        idx = km_find(m, nm);
        if (idx < 0) return -1;
        const km_tensor_t *pt = &m->tensors[idx];
        const int row_bytes = ((int)pt->n_cols + 4) / 5;
        if (!read_fn || row_bytes > (int)stage_cap) return -1;
        if (read_fn(ctx, pt->offset + (size_t)token * row_bytes, stage,
                    (uint32_t)row_bytes) != 0) return -1;
        const float spd = sqrtf((float)pd);
        const float inv2 = 0.7071067811865475f;   /* 2^-0.5 */
        /* 逐 trit 解码并就地合并（table_width=nl*pd 可能 > dim_p，不能暂存到 pleo） */
        for (int b = 0, oi = 0; b < row_bytes && oi < ple_rows; b++) {
            uint32_t v = stage[b];
            for (int k = 0; k < 5 && oi < ple_rows; k++, v /= 3u) {
                float tv = (float)((int)(v % 3u) - 1) * a->ple_gamma;
                ple[oi] = (ple[oi] + tv * spd) * inv2;
                oi++;
            }
        }
    }
    for (unsigned L = 0; L < (unsigned)a->n_layers; L++) {
        /* --- attention --- */
        snprintf(nm, sizeof(nm), "L%u.in_ln", L);
        idx = km_find(m, nm);
        if (idx < 0 || !f->t[idx].f32) return -1;
        rms_norm(x, f->t[idx].f32, dim, a->norm_eps, h);

        snprintf(nm, sizeof(nm), "L%u.q", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        gemv_q8(&f->t[idx], dim, dim, h, xq, q);

        snprintf(nm, sizeof(nm), "L%u.k", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        gemv_q8(&f->t[idx], kvw, dim, h, xq, kk);

        snprintf(nm, sizeof(nm), "L%u.v", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        gemv_q8(&f->t[idx], kvw, dim, h, xq, vv);

        rope_apply(q, nh, hd, (int)pos, a->rope_theta);
        rope_apply(kk, nkv, hd, (int)pos, a->rope_theta);

        float *kc = st->k_cache + ((size_t)L * st->n_ctx + pos) * kvw;
        float *vc = st->v_cache + ((size_t)L * st->n_ctx + pos) * kvw;
        memcpy(kc, kk, sizeof(float) * kvw);
        memcpy(vc, vv, sizeof(float) * kvw);

        const float scale = 1.0f / sqrtf((float)hd);
        const uint32_t npos = pos + 1;
        for (int hh = 0; hh < nh; hh++) {
            const float *qh = q + (size_t)hh * hd;
            const int kvh = hh / kv_rep;
            float mx = -1e30f;
            for (uint32_t t = 0; t < npos; t++) {
                const float *kt = st->k_cache + ((size_t)L * st->n_ctx + t) * kvw + (size_t)kvh * hd;
                float s = 0.0f;
                for (int d = 0; d < hd; d++) s += qh[d] * kt[d];
                s *= scale;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            float *oph = ao + (size_t)hh * hd;
            for (int d = 0; d < hd; d++) oph[d] = 0.0f;
            for (uint32_t t = 0; t < npos; t++) {
                const float *kt = st->k_cache + ((size_t)L * st->n_ctx + t) * kvw + (size_t)kvh * hd;
                float s = 0.0f;
                for (int d = 0; d < hd; d++) s += qh[d] * kt[d];
                s = expf(s * scale - mx);
                sum += s;
                const float *vt = st->v_cache + ((size_t)L * st->n_ctx + t) * kvw + (size_t)kvh * hd;
                for (int d = 0; d < hd; d++) oph[d] += s * vt[d];
            }
            float inv = 1.0f / sum;
            for (int d = 0; d < hd; d++) oph[d] *= inv;
        }

        snprintf(nm, sizeof(nm), "L%u.o", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        gemv_q8(&f->t[idx], dim, dim, ao, xq, h);   /* 复用 h 作输出 */

        for (int i = 0; i < dim; i++) x[i] += h[i];

        /* --- MLP (SwiGLU / MoE) --- */
        snprintf(nm, sizeof(nm), "L%u.post_ln", L);
        idx = km_find(m, nm); if (idx < 0 || !f->t[idx].f32) return -1;
        rms_norm(x, f->t[idx].f32, dim, a->norm_eps, h);

        if (a->n_expert > 0) {
            /* MoE：router(f32) → softmax → top-k 硬选择 → 专家加权累加 */
            const int ne = (int)a->n_expert;
            if (ne > 8) return -1;   /* 栈上固定缓冲上限，防御畸形文件 */
            int tk = (int)a->top_k;
            if (tk > ne) tk = ne;
            float rlog[8], p[8];
            snprintf(nm, sizeof(nm), "L%u.router", L);
            idx = km_find(m, nm); if (idx < 0 || !f->t[idx].f32) return -1;
            const float *rw = f->t[idx].f32;
            float mx = -1e30f;
            for (int e = 0; e < ne; e++) {
                const float *row = rw + (size_t)e * dim;
                float s = 0.0f;
                for (int j = 0; j < dim; j++) s += row[j] * h[j];
                rlog[e] = s;
                if (s > mx) mx = s;
            }
            float psum = 0.0f;
            for (int e = 0; e < ne; e++) { p[e] = expf(rlog[e] - mx); psum += p[e]; }
            for (int e = 0; e < ne; e++) p[e] /= psum;

            /* top-k 选择（n_expert 小，直接逐个挑最大） */
            int sel[8]; float wsel[8]; int chosen[8] = {0};
            for (int k = 0; k < tk; k++) {
                int bi = -1; float bv = -1e30f;
                for (int e = 0; e < ne; e++) {
                    if (chosen[e]) continue;
                    if (p[e] > bv) { bv = p[e]; bi = e; }
                }
                chosen[bi] = 1; sel[k] = bi; wsel[k] = p[bi];
            }
            float wsum = 0.0f;
            for (int k = 0; k < tk; k++) wsum += wsel[k];
            for (int k = 0; k < tk; k++) wsel[k] /= wsum;

            float *moe_out = ao;   /* 复用 attention 已用完的 ao 作累加缓冲 */
            float *eout = q;       /* 复用 q 作单专家 down 输出 */
            for (int i = 0; i < dim; i++) moe_out[i] = 0.0f;

            for (int k = 0; k < tk; k++) {
                const int e = sel[k];
                snprintf(nm, sizeof(nm), "L%u.e%d.gate", L, e);
                idx = km_find(m, nm); if (idx < 0) return -1;
                gemv_q8(&f->t[idx], ffn, dim, h, xq, g);

                snprintf(nm, sizeof(nm), "L%u.e%d.up", L, e);
                idx = km_find(m, nm); if (idx < 0) return -1;
                gemv_q8(&f->t[idx], ffn, dim, h, xq, u);

                for (int i = 0; i < ffn; i++) {
                    float sg = g[i] / (1.0f + expf(-g[i]));
                    act[i] = sg * u[i];
                }

                snprintf(nm, sizeof(nm), "L%u.e%d.down", L, e);
                idx = km_find(m, nm); if (idx < 0) return -1;
                gemv_q8(&f->t[idx], dim, ffn, act, xq, eout);
                for (int i = 0; i < dim; i++) moe_out[i] += wsel[k] * eout[i];
            }
            for (int i = 0; i < dim; i++) x[i] += moe_out[i];
        } else {
            snprintf(nm, sizeof(nm), "L%u.gate", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            gemv_q8(&f->t[idx], ffn, dim, h, xq, g);

            snprintf(nm, sizeof(nm), "L%u.up", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            gemv_q8(&f->t[idx], ffn, dim, h, xq, u);

            for (int i = 0; i < ffn; i++) {
                float sg = g[i] / (1.0f + expf(-g[i]));
                act[i] = sg * u[i];
            }

            snprintf(nm, sizeof(nm), "L%u.down", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            gemv_q8(&f->t[idx], dim, ffn, act, xq, h);

            for (int i = 0; i < dim; i++) x[i] += h[i];
        }

        /* --- PLE 分支（attention + ffn 之后） --- */
        if (pd > 0) {
            snprintf(nm, sizeof(nm), "L%u.ple_gate", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            gemv_q8(&f->t[idx], (int)pd, dim, x, xq, pleg);
            for (int i = 0; i < (int)pd; i++) {
                float v = pleg[i];
                pleg[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
            }
            float *pslice = ple + (size_t)L * pd;
            for (int i = 0; i < (int)pd; i++) pleg[i] *= pslice[i];

            snprintf(nm, sizeof(nm), "L%u.ple_proj", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            gemv_q8(&f->t[idx], dim, (int)pd, pleg, xq, pleo);

            snprintf(nm, sizeof(nm), "L%u.ple_norm", L);
            idx = km_find(m, nm); if (idx < 0 || !f->t[idx].f32) return -1;
            rms_norm(pleo, f->t[idx].f32, dim, a->norm_eps, pleo);
            for (int i = 0; i < dim; i++) x[i] += pleo[i];
        }
    }

    idx = km_find(m, "final_norm");
    if (idx < 0 || !f->t[idx].f32) return -1;
    rms_norm(x, f->t[idx].f32, dim, a->norm_eps, h);

    /* lm_head（tied=1 复用 tok_embed；tied=0 用独立 lm_head）→ 流式 argmax */
    const int vocab = (int)a->vocab_size;
    float best = -1e30f;
    int best_id = 0;
    if (topk > 8) topk = 8;
    for (int j = 0; j < topk; j++) { out_vals[j] = -1e30f; out_ids[j] = -1; }
    {
        const km_ft_t *oh = emb;   /* 输出头权重：默认 tied 复用 embed */
        if (!a->tied_embed) {
            idx = km_find(m, "lm_head");
            if (idx < 0) return -1;
            oh = &f->t[idx];
        }
        const int8_t   *q = oh->q;
        const int16_t  *d = oh->d;
        const int nblk  = (dim + 31) >> 5;
        const int align = nblk << 5;

        /* 量化 h → xq（输出头输入，dim 维，per-vector scale） */
        float maxa = 0.0f;
        for (int j = 0; j < dim; j++) {
            float ax = h[j] < 0.0f ? -h[j] : h[j];
            if (ax > maxa) maxa = ax;
        }
        float S = maxa > 1e-6f ? 32767.0f / maxa : 1.0f;
        for (int j = 0; j < dim; j++)
            xq[j] = (int16_t)(h[j] * S + (h[j] >= 0.0f ? 0.5f : -0.5f));
        for (int j = dim; j < align; j++) xq[j] = 0;
        const float coeff = (1.0f / S) * oh->d_scale;

        for (int v = 0; v < vocab; v++) {
            float acc = pie_gemv_row(q + (size_t)v * align, xq,
                                     d + (size_t)v * nblk, nblk, 0) * coeff;
#if REP_PENALTY
            acc = rep_apply(acc, v);
#endif
            if (acc > best) { best = acc; best_id = v; }
            if (topk > 0 && acc > out_vals[topk - 1]) {
                int j = topk - 1;
                while (j > 0 && acc > out_vals[j - 1]) {
                    out_vals[j] = out_vals[j - 1];
                    out_ids[j]  = out_ids[j - 1];
                    j--;
                }
                out_vals[j] = acc;
                out_ids[j]  = v;
            }
        }
    }

    st->pos = pos + 1;
    if (out_logit) *out_logit = best;
    return best_id;
}

/* ---------------- 决策头前向（单次前向 + 分类，不碰 lm_head） ----------------
 * 复用 decode 的 transformer 部分（embed → PLE → layers → final_norm），
 * 对整段 token 序列逐 token 跑，mean pooling last hidden，最后接 cls_head（f32）。
 * 独立分支：不影响 km_decode_step 的生成路径。
 */
int km_cls_forward(const km_model_t *m, const km_fast_t *f, km_state_t *st,
                   const int *tokens, int n_tokens, int n_cls,
                   int *out_id, float *out_logits,
                   km_read_fn read_fn, void *ctx, uint8_t *stage, size_t stage_cap) {
    const km_arch_t *a = &m->arch;
    const int dim = (int)a->dim;
    const int nh = (int)a->n_heads;
    const int nkv = (int)a->n_kv_heads;
    const int hd = (int)a->head_dim;
    const int ffn = (int)a->ffn_dim;
    const int kvw = nkv * hd;
    const int kv_rep = nh / nkv;
    const uint32_t nl = a->n_layers;
    const uint32_t pd = a->ple_dim;

    if (dim > 512 || n_tokens <= 0) return -1;

    const int dim_p = (dim + 31) & ~31;
    const uint32_t ffn_p = KM_ALIGN32(ffn);
    const uint32_t pd_p  = KM_ALIGN32(pd > 0 ? pd : 1u);
    uint8_t *sb = (uint8_t *)st->scratch;

    float *x  = (float *)sb;
    float *h  = x + dim_p;
    float *q  = h + dim_p;
    float *kk = q + dim_p;
    float *vv = kk + kvw;
    float *ao = vv + kvw;
    float *g  = ao + dim_p;
    float *u  = g + ffn;
    float *act= u + ffn;
    float *ple  = act + ffn_p;
    float *pleg = ple + (size_t)nl * pd;
    float *pleo = pleg + pd_p;

    size_t fbytes = scratch_floats(a) * sizeof(float);
    int16_t *xq = (int16_t *)(sb + ((fbytes + 15u) & ~(size_t)15u));

    /* 决策头是独立前向：从头清零 KV cache + scratch */
    memset(st->k_cache, 0, (size_t)st->n_ctx * nl * nkv * hd * sizeof(float));
    memset(st->v_cache, 0, (size_t)st->n_ctx * nl * nkv * hd * sizeof(float));
    memset(st->scratch, 0, km_scratch_bytes(a));

    int idx = km_find(m, "tok_embed");
    if (idx < 0) return -1;
    const km_ft_t *emb = &f->t[idx];
    char nm[32];

    static float mean_pooled[512];
    for (int i = 0; i < dim; i++) mean_pooled[i] = 0.0f;

    for (int pos = 0; pos < n_tokens; pos++) {
        const int token = tokens[pos];

        /* 输入嵌入 */
        if (emb->q != NULL) {
            q8_row_get(emb, dim, token, x);
        } else {
            if (!read_fn) return -1;
            if (km_embed_row(m, &m->tensors[idx], token, dim, x,
                             read_fn, ctx, stage, stage_cap) != 0) return -1;
        }

        /* PLE 底部 */
        if (pd > 0) {
            const int ple_rows = (int)nl * (int)pd;
            snprintf(nm, sizeof(nm), "ple_model_proj");
            idx = km_find(m, nm);
            if (idx < 0) return -1;
            gemv_q8(&f->t[idx], ple_rows, dim, x, xq, ple);

            const float inv_dim = 1.0f / sqrtf((float)dim);
            for (int i = 0; i < ple_rows; i++) ple[i] *= inv_dim;

            snprintf(nm, sizeof(nm), "ple_proj_norm");
            idx = km_find(m, nm);
            if (idx < 0 || !f->t[idx].f32) return -1;
            const float *pn = f->t[idx].f32;
            for (uint32_t l = 0; l < nl; l++)
                rms_norm(ple + (size_t)l * pd, pn, (int)pd, a->norm_eps, ple + (size_t)l * pd);

            snprintf(nm, sizeof(nm), "ple_table");
            idx = km_find(m, nm);
            if (idx < 0) return -1;
            const km_tensor_t *pt = &m->tensors[idx];
            const int row_bytes = ((int)pt->n_cols + 4) / 5;
            if (!read_fn || row_bytes > (int)stage_cap) return -1;
            if (read_fn(ctx, pt->offset + (size_t)token * row_bytes, stage,
                        (uint32_t)row_bytes) != 0) return -1;
            const float spd = sqrtf((float)pd);
            const float inv2 = 0.7071067811865475f;
            for (int b = 0, oi = 0; b < row_bytes && oi < ple_rows; b++) {
                uint32_t v = stage[b];
                for (int k = 0; k < 5 && oi < ple_rows; k++, v /= 3u) {
                    float tv = (float)((int)(v % 3u) - 1) * a->ple_gamma;
                    ple[oi] = (ple[oi] + tv * spd) * inv2;
                    oi++;
                }
            }
        }

        for (unsigned L = 0; L < nl; L++) {
            /* --- attention --- */
            snprintf(nm, sizeof(nm), "L%u.in_ln", L);
            idx = km_find(m, nm);
            if (idx < 0 || !f->t[idx].f32) return -1;
            rms_norm(x, f->t[idx].f32, dim, a->norm_eps, h);

            snprintf(nm, sizeof(nm), "L%u.q", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            gemv_q8(&f->t[idx], dim, dim, h, xq, q);

            snprintf(nm, sizeof(nm), "L%u.k", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            gemv_q8(&f->t[idx], kvw, dim, h, xq, kk);

            snprintf(nm, sizeof(nm), "L%u.v", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            gemv_q8(&f->t[idx], kvw, dim, h, xq, vv);

            rope_apply(q, nh, hd, pos, a->rope_theta);
            rope_apply(kk, nkv, hd, pos, a->rope_theta);

            float *kc = st->k_cache + ((size_t)L * st->n_ctx + pos) * kvw;
            float *vc = st->v_cache + ((size_t)L * st->n_ctx + pos) * kvw;
            memcpy(kc, kk, sizeof(float) * kvw);
            memcpy(vc, vv, sizeof(float) * kvw);

            const float scale = 1.0f / sqrtf((float)hd);
            const uint32_t npos = (uint32_t)pos + 1;
            for (int hh = 0; hh < nh; hh++) {
                const float *qh = q + (size_t)hh * hd;
                const int kvh = hh / kv_rep;
                float mx = -1e30f;
                for (uint32_t t = 0; t < npos; t++) {
                    const float *kt = st->k_cache + ((size_t)L * st->n_ctx + t) * kvw + (size_t)kvh * hd;
                    float s = 0.0f;
                    for (int d = 0; d < hd; d++) s += qh[d] * kt[d];
                    s *= scale;
                    if (s > mx) mx = s;
                }
                float sum = 0.0f;
                float *oph = ao + (size_t)hh * hd;
                for (int d = 0; d < hd; d++) oph[d] = 0.0f;
                for (uint32_t t = 0; t < npos; t++) {
                    const float *kt = st->k_cache + ((size_t)L * st->n_ctx + t) * kvw + (size_t)kvh * hd;
                    float s = 0.0f;
                    for (int d = 0; d < hd; d++) s += qh[d] * kt[d];
                    s = expf(s * scale - mx);
                    sum += s;
                    const float *vt = st->v_cache + ((size_t)L * st->n_ctx + t) * kvw + (size_t)kvh * hd;
                    for (int d = 0; d < hd; d++) oph[d] += s * vt[d];
                }
                float inv = 1.0f / sum;
                for (int d = 0; d < hd; d++) oph[d] *= inv;
            }

            snprintf(nm, sizeof(nm), "L%u.o", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            gemv_q8(&f->t[idx], dim, dim, ao, xq, h);
            for (int i = 0; i < dim; i++) x[i] += h[i];

            /* --- MLP (SwiGLU / MoE) --- */
            snprintf(nm, sizeof(nm), "L%u.post_ln", L);
            idx = km_find(m, nm); if (idx < 0 || !f->t[idx].f32) return -1;
            rms_norm(x, f->t[idx].f32, dim, a->norm_eps, h);

            if (a->n_expert > 0) {
                const int ne = (int)a->n_expert;
                if (ne > 8) return -1;
                int tk = (int)a->top_k;
                if (tk > ne) tk = ne;
                float rlog[8], p[8];
                snprintf(nm, sizeof(nm), "L%u.router", L);
                idx = km_find(m, nm); if (idx < 0 || !f->t[idx].f32) return -1;
                const float *rw = f->t[idx].f32;
                float mx = -1e30f;
                for (int e = 0; e < ne; e++) {
                    const float *row = rw + (size_t)e * dim;
                    float s = 0.0f;
                    for (int j = 0; j < dim; j++) s += row[j] * h[j];
                    rlog[e] = s;
                    if (s > mx) mx = s;
                }
                float psum = 0.0f;
                for (int e = 0; e < ne; e++) { p[e] = expf(rlog[e] - mx); psum += p[e]; }
                for (int e = 0; e < ne; e++) p[e] /= psum;

                int sel[8]; float wsel[8]; int chosen[8] = {0};
                for (int k = 0; k < tk; k++) {
                    int bi = -1; float bv = -1e30f;
                    for (int e = 0; e < ne; e++) {
                        if (chosen[e]) continue;
                        if (p[e] > bv) { bv = p[e]; bi = e; }
                    }
                    chosen[bi] = 1; sel[k] = bi; wsel[k] = p[bi];
                }
                float wsum = 0.0f;
                for (int k = 0; k < tk; k++) wsum += wsel[k];
                for (int k = 0; k < tk; k++) wsel[k] /= wsum;

                float *moe_out = ao;
                float *eout = q;
                for (int i = 0; i < dim; i++) moe_out[i] = 0.0f;
                for (int k = 0; k < tk; k++) {
                    const int e = sel[k];
                    snprintf(nm, sizeof(nm), "L%u.e%d.gate", L, e);
                    idx = km_find(m, nm); if (idx < 0) return -1;
                    gemv_q8(&f->t[idx], ffn, dim, h, xq, g);

                    snprintf(nm, sizeof(nm), "L%u.e%d.up", L, e);
                    idx = km_find(m, nm); if (idx < 0) return -1;
                    gemv_q8(&f->t[idx], ffn, dim, h, xq, u);

                    for (int i = 0; i < ffn; i++) {
                        float sg = g[i] / (1.0f + expf(-g[i]));
                        act[i] = sg * u[i];
                    }

                    snprintf(nm, sizeof(nm), "L%u.e%d.down", L, e);
                    idx = km_find(m, nm); if (idx < 0) return -1;
                    gemv_q8(&f->t[idx], dim, ffn, act, xq, eout);
                    for (int i = 0; i < dim; i++) moe_out[i] += wsel[k] * eout[i];
                }
                for (int i = 0; i < dim; i++) x[i] += moe_out[i];
            } else {
                snprintf(nm, sizeof(nm), "L%u.gate", L);
                idx = km_find(m, nm); if (idx < 0) return -1;
                gemv_q8(&f->t[idx], ffn, dim, h, xq, g);

                snprintf(nm, sizeof(nm), "L%u.up", L);
                idx = km_find(m, nm); if (idx < 0) return -1;
                gemv_q8(&f->t[idx], ffn, dim, h, xq, u);

                for (int i = 0; i < ffn; i++) {
                    float sg = g[i] / (1.0f + expf(-g[i]));
                    act[i] = sg * u[i];
                }

                snprintf(nm, sizeof(nm), "L%u.down", L);
                idx = km_find(m, nm); if (idx < 0) return -1;
                gemv_q8(&f->t[idx], dim, ffn, act, xq, h);
                for (int i = 0; i < dim; i++) x[i] += h[i];
            }

            /* --- PLE 分支 --- */
            if (pd > 0) {
                snprintf(nm, sizeof(nm), "L%u.ple_gate", L);
                idx = km_find(m, nm); if (idx < 0) return -1;
                gemv_q8(&f->t[idx], (int)pd, dim, x, xq, pleg);
                for (int i = 0; i < (int)pd; i++) {
                    float v = pleg[i];
                    pleg[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
                }
                float *pslice = ple + (size_t)L * pd;
                for (int i = 0; i < (int)pd; i++) pleg[i] *= pslice[i];

                snprintf(nm, sizeof(nm), "L%u.ple_proj", L);
                idx = km_find(m, nm); if (idx < 0) return -1;
                gemv_q8(&f->t[idx], dim, (int)pd, pleg, xq, pleo);

                snprintf(nm, sizeof(nm), "L%u.ple_norm", L);
                idx = km_find(m, nm); if (idx < 0 || !f->t[idx].f32) return -1;
                rms_norm(pleo, f->t[idx].f32, dim, a->norm_eps, pleo);
                for (int i = 0; i < dim; i++) x[i] += pleo[i];
            }
        }

        /* final_norm → last hidden，累加做 mean pooling */
        idx = km_find(m, "final_norm");
        if (idx < 0 || !f->t[idx].f32) return -1;
        rms_norm(x, f->t[idx].f32, dim, a->norm_eps, h);
        for (int i = 0; i < dim; i++) mean_pooled[i] += h[i];
    }

    const float inv_n = 1.0f / (float)n_tokens;
    for (int i = 0; i < dim; i++) mean_pooled[i] *= inv_n;

    /* cls_head（f32）GEMV + argmax */
    idx = km_find(m, "cls_head");
    if (idx < 0 || !f->t[idx].f32) return -1;
    const float *cw = f->t[idx].f32;
    float best = -1e30f;
    int best_id = 0;
    for (int c = 0; c < n_cls; c++) {
        const float *row = cw + (size_t)c * dim;
        float s = 0.0f;
        for (int i = 0; i < dim; i++) s += row[i] * mean_pooled[i];
        if (out_logits) out_logits[c] = s;
        if (s > best) { best = s; best_id = c; }
    }

    st->pos = n_tokens;
    if (out_id) *out_id = best_id;
    return best_id;
}

/* ================================================================
 * Agent 循环（观察回接）骨架
 *
 * 循环：决策（km_cls_forward）→ 动作分派 → 观察回接（生成 token 拼回
 * context）→ 再决策，直到 stop / max_steps / context 满。
 * 骨架阶段用「全量重算」保证正确性；增量 KV 复用留后续优化。
 * ================================================================ */
int km_agent_loop(const km_model_t *m, const km_fast_t *f, km_state_t *st,
                  const int *init_tokens, int n_init, int n_cls, int max_steps,
                  int max_ctx, int *out_actions, int *out_gen, float *out_logits,
                  km_read_fn read_fn, void *ctx, uint8_t *stage, size_t stage_cap) {
    const km_arch_t *a = &m->arch;
    if (n_init <= 0 || n_cls <= 0 || n_cls > KM_ACT_N) return -1;
    if (max_ctx > 512) max_ctx = 512;
    if (n_init >= max_ctx) return -1;

    static int cbuf[512];
    int n = n_init;
    for (int i = 0; i < n; i++) cbuf[i] = init_tokens[i];

    const size_t kv_bytes = (size_t)st->n_ctx * (size_t)a->n_layers
                          * (size_t)a->n_kv_heads * (size_t)a->head_dim
                          * sizeof(float);

    int steps = 0;
    for (int step = 0; step < max_steps; step++) {
        /* 1. 决策 */
        int action = 0;
        float logits[KM_ACT_N];
        km_cls_forward(m, f, st, cbuf, n, n_cls, &action, logits,
                       read_fn, ctx, stage, stage_cap);
        if (out_actions) out_actions[step] = action;
        if (out_logits) {
            for (int c = 0; c < n_cls; c++) out_logits[step * n_cls + c] = logits[c];
        }
        steps = step + 1;

        /* 2. 动作分派 */
        if (action == KM_ACT_STOP) {
            if (out_gen) out_gen[step] = -1;
            break;
        }

        if (action == KM_ACT_GENERATE) {
            /* 观察回接：全量重算 context，生成下一个 token 作为观察 */
            st->pos = 0;
            memset(st->k_cache, 0, kv_bytes);
            memset(st->v_cache, 0, kv_bytes);
            memset(st->scratch, 0, km_scratch_bytes(a));
            int tok = 0;
            for (int i = 0; i < n; i++) {
                tok = km_decode_step(m, f, st, cbuf[i], NULL, NULL, NULL, 0,
                                     read_fn, ctx, stage, stage_cap);
            }
            if (out_gen) out_gen[step] = tok;
            if (n < max_ctx) cbuf[n++] = tok;
            continue;
        }

        /* route / ask_human：骨架阶段无环境，占位（后续接工具/人工回调） */
        if (out_gen) out_gen[step] = -1;
        break;
    }
    return steps;
}

/* ================================================================
 * 逐层流式 decode（0.1B 分层驻留路径）
 *
 * 整份展开（km_fast_build_ex）需把全部权重展开进 PSRAM，0.1B 模型展开后
 * ~150MB 超出 32MB。流式路径把内存峰值压到「单层最大张量 + KV cache」：
 *   - embed 与输出头按行流式读（不整份展开 tok_embed，其展开后 ~30MB）
 *   - 每层 9 个张量「读入 → 展开 → 计算 → 释放」，复用同一 layer_arena
 * 因每个权重张量都只用一次（gemv/rms_norm 后即弃），arena 只需容纳最大的
 * 单个张量（gate/up/down），而非整层。
 * ================================================================ */

/* 单个张量流式展开到 arena 起始。返回对齐后占用字节数（>0 成功，0 失败）。 */
size_t km_build_tensor(const km_model_t *m, uint32_t idx, km_ft_t *ft,
                       uint8_t *arena, size_t cap,
                       km_read_fn read_fn, void *ctx,
                       uint8_t *stage, size_t stage_cap) {
    const km_tensor_t *t = &m->tensors[idx];
    ft->q = NULL; ft->d = NULL; ft->f32 = NULL;
    ft->n = 0; ft->is_q4 = 0;

    size_t off = 0;
    if (t->dtype != KM_DT_Q4) {
        uint32_t n = t->n_rows * t->n_cols;
        off = align_up(off, 64);
        float *dst = (float *)(arena + off);
        off += align_up((size_t)n * 4, 64);
        if (off > cap) return 0;
        if (read_fn(ctx, t->offset, dst, (uint32_t)((size_t)n * 4)) != 0) return 0;
        ft->f32 = dst;
        ft->n = n;
        return off;
    }

    uint32_t nb = t->n_rows * KM_NBLK(t->n_cols);
    uint32_t n  = nb * 32u;
    off = align_up(off, 64);
    int8_t *q = (int8_t *)(arena + off);
    off += align_up(n, 64);
    off = align_up(off, 64);
    int16_t *d = (int16_t *)(arena + off);
    off += align_up((size_t)nb * 2, 64);
    if (off > cap) return 0;

    uint32_t bpc = (uint32_t)(stage_cap / 18u);
    if (bpc == 0) return 0;
    float max_d = 0.0f;
    for (uint32_t b0 = 0; b0 < nb; b0 += bpc) {
        uint32_t cnt = nb - b0;
        if (cnt > bpc) cnt = bpc;
        if (read_fn(ctx, t->offset + (size_t)b0 * 18u, stage, cnt * 18u) != 0) return 0;
        for (uint32_t k = 0; k < cnt; k++) {
            const uint8_t *blk = stage + (size_t)k * 18u;
            float df = km_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            if (df > max_d) max_d = df;
        }
    }
    float d_scale = max_d / 32767.0f;
    if (d_scale < 1e-30f) d_scale = 1.0f;

    for (uint32_t b0 = 0; b0 < nb; b0 += bpc) {
        uint32_t cnt = nb - b0;
        if (cnt > bpc) cnt = bpc;
        if (read_fn(ctx, t->offset + (size_t)b0 * 18u, stage, cnt * 18u) != 0) return 0;
        for (uint32_t k = 0; k < cnt; k++) {
            uint32_t b = b0 + k;
            const uint8_t *blk = stage + (size_t)k * 18u;
            float df = km_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            d[b] = (int16_t)(df / d_scale + 0.5f);
            int8_t *dst = q + (size_t)b * 32u;
            for (uint32_t kk = 0; kk < 32u; kk++) {
                uint8_t byte = blk[2 + (kk >> 1)];
                int nib = (kk & 1) ? (byte >> 4) : (byte & 0x0F);
                dst[kk] = (int8_t)(nib - 8);
            }
        }
    }
    ft->q = q;
    ft->d = d;
    ft->d_scale = d_scale;
    ft->n = n;
    ft->is_q4 = 1;
    return off;
}

/* 单层流式 arena 所需字节数 = 最大单个张量展开大小（排除按行流式的 tok_embed）。 */
size_t km_layer_bytes(const km_model_t *m) {
    size_t max_sz = 0;
    for (uint32_t i = 0; i < m->arch.n_tensors; i++) {
        const km_tensor_t *t = &m->tensors[i];
        if (strcmp(t->name, "tok_embed") == 0) continue;
        if (t->dtype == KM_DT_TERNARY) continue;   /* 三值查表不展开 */
        size_t sz;
        if (t->dtype == KM_DT_Q4) {
            uint32_t nb = t->n_rows * KM_NBLK(t->n_cols);
            sz = align_up((size_t)nb * 32u, 64) + align_up((size_t)nb * 2, 64);
        } else {
            sz = align_up((size_t)t->n_rows * t->n_cols * 4, 64);
        }
        if (sz > max_sz) max_sz = sz;
    }
    return max_sz;
}

/* 流式读 embed 第 row 行（q4 反量化到 fp32，out 长度 dim，尾部填 0）。 */
int km_embed_row(const km_model_t *m, const km_tensor_t *emb, int row, int dim,
                 float *out, km_read_fn read_fn, void *ctx,
                 uint8_t *stage, size_t stage_cap) {
    (void)m;
    const int nblk = (int)KM_NBLK(emb->n_cols);
    const size_t row_bytes = (size_t)nblk * 18u;
    if (row_bytes > stage_cap) return -1;
    if (read_fn(ctx, emb->offset + (size_t)row * row_bytes, stage,
                (uint32_t)row_bytes) != 0) return -1;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *blk = stage + (size_t)b * 18u;
        float dsc = km_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        int cnt = (int)emb->n_cols - (b << 5);
        if (cnt > 32) cnt = 32;
        for (int j = 0; j < cnt; j++) {
            uint8_t byte = blk[2 + (j >> 1)];
            int nib = (j & 1) ? (byte >> 4) : (byte & 0x0F);
            out[(b << 5) + j] = (float)(nib - 8) * dsc;
        }
        for (int j = cnt; j < 32; j++) out[(b << 5) + j] = 0.0f;
    }
    return 0;
}

int km_decode_step_stream(const km_model_t *m, km_state_t *st,
                          int token, float *out_logit,
                          int *out_ids, float *out_vals, int topk,
                          km_read_fn read_fn, void *ctx,
                          uint8_t *layer_arena, size_t layer_cap,
                          uint8_t *stage, size_t stage_cap) {
    const km_arch_t *a = &m->arch;
    const int dim = (int)a->dim;
    const int nh = (int)a->n_heads;
    const int nkv = (int)a->n_kv_heads;
    const int hd = (int)a->head_dim;
    const int ffn = (int)a->ffn_dim;
    const int kvw = nkv * hd;
    const int kv_rep = nh / nkv;
    const uint32_t pos = st->pos;

#if REP_PENALTY
    if (pos == 0) rep_clear();
    rep_mark(token);
#endif

    const int dim_p = (dim + 31) & ~31;
    uint8_t *sb = (uint8_t *)st->scratch;

    float *x  = (float *)sb;
    float *h  = x + dim_p;
    float *q  = h + dim_p;
    float *kk = q + dim_p;
    float *vv = kk + kvw;
    float *ao = vv + kvw;
    float *g  = ao + dim_p;
    float *u  = g + ffn;
    float *act= u + ffn;

    const uint32_t ffn_p = KM_ALIGN32(ffn);
    const uint32_t pd    = a->ple_dim;
    const uint32_t pd_p  = KM_ALIGN32(pd > 0 ? pd : 1u);
    float *ple  = act + ffn_p;                            /* [nl*pd] 每 token PLE 输入 */
    float *pleg = ple + (size_t)a->n_layers * pd;         /* [pd_p] gelu/乘积缓冲 */
    float *pleo = pleg + pd_p;                            /* [dim_p] ple_proj 输出 */

    size_t fbytes = scratch_floats(a) * sizeof(float);
    int16_t *xq = (int16_t *)(sb + ((fbytes + 15u) & ~(size_t)15u));

    int idx = km_find(m, "tok_embed");
    if (idx < 0) return -1;
    const km_tensor_t *emb = &m->tensors[idx];

    /* 输入嵌入（流式按行读） */
    if (km_embed_row(m, emb, token, dim, x, read_fn, ctx, stage, stage_cap) != 0)
        return -1;

    km_ft_t ft;
    char nm[32];

    /* --- PLE 底部：ple = (RMSNorm(ple_model_proj(x)·dim^-0.5) + table·pd^0.5) · 2^-0.5 --- */
    if (pd > 0) {
        const int ple_rows = (int)a->n_layers * (int)pd;
        snprintf(nm, sizeof(nm), "ple_model_proj");
        idx = km_find(m, nm); if (idx < 0) return -1;
        if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
        gemv_q8(&ft, ple_rows, dim, x, xq, ple);

        const float inv_dim = 1.0f / sqrtf((float)dim);
        for (int i = 0; i < ple_rows; i++) ple[i] *= inv_dim;

        snprintf(nm, sizeof(nm), "ple_proj_norm");
        idx = km_find(m, nm); if (idx < 0) return -1;
        if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
        const float *pn = ft.f32;
        for (int l = 0; l < (int)a->n_layers; l++)
            rms_norm(ple + (size_t)l * pd, pn, (int)pd, a->norm_eps, ple + (size_t)l * pd);

        snprintf(nm, sizeof(nm), "ple_table");
        idx = km_find(m, nm); if (idx < 0) return -1;
        const km_tensor_t *pt = &m->tensors[idx];
        const int row_bytes = ((int)pt->n_cols + 4) / 5;
        if (row_bytes > (int)stage_cap) return -1;
        if (read_fn(ctx, pt->offset + (size_t)token * row_bytes, stage,
                    (uint32_t)row_bytes) != 0) return -1;
        const float spd = sqrtf((float)pd);
        const float inv2 = 0.7071067811865475f;   /* 2^-0.5 */
        /* 逐 trit 解码并就地合并（table_width=nl*pd 可能 > dim_p，不能暂存到 pleo） */
        for (int b = 0, oi = 0; b < row_bytes && oi < ple_rows; b++) {
            uint32_t v = stage[b];
            for (int k = 0; k < 5 && oi < ple_rows; k++, v /= 3u) {
                float tv = (float)((int)(v % 3u) - 1) * a->ple_gamma;
                ple[oi] = (ple[oi] + tv * spd) * inv2;
                oi++;
            }
        }
    }

    for (unsigned L = 0; L < (unsigned)a->n_layers; L++) {
        /* --- attention --- */
        snprintf(nm, sizeof(nm), "L%u.in_ln", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
        rms_norm(x, ft.f32, dim, a->norm_eps, h);

        snprintf(nm, sizeof(nm), "L%u.q", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
        gemv_q8(&ft, dim, dim, h, xq, q);

        snprintf(nm, sizeof(nm), "L%u.k", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
        gemv_q8(&ft, kvw, dim, h, xq, kk);

        snprintf(nm, sizeof(nm), "L%u.v", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
        gemv_q8(&ft, kvw, dim, h, xq, vv);

        rope_apply(q, nh, hd, (int)pos, a->rope_theta);
        rope_apply(kk, nkv, hd, (int)pos, a->rope_theta);

        float *kc = st->k_cache + ((size_t)L * st->n_ctx + pos) * kvw;
        float *vc = st->v_cache + ((size_t)L * st->n_ctx + pos) * kvw;
        memcpy(kc, kk, sizeof(float) * kvw);
        memcpy(vc, vv, sizeof(float) * kvw);

        const float scale = 1.0f / sqrtf((float)hd);
        const uint32_t npos = pos + 1;
        for (int hh = 0; hh < nh; hh++) {
            const float *qh = q + (size_t)hh * hd;
            const int kvh = hh / kv_rep;
            float mx = -1e30f;
            for (uint32_t t = 0; t < npos; t++) {
                const float *kt = st->k_cache + ((size_t)L * st->n_ctx + t) * kvw + (size_t)kvh * hd;
                float s = 0.0f;
                for (int d = 0; d < hd; d++) s += qh[d] * kt[d];
                s *= scale;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            float *oph = ao + (size_t)hh * hd;
            for (int d = 0; d < hd; d++) oph[d] = 0.0f;
            for (uint32_t t = 0; t < npos; t++) {
                const float *kt = st->k_cache + ((size_t)L * st->n_ctx + t) * kvw + (size_t)kvh * hd;
                float s = 0.0f;
                for (int d = 0; d < hd; d++) s += qh[d] * kt[d];
                s = expf(s * scale - mx);
                sum += s;
                const float *vt = st->v_cache + ((size_t)L * st->n_ctx + t) * kvw + (size_t)kvh * hd;
                for (int d = 0; d < hd; d++) oph[d] += s * vt[d];
            }
            float inv = 1.0f / sum;
            for (int d = 0; d < hd; d++) oph[d] *= inv;
        }

        snprintf(nm, sizeof(nm), "L%u.o", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
        gemv_q8(&ft, dim, dim, ao, xq, h);
        for (int i = 0; i < dim; i++) x[i] += h[i];

        /* --- MLP (SwiGLU / MoE) --- */
        snprintf(nm, sizeof(nm), "L%u.post_ln", L);
        idx = km_find(m, nm); if (idx < 0) return -1;
        if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
        rms_norm(x, ft.f32, dim, a->norm_eps, h);

        if (a->n_expert > 0) {
            const int ne = (int)a->n_expert;
            if (ne > 8) return -1;   /* 栈上固定缓冲上限，防御畸形文件 */
            int tk = (int)a->top_k;
            if (tk > ne) tk = ne;
            float rlog[8], p[8];
            snprintf(nm, sizeof(nm), "L%u.router", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
            const float *rw = ft.f32;
            float mx = -1e30f;
            for (int e = 0; e < ne; e++) {
                const float *row = rw + (size_t)e * dim;
                float s = 0.0f;
                for (int j = 0; j < dim; j++) s += row[j] * h[j];
                rlog[e] = s;
                if (s > mx) mx = s;
            }
            float psum = 0.0f;
            for (int e = 0; e < ne; e++) { p[e] = expf(rlog[e] - mx); psum += p[e]; }
            for (int e = 0; e < ne; e++) p[e] /= psum;

            int sel[8]; float wsel[8]; int chosen[8] = {0};
            for (int k = 0; k < tk; k++) {
                int bi = -1; float bv = -1e30f;
                for (int e = 0; e < ne; e++) {
                    if (chosen[e]) continue;
                    if (p[e] > bv) { bv = p[e]; bi = e; }
                }
                chosen[bi] = 1; sel[k] = bi; wsel[k] = p[bi];
            }
            float wsum = 0.0f;
            for (int k = 0; k < tk; k++) wsum += wsel[k];
            for (int k = 0; k < tk; k++) wsel[k] /= wsum;

            float *moe_out = ao;
            float *eout = q;
            for (int i = 0; i < dim; i++) moe_out[i] = 0.0f;

            for (int k = 0; k < tk; k++) {
                const int e = sel[k];
                snprintf(nm, sizeof(nm), "L%u.e%d.gate", L, e);
                idx = km_find(m, nm); if (idx < 0) return -1;
                if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
                gemv_q8(&ft, ffn, dim, h, xq, g);

                snprintf(nm, sizeof(nm), "L%u.e%d.up", L, e);
                idx = km_find(m, nm); if (idx < 0) return -1;
                if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
                gemv_q8(&ft, ffn, dim, h, xq, u);

                for (int i = 0; i < ffn; i++) {
                    float sg = g[i] / (1.0f + expf(-g[i]));
                    act[i] = sg * u[i];
                }

                snprintf(nm, sizeof(nm), "L%u.e%d.down", L, e);
                idx = km_find(m, nm); if (idx < 0) return -1;
                if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
                gemv_q8(&ft, dim, ffn, act, xq, eout);
                for (int i = 0; i < dim; i++) moe_out[i] += wsel[k] * eout[i];
            }
            for (int i = 0; i < dim; i++) x[i] += moe_out[i];
        } else {
            snprintf(nm, sizeof(nm), "L%u.gate", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
            gemv_q8(&ft, ffn, dim, h, xq, g);

            snprintf(nm, sizeof(nm), "L%u.up", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
            gemv_q8(&ft, ffn, dim, h, xq, u);

            for (int i = 0; i < ffn; i++) {
                float sg = g[i] / (1.0f + expf(-g[i]));
                act[i] = sg * u[i];
            }

            snprintf(nm, sizeof(nm), "L%u.down", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
            gemv_q8(&ft, dim, ffn, act, xq, h);
            for (int i = 0; i < dim; i++) x[i] += h[i];
        }

        /* --- PLE 分支（attention + ffn 之后） --- */
        if (pd > 0) {
            snprintf(nm, sizeof(nm), "L%u.ple_gate", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
            gemv_q8(&ft, (int)pd, dim, x, xq, pleg);
            for (int i = 0; i < (int)pd; i++) {
                float v = pleg[i];
                pleg[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
            }
            float *pslice = ple + (size_t)L * pd;
            for (int i = 0; i < (int)pd; i++) pleg[i] *= pslice[i];

            snprintf(nm, sizeof(nm), "L%u.ple_proj", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
            gemv_q8(&ft, dim, (int)pd, pleg, xq, pleo);

            snprintf(nm, sizeof(nm), "L%u.ple_norm", L);
            idx = km_find(m, nm); if (idx < 0) return -1;
            if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
            rms_norm(pleo, ft.f32, dim, a->norm_eps, pleo);
            for (int i = 0; i < dim; i++) x[i] += pleo[i];
        }
    }

    idx = km_find(m, "final_norm");
    if (idx < 0) return -1;
    if (!km_build_tensor(m, idx, &ft, layer_arena, layer_cap, read_fn, ctx, stage, stage_cap)) return -1;
    rms_norm(x, ft.f32, dim, a->norm_eps, h);

    /* 输出头（tied=1 复用 tok_embed；tied=0 用独立 lm_head）→ 流式 argmax */
    const int vocab = (int)a->vocab_size;
    float best = -1e30f;
    int best_id = 0;
    if (topk > 8) topk = 8;
    for (int j = 0; j < topk; j++) { out_vals[j] = -1e30f; out_ids[j] = -1; }

    const km_tensor_t *oh = emb;   /* 输出头张量：默认 tied 复用 embed */
    if (!a->tied_embed) {
        idx = km_find(m, "lm_head");
        if (idx < 0) return -1;
        oh = &m->tensors[idx];
    }

    const int nblk = (int)KM_NBLK(oh->n_cols);
    const int align = nblk << 5;
    float maxa = 0.0f;
    for (int j = 0; j < dim; j++) {
        float ax = h[j] < 0.0f ? -h[j] : h[j];
        if (ax > maxa) maxa = ax;
    }
    float S = maxa > 1e-6f ? 32767.0f / maxa : 1.0f;
    for (int j = 0; j < dim; j++)
        xq[j] = (int16_t)(h[j] * S + (h[j] >= 0.0f ? 0.5f : -0.5f));
    for (int j = dim; j < align; j++) xq[j] = 0;
    const float invS = 1.0f / S;

    /* 复用 layer_arena 存单行 embed 的反量化 int8 + int16 定点 scale。
     * 需 cap ≥ align_up(align,64) + align_up(nblk*2,64)。 */
    int8_t *eq = (int8_t *)layer_arena;
    int16_t *ed = (int16_t *)(layer_arena + align_up((size_t)align, 64));
    const size_t row_bytes = (size_t)nblk * 18u;
    for (int v = 0; v < vocab; v++) {
        if (row_bytes > stage_cap) return -1;
        if (read_fn(ctx, oh->offset + (size_t)v * row_bytes, stage,
                    (uint32_t)row_bytes) != 0) return -1;
        float max_d = 0.0f;
        for (int b = 0; b < nblk; b++) {
            const uint8_t *blk = stage + (size_t)b * 18u;
            float df = km_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            if (df > max_d) max_d = df;
        }
        float d_scale = max_d / 32767.0f;
        if (d_scale < 1e-30f) d_scale = 1.0f;
        for (int b = 0; b < nblk; b++) {
            const uint8_t *blk = stage + (size_t)b * 18u;
            float df = km_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            ed[b] = (int16_t)(df / d_scale + 0.5f);
            int8_t *dst = eq + (b << 5);
            for (int kk = 0; kk < 32; kk++) {
                uint8_t byte = blk[2 + (kk >> 1)];
                int nib = (kk & 1) ? (byte >> 4) : (byte & 0x0F);
                dst[kk] = (int8_t)(nib - 8);
            }
        }
        float acc = pie_gemv_row(eq, xq, ed, nblk, 0) * invS * d_scale;
#if REP_PENALTY
        acc = rep_apply(acc, v);
#endif
        if (acc > best) { best = acc; best_id = v; }
        if (topk > 0 && acc > out_vals[topk - 1]) {
            int j = topk - 1;
            while (j > 0 && acc > out_vals[j - 1]) {
                out_vals[j] = out_vals[j - 1];
                out_ids[j]  = out_ids[j - 1];
                j--;
            }
            out_vals[j] = acc;
            out_ids[j]  = v;
        }
    }

    st->pos = pos + 1;
    if (out_logit) *out_logit = best;
    return best_id;
}
