/* host_verify.c — PC 端对拍：用标量 pie_gemv_row stub 编译 kmcu.c，
 * 加载 KMCU 镜像，跑与 ref_forward.py 相同的贪心 decode，打印 token 序列。
 * 供与 tools/ref_forward.py 的输出逐 token 比对（argmax 一致）。
 *
 * 编译： gcc -O2 -std=c11 -I main -o host_verify host_verify.c main/kmcu.c -lm
 * 用法： ./host_verify <model.kmcu>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kmcu.h"

/* 标量版 pie_gemv_row（ESP32-P4 上为 pie_dotprod.S 的 PIE 汇编）。
 * 语义：sum_b d[b] * sum_j q[b*32+j]*x[b*32+j]，shift=0。 */
float pie_gemv_row(const int8_t *q, const int16_t *x, const int16_t *d, int nblk, int shift) {
    (void)shift;
    float acc = 0.0f;
    for (int b = 0; b < nblk; b++) {
        int32_t s = 0;
        for (int j = 0; j < 32; j++)
            s += (int32_t)q[(size_t)b * 32 + j] * (int32_t)x[(size_t)b * 32 + j];
        acc += (float)d[b] * (float)s;
    }
    return acc;
}

typedef struct { const uint8_t *base; } mem_ctx_t;
static int mem_read(void *ctx, uint32_t off, void *dst, uint32_t len) {
    memcpy(dst, ((mem_ctx_t *)ctx)->base + off, len);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.kmcu>\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "open fail\n"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *base = (uint8_t *)malloc((size_t)sz);
    if (fread(base, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "read fail\n"); return 2; }
    fclose(f);

    uint32_t n_tensors;
    memcpy(&n_tensors, base + 52, 4);

    km_model_t m;
    memset(&m, 0, sizeof(m));
    m.tensors = (km_tensor_t *)calloc(n_tensors, sizeof(km_tensor_t));
    km_err_t err;
    if (km_open(&m, base, &err) != 0) { fprintf(stderr, "km_open: %s\n", err.buf); return 2; }

    const km_arch_t *A = &m.arch;
    printf("arch: layers=%u dim=%u nh=%u nkv=%u hd=%u ffn=%u vocab=%u "
           "tied=%u n_expert=%u top_k=%u ple_dim=%u ple_gamma=%.6f n_cls=%u\n",
           A->n_layers, A->dim, A->n_heads, A->n_kv_heads, A->head_dim,
           A->ffn_dim, A->vocab_size, A->tied_embed, A->n_expert, A->top_k,
           A->ple_dim, (double)A->ple_gamma, A->n_cls);

    size_t fbytes = km_fast_bytes(&m);
    uint8_t *arena = (uint8_t *)malloc(fbytes);
    km_fast_t fast;
    fast.t = (km_ft_t *)calloc(n_tensors, sizeof(km_ft_t));
    km_fast_build(&m, &fast, arena, fbytes);
    printf("fast weights: %zu bytes\n", fbytes);

    size_t kv = (size_t)A->n_layers * 128 * A->n_kv_heads * A->head_dim;
    float *kc = (float *)calloc(kv, sizeof(float));
    float *vc = (float *)calloc(kv, sizeof(float));
    float *sc = (float *)malloc(km_scratch_bytes(A));
    km_state_t st;
    km_state_init(&st, A, 128, kc, vc, sc);

    uint8_t stage[65536];
    mem_ctx_t ctx = { base };

    int prompt[4] = { 1, 450, 2217, 4996 };
    int seq[20];
    int ids[8]; float vals[8];

    int last = 0;
    for (int i = 0; i < 4; i++) {
        seq[i] = prompt[i];
        last = km_decode_step(&m, &fast, &st, prompt[i], NULL, ids, vals, 8,
                              mem_read, &ctx, stage, sizeof(stage));
    }
    seq[4] = last;
    printf("top8 @first-gen:\n");
    for (int j = 0; j < 8; j++) printf("  #%d id=%d logit=%.6f\n", j, ids[j], (double)vals[j]);

    for (int i = 4; i < 19; i++) {
        last = km_decode_step(&m, &fast, &st, last, NULL, ids, vals, 8,
                              mem_read, &ctx, stage, sizeof(stage));
        seq[i + 1] = last;
    }
    printf("seq:");
    for (int i = 0; i < 20; i++) printf("%s%d", i ? "," : "", seq[i]);
    printf("\n");

    /* 决策头测试（n_cls>0 时：单次前向分类，对比 ref_forward 的 cls_forward） */
    if (A->n_cls > 0) {
        int cls_tokens[8] = { 1, 450, 2217, 4996, 522, 28723, 415, 907 };
        float cls_logits[16];
        int cls_id = 0;
        int rc = km_cls_forward(&m, &fast, &st, cls_tokens, 8, (int)A->n_cls,
                                &cls_id, cls_logits, mem_read, &ctx, stage, sizeof(stage));
        printf("cls: rc=%d id=%d logits:", rc, cls_id);
        for (int c = 0; c < (int)A->n_cls; c++) printf(" %.6f", (double)cls_logits[c]);
        printf("\n");

        /* Agent 循环测试（决策 → 动作分派 → 观察回接 → 再决策） */
        int init_tokens[4] = { 1, 450, 2217, 4996 };
        int actions[16], gens[16];
        float act_logits[64];
        int nsteps = km_agent_loop(&m, &fast, &st, init_tokens, 4, (int)A->n_cls, 8, 64,
                                   actions, gens, act_logits,
                                   mem_read, &ctx, stage, sizeof(stage));
        printf("agent: steps=%d\n", nsteps);
        for (int s = 0; s < nsteps; s++) {
            printf("  step %d: action=%d gen=%d logits:", s, actions[s], gens[s]);
            for (int c = 0; c < (int)A->n_cls; c++)
                printf(" %.6f", (double)act_logits[s * A->n_cls + c]);
            printf("\n");
        }
    }
    return 0;
}
