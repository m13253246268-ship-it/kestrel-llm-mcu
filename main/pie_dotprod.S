/* ================================================================
 * pie_dotprod.S - ESP32-P4 PIE（xespv）定点点积内核
 *
 * 移植自乐鑫 ESP-DL dl_esp32p4_dotprod_i16k8o16（Apache-2.0），
 * 仅改函数名/签名：返回 int32（不饱和），供 GEMV 乘 fp32 scale 后累加。
 *
 * 上游出处 / Upstream: https://github.com/espressif/esp-dl
 * 上游版权 / Upstream copyright: Espressif Systems (Shanghai) CO LTD
 * 本文件许可 / License: Apache-2.0（见 ../LICENSE 与 ../NOTICE）
 *
 * 数据格式：int8 权重 w × int16 激活 x → int32 累加（w8a16）。
 * PIE 只支持整数类型，本内核是「激活定点化」路径的最小验证件。
 * ================================================================ */

/* int32_t pie_dotprod_s8s16(const int8_t *w, const int16_t *x, int shift, int n);
 *   a0 = int8_t  *w
 *   a1 = int16_t *x
 *   a2 = int shift（右移，0 表示不缩放）
 *   a3 = int n（元素数）
 *   返回 int32（不饱和）。
 *   约束：w、x 需 16 字节对齐；n 任意（内部处理 16 元素/轮 + 尾块）。 */
    .text
    .global pie_dotprod_s8s16
    .type   pie_dotprod_s8s16, @function
    .align 4
pie_dotprod_s8s16:
    esp.zero.xacc
    srli a5, a3, 4                          /* a5 = n / 16 */
    slli t3, a5, 4                          /* t3 = (n/16)*16 */
    esp.vldext.s8.ip q0, q1, a0, 16         /* 加载 16 个 int8 并符号扩展成 q0,q1 */
    esp.vld.128.ip q2, a1, 16               /* 加载 16 个 int16 到 q2 */
    li t0, 0
    beq a5, t0, pie_dotprod_s8s16_small
    add t0, a5, 0
pie_dotprod_s8s16_loop:
    esp.vld.128.ip q3, a1, 16
    esp.vmulas.s16.xacc.ld.ip q4, a0, 16, q0, q2
    esp.vmulas.s16.xacc.ld.ip q2, a1, 16, q1, q3
    esp.vext.s8 q0, q1, q4
    addi t0, t0, -1
    bnez t0, pie_dotprod_s8s16_loop
pie_dotprod_s8s16_small:
    beq a3, t3, pie_dotprod_s8s16_end
    esp.vmulas.s16.xacc q0, q2
pie_dotprod_s8s16_end:
    esp.srs.s.xacc a0, a2                   /* 结果右移 a2 位到 a0（不饱和） */
    ret

/* float pie_gemv_row(const int8_t *q, const int16_t *x, const int16_t *d, int nblk, int shift);
 * 对一行做分块 GEMV（w8a16）：
 *   acc = Σ_b d[b] * dot(q[b*32 .. b*32+32], x[b*32 .. b*32+32])
 * 每块 32 元素，q 为 int8、x 为 int16（调用方已量化）、d 为 int16 定点 scale。
 *   a0 = int8_t   *q（nblk*32 字节，16 字节对齐）
 *   a1 = int16_t  *x（nblk*32 个 int16）
 *   a2 = int16_t  *d（nblk 个 int16 定点）
 *   a3 = int nblk
 *   a4 = int shift（右移，0 表示不缩放）
 *   返回 fp32 acc（fa0），不含 per-tensor d_scale（由调用方乘）。 */
    .text
    .global pie_gemv_row
    .type   pie_gemv_row, @function
    .align 4
pie_gemv_row:
    fmv.w.x ft0, zero                       /* fp32 acc = 0 */
    li t4, 0                                /* b = 0 */
pie_gemv_row_blk:
    esp.zero.xacc                           /* 每块清零累加器 */
    /* 乐鑫 .ld.ip 流水线（32 元素 = 2 轮 16），地址用 a0/a1 后增 */
    esp.vldext.s8.ip q0, q1, a0, 16         /* 轮1: 16 int8 -> q0,q1 */
    esp.vld.128.ip q2, a1, 16               /* 轮1: 8 int16 -> q2 */
    esp.vld.128.ip q3, a1, 16               /* 轮1: 8 int16 -> q3 */
    esp.vmulas.s16.xacc.ld.ip q4, a0, 16, q0, q2   /* 轮1: += q0*q2; q4=轮2 int8 */
    esp.vmulas.s16.xacc.ld.ip q2, a1, 16, q1, q3   /* 轮1: += q1*q3; q2=轮2 int16 */
    esp.vext.s8 q0, q1, q4                  /* q4 -> q0,q1（轮2 int8） */
    esp.vld.128.ip q3, a1, 16               /* 轮2: 8 int16 -> q3 */
    esp.vmulas.s16.xacc.ld.ip q4, a0, 16, q0, q2   /* 轮2: += q0*q2 */
    esp.vmulas.s16.xacc.ld.ip q2, a1, 16, q1, q3   /* 轮2: += q1*q3 */
    addi a0, a0, -16                        /* 回退多加载（a0 前进 48，应 32） */
    addi a1, a1, -16                        /* 回退多加载（a1 前进 80，应 64） */
    /* 取 dot_b（int32），乘 d[b]（int16 定点）累加到 fp32 */
    esp.srs.s.xacc t3, a4                   /* t3 = dot_b >> shift */
    fcvt.s.w ft1, t3                        /* ft1 = (float)dot_b */
    lh t0, 0(a2)                            /* t0 = d[b] int16 定点 */
    fcvt.s.w ft2, t0                        /* ft2 = (float)d[b] */
    fmadd.s ft0, ft1, ft2, ft0              /* acc += dot_b * d[b] */
    addi a2, a2, 2
    addi t4, t4, 1
    blt t4, a3, pie_gemv_row_blk
    fmv.s fa0, ft0
    ret
