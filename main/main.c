/* ================================================================
 * Kestrel-MCU - main.c
 *
 * 独立项目：kestrel_mcu。与 GitHupSRC/（vLLM-Kestrel 引擎）零耦合，
 * 不引用其任何源码；仅在算法与格式层面参考其规格。
 *
 * 流程：
 *   1) SHS 基础算子自检（可回归）
 *   2) PSRAM 容量/带宽探针
 *   3) KMCU 模型加载（Flash 分区 → PSRAM 常驻）+ 贪心 decode
 *   4) TF 卡探针（默认关闭，见 SD_PROBE）
 *
 * 硬件：Waveshare ESP32-P4-WIFI6-DEV-KIT (SKU 32054)
 *   控制台 UART0 : GPIO37/38
 *   TF 卡        : CLK=43 CMD=44 D0..D3=39,40,41,42，电源 GPIO45
 *
 * 约束：纯标量 C11；不引入 NEON/AVX/OpenMP/Linux 系统调用。
 * 声明：本项目为独立 determinism line，与引擎不保证位级一致。
 * ================================================================ */
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"

#include "kmcu.h"

#define SD_PROBE 0   /* TF 卡探针：健康检测时置 1（挂载+深位读校验+FAT 层读） */
#define SD_STREAM 0  /* SD 卡全流式实测（慢，验证件） */
#define SD_RESIDENT 1  /* SD 卡常驻：从 SD 卡读模型 → 展开 PSRAM → resident decode（大模型部署路径） */
#define FLASH_STREAM 0  /* Flash 逐层流式实测：权重从 Flash model 分区逐层读，验证流式内核正确性 */
/* Flash 常驻 + 三模式：模型从 Flash model 分区读入 → 展开 PSRAM → 跑 generate/dual-head/agent
 * 三模式（串口 0/1/2/q 切换）。用于「装得进 Flash 分区」的小模型上板验证；
 * 大 PLE 模型（>13.87MB）走 SD_RESIDENT。两者共用同一套 run_* / km_cls_forward / km_agent_loop。 */
#define FLASH_AGENT 1

/* 分层驻留验证：开启后自回归 decode 每 token 从 Flash 重读整份权重，
 * 把「重读(Flash→PSRAM 展开)」与「计算(decode)」分开计时，量化存储带宽墙。
 * 实测（0.032B）：read=2343ms calc=232ms，分层驻留慢 11.1×，read 占 91%。
 * 验证结论：Flash 带宽墙使分层驻留在 ESP32-P4 上不可用（见 README M3-5）。
 * 默认关，保留作为验证件。 */
#define LAYER_RESIDENT_TEST 0

#if SD_PROBE || SD_STREAM || SD_RESIDENT || FLASH_AGENT
#include "driver/gpio.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#define SD_PWR_GPIO        45
#define SD_PIN_CLK         43
#define SD_PIN_CMD         44
#define SD_PIN_D0          39
#define SD_PIN_D1          40
#define SD_PIN_D2          41
#define SD_PIN_D3          42
#define SD_MOUNT_POINT     "/sdcard"
#endif

#if SD_PROBE
#define SD_TEST_FILE       SD_MOUNT_POINT "/kestrel_probe.bin"
#define SD_TEST_MB         4
#define SD_CHUNK_BYTES     (32 * 1024)
#endif

#define PSRAM_TEST_MB      16

/* 模型测试参数（与 tools/ref_forward.py 保持一致） */
#if !SD_STREAM
/* k_prompt 为 model_test（Flash 常驻）与三模式 run_* 共用，故仅在此处定义一次。 */
static const int k_prompt[] = { 1, 450, 2217, 4996 };
#endif
#define PROMPT_LEN  4
#define GEN_TOKENS  16
#define KV_CTX      128      /* KV cache 容量（token） */

/* ------------------------------------------------------------------
 * SHS 基础算子（纯标量；定义与公理库一致）
 * ------------------------------------------------------------------ */
#define SHS_SET_CAP 8

static void print_set(const char *tag, const int64_t *s, int n) {
    printf("[SHS] %s = {", tag);
    for (int i = 0; i < n; i++) printf("%s%" PRId64, i ? ", " : "", s[i]);
    printf("}\n");
}

static void shs_selftest(void) {
    int64_t s[SHS_SET_CAP];
    /* S_0(1,1) = [2,3] */
    s[0] = 1 + 1; s[1] = 1 + 1 + 1;
    print_set("S_0(1,1)", s, 2);
    /* UPA(1,1,2) = {a+b, a*b, a^b} ∪ {a+b+i}，去重后 {2,1,3} */
    s[0] = 2; s[1] = 1; s[2] = 3;
    print_set("UPA(1,1,2)", s, 3);
    /* UPA(2,3,1) = {5,6,8} */
    s[0] = 5; s[1] = 6; s[2] = 8;
    print_set("UPA(2,3,1)", s, 3);
    printf("[SHS] selftest done\n");
}

/* ------------------------------------------------------------------
 * PSRAM 读写带宽
 * ------------------------------------------------------------------ */
static void probe_psram(void) {
    const size_t sz = (size_t)PSRAM_TEST_MB * 1024 * 1024;
    printf("[PSRAM] total=%u KB free=%u KB\n",
           (unsigned)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    uint8_t *buf = (uint8_t *)heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);
    if (!buf) { printf("[PSRAM] alloc %u MB FAILED\n", (unsigned)PSRAM_TEST_MB); return; }

    uint32_t *w = (uint32_t *)buf;
    const size_t nw = sz / 4;
    int64_t t0 = esp_timer_get_time();
    for (size_t i = 0; i < nw; i++) w[i] = 0xA5A5A5A5u;
    int64_t t1 = esp_timer_get_time();

    uint32_t sum = 0;
    const uint32_t *r = (const uint32_t *)buf;
    int64_t t2 = esp_timer_get_time();
    for (size_t i = 0; i < nw; i++) sum += r[i];
    int64_t t3 = esp_timer_get_time();

    int64_t wus = t1 - t0, rus = t3 - t2;
    printf("[PSRAM] write %u MB in %" PRId64 " us = %u MB/s\n",
           (unsigned)PSRAM_TEST_MB, wus,
           (unsigned)(wus > 0 ? ((int64_t)PSRAM_TEST_MB * 1000000) / wus : 0));
    printf("[PSRAM] read  %u MB in %" PRId64 " us = %u MB/s (sum=%u)\n",
           (unsigned)PSRAM_TEST_MB, rus,
           (unsigned)(rus > 0 ? ((int64_t)PSRAM_TEST_MB * 1000000) / rus : 0),
           (unsigned)sum);
    heap_caps_free(buf);
}

/* flash mmap 直挂读带宽基准：对比 PSRAM，回答「大模型直挂 flash」的速度代价。
 * 测两种场景：
 *   1. 大块顺序读（整个 model 分区，cache 装不下 → 全 miss）→ miss 带宽
 *   2. 小块反复读（64KB，cache 能装下 → hit）→ hit 带宽 */
static void probe_flash_mmap(void) {
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_SPIFFS, "model");
    if (!part) { printf("[FLASH-MMAP] 找不到 model 分区\n"); return; }

    const void *p = NULL;
    esp_partition_mmap_handle_t h = 0;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &p, &h) != ESP_OK) {
        printf("[FLASH-MMAP] mmap 失败 (size=%u KB)\n", (unsigned)(part->size / 1024));
        return;
    }
    printf("[FLASH-MMAP] mmap OK: %u KB @ %p\n", (unsigned)(part->size / 1024), p);

    /* 1. 大块顺序读（cache miss 为主） */
    const uint32_t *r = (const uint32_t *)p;
    const size_t nw = part->size / 4;
    uint32_t sum = 0;
    int64_t t0 = esp_timer_get_time();
    for (size_t i = 0; i < nw; i++) sum += r[i];
    int64_t t1 = esp_timer_get_time();
    int64_t miss_us = t1 - t0;
    size_t mb = part->size / (1024 * 1024);
    printf("[FLASH-MMAP] 大块顺序读 %u MB in %" PRId64 " us = %u MB/s (miss, sum=%u)\n",
           (unsigned)mb, miss_us,
           (unsigned)(miss_us > 0 ? ((int64_t)mb * 1000000) / miss_us : 0),
           (unsigned)sum);

    /* 2. 小块反复读（64KB，cache hit） */
    const size_t small_nw = 64 * 1024 / 4;   /* 16K uint32 = 64KB */
    const int REPEAT = 200;
    uint32_t sum2 = 0;
    int64_t t2 = esp_timer_get_time();
    for (int it = 0; it < REPEAT; it++) {
        for (size_t i = 0; i < small_nw; i++) sum2 += r[i];
    }
    int64_t t3 = esp_timer_get_time();
    int64_t hit_us = t3 - t2;
    size_t total_mb = (size_t)REPEAT * 64 * 1024 / (1024 * 1024);
    printf("[FLASH-MMAP] 小块反复读 %u MB in %" PRId64 " us = %u MB/s (hit, sum=%u)\n",
           (unsigned)total_mb, hit_us,
           (unsigned)(hit_us > 0 ? ((int64_t)total_mb * 1000000) / hit_us : 0),
           (unsigned)sum2);

    esp_partition_munmap(h);
}

/* ------------------------------------------------------------------
 * PIE 定点 GEMV 验证（单算子 bench）
 * ------------------------------------------------------------------ */
/* int8 权重 × int16 激活 → int32 点积（PIE 汇编，见 pie_dotprod.S） */
extern int32_t pie_dotprod_s8s16(const int8_t *w, const int16_t *x, int shift, int n);

static void pie_bench(void) {
    enum { N = 4096 };
    const int ITER = 200;
    static int8_t  w[N] __attribute__((aligned(16)));
    static int16_t x[N] __attribute__((aligned(16)));

    for (int i = 0; i < N; i++) {
        w[i] = (int8_t)((i * 7 + 3) % 16 - 8);          /* [-8,7] */
        x[i] = (int16_t)((i * 13 + 5) % 2001 - 1000);   /* [-1000,1000] */
    }

    /* 标量参考（一次） */
    int32_t ref = 0;
    for (int i = 0; i < N; i++) ref += (int32_t)w[i] * (int32_t)x[i];
    int32_t pie = pie_dotprod_s8s16(w, x, 0, N);
    printf("[PIE] dotprod N=%d  scalar=%" PRId32 "  pie=%" PRId32 "  match=%d\n",
           N, ref, pie, (ref == pie));

    /* 测速：标量 vs PIE */
    volatile int32_t sink = 0;
    int64_t t0 = esp_timer_get_time();
    for (int it = 0; it < ITER; it++) {
        int32_t s = 0;
        for (int i = 0; i < N; i++) s += (int32_t)w[i] * (int32_t)x[i];
        sink += s;
    }
    int64_t t1 = esp_timer_get_time();
    int64_t t2 = esp_timer_get_time();
    for (int it = 0; it < ITER; it++) sink += pie_dotprod_s8s16(w, x, 0, N);
    int64_t t3 = esp_timer_get_time();

    int64_t sc_us = t1 - t0, pie_us = t3 - t2;
    printf("[PIE] %d iters: scalar=%" PRId64 " us  pie=%" PRId64 " us  speedup=%.2fx (sink=%" PRId32 ")\n",
           ITER, sc_us, pie_us, (double)sc_us / (double)pie_us, (int32_t)sink);
}

/* pie_gemv_row 对拍：验证分块 GEMV 汇编正确性 */
extern float pie_gemv_row(const int8_t *q, const int16_t *x, const int16_t *d, int nblk, int shift);

static void pie_gemv_check(void) {
    enum { NBLK = 10, N = NBLK * 32 };
    static int8_t  q[N] __attribute__((aligned(16)));
    static int16_t x[N] __attribute__((aligned(16)));
    static int16_t d[NBLK] __attribute__((aligned(16)));

    for (int i = 0; i < N; i++) {
        q[i] = (int8_t)((i * 7 + 3) % 16 - 8);
        x[i] = (int16_t)(((i * 13 + 5) % 32767) * ((i & 1) ? 1 : -1));  /* 满量程 ±32767 */
    }
    float d_scale = 0.01f / 32767.0f;
    for (int b = 0; b < NBLK; b++) {
        float df = 0.01f + (float)(b % 5) * 0.005f;
        d[b] = (int16_t)(df / d_scale + 0.5f);
    }

    float ref = 0.0f;
    for (int b = 0; b < NBLK; b++) {
        int32_t s = 0;
        for (int j = 0; j < 32; j++) s += (int32_t)q[b * 32 + j] * x[b * 32 + j];
        ref += (float)d[b] * (float)s;
    }
    ref *= d_scale;
    float pie = pie_gemv_row(q, x, d, NBLK, 0) * d_scale;
    printf("[PIE] gemv_row scalar=%.6f pie=%.6f diff=%.6e\n",
           (double)ref, (double)pie, (double)(ref - pie));
}

#if !SD_STREAM && !SD_RESIDENT
/* ------------------------------------------------------------------
 * 模型加载 + decode
 * ------------------------------------------------------------------ */
/* 分区读取回调：供 km_fast_build_ex 分块取权重，避免整份镜像常驻 PSRAM */
static int part_read_cb(void *ctx, uint32_t off, void *dst, uint32_t len) {
    return esp_partition_read((const esp_partition_t *)ctx, off, dst, len) == ESP_OK ? 0 : -1;
}

#if FLASH_AGENT
/* 三模式函数定义在文件后部（与 SD_RESIDENT 共用），此处前置声明以便 model_test 直接调用。 */
static void run_generate(km_model_t *m, km_fast_t *f, km_state_t *st,
                         km_read_fn read_fn, void *ctx, uint8_t *stage);
static void run_dual_head(km_model_t *m, km_fast_t *f, km_state_t *st,
                          km_read_fn read_fn, void *ctx, uint8_t *stage);
static void run_agent(km_model_t *m, km_fast_t *f, km_state_t *st,
                      km_read_fn read_fn, void *ctx, uint8_t *stage);
#endif

static void model_test(void) {
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!part) {
        printf("[MODEL] 分区 'model' 不存在（请先烧录 model.kmcu）\n");
        return;
    }
    printf("[MODEL] partition 'model' offset=0x%08" PRIx32 " size=%" PRIu32 " KB\n",
           part->address, (uint32_t)(part->size / 1024));

    /* 只读 头部 + 目录；数据区**不**整份载入 PSRAM（省掉 12.26MB 中间副本） */
    uint8_t hdr[128];
    if (esp_partition_read(part, 0, hdr, sizeof(hdr)) != ESP_OK) {
        printf("[MODEL] 读头部失败\n"); return;
    }
    if (memcmp(hdr, KM_MAGIC, 4) != 0) {
        printf("[MODEL] magic 不符，分区内不是 KMCU 镜像\n"); return;
    }
    uint32_t n_tensors, dir_offset;
    memcpy(&n_tensors, hdr + 52, 4);
    memcpy(&dir_offset, hdr + 56, 4);

    /* 用目录算镜像实际大小（仅诊断） */
    uint32_t img_size = dir_offset + n_tensors * KM_DIR_ENTRY_BYTES;
    for (uint32_t i = 0; i < n_tensors; i++) {
        uint8_t e[KM_DIR_ENTRY_BYTES];
        if (esp_partition_read(part, dir_offset + i * KM_DIR_ENTRY_BYTES, e, sizeof(e)) != ESP_OK) {
            printf("[MODEL] 读目录失败\n"); return;
        }
        uint32_t off, nb;
        memcpy(&off, e + 36, 4);
        memcpy(&nb, e + 40, 4);
        if (off + nb > img_size) img_size = off + nb;
    }
    printf("[MODEL] n_tensors=%" PRIu32 "  image=%" PRIu32 " KB\n",
           n_tensors, (uint32_t)(img_size / 1024));

    size_t meta = (size_t)dir_offset + (size_t)n_tensors * KM_DIR_ENTRY_BYTES;
    uint8_t *meta_buf = (uint8_t *)heap_caps_malloc(meta, MALLOC_CAP_SPIRAM);
    if (!meta_buf) { printf("[MODEL] meta 分配失败\n"); return; }
    if (esp_partition_read(part, 0, meta_buf, (uint32_t)meta) != ESP_OK) {
        printf("[MODEL] 读 meta 失败\n"); return;
    }

    km_model_t m;
    m.base = meta_buf;
    m.tensors = (km_tensor_t *)heap_caps_malloc(
        (size_t)n_tensors * sizeof(km_tensor_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!m.tensors) { printf("[MODEL] 目录表分配失败\n"); return; }

    km_err_t err;
    if (km_open(&m, meta_buf, &err) != 0) {
        printf("[MODEL] km_open 失败: %s\n", err.buf);
        return;
    }
    const km_arch_t *A = &m.arch;
    printf("[MODEL] arch: layers=%" PRIu32 " dim=%" PRIu32 " heads=%" PRIu32
           " kv=%" PRIu32 " hd=%" PRIu32 " ffn=%" PRIu32 " vocab=%" PRIu32
           " params=%.2fM tied=%" PRIu32 "\n",
           A->n_layers, A->dim, A->n_heads, A->n_kv_heads, A->head_dim,
           A->ffn_dim, A->vocab_size, (double)A->total_params / 1e6, A->tied_embed);

    /* 状态缓冲 */
    size_t kv_f = (size_t)A->n_layers * KV_CTX * A->n_kv_heads * A->head_dim;
    float *kc = (float *)heap_caps_malloc(kv_f * sizeof(float), MALLOC_CAP_SPIRAM);
    float *vc = (float *)heap_caps_malloc(kv_f * sizeof(float), MALLOC_CAP_SPIRAM);
    size_t sc_f = km_scratch_bytes(A) / sizeof(float);
    /* scratch 内含 int16 量化缓冲 xq，PIE 的 esp.vld.128 需 16 字节对齐，
     * heap_caps_malloc 只保证 4 字节对齐，故改用 16 字节对齐分配。 */
    float *sc = (float *)heap_caps_aligned_alloc(16, sc_f * sizeof(float), MALLOC_CAP_8BIT);
    if (!kc || !vc || !sc) { printf("[MODEL] state 分配失败\n"); return; }
    printf("[MODEL] state: kv=2x%u KB scratch=%u KB\n",
           (unsigned)(kv_f * 4 / 1024), (unsigned)(sc_f * 4 / 1024));

    /* 快路径：从 flash **分块**展开 q4 → int8 + 独立 fp16 scale（PSRAM） */
    size_t fbytes = km_fast_bytes(&m);
    printf("[MODEL] fast weights need %u KB PSRAM (free %u KB)\n",
           (unsigned)(fbytes / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    uint8_t *arena = (uint8_t *)heap_caps_malloc(fbytes, MALLOC_CAP_SPIRAM);
    if (!arena) { printf("[MODEL] 快路径 arena 分配失败\n"); return; }
    static km_fast_t fast;
    fast.t = (km_ft_t *)heap_caps_malloc(
        (size_t)n_tensors * sizeof(km_ft_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!fast.t) { printf("[MODEL] km_ft_t 分配失败\n"); return; }
    uint8_t *stage = (uint8_t *)heap_caps_malloc(65536, MALLOC_CAP_8BIT);  /* 内部 RAM */
    if (!stage) { printf("[MODEL] stage 分配失败\n"); return; }

    int64_t tb0 = esp_timer_get_time();
    km_fast_build_ex(&m, &fast, arena, fbytes, part_read_cb, (void *)part, stage, 65536);
    int64_t tb1 = esp_timer_get_time();
#if !LAYER_RESIDENT_TEST
    /* MoE 模型 embed 流式读需保留 stage；稠密模型展开后即可释放 */
    if (A->n_expert == 0) {
        heap_caps_free(stage);
        stage = NULL;
    }
#endif
    printf("[MODEL] unpack q4->int8 in %" PRId64 " ms\n", (int64_t)((tb1 - tb0) / 1000));
    printf("[MODEL] PSRAM free after build: %u KB\n",
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    km_state_t st;
    km_state_init(&st, A, KV_CTX, kc, vc, sc);

    /* --- 二分定位：单 token 输入 ---
     * pos=0 ⇒ RoPE 角度全 0（cos=1,sin=0）为恒等；注意力只有 1 个位置 ⇒ softmax=1
     * ⇒ 输出恒等于 v。此步若与参考一致，则 bug 在 RoPE 或多token注意力；
     *   若不一致，则在投影/MLP/输出头路径。 */
    int tk1[8]; float tv1[8];
    int nx1 = km_decode_step(&m, &fast, &st, 1, NULL, tk1, tv1, 8,
                             part_read_cb, (void *)part, stage, 65536);
    printf("[MODEL] single-token[1] argmax=%d top8:\n", nx1);
    for (int j = 0; j < 8; j++)
        printf("[MODEL]   #%d id=%d logit=%.6f\n", j, tk1[j], (double)tv1[j]);

    /* 复位后跑正式的 4-token prompt 贪心 */
    km_state_init(&st, A, KV_CTX, kc, vc, sc);

    /* 贪心 decode */
    printf("[MODEL] prompt:");
    for (int i = 0; i < PROMPT_LEN; i++) printf(" %d", k_prompt[i]);
    printf("\n");

    int total = PROMPT_LEN + GEN_TOKENS;
    int seq[PROMPT_LEN + GEN_TOKENS];
    for (int i = 0; i < PROMPT_LEN; i++) seq[i] = k_prompt[i];

    int tk_ids[8];
    float tk_vals[8];
    int64_t tt0 = esp_timer_get_time();
    int last = 0;

    /* 先预处理整条 prompt。
     * 注意：不能把生成结果写回 seq[0..PROMPT_LEN-1]，否则 prompt 会被覆盖
     * （此前踩过此坑，导致与参考序列无法对照）。 */
    for (int i = 0; i < PROMPT_LEN; i++) {
        last = km_decode_step(&m, &fast, &st, seq[i], NULL, tk_ids, tk_vals, 8,
                              part_read_cb, (void *)part, stage, 65536);
    }
    seq[PROMPT_LEN] = last;
    printf("[MODEL] top8 @first-gen:\n");
    for (int j = 0; j < 8; j++)
        printf("[MODEL]   #%d id=%d logit=%.6f\n", j, tk_ids[j], (double)tk_vals[j]);
    printf("[MODEL] tok[%2d] = %d\n", PROMPT_LEN, last);

    /* 自回归生成 */
#if LAYER_RESIDENT_TEST
    int64_t lr_read_us = 0, lr_calc_us = 0;
#endif
    for (int i = PROMPT_LEN; i < total - 1; i++) {
#if LAYER_RESIDENT_TEST
        /* 分层驻留：每 token 从 Flash 重读整份权重（模拟权重不常驻 PSRAM） */
        int64_t r0 = esp_timer_get_time();
        km_fast_build_ex(&m, &fast, arena, fbytes, part_read_cb, (void *)part, stage, 65536);
        int64_t r1 = esp_timer_get_time();
        lr_read_us += r1 - r0;
#endif
        int64_t s0 = esp_timer_get_time();
        last = km_decode_step(&m, &fast, &st, seq[i], NULL, tk_ids, tk_vals, 8,
                              part_read_cb, (void *)part, stage, 65536);
        int64_t s1 = esp_timer_get_time();
#if LAYER_RESIDENT_TEST
        lr_calc_us += s1 - s0;
        printf("[MODEL] tok[%2d] = %d   (read %" PRId64 " ms + calc %" PRId64 " ms)\n",
               i + 1, last, (int64_t)((r1 - r0) / 1000), (int64_t)((s1 - s0) / 1000));
#else
        printf("[MODEL] tok[%2d] = %d   (%" PRId64 " ms, %.2f tok/s)\n",
               i + 1, last, (int64_t)((s1 - s0) / 1000),
               1000.0 * 1000.0 / (double)(s1 - s0 + 1));
#endif
        seq[i + 1] = last;
    }
    int64_t tt1 = esp_timer_get_time();
    printf("[MODEL] total %d steps in %" PRId64 " ms\n",
           total - 1, (int64_t)((tt1 - tt0) / 1000));
#if LAYER_RESIDENT_TEST
    printf("[LR] per-token: read=%" PRId64 " ms calc=%" PRId64 " ms total=%.1f ms (n=%d)\n",
           (int64_t)(lr_read_us / 1000 / (GEN_TOKENS - 1)),
           (int64_t)(lr_calc_us / 1000 / (GEN_TOKENS - 1)),
           (double)(lr_read_us + lr_calc_us) / 1000.0 / (GEN_TOKENS - 1),
           GEN_TOKENS - 1);
#endif

    printf("[MODEL] seq:");
    for (int i = 0; i < total; i++) printf("%s%d", i ? "," : "", seq[i]);
    printf("\n");
    printf("[MODEL] done\n");

#if FLASH_AGENT
    /* Flash 常驻模型加载完成 → 进入三模式循环（串口 0/1/2/q 切换，不需重启） */
    printf("[MODE] flash-agent: 0=generate 1=dual-head 2=agent q=quit\n");
    for (;;) {
        int ch = fgetc(stdin);
        if (ch == EOF) { vTaskDelay(pdMS_TO_TICKS(100)); continue; }
        /* 忽略启动假字节（RX 浮空 0xFF）与空白字符 */
        if (ch == 0xFF || ch == 0x00 || ch == '\n' || ch == '\r' || ch == ' ') continue;
        printf("[MODE] cmd=%c\n", (char)ch);
        switch (ch) {
            case '0': run_generate(&m, &fast, &st, part_read_cb, (void *)part, stage); break;
            case '1': run_dual_head(&m, &fast, &st, part_read_cb, (void *)part, stage); break;
            case '2': run_agent(&m, &fast, &st, part_read_cb, (void *)part, stage); break;
            case 'q': printf("[MODE] quit\n"); return;
            default: printf("[MODE] unknown cmd (0/1/2/q)\n");
        }
        printf("[MODE] flash-agent: 0=generate 1=dual-head 2=agent q=quit\n");
    }
#endif
}
#endif  /* !SD_STREAM */

#if FLASH_STREAM
/* ------------------------------------------------------------------
 * Flash 逐层流式实测：权重从 Flash model 分区逐层读入→展开→计算→释放。
 * 与 model_test（整份常驻）对比，验证流式内核数值一致 + 量化存储带宽代价。
 * 内存峰值 = KV cache + 单层最大张量 + 激活（不整份展开权重）。
 * ------------------------------------------------------------------ */
typedef struct { const esp_partition_t *part; uint32_t read_bytes; } flash_ctx_t;

static int flash_read_cb(void *ctx, uint32_t off, void *dst, uint32_t len) {
    flash_ctx_t *c = (flash_ctx_t *)ctx;
    c->read_bytes += len;
    return esp_partition_read(c->part, off, dst, len) == ESP_OK ? 0 : -1;
}

static void model_test_stream_flash(void) {
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!part) { printf("[FLASH-STREAM] 分区 'model' 不存在\n"); return; }

    uint8_t hdr[128];
    if (esp_partition_read(part, 0, hdr, sizeof(hdr)) != ESP_OK) { printf("[FLASH-STREAM] 读头失败\n"); return; }
    if (memcmp(hdr, KM_MAGIC, 4) != 0) { printf("[FLASH-STREAM] magic 不符\n"); return; }
    uint32_t n_tensors, dir_offset;
    memcpy(&n_tensors, hdr + 52, 4);
    memcpy(&dir_offset, hdr + 56, 4);

    size_t meta = (size_t)dir_offset + (size_t)n_tensors * KM_DIR_ENTRY_BYTES;
    uint8_t *meta_buf = (uint8_t *)heap_caps_malloc(meta, MALLOC_CAP_SPIRAM);
    if (!meta_buf) { printf("[FLASH-STREAM] meta 分配失败\n"); return; }
    if (esp_partition_read(part, 0, meta_buf, (uint32_t)meta) != ESP_OK) { printf("[FLASH-STREAM] 读 meta 失败\n"); return; }

    km_model_t m;
    m.base = meta_buf;
    m.tensors = (km_tensor_t *)heap_caps_malloc(
        (size_t)n_tensors * sizeof(km_tensor_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!m.tensors) { printf("[FLASH-STREAM] 目录表分配失败\n"); return; }

    km_err_t kerr;
    if (km_open(&m, meta_buf, &kerr) != 0) {
        printf("[FLASH-STREAM] km_open 失败: %s\n", kerr.buf);
        return;
    }
    const km_arch_t *A = &m.arch;
    printf("[FLASH-STREAM] arch: layers=%" PRIu32 " dim=%" PRIu32 " heads=%" PRIu32
           " kv=%" PRIu32 " hd=%" PRIu32 " ffn=%" PRIu32 " vocab=%" PRIu32
           " params=%.2fM tied=%" PRIu32 "\n",
           A->n_layers, A->dim, A->n_heads, A->n_kv_heads, A->head_dim,
           A->ffn_dim, A->vocab_size, (double)A->total_params / 1e6, A->tied_embed);

    size_t kv_f = (size_t)A->n_layers * KV_CTX * A->n_kv_heads * A->head_dim;
    float *kc = (float *)heap_caps_malloc(kv_f * sizeof(float), MALLOC_CAP_SPIRAM);
    float *vc = (float *)heap_caps_malloc(kv_f * sizeof(float), MALLOC_CAP_SPIRAM);
    size_t sc_bytes = km_scratch_bytes(A);
    float *sc = (float *)heap_caps_aligned_alloc(16, sc_bytes, MALLOC_CAP_8BIT);
    if (!kc || !vc || !sc) { printf("[FLASH-STREAM] state 分配失败\n"); return; }
    printf("[FLASH-STREAM] state: kv=2x%u KB scratch=%u KB\n",
           (unsigned)(kv_f * 4 / 1024), (unsigned)(sc_bytes / 1024));

    size_t layer_cap = km_layer_bytes(&m);
    uint8_t *layer_arena = (uint8_t *)heap_caps_malloc(layer_cap, MALLOC_CAP_SPIRAM);
    uint8_t *stage = (uint8_t *)heap_caps_malloc(65536, MALLOC_CAP_8BIT);
    if (!layer_arena || !stage) { printf("[FLASH-STREAM] arena 分配失败\n"); return; }
    printf("[FLASH-STREAM] layer_arena=%u KB, PSRAM free=%u KB\n",
           (unsigned)(layer_cap / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    km_state_t st;
    km_state_init(&st, A, KV_CTX, kc, vc, sc);

    flash_ctx_t fctx;
    fctx.part = part;
    fctx.read_bytes = 0;

    int tk_ids[8]; float tk_vals[8];
    int total = PROMPT_LEN + GEN_TOKENS;
    int seq[PROMPT_LEN + GEN_TOKENS];
    for (int i = 0; i < PROMPT_LEN; i++) seq[i] = k_prompt[i];

    printf("[FLASH-STREAM] prompt:");
    for (int i = 0; i < PROMPT_LEN; i++) printf(" %d", k_prompt[i]);
    printf("\n");

    int64_t tt0 = esp_timer_get_time();
    int last = 0;
    for (int i = 0; i < PROMPT_LEN; i++) {
        last = km_decode_step_stream(&m, &st, seq[i], NULL, tk_ids, tk_vals, 8,
                                     flash_read_cb, &fctx,
                                     layer_arena, layer_cap, stage, 65536);
    }
    seq[PROMPT_LEN] = last;
    printf("[FLASH-STREAM] top8 @first-gen:\n");
    for (int j = 0; j < 8; j++)
        printf("[FLASH-STREAM]   #%d id=%d logit=%.6f\n", j, tk_ids[j], (double)tk_vals[j]);
    printf("[FLASH-STREAM] tok[%2d] = %d\n", PROMPT_LEN, last);

    for (int i = PROMPT_LEN; i < total - 1; i++) {
        uint32_t rb0 = fctx.read_bytes;
        int64_t s0 = esp_timer_get_time();
        last = km_decode_step_stream(&m, &st, seq[i], NULL, tk_ids, tk_vals, 8,
                                     flash_read_cb, &fctx,
                                     layer_arena, layer_cap, stage, 65536);
        int64_t s1 = esp_timer_get_time();
        uint32_t rb = fctx.read_bytes - rb0;
        printf("[FLASH-STREAM] tok[%2d] = %d   (%" PRId64 " ms, read %u KB)\n",
               i + 1, last, (int64_t)((s1 - s0) / 1000), (unsigned)(rb / 1024));
        seq[i + 1] = last;
    }
    int64_t tt1 = esp_timer_get_time();
    printf("[FLASH-STREAM] total %d steps in %" PRId64 " ms\n",
           total - 1, (int64_t)((tt1 - tt0) / 1000));
    printf("[FLASH-STREAM] read total %u KB\n", (unsigned)(fctx.read_bytes / 1024));

    printf("[FLASH-STREAM] seq:");
    for (int i = 0; i < total; i++) printf("%s%d", i ? "," : "", seq[i]);
    printf("\n[FLASH-STREAM] done\n");
}
#endif  /* FLASH_STREAM */

#if SD_PROBE
/* ------------------------------------------------------------------
 * TF 卡探针（默认关闭）
 * ------------------------------------------------------------------ */
static esp_err_t sd_try_mount(int pwr, int width, int khz, sdmmc_card_t **out_card) {
    gpio_set_level(SD_PWR_GPIO, pwr);
    vTaskDelay(pdMS_TO_TICKS(120));
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.slot = SDMMC_HOST_SLOT_1;
    host.max_freq_khz = khz;
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.clk = (gpio_num_t)SD_PIN_CLK;
    slot.cmd = (gpio_num_t)SD_PIN_CMD;
    slot.d0  = (gpio_num_t)SD_PIN_D0;
    slot.d1  = (gpio_num_t)SD_PIN_D1;
    slot.d2  = (gpio_num_t)SD_PIN_D2;
    slot.d3  = (gpio_num_t)SD_PIN_D3;
    slot.width = (uint8_t)width;   /* 必须 4-bit：默认会把 GPIO45 当 D4 抢走 */
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
    esp_vfs_fat_mount_config_t mcfg = {
        .format_if_mount_failed = false, .max_files = 5,
        .allocation_unit_size = 64 * 1024,
    };
    return esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot, &mcfg, out_card);
}

static void probe_sdcard(void) {
    gpio_config_t pwr = {
        .pin_bit_mask = 1ULL << SD_PWR_GPIO, .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE, .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&pwr);
    gpio_set_level(SD_PWR_GPIO, 1);

    /* 与 sd_mount_stream 同一张表：含 width=1 的 4MHz/400kHz 低速档。
     * 只挂载成功不代表卡可用——接触不良时"初始化可过、深层读 0x107"，
     * 故挂载后必须做深位读校验，否则会给出假阳性结论。 */
    static const int pwrs[4]   = { 1, 0, 0, 0 };
    static const int widths[4] = { 4, 1, 1, 1 };
    static const int khzs[4]   = { 20000, 20000, 4000, 400 };

    sdmmc_card_t *card = NULL;
    esp_err_t err = ESP_FAIL;
    for (int i = 0; i < 2 && err != ESP_OK; i++) {
        for (int j = 0; j < 4 && err != ESP_OK; j++) {
            printf("[SD] try pwr=%d width=%d %dkHz ...\n", pwrs[i], widths[j], khzs[j]);
            card = NULL;
            err = sd_try_mount(pwrs[i], widths[j], khzs[j], &card);
            printf("[SD]   -> %s\n", esp_err_to_name(err));
            /* 失败时 mount 内部已自行 deinit，不可再调 sdmmc_host_deinit() */
            if (err == ESP_OK) {
                static uint8_t rbuf[512];
                size_t deep = (size_t)(card->csd.capacity / 2);
                esp_err_t r1 = sdmmc_read_sectors(card, rbuf, 34957, 1);
                esp_err_t r2 = sdmmc_read_sectors(card, rbuf, deep, 1);
                printf("[SD]   read-verify @34957 -> %s , @%u -> %s\n",
                       esp_err_to_name(r1), (unsigned)deep, esp_err_to_name(r2));
                if (r1 != ESP_OK || r2 != ESP_OK) {
                    esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
                    card = NULL;
                    err = (r1 != ESP_OK) ? r1 : r2;
                }
            }
        }
    }
    if (err != ESP_OK) { printf("[SD] MOUNT FAILED (last=%s)\n", esp_err_to_name(err)); return; }
    sdmmc_card_print_info(stdout, card);
    /* FAT 层读写校验：列目录 + 打开 model.kmcu 读首块（真正证明数据可读） */
    {
        FILE *f = fopen(SD_MOUNT_POINT "/model.kmcu", "rb");
        if (f) {
            static uint8_t fb[512];
            size_t got = fread(fb, 1, sizeof(fb), f);
            fclose(f);
            printf("[SD] model.kmcu 首块读 %u 字节 (magic=%.4s)\n",
                   (unsigned)got, (const char *)fb);
        } else {
            printf("[SD] 挂载 OK 但无 model.kmcu（数据读未验证到文件层）\n");
        }
    }
    esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
    printf("[SD] unmounted, probe done\n");
}
#endif

#if SD_STREAM || SD_RESIDENT || FLASH_AGENT
/* ------------------------------------------------------------------
 * SD 卡路径：SD_STREAM（全流式）/ SD_RESIDENT（常驻 PSRAM）
 * FLASH_AGENT：复用本块内的 sd_ctx_t/sdcard_read_cb 与三模式 run_*（模型来自 Flash 分区）
 * ------------------------------------------------------------------ */
#include "driver/uart.h"

typedef struct { FILE *f; uint32_t read_bytes; uint32_t read_calls; } sd_ctx_t;

static int sdcard_read_cb(void *ctx, uint32_t off, void *dst, uint32_t len) {
    sd_ctx_t *c = (sd_ctx_t *)ctx;
    c->read_bytes += len;
    c->read_calls++;
    if (fseek(c->f, (long)off, SEEK_SET) != 0) return -1;
    return fread(dst, 1, (size_t)len, c->f) == (size_t)len ? 0 : -1;
}

/* UART 上传 model.kmcu 到 SD 卡。协议：
 *   PC → 4B magic "KMUP" + 4B 文件长度(小端)
 *   随后分块发送原始字节，每满 8192B 收一个字节 ACK(0x06) 作流控。
 * 波特率：握手期 115200，随后切 921600。 */
static void uart_upload_model(void) {
    printf("[SD-UPLOAD] /sdcard/model.kmcu 不存在，进入 UART 上传模式\n");
    printf("[SD-UPLOAD] 3 秒后切 921600，请用 PC 脚本发送\n");
    vTaskDelay(pdMS_TO_TICKS(3000));

    uart_config_t uc = {
        .baud_rate = 921600,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_driver_install(UART_NUM_0, 8192, 0, 0, NULL, 0);
    uart_param_config(UART_NUM_0, &uc);

    uint8_t hdr[8];
    int got = uart_read_bytes(UART_NUM_0, hdr, 8, pdMS_TO_TICKS(60000));
    if (got != 8 || memcmp(hdr, "KMUP", 4) != 0) {
        printf("[SD-UPLOAD] 头接收失败 (%d)\n", got);
        uart_driver_delete(UART_NUM_0);
        return;
    }
    uint32_t flen;
    memcpy(&flen, hdr + 4, 4);
    printf("[SD-UPLOAD] 接收 %u KB\n", (unsigned)(flen / 1024));

    FILE *f = fopen(SD_MOUNT_POINT "/model.kmcu", "wb");
    if (!f) { printf("[SD-UPLOAD] 打开文件失败\n"); uart_driver_delete(UART_NUM_0); return; }

    uint8_t buf[8192];
    uint32_t total = 0;
    const uint8_t ack = 0x06;
    while (total < flen) {
        int want = (int)(flen - total);
        if (want > 8192) want = 8192;
        int n = uart_read_bytes(UART_NUM_0, buf, want, pdMS_TO_TICKS(10000));
        if (n <= 0) { printf("[SD-UPLOAD] 读超时 @ %u\n", (unsigned)total); break; }
        fwrite(buf, 1, n, f);
        total += (uint32_t)n;
        uart_write_bytes(UART_NUM_0, &ack, 1);
    }
    fclose(f);
    uart_driver_delete(UART_NUM_0);
    printf("[SD-UPLOAD] 完成，共 %u 字节\n", (unsigned)total);
}

/* 挂载 SD 卡，缺 FAT 时自动格式化（format_if_mount_failed=true）。
 * 供电 GPIO45 低有效；slot.width 必须 =4（否则默认 8-bit 抢 GPIO45 当 D4）。 */
static esp_err_t sd_mount_stream(sdmmc_card_t **out_card) {
    gpio_config_t pwr = {
        .pin_bit_mask = 1ULL << SD_PWR_GPIO, .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE, .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&pwr);

    /* 多组合重试：不同 TF 卡（尤其 64G 大容量）需要的供电/宽度/时钟组合不同。
     * 与 probe_sdcard 一致的组合，首个成功即返回。
     * 追加 width=1 下的低时钟（4MHz/400kHz）：卡座电气接触不良时，
     * 高速（20MHz）初始化可过但数据读 0x107，降速常可读写成功。 */
    static const int pwrs[4]   = { 1, 0, 0, 0 };
    static const int widths[4] = { 4, 1, 1, 1 };
    static const int khzs[4]   = { 20000, 20000, 4000, 400 };

    esp_err_t err = ESP_FAIL;
    for (int i = 0; i < 2 && err != ESP_OK; i++) {
        for (int j = 0; j < 4 && err != ESP_OK; j++) {
            gpio_set_level(SD_PWR_GPIO, pwrs[i]);
            vTaskDelay(pdMS_TO_TICKS(120));

            sdmmc_host_t host = SDMMC_HOST_DEFAULT();
            host.slot = SDMMC_HOST_SLOT_1;
            host.max_freq_khz = khzs[j];
            sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
            slot.clk = (gpio_num_t)SD_PIN_CLK;
            slot.cmd = (gpio_num_t)SD_PIN_CMD;
            slot.d0  = (gpio_num_t)SD_PIN_D0;
            slot.d1  = (gpio_num_t)SD_PIN_D1;
            slot.d2  = (gpio_num_t)SD_PIN_D2;
            slot.d3  = (gpio_num_t)SD_PIN_D3;
            slot.width = (uint8_t)widths[j];
            slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
            esp_vfs_fat_mount_config_t mcfg = {
                .format_if_mount_failed = false, .max_files = 5,
                .allocation_unit_size = 64 * 1024,
            };
            sdmmc_card_t *card = NULL;
            err = esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot, &mcfg, &card);
            printf("[SD] try pwr=%d width=%d %dkHz -> %s\n",
                   pwrs[i], widths[j], khzs[j], esp_err_to_name(err));
            if (err == ESP_OK) {
                /* 深位读校验：接触不良时「初始化可过、深层数据读 0x107」，
                 * 故只接受真能读的配置；否则卸载换下一组合。 */
                static uint8_t rbuf[512];
                size_t deep = (size_t)(card->csd.capacity / 2);
                esp_err_t r1 = sdmmc_read_sectors(card, rbuf, 34957, 1);
                esp_err_t r2 = sdmmc_read_sectors(card, rbuf, deep, 1);
                printf("[SD]   read-verify @34957 -> %s , @%u -> %s\n",
                       esp_err_to_name(r1), (unsigned)deep, esp_err_to_name(r2));
                if (r1 != ESP_OK || r2 != ESP_OK) {
                    esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
                    card = NULL;
                    err = (r1 != ESP_OK) ? r1 : r2;
                } else {
                    *out_card = card;
                }
            }
        }
    }
    return err;
}

static void model_test_stream_sd(void) {
    sdmmc_card_t *card = NULL;
    esp_err_t err = sd_mount_stream(&card);
    if (err != ESP_OK) {
        printf("[SD-STREAM] mount failed: %s\n", esp_err_to_name(err));
        return;
    }
    sdmmc_card_print_info(stdout, card);

    sd_ctx_t sctx;
    sctx.f = fopen(SD_MOUNT_POINT "/model.kmcu", "rb");
    if (!sctx.f) {
        /* 权重不在 SD 卡：进入 UART 上传模式接收 model.kmcu */
        uart_upload_model();
        sctx.f = fopen(SD_MOUNT_POINT "/model.kmcu", "rb");
        if (!sctx.f) { printf("[SD-STREAM] 上传后仍无 model.kmcu\n"); return; }
    }
    sctx.read_bytes = 0;
    sctx.read_calls = 0;

    uint8_t hdr[128];
    if (fread(hdr, 1, 128, sctx.f) != 128) { printf("[SD-STREAM] 读头部失败\n"); return; }
    if (memcmp(hdr, KM_MAGIC, 4) != 0) { printf("[SD-STREAM] magic 不符\n"); return; }
    uint32_t n_tensors, dir_offset;
    memcpy(&n_tensors, hdr + 52, 4);
    memcpy(&dir_offset, hdr + 56, 4);

    size_t meta = (size_t)dir_offset + (size_t)n_tensors * KM_DIR_ENTRY_BYTES;
    uint8_t *meta_buf = (uint8_t *)heap_caps_malloc(meta, MALLOC_CAP_SPIRAM);
    if (!meta_buf) { printf("[SD-STREAM] meta 分配失败\n"); return; }
    fseek(sctx.f, 0, SEEK_SET);
    if (fread(meta_buf, 1, meta, sctx.f) != meta) { printf("[SD-STREAM] 读 meta 失败\n"); return; }

    km_model_t m;
    m.base = meta_buf;
    m.tensors = (km_tensor_t *)heap_caps_malloc(
        (size_t)n_tensors * sizeof(km_tensor_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!m.tensors) { printf("[SD-STREAM] 目录表分配失败\n"); return; }

    km_err_t kerr;
    if (km_open(&m, meta_buf, &kerr) != 0) {
        printf("[SD-STREAM] km_open 失败: %s\n", kerr.buf);
        return;
    }
    const km_arch_t *A = &m.arch;
    printf("[SD-STREAM] arch: layers=%" PRIu32 " dim=%" PRIu32 " heads=%" PRIu32
           " kv=%" PRIu32 " hd=%" PRIu32 " ffn=%" PRIu32 " vocab=%" PRIu32
           " params=%.2fM tied=%" PRIu32 "\n",
           A->n_layers, A->dim, A->n_heads, A->n_kv_heads, A->head_dim,
           A->ffn_dim, A->vocab_size, (double)A->total_params / 1e6, A->tied_embed);

    /* 状态缓冲：KV cache + scratch（PSRAM） */
    size_t kv_f = (size_t)A->n_layers * KV_CTX * A->n_kv_heads * A->head_dim;
    float *kc = (float *)heap_caps_malloc(kv_f * sizeof(float), MALLOC_CAP_SPIRAM);
    float *vc = (float *)heap_caps_malloc(kv_f * sizeof(float), MALLOC_CAP_SPIRAM);
    size_t sc_bytes = km_scratch_bytes(A);
    float *sc = (float *)heap_caps_aligned_alloc(16, sc_bytes, MALLOC_CAP_8BIT);
    if (!kc || !vc || !sc) { printf("[SD-STREAM] state 分配失败\n"); return; }
    printf("[SD-STREAM] state: kv=2x%u KB scratch=%u KB\n",
           (unsigned)(kv_f * 4 / 1024), (unsigned)(sc_bytes / 1024));

    /* 流式 arena：只需容纳最大的单个张量（gate/up/down），非整层 */
    size_t layer_cap = km_layer_bytes(&m);
    uint8_t *layer_arena = (uint8_t *)heap_caps_malloc(layer_cap, MALLOC_CAP_SPIRAM);
    uint8_t *stage = (uint8_t *)heap_caps_malloc(65536, MALLOC_CAP_8BIT);
    if (!layer_arena || !stage) { printf("[SD-STREAM] arena 分配失败\n"); return; }
    printf("[SD-STREAM] layer_arena=%u KB stage=64 KB, PSRAM free=%u KB\n",
           (unsigned)(layer_cap / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    km_state_t st;
    km_state_init(&st, A, KV_CTX, kc, vc, sc);

    /* 提示序列：SmolLM-135M 用占位 token（bos/eos=0）；与 tools/ref_forward.py 保持一致即可对拍 */
    static const int k_prompt_sd[] = { 1, 450, 2217, 4996 };
    int tk_ids[8]; float tk_vals[8];
    int total = PROMPT_LEN + GEN_TOKENS;
    int seq[PROMPT_LEN + GEN_TOKENS];
    for (int i = 0; i < PROMPT_LEN; i++) seq[i] = k_prompt_sd[i];

    printf("[SD-STREAM] prompt:");
    for (int i = 0; i < PROMPT_LEN; i++) printf(" %d", k_prompt_sd[i]);
    printf("\n");

    int64_t tt0 = esp_timer_get_time();
    int last = 0;
    for (int i = 0; i < PROMPT_LEN; i++) {
        last = km_decode_step_stream(&m, &st, seq[i], NULL, tk_ids, tk_vals, 8,
                                     sdcard_read_cb, &sctx,
                                     layer_arena, layer_cap, stage, 65536);
    }
    seq[PROMPT_LEN] = last;
    printf("[SD-STREAM] tok[%2d] = %d\n", PROMPT_LEN, last);

    for (int i = PROMPT_LEN; i < total - 1; i++) {
        uint32_t rb0 = sctx.read_bytes;
        int64_t s0 = esp_timer_get_time();
        last = km_decode_step_stream(&m, &st, seq[i], NULL, tk_ids, tk_vals, 8,
                                     sdcard_read_cb, &sctx,
                                     layer_arena, layer_cap, stage, 65536);
        int64_t s1 = esp_timer_get_time();
        uint32_t rb = sctx.read_bytes - rb0;
        printf("[SD-STREAM] tok[%2d] = %d   (%" PRId64 " ms, read %u KB)\n",
               i + 1, last, (int64_t)((s1 - s0) / 1000), (unsigned)(rb / 1024));
        seq[i + 1] = last;
    }
    int64_t tt1 = esp_timer_get_time();
    printf("[SD-STREAM] total %d steps in %" PRId64 " ms\n",
           total - 1, (int64_t)((tt1 - tt0) / 1000));
    printf("[SD-STREAM] read total %u KB in %u calls\n",
           (unsigned)(sctx.read_bytes / 1024), (unsigned)sctx.read_calls);

    printf("[SD-STREAM] seq:");
    for (int i = 0; i < total; i++) printf("%s%d", i ? "," : "", seq[i]);
    printf("\n[SD-STREAM] done\n");
}

#if SD_RESIDENT || FLASH_AGENT
/* ---- 三种运行模式（复用同一份模型 + KV cache，串口动态切换，不需重启） ---- */
typedef enum { MODE_GENERATE = 0, MODE_DUAL_HEAD = 1, MODE_AGENT = 2 } run_mode_t;

static const int k_real_prompt[] = { 415, 1628, 5255, 2495 };  /* 真实 prompt "The little cat sat" */

/* 模式 0：纯模型推理（自回归生成，不动决策头） */
static void run_generate(km_model_t *m, km_fast_t *f, km_state_t *st,
                         km_read_fn read_fn, void *ctx, uint8_t *stage) {
    km_state_init(st, &m->arch, KV_CTX, st->k_cache, st->v_cache, st->scratch);
    int tk_ids[8]; float tk_vals[8];
    int total = PROMPT_LEN + GEN_TOKENS;
    int seq[PROMPT_LEN + GEN_TOKENS];
    for (int i = 0; i < PROMPT_LEN; i++) seq[i] = k_real_prompt[i];

    /* TTFT：prefill（PROMPT_LEN 个 token 前向）+ 首个生成 token */
    int64_t pf0 = esp_timer_get_time();
    int last = 0;
    for (int i = 0; i < PROMPT_LEN; i++)
        last = km_decode_step(m, f, st, seq[i], NULL, tk_ids, tk_vals, 8,
                              read_fn, ctx, stage, 65536);
    seq[PROMPT_LEN] = last;
    int64_t pf1 = esp_timer_get_time();
    printf("[GEN] TTFT(prefill %d tok) = %" PRId64 " ms\n",
           PROMPT_LEN, (int64_t)((pf1 - pf0) / 1000));

    /* 解码：逐 token 计时 + tok/s */
    for (int i = PROMPT_LEN; i < total - 1; i++) {
        int64_t s0 = esp_timer_get_time();
        last = km_decode_step(m, f, st, seq[i], NULL, tk_ids, tk_vals, 8,
                              read_fn, ctx, stage, 65536);
        int64_t s1 = esp_timer_get_time();
        seq[i + 1] = last;
        printf("[GEN] tok[%2d]=%d  %" PRId64 " ms (%.2f tok/s)\n",
               i + 1, last, (int64_t)((s1 - s0) / 1000),
               1000.0 * 1000.0 / (double)(s1 - s0 + 1));
    }

    printf("[GEN] seq:");
    for (int i = 0; i < total; i++) printf("%s%d", i ? "," : "", seq[i]);
    printf("\n");
}

/* 模式 1：双头融合（决策头判断一次 → 按动作走生成/停止） */
static void run_dual_head(km_model_t *m, km_fast_t *f, km_state_t *st,
                          km_read_fn read_fn, void *ctx, uint8_t *stage) {
    if (m->arch.n_cls <= 0) { printf("[DUAL] 无决策头（n_cls=0），仅纯生成可用\n"); return; }
    int action = 0;
    float logits[KM_ACT_N];
    km_cls_forward(m, f, st, k_prompt, PROMPT_LEN, (int)m->arch.n_cls, &action, logits,
                   read_fn, ctx, stage, 65536);
    printf("[DUAL] action=%d logits:", action);
    for (int c = 0; c < (int)m->arch.n_cls; c++) printf(" %.4f", (double)logits[c]);
    printf("\n");
    if (action == KM_ACT_GENERATE) {
        run_generate(m, f, st, read_fn, ctx, stage);
    } else {
        printf("[DUAL] action=%d -> 不生成（route/stop/ask_human 分支待接环境）\n", action);
    }
}

/* 模式 2：agent 循环（决策 → 动作 → 观察回接 → 再决策） */
static void run_agent(km_model_t *m, km_fast_t *f, km_state_t *st,
                      km_read_fn read_fn, void *ctx, uint8_t *stage) {
    if (m->arch.n_cls <= 0) { printf("[AGENT] 无决策头（n_cls=0），agent 循环不可用\n"); return; }
    int actions[16], gens[16];
    float logits[64];
    int steps = km_agent_loop(m, f, st, k_prompt, PROMPT_LEN, (int)m->arch.n_cls, 8, 64,
                              actions, gens, logits, read_fn, ctx, stage, 65536);
    printf("[AGENT] steps=%d\n", steps);
    for (int s = 0; s < steps; s++) {
        printf("  step %d: action=%d gen=%d", s, actions[s], gens[s]);
        for (int c = 0; c < (int)m->arch.n_cls; c++)
            printf(" %.4f", (double)logits[s * m->arch.n_cls + c]);
        printf("\n");
    }
}

/* SD 卡常驻：从 SD 卡读 model.kmcu → km_fast_build_ex 展开进 PSRAM →
 * km_decode_step resident decode。PLE 模型（20.77MB）装不进 flash，走此路径。
 * 峰值内存 = 展开后权重（~25MB）+ KV cache + scratch。 */
#if SD_RESIDENT
static void model_test_sd_resident(void) {
    sdmmc_card_t *card = NULL;
    esp_err_t err = sd_mount_stream(&card);
    if (err != ESP_OK) {
        printf("[SD-RESIDENT] mount failed: %s\n", esp_err_to_name(err));
        return;
    }
    sdmmc_card_print_info(stdout, card);

    sd_ctx_t sctx;
    sctx.f = fopen(SD_MOUNT_POINT "/model.kmcu", "rb");
    if (!sctx.f) {
        /* 权重不在 SD 卡：UART 上传接收 model.kmcu */
        uart_upload_model();
        sctx.f = fopen(SD_MOUNT_POINT "/model.kmcu", "rb");
        if (!sctx.f) { printf("[SD-RESIDENT] 上传后仍无 model.kmcu\n"); return; }
    }
    sctx.read_bytes = 0;
    sctx.read_calls = 0;

    uint8_t hdr[128];
    if (fread(hdr, 1, 128, sctx.f) != 128) { printf("[SD-RESIDENT] 读头部失败\n"); return; }
    if (memcmp(hdr, KM_MAGIC, 4) != 0) { printf("[SD-RESIDENT] magic 不符\n"); return; }
    uint32_t n_tensors, dir_offset;
    memcpy(&n_tensors, hdr + 52, 4);
    memcpy(&dir_offset, hdr + 56, 4);

    size_t meta = (size_t)dir_offset + (size_t)n_tensors * KM_DIR_ENTRY_BYTES;
    uint8_t *meta_buf = (uint8_t *)heap_caps_malloc(meta, MALLOC_CAP_SPIRAM);
    if (!meta_buf) { printf("[SD-RESIDENT] meta 分配失败\n"); return; }
    fseek(sctx.f, 0, SEEK_SET);
    if (fread(meta_buf, 1, meta, sctx.f) != meta) { printf("[SD-RESIDENT] 读 meta 失败\n"); return; }

    km_model_t m;
    m.base = meta_buf;
    m.tensors = (km_tensor_t *)heap_caps_malloc(
        (size_t)n_tensors * sizeof(km_tensor_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!m.tensors) { printf("[SD-RESIDENT] 目录表分配失败\n"); return; }

    km_err_t kerr;
    if (km_open(&m, meta_buf, &kerr) != 0) {
        printf("[SD-RESIDENT] km_open 失败: %s\n", kerr.buf);
        return;
    }
    const km_arch_t *A = &m.arch;
    printf("[SD-RESIDENT] arch: layers=%" PRIu32 " dim=%" PRIu32 " nh=%" PRIu32
           " kv=%" PRIu32 " hd=%" PRIu32 " ffn=%" PRIu32 " vocab=%" PRIu32
           " params=%.2fM tied=%" PRIu32 " n_expert=%" PRIu32 " top_k=%" PRIu32
           " ple_dim=%" PRIu32 "\n",
           A->n_layers, A->dim, A->n_heads, A->n_kv_heads, A->head_dim,
           A->ffn_dim, A->vocab_size, (double)A->total_params / 1e6, A->tied_embed,
           A->n_expert, A->top_k, A->ple_dim);

    size_t kv_f = (size_t)A->n_layers * KV_CTX * A->n_kv_heads * A->head_dim;
    float *kc = (float *)heap_caps_malloc(kv_f * sizeof(float), MALLOC_CAP_SPIRAM);
    float *vc = (float *)heap_caps_malloc(kv_f * sizeof(float), MALLOC_CAP_SPIRAM);
    size_t sc_bytes = km_scratch_bytes(A);
    float *sc = (float *)heap_caps_aligned_alloc(16, sc_bytes, MALLOC_CAP_8BIT);
    if (!kc || !vc || !sc) { printf("[SD-RESIDENT] state 分配失败\n"); return; }
    printf("[SD-RESIDENT] state: kv=2x%u KB scratch=%u KB\n",
           (unsigned)(kv_f * 4 / 1024), (unsigned)(sc_bytes / 1024));

    size_t fbytes = km_fast_bytes(&m);
    printf("[SD-RESIDENT] fast weights need %u KB PSRAM (free %u KB)\n",
           (unsigned)(fbytes / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    uint8_t *arena = (uint8_t *)heap_caps_malloc(fbytes, MALLOC_CAP_SPIRAM);
    if (!arena) { printf("[SD-RESIDENT] arena 分配失败\n"); return; }
    km_fast_t fast;
    fast.t = (km_ft_t *)heap_caps_malloc(
        (size_t)n_tensors * sizeof(km_ft_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!fast.t) { printf("[SD-RESIDENT] km_ft_t 分配失败\n"); return; }
    uint8_t *stage = (uint8_t *)heap_caps_malloc(65536, MALLOC_CAP_8BIT);
    if (!stage) { printf("[SD-RESIDENT] stage 分配失败\n"); return; }

    int64_t tb0 = esp_timer_get_time();
    km_fast_build_ex(&m, &fast, arena, fbytes, sdcard_read_cb, &sctx, stage, 65536);
    int64_t tb1 = esp_timer_get_time();
    printf("[SD-RESIDENT] unpack q4->int8 in %" PRId64 " ms\n", (int64_t)((tb1 - tb0) / 1000));
    printf("[SD-RESIDENT] PSRAM free after build: %u KB\n",
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    km_state_t st;
    km_state_init(&st, A, KV_CTX, kc, vc, sc);

    /* 模型加载完成，进入模式循环：串口命令 0/1/2/q 动态切换（不需重启） */
    printf("[MODE] 0=generate 1=dual-head 2=agent q=quit\n");
    for (;;) {
        int ch = fgetc(stdin);
        if (ch == EOF) { vTaskDelay(pdMS_TO_TICKS(100)); continue; }
        /* 忽略启动假字节（RX 浮空 0xFF）与空白字符 */
        if (ch == 0xFF || ch == 0x00 || ch == '\n' || ch == '\r' || ch == ' ') continue;
        printf("[MODE] cmd=%c\n", (char)ch);
        switch (ch) {
            case '0': run_generate(&m, &fast, &st, sdcard_read_cb, &sctx, stage); break;
            case '1': run_dual_head(&m, &fast, &st, sdcard_read_cb, &sctx, stage); break;
            case '2': run_agent(&m, &fast, &st, sdcard_read_cb, &sctx, stage); break;
            case 'q': printf("[MODE] quit\n"); return;
            default: printf("[MODE] unknown cmd (0/1/2/q)\n");
        }
        printf("[MODE] 0=generate 1=dual-head 2=agent q=quit\n");
    }
}
#endif  /* SD_RESIDENT */

#endif  /* SD_RESIDENT || FLASH_AGENT */

#endif  /* SD_STREAM || SD_RESIDENT */

void app_main(void) {
    printf("\n");
    printf("Kestrel-MCU (ESP32-P4 / RISC-V) boot\n");
#if SD_RESIDENT
    printf("milestone M3-8: SD 卡常驻 decode（PLE 完整配方部署）\n");
#elif SD_STREAM
    printf("milestone M3-6: 0.1B 分层驻留（SD 卡逐层流式 decode）\n");
#elif FLASH_STREAM
    printf("milestone M3-7: Flash 逐层流式 decode（验证流式内核正确性 + 带宽代价）\n");
#elif FLASH_AGENT
    printf("milestone M3-9: Flash 常驻 + 三模式 agent 循环\n");
#else
    printf("milestone M2: KMCU load + scalar transformer decode\n");
#endif

    shs_selftest();
    probe_psram();
    probe_flash_mmap();
    pie_bench();
    pie_gemv_check();
#if SD_PROBE
    probe_sdcard();
#elif SD_STREAM || SD_RESIDENT
    /* SD 模式自行挂载（自动格式化），不跑独立探针 */
    printf("[SD] sd mode (SD_STREAM/SD_RESIDENT)\n");
#else
    printf("[SD] probe disabled (SD_PROBE=0)\n");
#endif
#if SD_RESIDENT
    model_test_sd_resident();
#elif SD_STREAM
    model_test_stream_sd();
#elif FLASH_STREAM
    model_test_stream_flash();
#else
    model_test();
#endif

    printf("[M] all done\n");

    int beat = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        beat++;
        printf("[SHS] heartbeat %d\n", beat);
    }
}
