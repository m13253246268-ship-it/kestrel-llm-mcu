#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_minueza.py - 把 HuggingFace safetensors 转成 KMCU 量化权重格式。

KMCU v1 布局（小端）：
  [0,128)   Header
  [dir_offset, dir_offset + n_tensors*48)  目录（每项 48B）
  [data_offset, ...)  数据区（每张量 64B 对齐）

Header:
  0  char magic[4] = "KMCU"
  4  u32 version = 1
  8  u32 n_layers
  12 u32 dim
  16 u32 n_heads
  20 u32 n_kv_heads
  24 u32 head_dim
  28 u32 ffn_dim
  32 u32 vocab_size
  36 u32 max_seq
  40 u32 bos_id
  44 u32 eos_id
  48 u32 tied_embed       (1 = lm_head 复用 embed_tokens，文件中不存 lm_head)
  52 u32 n_tensors
  56 u32 dir_offset
  60 u32 data_offset
  64 f32 rope_theta
  68 f32 norm_eps
  72 u32 total_params
  76 u32 reserved

目录项 (48B):
  0  char name[24]
  24 u8   dtype    (1=f32, 2=q4_0)
  25 u8   pad[3]
  28 u32  n_rows
  32 u32  n_cols          (元素数 N = n_rows*n_cols)
  36 u32  offset          (绝对文件偏移)
  40 u32  nbytes          (存储字节数)
  44 u32  reserved

q4_0 存储：**每行先按 32 对齐**（行尾零填充），故行与块一一对齐：
  第 r 行第 b 块 → 扁平块号 r*nblk + b（nblk = ceil(cols/32)）。
  每块 18 字节：
  [0,2)   fp16 scale d
  [2,18)  16 字节：第 k 字节低半字节 = 元素 2k，高半字节 = 元素 2k+1
  反量化： w = (nibble - 8) * d
  d = max_abs / 8，量化值 q = round(w/d) 截断到 [-8,7]
  目录中的 n_cols 仍记录**原始列数**；MCU 按 align32(n_cols) 推导 nblk。

用法： python convert_minueza.py                          # 稠密 Minueza
      python convert_minueza.py --src E:/models/moe_full_ce_ffne128 --out E:/models/model_moe.kmcu
                            # MoE 模型（自动检测 router，专家分类存储 + 独立 lm_head）
"""
import json
import os
import struct
import sys

import numpy as np

SRC_DIR = r"E:\models"
OUT_PATH = r"E:\models\model.kmcu"

BLOCK = 32
Q4_BYTES = 18
HEADER_BYTES = 128
DIR_ENTRY_BYTES = 48
ALIGN = 64

DT_F32 = 1
DT_Q4 = 2
DT_TERNARY = 3   # 三值查表（PLE table）：base-3 打包，5 trit/字节，行对齐


def q4_quantize(w: np.ndarray):
    """扁平 q4_0 量化。返回 (bytes, nbytes, n_blocks)。"""
    flat = np.asarray(w, dtype=np.float32).reshape(-1)
    n = flat.size
    nb = (n + BLOCK - 1) // BLOCK
    if nb * BLOCK > n:
        flat = np.concatenate([flat, np.zeros(nb * BLOCK - n, np.float32)])
    blocks = flat.reshape(nb, BLOCK)

    amax = np.abs(blocks).max(axis=1).astype(np.float32)
    d = np.where(amax > 0.0, amax / 8.0, 1.0).astype(np.float32)

    q = np.rint(blocks / d[:, None])
    q = np.clip(q, -8.0, 7.0).astype(np.int32)
    nib = (q + 8).astype(np.uint8)          # [0,15]

    out = np.zeros((nb, Q4_BYTES), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    lo = nib[:, 0::2]                        # 元素 2k
    hi = nib[:, 1::2]                        # 元素 2k+1
    out[:, 2:18] = (lo | (hi << 4)).astype(np.uint8)
    return out.tobytes(), nb * Q4_BYTES


def f32_bytes(w: np.ndarray) -> bytes:
    return np.asarray(w, dtype=np.float32).tobytes()


def ternary_pack(w: np.ndarray):
    """三值量化 PLE 查表：{-1,0,+1} × gamma，base-3 打包（5 trit/字节，行对齐）。

    返回 (payload_bytes, gamma)。gamma = mean(|w|)（与训练 STE 口径一致）；
    code = clip(round(w/gamma), -1, 1)；trit = code + 1 ∈ {0,1,2}；
    每字节装 5 个 trit：byte = sum(trit_i * 3^i)，每行独立对齐到 ceil(cols/5) 字节。
    """
    w = np.asarray(w, dtype=np.float32)
    assert w.ndim == 2, "ternary table 必须是 [vocab, table_width]"
    vocab, width = w.shape
    gamma = float(np.abs(w).mean())
    codes = np.clip(np.rint(w / gamma), -1.0, 1.0).astype(np.int8)   # {-1,0,1}
    trit = (codes + 1).astype(np.uint8)                              # {0,1,2}
    row_bytes = (width + 4) // 5
    padded = np.zeros((vocab, row_bytes * 5), dtype=np.uint8)
    padded[:, :width] = trit
    trit3 = padded.reshape(vocab, row_bytes, 5)
    powers = np.array([1, 3, 9, 27, 81], dtype=np.uint32)
    byte = (trit3.astype(np.uint32) * powers).sum(axis=2).astype(np.uint8)
    return byte.tobytes(), gamma


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DIR, help="源模型目录（含 config.json + model.safetensors）")
    ap.add_argument("--out", default=OUT_PATH, help="输出 KMCU 文件路径")
    ap.add_argument("--top_k", type=int, default=2, help="MoE top-k（稠密忽略）")
    args = ap.parse_args()

    cfg_path = os.path.join(args.src, "config.json")
    st_path = os.path.join(args.src, "model.safetensors")
    if not os.path.isfile(cfg_path) or not os.path.isfile(st_path):
        print("缺少 config.json 或 model.safetensors，请先下载")
        return 1

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dim = cfg["hidden_size"]
    nl = cfg["num_hidden_layers"]
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    hd = dim // nh
    ffn = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    max_seq = cfg["max_position_embeddings"]
    rope_theta = float(cfg.get("rope_theta",
                               (cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0)))
    eps = float(cfg["rms_norm_eps"])
    bos = int(cfg.get("bos_token_id", 1))
    eos = int(cfg.get("eos_token_id", 2))

    from safetensors.numpy import load_file
    print("loading safetensors ...")
    sd = load_file(st_path)
    keys = set(sd.keys())

    # MoE 检测：存在 router 权重即为 MoE 模型
    is_moe = "model.layers.0.mlp.router.weight" in keys
    n_expert = 0
    top_k = 0
    ffn_e = 0
    if is_moe:
        n_expert = int(cfg.get("n_expert") or sd["model.layers.0.mlp.router.weight"].shape[0])
        top_k = int(cfg.get("top_k", args.top_k))
        ffn_e = int(sd["model.layers.0.mlp.experts.0.gate.weight"].shape[0])
        ffn = ffn_e  # header 的 ffn_dim 存专家维度
        print(f"[moe] n_expert={n_expert} top_k={top_k} ffn_e={ffn_e}")

    # PLE：ple_dim > 0 表示存在 per-layer embedding 查表
    ple_dim = int(cfg.get("ple_dim", 0))
    has_ple = ple_dim > 0

    # tied head：优先读配置，回退到历史行为（稠密 tied、MoE 独立）
    if "tie_word_embeddings" in cfg:
        tied = 1 if cfg["tie_word_embeddings"] else 0
    else:
        tied = 0 if is_moe else 1

    has_lm_head = "lm_head.weight" in keys

    # 决策头：cls_head.weight 存在即挂决策头（分类头，可选分支）
    n_cls = 0
    if "cls_head.weight" in keys:
        n_cls = int(sd["cls_head.weight"].shape[0])
        print(f"[cls] n_cls={n_cls}")

    tensors = []   # (name, rows, cols, dtype, payload_bytes)

    def add(name, w, dtype):
        w = np.asarray(w, dtype=np.float32)
        assert w.ndim == 2 or w.ndim == 1, name
        if w.ndim == 1:
            rows, cols = 1, w.size
        else:
            rows, cols = w.shape
        if dtype == DT_Q4:
            # 行对齐到 32：每行补齐成整数个 32 元素块（零填充），使每块只有
            # 一个 scale、且块不跨行。MCU 侧 GEMV 的块循环因此变统一，
            # 可完全展开，并能按 (row, block) 规整寻址。
            w2 = w.reshape(rows, cols)
            if cols % 32 != 0:
                w2 = np.pad(w2, ((0, 0), (0, 32 - (cols % 32))))
            payload, nb = q4_quantize(w2)
        else:
            payload = f32_bytes(w)
            nb = len(payload)
        tensors.append((name, rows, cols, dtype, payload, nb))

    # embed
    add("tok_embed", sd["model.embed_tokens.weight"], DT_Q4)

    for i in range(nl):
        p = f"model.layers.{i}."
        add(f"L{i}.in_ln", sd[p + "input_layernorm.weight"], DT_F32)
        add(f"L{i}.q", sd[p + "self_attn.q_proj.weight"], DT_Q4)
        add(f"L{i}.k", sd[p + "self_attn.k_proj.weight"], DT_Q4)
        add(f"L{i}.v", sd[p + "self_attn.v_proj.weight"], DT_Q4)
        add(f"L{i}.o", sd[p + "self_attn.o_proj.weight"], DT_Q4)
        add(f"L{i}.post_ln", sd[p + "post_attention_layernorm.weight"], DT_F32)
        if is_moe:
            # 专家分类存储：router(f32) + 每个专家 gate/up/down(q4)
            add(f"L{i}.router", sd[p + "mlp.router.weight"], DT_F32)
            for e in range(n_expert):
                ep = p + f"mlp.experts.{e}."
                add(f"L{i}.e{e}.gate", sd[ep + "gate.weight"], DT_Q4)
                add(f"L{i}.e{e}.up", sd[ep + "up.weight"], DT_Q4)
                add(f"L{i}.e{e}.down", sd[ep + "down.weight"], DT_Q4)
        else:
            add(f"L{i}.gate", sd[p + "mlp.gate_proj.weight"], DT_Q4)
            add(f"L{i}.up", sd[p + "mlp.up_proj.weight"], DT_Q4)
            add(f"L{i}.down", sd[p + "mlp.down_proj.weight"], DT_Q4)
        if has_ple:
            add(f"L{i}.ple_gate", sd[p + "ple_gate.weight"], DT_Q4)
            add(f"L{i}.ple_proj", sd[p + "ple_proj.weight"], DT_Q4)
            add(f"L{i}.ple_norm", sd[p + "ple_norm.weight"], DT_F32)

    if has_ple:
        add("ple_model_proj", sd["model.ple_model_proj.weight"], DT_Q4)
        add("ple_proj_norm", sd["model.ple_proj_norm.weight"], DT_F32)

    add("final_norm", sd["model.norm.weight"], DT_F32)
    # 决策头（可选，n_cls=0 不存；f32 存储，决策头小无需量化）
    if n_cls > 0:
        add("cls_head", sd["cls_head.weight"], DT_F32)
    # 独立 lm_head（tied=0 时存储；tied=1 时复用 embed，不存）
    if not tied and has_lm_head:
        add("lm_head", sd["lm_head.weight"], DT_Q4)

    # PLE 三值查表（base-3 打包，行对齐，供 flash 逐行随机读）
    ple_gamma = 0.0
    if has_ple:
        table_w = sd["model.ple_table.weight"]
        payload, ple_gamma = ternary_pack(table_w)
        tensors.append(("ple_table", table_w.shape[0], table_w.shape[1],
                        DT_TERNARY, payload, len(payload)))

    # 统计
    n_tensors = len(tensors)
    dir_offset = HEADER_BYTES
    data_offset = dir_offset + n_tensors * DIR_ENTRY_BYTES
    data_offset = (data_offset + ALIGN - 1) // ALIGN * ALIGN

    # 计算布局
    entries = []
    off = data_offset
    n_params = 0
    n_q4 = 0
    for (name, rows, cols, dtype, payload, nb) in tensors:
        entries.append((name, rows, cols, dtype, off, nb))
        off = (off + nb + ALIGN - 1) // ALIGN * ALIGN
        n_params += rows * cols
        if dtype == DT_Q4:
            n_q4 += rows * cols

    total_bytes = off
    print(f"n_tensors={n_tensors}  params={n_params/1e6:.2f}M  q4={n_q4/1e6:.2f}M")
    print(f"file size = {total_bytes/1024/1024:.2f} MB   (tied_embed={tied}, "
          f"moe={is_moe} n_expert={n_expert} top_k={top_k} ple_dim={ple_dim} "
          f"ple_gamma={ple_gamma:.6f})")

    # 写文件
    with open(args.out, "wb") as f:
        hdr = bytearray(HEADER_BYTES)
        hdr[0:4] = b"KMCU"
        struct.pack_into("<I", hdr, 4, 1)
        struct.pack_into("<I", hdr, 8, nl)
        struct.pack_into("<I", hdr, 12, dim)
        struct.pack_into("<I", hdr, 16, nh)
        struct.pack_into("<I", hdr, 20, nkv)
        struct.pack_into("<I", hdr, 24, hd)
        struct.pack_into("<I", hdr, 28, ffn)
        struct.pack_into("<I", hdr, 32, vocab)
        struct.pack_into("<I", hdr, 36, max_seq)
        struct.pack_into("<I", hdr, 40, bos)
        struct.pack_into("<I", hdr, 44, eos)
        struct.pack_into("<I", hdr, 48, tied)
        struct.pack_into("<I", hdr, 52, n_tensors)
        struct.pack_into("<I", hdr, 56, dir_offset)
        struct.pack_into("<I", hdr, 60, data_offset)
        struct.pack_into("<f", hdr, 64, rope_theta)
        struct.pack_into("<f", hdr, 68, eps)
        struct.pack_into("<I", hdr, 72, n_params)
        struct.pack_into("<I", hdr, 76, n_expert)   # 0=稠密，>0=MoE
        struct.pack_into("<I", hdr, 80, top_k)      # MoE top-k
        struct.pack_into("<I", hdr, 84, ple_dim)    # PLE 每层查表宽度（0=无 PLE）
        struct.pack_into("<f", hdr, 88, ple_gamma)  # PLE 三值表 scale
        struct.pack_into("<I", hdr, 92, n_cls)      # 决策头类别数（0=无决策头）
        f.write(bytes(hdr))

        for (name, rows, cols, dtype, offset, nb) in entries:
            e = bytearray(DIR_ENTRY_BYTES)
            nb_name = name.encode("ascii")[:24]
            e[0:len(nb_name)] = nb_name
            e[24] = dtype
            struct.pack_into("<I", e, 28, rows)
            struct.pack_into("<I", e, 32, cols)
            struct.pack_into("<I", e, 36, offset)
            struct.pack_into("<I", e, 40, nb)
            f.write(bytes(e))

        # 数据区（按 entry 顺序，含对齐填充）
        # 注意：header+目录 到 data_offset 之间可能有对齐 gap，必须先补齐，
        # 否则目录项 offset 与实际文件位置错位（稠密模型 gap=0 未暴露，MoE 模型暴露）。
        gap = data_offset - (HEADER_BYTES + n_tensors * DIR_ENTRY_BYTES)
        if gap > 0:
            f.write(b"\x00" * gap)
        cur = data_offset
        for (t, e) in zip(tensors, entries):
            payload = t[4]
            assert e[4] == cur, (t[0], e[4], cur)
            f.write(payload)
            cur += len(payload)
            pad = (ALIGN - (cur % ALIGN)) % ALIGN
            if pad:
                f.write(b"\x00" * pad)
                cur += pad

    print(f"written: {args.out}")

    # 导出 manifest 供人工核对
    with open(os.path.join(args.src, "model.kmcu.manifest.txt"), "w", encoding="utf-8") as f:
        f.write(f"dim={dim} nl={nl} nh={nh} nkv={nkv} hd={hd} ffn={ffn} "
                f"vocab={vocab} max_seq={max_seq} theta={rope_theta} eps={eps}\n")
        f.write(f"tied_embed={tied} n_expert={n_expert} top_k={top_k} ple_dim={ple_dim} "
                f"ple_gamma={ple_gamma:.6f} n_cls={n_cls} n_tensors={n_tensors} "
                f"data_offset={data_offset} file_size={total_bytes}\n")
        for (name, rows, cols, dtype, offset, nb) in entries:
            f.write(f"{name:16s} dtype={dtype} shape=({rows},{cols}) off={offset} nb={nb}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
