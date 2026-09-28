#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ref_forward.py - 读取 KMCU 量化权重，用 NumPy 做 fp32 参考前向，
给出贪心解码的 token 序列，作为 MCU 端实现的唯一对拍基准。

重要：本参考与 MCU 使用**同一份量化权重**（q4_0 反量化后），
因此二者差异只应来自浮点累加顺序/舍入，不应来自权重本身。
"""
import os
import struct
import sys

import numpy as np

try:
    from scipy.special import erf
except ImportError:
    import math
    erf = np.vectorize(math.erf)

KM = os.environ.get("KMCU_KM", "model.kmcu")

DT_F32 = 1
DT_Q4 = 2
DT_TERNARY = 3
BLOCK = 32
Q4_BYTES = 18


def dequant_q4(payload: bytes, n_elem: int) -> np.ndarray:
    buf = np.frombuffer(payload, dtype=np.uint8)
    nb = buf.size // Q4_BYTES
    b = buf.reshape(nb, Q4_BYTES)
    d = b[:, 0:2].copy().view(np.float16).reshape(nb).astype(np.float32)
    packed = b[:, 2:18].astype(np.uint16)
    lo = (packed & 0x0F).astype(np.int32) - 8
    hi = ((packed >> 4) & 0x0F).astype(np.int32) - 8
    out = np.empty((nb, BLOCK), np.int32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    w = (out.astype(np.float32) * d[:, None]).reshape(-1)
    return w[:n_elem].astype(np.float32)


def dequant_ternary(payload: bytes, rows: int, cols: int, gamma: float) -> np.ndarray:
    """base-3 打包三值表 → [rows, cols] fp32（code × gamma）。"""
    buf = np.frombuffer(payload, dtype=np.uint8)
    row_bytes = (cols + 4) // 5
    b = buf.reshape(rows, row_bytes).astype(np.uint32)[:, :, None]
    powers = np.array([1, 3, 9, 27, 81], dtype=np.uint32)
    trit = (b // powers[None, None, :]) % 3                    # [rows, row_bytes, 5]
    code = trit.astype(np.float32) - 1.0                        # {-1,0,1}
    flat = code.reshape(rows, row_bytes * 5)
    return (flat[:, :cols] * np.float32(gamma)).astype(np.float32)


def load_kmcu(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[0:4] == b"KMCU", "bad magic"
    (ver, nl, dim, nh, nkv, hd, ffn, vocab, max_seq, bos, eos,
     tied, n_tensors, dir_off, data_off) = struct.unpack_from("<15I", raw, 4)
    rope_theta, eps = struct.unpack_from("<2f", raw, 64)
    total_params = struct.unpack_from("<I", raw, 72)[0]
    n_expert = struct.unpack_from("<I", raw, 76)[0]
    top_k = struct.unpack_from("<I", raw, 80)[0]
    ple_dim = struct.unpack_from("<I", raw, 84)[0]
    ple_gamma = struct.unpack_from("<f", raw, 88)[0]
    n_cls = struct.unpack_from("<I", raw, 92)[0]
    assert ver == 1

    arch = dict(nl=nl, dim=dim, nh=nh, nkv=nkv, hd=hd, ffn=ffn, vocab=vocab,
                max_seq=max_seq, bos=bos, eos=eos, tied=tied,
                rope_theta=rope_theta, eps=eps, total_params=total_params,
                n_expert=n_expert, top_k=top_k, ple_dim=ple_dim, ple_gamma=ple_gamma,
                n_cls=n_cls)

    T = {}
    for i in range(n_tensors):
        base = dir_off + i * 48
        name = raw[base:base + 24].split(b"\x00")[0].decode("ascii")
        dtype = raw[base + 24]
        rows, cols, off, nb = struct.unpack_from("<4I", raw, base + 28)
        payload = raw[off:off + nb]
        n = rows * cols
        if dtype == DT_Q4:
            # 行对齐到 32：存储长度 = rows * align32(cols)
            align = (cols + 31) // 32 * 32
            wpad = dequant_q4(payload, rows * align).reshape(rows, align)
            T[name] = wpad[:, :cols].copy()
        elif dtype == DT_TERNARY:
            T[name] = dequant_ternary(payload, rows, cols, ple_gamma)
        else:
            w = np.frombuffer(payload, dtype=np.float32)[:n].copy()
            T[name] = w.reshape(rows, cols) if rows != 1 else w.reshape(-1)
    return arch, T


def rms_norm(x, w, eps):
    # x: [..., dim]
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def gelu(x):
    # 精确 erf 版 GELU（与 F.gelu 默认 / C erff 一致）
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def rope_cache(hd, seq, theta):
    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / hd))
    pos = np.arange(seq, dtype=np.float32)
    fr = np.outer(pos, inv)              # [seq, half]
    emb = np.concatenate([fr, fr], axis=-1)   # [seq, hd]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x, cos, sin):
    # x: [seq, nh, hd]
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


def forward(arch, T, tokens):
    nl, dim, nh, nkv, hd, ffn = arch['nl'], arch['dim'], arch['nh'], arch['nkv'], arch['hd'], arch['ffn']
    eps, theta = arch['eps'], arch['rope_theta']
    seq = len(tokens)
    scale = 1.0 / np.sqrt(np.float32(hd))

    cos, sin = rope_cache(hd, seq, theta)

    x = T['tok_embed'][np.asarray(tokens, dtype=np.int64)]     # [seq, dim]

    # PLE 底部：ple = (RMSNorm(ple_model_proj(x)·dim^-0.5) + table·ple_dim^0.5) · 2^-0.5
    ple = None
    if arch.get('ple_dim', 0) > 0:
        pd = arch['ple_dim']
        ple = (x @ T['ple_model_proj'].T) * (dim ** -0.5)      # [seq, nl*pd]
        ple = rms_norm(ple.reshape(seq, nl, pd), T['ple_proj_norm'], eps)
        table = T['ple_table'][np.asarray(tokens, dtype=np.int64)].reshape(seq, nl, pd)
        ple = (ple + table * (pd ** 0.5)) * (2 ** -0.5)

    kv_rep = nh // nkv
    mask = np.triu(np.full((seq, seq), -1e30, np.float32), k=1)

    for L in range(nl):
        p = f"L{L}."
        h = rms_norm(x, T[p + 'in_ln'], eps)

        q = h @ T[p + 'q'].T          # [seq, nh*hd]
        k = h @ T[p + 'k'].T          # [seq, nkv*hd]
        v = h @ T[p + 'v'].T

        q = q.reshape(seq, nh, hd)
        k = k.reshape(seq, nkv, hd)
        v = v.reshape(seq, nkv, hd)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: 每 kv head 服务 kv_rep 个 q head
        k = np.repeat(k, kv_rep, axis=1)     # [seq, nh, hd]
        v = np.repeat(v, kv_rep, axis=1)

        # [nh, seq, seq]
        att = np.einsum('thd,shd->hts', q, k) * scale
        att = att + mask[None, :, :]
        att = att - att.max(axis=-1, keepdims=True)
        e = np.exp(att)
        att = e / e.sum(axis=-1, keepdims=True)

        o = np.einsum('hts,shd->thd', att, v).reshape(seq, nh * hd)
        x = x + o @ T[p + 'o'].T

        h = rms_norm(x, T[p + 'post_ln'], eps)
        if arch.get('n_expert', 0) > 0:
            # MoE FFN：router → softmax → top-k 硬选择 → 专家加权（逐行独立）
            rlog = h @ T[p + 'router'].T                      # [seq, n_expert]
            rp = np.exp(rlog - rlog.max(axis=-1, keepdims=True))
            rp = rp / rp.sum(axis=-1, keepdims=True)          # softmax
            topk = arch['top_k']
            idx = np.argsort(-rp, axis=-1)[:, :topk]          # top-k 专家索引
            moe_out = np.zeros_like(h)
            for s in range(h.shape[0]):
                for e in idx[s]:
                    w = rp[s, e] / rp[s, idx[s]].sum()        # 归一化 hard 权重
                    g = h[s] @ T[p + f'e{e}.gate'].T
                    u = h[s] @ T[p + f'e{e}.up'].T
                    sg = g / (1.0 + np.exp(-g))
                    moe_out[s] += w * ((sg * u) @ T[p + f'e{e}.down'].T)
            x = x + moe_out
        else:
            g = h @ T[p + 'gate'].T
            u = h @ T[p + 'up'].T
            sg = g / (1.0 + np.exp(-g))          # silu
            x = x + (sg * u) @ T[p + 'down'].T

        # PLE 分支（attention + ffn 之后）
        if ple is not None:
            g = gelu(x @ T[p + 'ple_gate'].T)               # [seq, pd]
            x = x + rms_norm((g * ple[:, L, :]) @ T[p + 'ple_proj'].T,
                             T[p + 'ple_norm'], eps)

    x = rms_norm(x, T['final_norm'], eps)
    # tied=1 时 lm_head 复用 tok_embed；tied=0 时用独立 lm_head
    if arch.get('tied', 1):
        logits = x @ T['tok_embed'].T            # [seq, vocab]
    else:
        logits = x @ T['lm_head'].T
    return logits


def cls_forward(arch, T, tokens):
    """决策头前向：transformer → final_norm → mean pooling → cls_head → [n_cls]。

    复用 forward 的 transformer 部分（不碰生成路径），最后一步不接 lm_head，
    而是 mean pool 后接决策头 cls_head（n_cls=0 时无此张量，调用方需保证 n_cls>0）。
    """
    nl, dim, nh, nkv, hd, ffn = arch['nl'], arch['dim'], arch['nh'], arch['nkv'], arch['hd'], arch['ffn']
    eps, theta = arch['eps'], arch['rope_theta']
    seq = len(tokens)
    scale = 1.0 / np.sqrt(np.float32(hd))

    cos, sin = rope_cache(hd, seq, theta)
    x = T['tok_embed'][np.asarray(tokens, dtype=np.int64)]     # [seq, dim]

    ple = None
    if arch.get('ple_dim', 0) > 0:
        pd = arch['ple_dim']
        ple = (x @ T['ple_model_proj'].T) * (dim ** -0.5)
        ple = rms_norm(ple.reshape(seq, nl, pd), T['ple_proj_norm'], eps)
        table = T['ple_table'][np.asarray(tokens, dtype=np.int64)].reshape(seq, nl, pd)
        ple = (ple + table * (pd ** 0.5)) * (2 ** -0.5)

    kv_rep = nh // nkv
    mask = np.triu(np.full((seq, seq), -1e30, np.float32), k=1)

    for L in range(nl):
        p = f"L{L}."
        h = rms_norm(x, T[p + 'in_ln'], eps)
        q = h @ T[p + 'q'].T
        k = h @ T[p + 'k'].T
        v = h @ T[p + 'v'].T
        q = q.reshape(seq, nh, hd)
        k = k.reshape(seq, nkv, hd)
        v = v.reshape(seq, nkv, hd)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = np.repeat(k, kv_rep, axis=1)
        v = np.repeat(v, kv_rep, axis=1)
        att = np.einsum('thd,shd->hts', q, k) * scale
        att = att + mask[None, :, :]
        att = att - att.max(axis=-1, keepdims=True)
        e = np.exp(att)
        att = e / e.sum(axis=-1, keepdims=True)
        o = np.einsum('hts,shd->thd', att, v).reshape(seq, nh * hd)
        x = x + o @ T[p + 'o'].T
        h = rms_norm(x, T[p + 'post_ln'], eps)
        if arch.get('n_expert', 0) > 0:
            rlog = h @ T[p + 'router'].T
            rp = np.exp(rlog - rlog.max(axis=-1, keepdims=True))
            rp = rp / rp.sum(axis=-1, keepdims=True)
            topk = arch['top_k']
            idx = np.argsort(-rp, axis=-1)[:, :topk]
            moe_out = np.zeros_like(h)
            for s in range(h.shape[0]):
                for e in idx[s]:
                    w = rp[s, e] / rp[s, idx[s]].sum()
                    g = h[s] @ T[p + f'e{e}.gate'].T
                    u = h[s] @ T[p + f'e{e}.up'].T
                    sg = g / (1.0 + np.exp(-g))
                    moe_out[s] += w * ((sg * u) @ T[p + f'e{e}.down'].T)
            x = x + moe_out
        else:
            g = h @ T[p + 'gate'].T
            u = h @ T[p + 'up'].T
            sg = g / (1.0 + np.exp(-g))
            x = x + (sg * u) @ T[p + 'down'].T
        if ple is not None:
            g = gelu(x @ T[p + 'ple_gate'].T)
            x = x + rms_norm((g * ple[:, L, :]) @ T[p + 'ple_proj'].T,
                             T[p + 'ple_norm'], eps)

    x = rms_norm(x, T['final_norm'], eps)          # [seq, dim]
    x = x.mean(axis=0)                             # mean pooling → [dim]
    return x @ T['cls_head'].T                     # [n_cls]


def generate(arch, T, prompt, n_new, rep_penalty=1.0):
    """贪心解码。rep_penalty>1.0 时对已出现 token 的 logit 做重复惩罚
    （正 logit 除以系数、负 logit 乘以系数，与 kmcu.c 的 REP_PENALTY 语义一致）。"""
    toks = list(prompt)
    for _ in range(n_new):
        logits = forward(arch, T, toks)
        last = logits[-1].copy()
        if rep_penalty > 1.0:
            for s in set(toks):
                if 0 <= s < last.shape[0]:
                    last[s] = last[s] / rep_penalty if last[s] >= 0.0 else last[s] * rep_penalty
        nxt = int(np.argmax(last))
        toks.append(nxt)
    return toks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--km", default=KM, help="KMCU 文件路径")
    ap.add_argument("--ref_out", default=os.environ.get("KMCU_REF_OUT", "ref_out.txt"))
    args = ap.parse_args()
    if not os.path.isfile(args.km):
        print("缺少", args.km)
        return 1
    arch, T = load_kmcu(args.km)
    print("arch:", {k: arch[k] for k in
                    ('nl', 'dim', 'nh', 'nkv', 'hd', 'ffn', 'vocab', 'tied',
                     'rope_theta', 'eps', 'total_params', 'n_expert', 'top_k')})
    print("tensors:", len(T))

    prompt = [1, 450, 2217, 4996]
    print("prompt:", prompt)

    # 二分定位基准：单 token（pos=0 ⇒ RoPE 恒等、注意力退化）
    lg1 = forward(arch, T, [1])
    l1 = lg1[-1]
    t1 = np.argsort(-l1)[:8]
    print("ref single-token[1] top8:")
    for r, v in enumerate(t1):
        print(f"  #{r} id={int(v)} logit={float(l1[v]):.6f}")

    # 单次 forward 到完整 prompt，打印末位 top-8 供与 MCU 逐值对拍
    lg = forward(arch, T, prompt)
    last = lg[-1]
    top = np.argsort(-last)[:8]
    print("ref top8 @first-gen:")
    for r, v in enumerate(top):
        print(f"  #{r} id={int(v)} logit={float(last[v]):.6f}")

    out = generate(arch, T, prompt, 16)
    print("ref greedy:", out)

    # 决策头对拍（n_cls>0 时：与 host_verify.c 的 km_cls_forward 逐值比对）
    if arch.get('n_cls', 0) > 0:
        cls_tokens = [1, 450, 2217, 4996, 522, 28723, 415, 907]
        cl = cls_forward(arch, T, cls_tokens)
        print("ref cls:", [round(float(v), 6) for v in cl],
              "id=", int(np.argmax(cl)))

    with open(args.ref_out, "w", encoding="ascii") as f:
        f.write(",".join(str(t) for t in out))
    print("saved", args.ref_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
