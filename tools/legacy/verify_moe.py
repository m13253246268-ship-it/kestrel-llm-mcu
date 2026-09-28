#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_moe.py - 稠密 → MoE 改造的精度上限验证（oracle router）。

核心问题：把 FFN 的 ffn=1092 个神经元聚类成 N=8 组，每 token 只激活 top-k=2 组
（即 2/8 的神经元），MoE 输出能否逼近稠密 FFN 输出？

用「oracle router」给出精度上限：对每个 token 穷举/贪心选 top-k 专家，使
y_moe 最接近 y_dense。若 oracle 精度都不够，训练 router 也无意义。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # tools/：ref_forward 等主线模块
sys.path.insert(0, _HERE)                   # legacy/：同目录模块
import ref_forward as rf
from ref_forward import rms_norm, rope_cache, apply_rope
from verify_ann_head import kmeans


def collect_layer_acts(arch, T, tokens):
    """跑 forward，收集每层 MLP 输入 h 与稠密 FFN 输出 y_dense（[seq, dim]）。"""
    nl, dim, nh, nkv, hd, ffn = (arch['nl'], arch['dim'], arch['nh'],
                                 arch['nkv'], arch['hd'], arch['ffn'])
    eps, theta = arch['eps'], arch['rope_theta']
    seq = len(tokens)
    scale = 1.0 / np.sqrt(np.float32(hd))
    cos, sin = rope_cache(hd, seq, theta)
    x = T['tok_embed'][np.asarray(tokens, dtype=np.int64)]
    kv_rep = nh // nkv
    mask = np.triu(np.full((seq, seq), -1e30, np.float32), k=1)

    acts = [None] * nl   # acts[L] = (h[seq,dim], y_dense[seq,dim])

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
        g = h @ T[p + 'gate'].T
        u = h @ T[p + 'up'].T
        sg = g / (1.0 + np.exp(-g))
        y_dense = (sg * u) @ T[p + 'down'].T
        acts[L] = (h, y_dense)

        x = x + y_dense

    return acts


def main():
    arch, T = rf.load_kmcu(rf.KM)
    nl = arch['nl']

    # 生成 16 token，收集每层激活
    prompt = [1, 450, 2217, 4996]
    toks = list(prompt)
    # 用完整 forward 生成，同时收集激活（每步重新收集）
    acts_all = []
    for _ in range(16):
        acts = collect_layer_acts(arch, T, toks)
        acts_all.append([a[0][-1] for a in acts])   # 每层最后一个 token 的 h
        nxt = int(np.argmax(acts[-1][1][-1] @ T['tok_embed'].T))
        toks.append(nxt)
    # 需要每层 h 和 y_dense 的完整序列；重新收集一次完整 20 token
    acts = collect_layer_acts(arch, T, toks)
    seq = len(toks)

    N = 8
    top_k = 2
    print(f"layers={nl} N={N} top_k={top_k} seq={seq}")

    rel_err_all = []
    for L in range(nl):
        h, y_dense = acts[L]                 # [seq,dim]
        p = f"L{L}."
        gate = T[p + 'gate']                 # [ffn, dim]
        up = T[p + 'up']                     # [ffn, dim]
        down = T[p + 'down']                 # [dim, ffn]
        ffn = gate.shape[0]

        # 神经元特征 = gate/up 行拼接，归一化后 K-means 聚类
        feat = np.concatenate([gate, up], axis=1)       # [ffn, 2*dim]
        feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
        labels, _ = kmeans(feat.astype(np.float32), N)

        # 每个专家的神经元索引
        # oracle：对每个 token，穷举 C(N,k) 组合选使 y_moe 最接近 y_dense 的 top-k 专家
        import itertools
        combos = list(itertools.combinations(range(N), top_k))   # C(8,2)=28

        def expert_ffn(e_idx, hh):
            mem = labels == e_idx
            if not mem.any():
                return np.zeros_like(hh[:, :1] @ down[:1, :1].T)
            g = hh @ gate[mem].T
            u = hh @ up[mem].T
            sg = g / (1.0 + np.exp(-g))
            return (sg * u) @ down[:, mem].T

        # 预计算每个专家的输出 [seq, dim]
        expert_outs = [expert_ffn(e, h) for e in range(N)]

        # oracle 选 top-k（穷举 28 组合）
        y_moe = np.zeros_like(y_dense)
        for t in range(seq):
            best = None
            best_comb = None
            for comb in combos:
                yc = sum(expert_outs[e][t] for e in comb) / top_k
                err = np.sum((yc - y_dense[t]) ** 2)
                if best is None or err < best:
                    best = err
                    best_comb = comb
            y_moe[t] = sum(expert_outs[e][t] for e in best_comb) / top_k

        rel = np.linalg.norm(y_moe - y_dense, axis=1) / np.linalg.norm(y_dense, axis=1)
        rel_err_all.append(rel.mean())
        if L < 2 or L == nl - 1:
            print(f"  L{L}: mean rel err = {rel.mean():.4f}  (max {rel.max():.4f})")

    print(f"\n整体平均相对误差（oracle top-{top_k}）：{np.mean(rel_err_all):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
