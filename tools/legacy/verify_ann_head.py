#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_ann_head.py - 输出头 ANN 稀疏查找可行性验证（PLE 思想，H4）。

把 tied lm_head（32002 行的嵌入表）从「密集 argmax」改成「K-means 聚类 + 选簇 + 精确算候选」，
测召回率（真 top-1 是否在候选内）与剪枝率（候选行数 / vocab）。

验证目标：找到「召回率 100%（或可控降级）下剪枝率最大」的 (C, K) 组合。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # tools/：ref_forward 等主线模块
sys.path.insert(0, _HERE)                   # legacy/：同目录模块
import ref_forward as rf
from ref_forward import rms_norm, rope_cache, apply_rope


def get_hs(arch, T, tokens):
    """跑 forward 到 final_norm，返回每 token 的隐藏状态 x（[seq, dim]）。"""
    nl, dim, nh, nkv, hd, ffn = (arch['nl'], arch['dim'], arch['nh'],
                                arch['nkv'], arch['hd'], arch['ffn'])
    eps, theta = arch['eps'], arch['rope_theta']
    seq = len(tokens)
    scale = 1.0 / np.sqrt(np.float32(hd))
    cos, sin = rope_cache(hd, seq, theta)

    x = T['tok_embed'][np.asarray(tokens, dtype=np.int64)]
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
        g = h @ T[p + 'gate'].T
        u = h @ T[p + 'up'].T
        sg = g / (1.0 + np.exp(-g))
        x = x + (sg * u) @ T[p + 'down'].T

    return rms_norm(x, T['final_norm'], eps)


def kmeans(X, C, iters=20, seed=0):
    """手写 K-means（K-means++ 初始化），返回 (labels, centers)。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, C):
        dots = X @ np.asarray(centers).T
        xn = np.sum(X * X, axis=1)
        cn = np.sum(np.asarray(centers) ** 2, axis=1)
        d2 = np.min(xn[:, None] - 2 * dots + cn[None, :], axis=1)
        d2 = np.clip(d2, 0, None)
        if d2.sum() <= 0:
            d2 = np.ones(n)
        centers.append(X[rng.choice(n, p=d2 / d2.sum())])
    centers = np.asarray(centers, dtype=np.float32)

    for _ in range(iters):
        dots = X @ centers.T
        xn = np.sum(X * X, axis=1)
        cn = np.sum(centers * centers, axis=1)
        d2 = xn[:, None] - 2 * dots + cn[None, :]
        labels = np.argmin(d2, axis=1)
        new_centers = np.empty_like(centers)
        for c in range(C):
            mem = labels == c
            new_centers[c] = X[mem].mean(axis=0) if mem.any() else centers[c]
        if np.allclose(centers, new_centers, rtol=1e-4, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    dots = X @ centers.T
    xn = np.sum(X * X, axis=1)
    cn = np.sum(centers * centers, axis=1)
    d2 = xn[:, None] - 2 * dots + cn[None, :]
    labels = np.argmin(d2, axis=1)
    return labels, centers


def main():
    arch, T = rf.load_kmcu(rf.KM)
    embed = T['tok_embed'].astype(np.float32)   # [vocab, dim]
    vocab, dim = embed.shape
    print(f"vocab={vocab} dim={dim}")

    prompt = [1, 450, 2217, 4996]
    # 生成 16 token，共 20 个 h
    toks = list(prompt)
    hs = []
    for _ in range(16):
        h = get_hs(arch, T, toks)[-1]
        hs.append(h)
        nxt = int(np.argmax(h @ embed.T))
        toks.append(nxt)
    hs = np.asarray(hs)          # [16, dim]
    n_tok = hs.shape[0]
    print(f"n_tokens={n_tok}")

    # 真 top-1 基线
    true_logits = hs @ embed.T
    true_top1 = np.argmax(true_logits, axis=1)

    # 归一化 embed / h（模长 1），spherical k-means（此时欧氏距离 = 余弦距离）
    embed_n = embed / np.linalg.norm(embed, axis=1, keepdims=True)
    hs_n = hs / np.linalg.norm(hs, axis=1, keepdims=True)

    print(f"\n{'C':>5} {'K':>3} {'召回':>6} {'平均候选':>8} {'剪枝率':>8} {'ANN正确':>8}")
    for C in (64, 128, 256, 512):
        labels, centers = kmeans(embed_n, C)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)  # 中心归一化
        cluster_size = np.bincount(labels, minlength=C)
        # h 与中心的余弦（归一化后点积=余弦），选 top-K 簇
        center_dots = hs_n @ centers.T            # [n_tok, C] 余弦
        order = np.argsort(-center_dots, axis=1)
        for K in (1, 2, 4, 8):
            picked = order[:, :K]
            cand_count = cluster_size[picked].sum(axis=1)
            recall = 0
            ann_correct = 0
            for t in range(n_tok):
                cand_labels = picked[t]
                if labels[true_top1[t]] in cand_labels:
                    recall += 1
                # ANN：精确算候选行 logit（原始 embed，含模长），看 argmax 是否 = 真 top-1
                cand_idx = np.where(np.isin(labels, cand_labels))[0]
                ann_top1 = cand_idx[np.argmax(hs[t] @ embed[cand_idx].T)]
                if ann_top1 == true_top1[t]:
                    ann_correct += 1
            avg_cand = cand_count.mean()
            print(f"{C:>5} {K:>3} {recall:>3}/{n_tok} {avg_cand:>8.1f} "
                  f"{1 - avg_cand / vocab:>7.1%} {ann_correct:>3}/{n_tok}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
