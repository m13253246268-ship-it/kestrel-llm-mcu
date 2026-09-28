#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_q2.py - H4 授权下有损量化（q2/q3）的精度边界验证。

在现有 q4_0 反量化权重基础上，进一步量化到 q2（{-2,-1,0,1}）或 q3，
验证极端量化对贪心解码 argmax 的影响。

说明：这是「q4 → q2」的再量化（比「原始 fp32 → q2」更保守，因为 q4
已损失部分精度，q2 再量化会叠加损失）。目的是给出有损量化的精度下界。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # tools/：ref_forward 等主线模块
sys.path.insert(0, _HERE)                   # legacy/：同目录模块
import ref_forward as rf

BLOCK = 32


def quantize_to_int(W, bits):
    """把 fp32 权重按 block 量化到 [-2^(bits-1), 2^(bits-1)-1]，返回反量化后的 fp32。

    bits=2 → q2（值域 -2..1），bits=3 → q3（值域 -4..3）。
    """
    rows, cols = W.shape
    nblk = cols // BLOCK
    out = np.empty_like(W)
    hi = 2 ** (bits - 1) - 1
    for r in range(rows):
        for b in range(nblk):
            seg = W[r, b * BLOCK:(b + 1) * BLOCK]
            scale = float(np.max(np.abs(seg))) / hi
            if scale == 0.0:
                scale = 1.0
            q = np.clip(np.round(seg / scale), -hi - 1, hi).astype(np.int32)
            out[r, b * BLOCK:(b + 1) * BLOCK] = q.astype(np.float32) * scale
    return out


def main():
    km = rf.KM
    arch, T = rf.load_kmcu(km)
    prompt = [1, 450, 2217, 4996]

    # 基准（q4 反量化，未再量化）
    base_toks = rf.generate(arch, T, prompt, 16)

    for bits in (3, 2):
        Tq = dict(T)
        # 只量化权重张量（q/k/v/o/gate/up/down/tok_embed），跳过 norm（fp32）
        for name, w in T.items():
            if name.endswith(('_ln',)) or name == 'final_norm':
                continue
            if w.ndim == 2 and w.shape[1] >= BLOCK:
                Tq[name] = quantize_to_int(w, bits)
            else:
                Tq[name] = w
        toks = rf.generate(arch, Tq, prompt, 16)
        match = sum(1 for a, b in zip(base_toks, toks) if a == b)
        n = len(base_toks)
        print(f"q{bits}: match {match}/{n}  seq={toks}")
        # 首 token logit 对比
        lg = rf.forward(arch, Tq, prompt)[-1]
        top = np.argsort(-lg)[:3]
        print(f"  top3: " + " ".join(f"id={int(v)} logit={float(lg[v]):.4f}" for v in top))

    return 0


if __name__ == "__main__":
    sys.exit(main())
