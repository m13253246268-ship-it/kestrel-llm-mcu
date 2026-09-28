#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_moe_qat.py — 技术点 2：λ_q 量化感知训练（QAT vs PTQ）。

验证假设：训练后量化（PTQ）精度损失大，λ_q 平滑过渡的 QAT 让模型在训练中适配 q4_0，
精度损失更小。

核心：
  W_eff = (1-λ_q)·W_fp32  ⊕  λ_q·round(W/d)·d     λ_q: 0 → 1

fake-quant 与 convert_minueza.py 的 q4_0 规则**位级一致**：
  每块 32 元素；d = max_abs/8；q = round(w/d) 截断 [-8,7]；反量化 w = q*d。
  用 straight-through estimator（前向量化值，反向梯度恒等传回 fp32 主权重）。

判定标准：
  ✅ PASS：QAT 的 q4 模型 ppl 明显优于 PTQ；
  ❌ FAIL：QAT 无增益或训练不稳定，退回 PTQ + 更保守位宽。
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from transformers import AutoTokenizer, MistralForCausalLM

from train_moe import build_student, make_data_iterator, collate


# ---------------------------------------------------------------------------
# q4_0 straight-through fake-quant（与 convert_minueza 规则一致）
# ---------------------------------------------------------------------------
def q4_0_ste(w: torch.Tensor):
    """q4_0 量化 + 反量化，straight-through。w: [rows, cols]（每行按 32 分块）。"""
    rows, cols = w.shape
    pad = (32 - cols % 32) % 32
    wp = F.pad(w, (0, pad)) if pad else w
    wp = wp.view(rows, -1, 32)                      # [rows, nblk, 32]
    amax = wp.abs().amax(dim=-1, keepdim=True)      # [rows, nblk, 1]
    d = torch.where(amax > 0.0, amax / 8.0, torch.ones_like(amax))
    q = torch.round(wp / d).clamp(-8.0, 7.0)
    dq = (q * d).view(rows, -1)[:, :cols]           # 去掉 padding
    # STE：前向 = dq，反向梯度恒等传回 w
    return w + (dq - w).detach()


class Q4Param(nn.Module):
    """parametrize 包装：forward 用 λ_q 混合 fp32 与 q4 量化值。"""

    def __init__(self):
        super().__init__()
        self.lam_q = 0.0

    def forward(self, w):
        if self.lam_q <= 0.0:
            return w
        wq = q4_0_ste(w)
        return (1.0 - self.lam_q) * w + self.lam_q * wq


def apply_qat(model):
    """对所有 Linear/Embedding 的 weight 注册 q4_0 fake-quant，返回 Q4Param 引用列表。"""
    qparams = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            qp = Q4Param()
            P.register_parametrization(module, "weight", qp)
            qparams.append(qp)
    return qparams


def set_lam_q(qparams, lam_q):
    for qp in qparams:
        qp.lam_q = lam_q


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_ppl(model, tok, texts, device, max_chars=500_000):
    model.eval()
    nll = 0.0
    n = 0
    chars = 0
    for t in texts:
        if chars >= max_chars:
            break
        chars += len(t)
        ids = tok.encode(t, add_special_tokens=False)
        for i in range(0, max(1, len(ids) - 1), 256):
            c = ids[i:i + 257]
            if len(c) < 2:
                continue
            x = torch.tensor(c[:-1], device=device).unsqueeze(0)
            y = torch.tensor(c[1:], device=device)
            ce = F.cross_entropy(model(input_ids=x).logits[0], y, reduction="sum")
            nll += ce.item()
            n += y.numel()
    return torch.exp(torch.tensor(nll / n)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=r"E:\models\full_ft_probe")
    ap.add_argument("--moe_ckpt", default=r"E:\models\moe_full_ce_ffne128",
                    help="已训练的全量 MoE 模型目录（fp32）")
    ap.add_argument("--out_dir", default=r"E:\models\moe_qat_ffne128")
    ap.add_argument("--data_file", default=r"E:\models\tinystories\train_full.txt")
    ap.add_argument("--valid", default=r"E:\models\tinystories\data\validation-00000-of-00001-869c898b519ad725.parquet")
    ap.add_argument("--n_expert", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--ffn_e", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lam_q_floor", type=float, default=0.0,
                    help="λ_q 结构过渡起步（0=从 fp32 纯态起步）")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    import pyarrow.parquet as pq
    tbl = pq.read_table(args.valid)
    val_texts = [t for t in tbl.column("text").to_pylist() if t and len(t.strip()) > 20]

    # 构建 MoE 结构 + 加载已训练权重
    print(f"[init] building MoE structure (ffn_e={args.ffn_e}) ...")
    teacher = MistralForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32).to(device)
    teacher.eval()
    model = build_student(teacher, args.n_expert, args.top_k, args.ffn_e).to(device)

    from safetensors.torch import load_file
    sd = load_file(os.path.join(args.moe_ckpt, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[init] loaded MoE ckpt, missing={len(missing)} unexpected={len(unexpected)}")

    # 固定 router 为 hard 推理态（量化只适配权重，不再动路由）
    for L in model.model.layers:
        L.mlp.set_lambda(0.0)

    # 1) fp32 基线
    print("[eval] fp32 基线 ...")
    fp32_ppl = val_ppl(model, tok, val_texts, device)
    print(f"[eval] fp32 ppl = {fp32_ppl:.2f}")

    # 2) PTQ 基线：λ_q=1 直接量化，不训练
    qparams = apply_qat(model)
    set_lam_q(qparams, 1.0)
    ptq_ppl = val_ppl(model, tok, val_texts, device)
    print(f"[eval] PTQ (λ_q=1, 直接量化) ppl = {ptq_ppl:.2f}  损失={ptq_ppl/fp32_ppl-1:.1%}")

    # 3) QAT：λ_q 从 0 平滑退火到 1
    print(f"[train] QAT λ_q: 0 -> 1, max_steps={args.max_steps}, lr={args.lr}")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * p))

    def lam_q_at(step):
        if step >= args.max_steps:
            return 1.0
        p = step / args.max_steps
        return args.lam_q_floor + (1.0 - args.lam_q_floor) * 0.5 * (1.0 - math.cos(math.pi * p))

    it = make_data_iterator(tok, args.block_size, args.data_file)
    block_buf = []
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        while len(block_buf) < args.batch_size:
            try:
                block_buf.append(next(it))
            except StopIteration:
                print(f"[train] data exhausted at step {step}, stopping.", flush=True)
                step = args.max_steps
                break
        if step >= args.max_steps:
            break
        blocks = block_buf[: args.batch_size]
        block_buf = block_buf[args.batch_size:]
        x, y = collate(blocks, device)

        lam_q = lam_q_at(step)
        set_lam_q(qparams, lam_q)

        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == args.max_steps - 1:
            print(f"step {step:5d}/{args.max_steps}  loss={loss.item():.4f} "
                  f"λ_q={lam_q:.3f} lr={lr:.2e}  "
                  f"{(time.time()-t0)/max(1,step+1):.3f} s/step", flush=True)

        if (step + 1) % args.eval_every == 0:
            set_lam_q(qparams, 1.0)
            ppl = val_ppl(model, tok, val_texts, device)
            print(f"[eval] step {step+1}  q4 ppl = {ppl:.2f}  "
                  f"(fp32={fp32_ppl:.2f}, PTQ={ptq_ppl:.2f})", flush=True)
            set_lam_q(qparams, lam_q)
            model.train()
        step += 1

    # 最终 QAT q4 评估
    set_lam_q(qparams, 1.0)
    qat_ppl = val_ppl(model, tok, val_texts, device)
    print(f"[done] QAT q4 ppl = {qat_ppl:.2f}")
    print(f"[summary] fp32={fp32_ppl:.2f}  PTQ={ptq_ppl:.2f} (+{ptq_ppl/fp32_ppl-1:.1%})  "
          f"QAT={qat_ppl:.2f} (+{qat_ppl/fp32_ppl-1:.1%})")

    os.makedirs(args.out_dir, exist_ok=True)
    # 去掉所有模块的 fake-quant parametrization，暴露 QAT 训练后的 fp32 主权重
    # （后续 convert 再按 q4_0 规则正式量化）
    for module in model.modules():
        if P.is_parametrized(module, "weight"):
            P.remove_parametrizations(module, "weight", leave_parametrized=False)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[done] saved fp32 master weights to {args.out_dir}")


if __name__ == "__main__":
    main()
