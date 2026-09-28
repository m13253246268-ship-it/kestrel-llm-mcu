#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_moe_ple.py — 正式训练：Mistral + MoE + PLE 三值查表 + tied head，FineWeb-Edu。

数据：_tok_fineweb.py 产出的 fw_train.bin / fw_val.bin（uint16，Mistral vocab 32002）。
范式：全量训练 + 纯 CE + AdamW + cosine（与已验证配方一致，无冻结、无 KD、无 dense 兜底）。

用法：
  python train_moe_ple.py --out_dir E:/models/moe_ple_fw --max_steps 30000
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from mistral_ple import MistralPLE, MistralPLEConfig  # noqa: E402

TRAIN_BIN = r"E:\models\fw_train.bin"
VAL_BIN = r"E:\models\fw_val.bin"


class Batcher:
    def __init__(self, path, bs, sl, device, seed):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.bs, self.sl, self.device = bs, sl, device
        self.rng = np.random.default_rng(1234 if "val" in path else seed)

    def __call__(self):
        ix = self.rng.integers(0, len(self.data) - self.sl - 1, self.bs)
        x = np.stack([self.data[i:i + self.sl] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.sl] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)


@torch.no_grad()
def evaluate(model, batcher, iters):
    model.eval()
    batcher.rng = np.random.default_rng(1234)
    losses = [model(*batcher())[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=r"E:\models\moe_ple_fw")
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    # 架构
    ap.add_argument("--dim", type=int, default=312)
    ap.add_argument("--n_layers", type=int, default=10)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--n_expert", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--ffn_e", type=int, default=128)
    ap.add_argument("--ple_dim", type=int, default=128)
    ap.add_argument("--no_ternary", action="store_true", help="PLE 表用 fp32 而非三值 STE")
    ap.add_argument("--smoke", action="store_true", help="仅跑 1 步 fwd+bwd 验证可运行")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_bin", default=TRAIN_BIN)
    ap.add_argument("--val_bin", default=VAL_BIN)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}", flush=True)

    cfg = MistralPLEConfig(
        vocab_size=32002, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, ffn_e=args.ffn_e, n_expert=args.n_expert,
        top_k=args.top_k, ple_dim=args.ple_dim, seq_len=args.seq_len,
        ternary_table=not args.no_ternary)
    model = MistralPLE(cfg).to(device)
    b = model.param_budget()
    print(f"[model] core={b['core']:,} stream={b['stream']:,} table={b['table']:,} "
          f"total={b['total']:,}", flush=True)

    if args.smoke:
        x = torch.randint(0, 32002, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, 32002, (args.batch_size, args.seq_len), device=device)
        _, loss = model(x, y)
        loss.backward()
        print(f"[smoke] fwd+bwd OK loss={loss.item():.4f}", flush=True)
        return 0

    if not (os.path.isfile(args.train_bin) and os.path.isfile(args.val_bin)):
        print(f"[error] 缺少 {args.train_bin} / {args.val_bin}，先运行 tokenize 脚本", flush=True)
        return 1

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "table" in n or "embed_tokens" in n else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    train_b = Batcher(args.train_bin, args.batch_size, args.seq_len, device, seed=args.seed)
    val_b = Batcher(args.val_bin, args.batch_size, args.seq_len, device, seed=0)

    os.makedirs(args.out_dir, exist_ok=True)
    best = float("inf")
    t0 = time.time()
    for step in range(args.max_steps):
        lr = lr_at(step, args.max_steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_b()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.max_steps - 1:
            vl = evaluate(model, val_b, args.eval_iters)
            best = min(best, vl)
            print(f"step {step:6d}/{args.max_steps} train {loss.item():.4f} "
                  f"val {vl:.4f} ppl {math.exp(vl):7.2f} lr {lr:.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0:
            ckpt = os.path.join(args.out_dir, f"ckpt_step{step+1}")
            model.save_pretrained(ckpt)
            print(f"[ckpt] {ckpt}", flush=True)

    model.save_pretrained(args.out_dir)
    print(f"[done] best_ppl={math.exp(best):.2f} total={time.time()-t0:.0f}s -> {args.out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
