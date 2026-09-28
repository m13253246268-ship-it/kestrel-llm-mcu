#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_finetune.py — 全量训练试点：teacher 权重初始化 + 全量更新（不冻结任何参数）+ 纯 CE。

背景：之前「复用 attention 冻结 + 从零训 FFN」证伪（KD 能力不足 ppl77，纯 CE 过拟合 ppl1711）。
本脚本验证「全量训练」能否突破 ppl 天花板——attention 与 FFN 联合更新。

关键：定期在 validation 集测 perplexity，实时观察是否过拟合 / 是否突破 ppl77。
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, MistralForCausalLM

from train_moe import make_data_iterator, collate


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
    ap.add_argument("--teacher", default=r"E:\models")
    ap.add_argument("--out_dir", default=r"E:\models\full_ft_probe")
    ap.add_argument("--data_file", default=r"E:\models\tinystories\train_full.txt")
    ap.add_argument("--valid", default=r"E:\models\tinystories\data\validation-00000-of-00001-869c898b519ad725.parquet")
    ap.add_argument("--max_steps", type=int, default=30000)   # 1 epoch ≈ 30500 步
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--ckpt_every", type=int, default=5000,
                    help="每隔多少步保存一次 checkpoint（0=不保存中间 checkpoint）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # teacher 权重初始化，但所有参数可训练（不冻结）
    print("[model] loading teacher weights as init, full trainable ...")
    model = MistralForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32).to(device)
    model.train()
    print(f"[model] params = {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * p))

    # validation 文本（子集，快速评估）
    import pyarrow.parquet as pq
    tbl = pq.read_table(args.valid)
    val_texts = [t for t in tbl.column("text").to_pylist() if t and len(t.strip()) > 20]

    it = make_data_iterator(tok, args.block_size, args.data_file)
    block_buf = []

    print(f"[train] max_steps={args.max_steps} batch={args.batch_size} block={args.block_size} "
          f"lr={args.lr} 纯CE 全量训练", flush=True)
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        while len(block_buf) < args.batch_size:
            try:
                block_buf.append(next(it))
            except StopIteration:
                # 数据耗尽：单 epoch，直接退出（避免反复重读 1.8GB 文件导致假死）
                print(f"[train] data exhausted at step {step}, stopping (single-epoch).", flush=True)
                step = args.max_steps
                break
        if step >= args.max_steps:
            break
        blocks = block_buf[: args.batch_size]
        block_buf = block_buf[args.batch_size:]
        x, y = collate(blocks, device)

        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == args.max_steps - 1:
            print(f"step {step:5d}/{args.max_steps}  ce={loss.item():.4f} lr={lr:.2e}  "
                  f"{(time.time()-t0)/max(1,step+1):.3f} s/step", flush=True)

        if (step + 1) % args.eval_every == 0:
            ppl = val_ppl(model, tok, val_texts, device)
            print(f"[eval] step {step+1}  validation perplexity = {ppl:.2f}", flush=True)
            model.train()
        step += 1

        # 周期 checkpoint：防止进程被外部终止时进度全丢
        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            ckpt_dir = os.path.join(args.out_dir, f"ckpt_step{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            print(f"[ckpt] saved {ckpt_dir}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[done] saved to {args.out_dir} total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
