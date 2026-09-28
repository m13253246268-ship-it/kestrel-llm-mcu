#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dense_baseline.py — 稠密 FFN 对照：复用 teacher 除 FFN 外全部权重，FFN 从零稠密训练。

目的：分离「MoE 稀疏激活」这一变量。若稠密 FFN 从零训练 argmax match 也上不去，
则问题在「从头训练 FFN + KD 蒸馏」本身；若稠密能上 0.9+，则是 MoE 稀疏容量损失。
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, MistralForCausalLM

from train_moe import make_data_iterator, collate, kd_loss_fn


def build_dense_student(teacher):
    cfg = teacher.config
    student = MistralForCausalLM(cfg)
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    copied = skipped = 0
    with torch.no_grad():
        for name, p in s_sd.items():
            if name in t_sd and t_sd[name].shape == p.shape and "mlp" not in name:
                p.copy_(t_sd[name])
                copied += 1
            else:
                # 初始化：默认已有随机值，MLP 从零
                skipped += 1
    print(f"[init] copied {copied} params from teacher (非 FFN), left {skipped} from scratch (FFN)")
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=r"E:\models")
    ap.add_argument("--out_dir", default=r"E:\models\dense_baseline")
    ap.add_argument("--data_file", default=r"E:\models\tinystories\train.txt")
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher = MistralForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    student = build_dense_student(teacher).to(device)
    student.train()

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * p))

    it = make_data_iterator(tok, args.block_size, args.data_file)
    block_buf = []
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        while len(block_buf) < args.batch_size:
            try:
                block_buf.append(next(it))
            except StopIteration:
                it = make_data_iterator(tok, args.block_size, args.data_file)
                if not block_buf:
                    print("[train] data exhausted, restarting ...")
        blocks = block_buf[: args.batch_size]
        block_buf = block_buf[args.batch_size:]
        x, y = collate(blocks, device)

        with torch.no_grad():
            t_out = teacher(input_ids=x).logits
        s_out = student(input_ids=x).logits
        loss, logs = kd_loss_fn(s_out, t_out, y, args.T, args.alpha)

        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == args.max_steps - 1:
            print(f"step {step:5d}/{args.max_steps}  loss={loss.item():.4f} "
                  f"kd={logs['kd']:.4f} ce={logs['ce']:.4f} lr={lr:.2e}  "
                  f"{(time.time()-t0)/max(1,step+1):.3f} s/step")

        if (step + 1) % args.eval_every == 0:
            match = eval_match(student, teacher, tok, device)
            print(f"[eval] step {step+1} argmax match = {match:.4f}")
            student.train()
        step += 1

    os.makedirs(args.out_dir, exist_ok=True)
    student.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[done] saved to {args.out_dir} total={time.time()-t0:.1f}s")


@torch.no_grad()
def eval_match(student, teacher, tok, device, n_prompt=6, n_gen=16):
    prompts = ["Once upon a time", "The little cat", "A boy named",
               "In a faraway", "One day, a", "The old man"][:n_prompt]
    student.eval()
    total, hit = 0, 0
    for p in prompts:
        ids = tok.encode(p, return_tensors="pt").to(device)
        attn = torch.ones_like(ids)
        s_gen = student.generate(ids, attention_mask=attn, max_new_tokens=n_gen,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        t_gen = teacher.generate(ids, attention_mask=attn, max_new_tokens=n_gen,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        s_tok, t_tok = s_gen[0].tolist(), t_gen[0].tolist()
        L = min(len(s_tok), len(t_tok))
        total += L
        hit += sum(1 for i in range(L) if s_tok[i] == t_tok[i])
    return hit / max(1, total)


if __name__ == "__main__":
    main()
