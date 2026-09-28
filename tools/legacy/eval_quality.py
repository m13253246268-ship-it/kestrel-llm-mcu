#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_quality.py — 用生成质量指标（perplexity + 生成文本）评估 MoE student vs 稠密 teacher。

依据：argmax match 已在 M3-13 证伪为不可达指标（低 margin 下 cosine 0.99 仍 argmax 全翻），
改用语言建模标准指标：
  · perplexity = exp(CE)，衡量「预测真实下一 token」的能力，越低越好；
  · 生成文本：teacher 与 student 各贪心生成若干段，人工/自动比对连贯性。

对比对象：
  · teacher（稠密 Minueza）
  · moe_student（ffn_e=128）
  · moe_student_ffn256（ffn_e=256）
  · dense_baseline（稠密 FFN 从零训练，对照）
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, MistralForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_moe import build_student
from dense_baseline import build_dense_student


@torch.no_grad()
def perplexity(model, tok, texts, device, block_size=256, max_chars=2_000_000):
    """在给定文本上算平均 CE 损失 → perplexity。"""
    model.eval()
    nll_sum = 0.0
    n_tok = 0
    chars = 0
    for t in texts:
        if chars >= max_chars:
            break
        chars += len(t)
        ids = tok.encode(t, add_special_tokens=False)
        # 切块
        for i in range(0, max(1, len(ids) - 1), block_size):
            chunk = ids[i:i + block_size + 1]
            if len(chunk) < 2:
                continue
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:], dtype=torch.long, device=device)
            logits = model(input_ids=x).logits  # [1, S, V]
            ce = F.cross_entropy(logits[0], y, reduction="sum")
            nll_sum += ce.item()
            n_tok += y.numel()
    return torch.exp(torch.tensor(nll_sum / n_tok)).item(), n_tok


@torch.no_grad()
def gen_samples(model, tok, prompts, device, n_gen=40):
    model.eval()
    out = []
    for p in prompts:
        ids = tok.encode(p, return_tensors="pt").to(device)
        attn = torch.ones_like(ids)
        g = model.generate(ids, attention_mask=attn, max_new_tokens=n_gen,
                           do_sample=False, pad_token_id=tok.eos_token_id)
        out.append(tok.decode(g[0], skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=r"E:\models")
    ap.add_argument("--valid", default=r"E:\models\tinystories\data\validation-00000-of-00001-869c898b519ad725.parquet")
    ap.add_argument("--moe128", default=r"E:\models\moe_student")
    ap.add_argument("--moe256", default=r"E:\models\moe_student_ffn256")
    ap.add_argument("--dense", default=r"E:\models\dense_baseline")
    ap.add_argument("--n_expert", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=2_000_000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # 读 validation 文本
    import pyarrow.parquet as pq
    tbl = pq.read_table(args.valid)
    texts = [t for t in tbl.column("text").to_pylist() if t and len(t.strip()) > 20]

    print(f"[eval] validation stories = {len(texts)}")

    # teacher
    teacher = MistralForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32).to(device).eval()

    models = {"teacher(稠密)": teacher}

    if os.path.isdir(args.moe128):
        m = build_student(teacher, args.n_expert, args.top_k, 128).to(device)
        from safetensors.torch import load_file
        m.load_state_dict(load_file(os.path.join(args.moe128, "model.safetensors")), strict=False)
        for L in m.model.layers:
            L.mlp.set_lambda(0.0)
        models["moe_ffn128"] = m
    if os.path.isdir(args.moe256):
        m = build_student(teacher, args.n_expert, args.top_k, 256).to(device)
        from safetensors.torch import load_file
        m.load_state_dict(load_file(os.path.join(args.moe256, "model.safetensors")), strict=False)
        for L in m.model.layers:
            L.mlp.set_lambda(0.0)
        models["moe_ffn256"] = m
    if os.path.isdir(args.dense):
        m = build_dense_student(teacher).to(device)
        from safetensors.torch import load_file
        m.load_state_dict(load_file(os.path.join(args.dense, "model.safetensors")), strict=False)
        models["dense_从零FFN"] = m

    # perplexity
    print("\n=== Perplexity（越低越好）===")
    for name, m in models.items():
        ppl, n = perplexity(m, tok, texts, device, max_chars=args.max_chars)
        print(f"{name:20s}  perplexity={ppl:.2f}  (tokens={n})")

    # 生成样本
    prompts = ["Once upon a time", "The little cat", "A boy named Tom"]
    print("\n=== 生成样本（贪心，40 token）===")
    for name, m in models.items():
        print(f"\n--- {name} ---")
        for s in gen_samples(m, tok, prompts, device):
            print("  " + s)


if __name__ == "__main__":
    main()
