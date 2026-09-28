#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lambda.py — 技术点 1：λ_s 结构稀疏化（稠密 FFN → MoE 的结构蒸馏）。

核心：阶段 0 全量训练出稠密模型（teacher），本脚本用 λ_s 平滑过渡把稠密 FFN
稀疏化到 MoE，验证「蒸馏是否可行」：

  out = (1-λ_s)·dense_ffn(h)  ⊕  λ_s·moe_ffn(h)      λ_s: 0 → 1

- λ_s=0：纯稠密（阶段 0 收敛态，天然不崩）
- λ_s=1：纯 MoE（稀疏激活，目标态）
- 稠密旁路冻结作为 teacher 参考，只训练 MoE 旁路（router + 专家从零）

对应公理 SHS-3 连续可还原律：结构变换走 λ_s 平滑过渡，禁止硬切。

判定标准（技术点 1 闸门）：
  ✅ PASS：λ_s→1 时 router 熵 < 1.8（分化）且 validation ppl 相对稠密基线上涨 < 20%
  ❌ FAIL：router 不分化（熵≈log8）或 ppl 崩溃 → 止损，转向全量训练 MoE
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, MistralForCausalLM

from train_moe import LambdaMoE, make_data_iterator, collate


# ---------------------------------------------------------------------------
# λ_s 结构过渡模块：稠密旁路 + MoE 旁路
# ---------------------------------------------------------------------------
class DenseMoEHybrid(nn.Module):
    """FFN = (1-λ_s)·dense + λ_s·moe。稠密旁路冻结（teacher），MoE 旁路训练（student）。"""

    def __init__(self, dim: int, ffn: int, n_expert: int, top_k: int, ffn_e: int):
        super().__init__()
        # 稠密旁路（阶段 0 收敛权重，冻结）
        self.dense_gate = nn.Linear(dim, ffn, bias=False)
        self.dense_up = nn.Linear(dim, ffn, bias=False)
        self.dense_down = nn.Linear(ffn, dim, bias=False)
        # MoE 旁路（从零训练）
        self.moe = LambdaMoE(dim, n_expert, top_k, ffn_e)
        self.register_buffer("lam_s", torch.tensor(0.0))

    def set_lambda_s(self, lam_s: float):
        self.lam_s.fill_(lam_s)

    def forward(self, h):
        g = F.silu(self.dense_gate(h))
        u = self.dense_up(h)
        dense_out = self.dense_down(g * u)
        moe_out = self.moe(h)
        return (1.0 - self.lam_s) * dense_out + self.lam_s * moe_out


# ---------------------------------------------------------------------------
# 模型构建：从阶段 0 稠密模型加载，替换 mlp 为 DenseMoEHybrid
# ---------------------------------------------------------------------------
def build_hybrid(dense_model, n_expert, top_k, ffn_e):
    """复用阶段 0 稠密模型，每层 mlp 替换为 DenseMoEHybrid（稠密旁路冻结 + MoE 训练）。"""
    cfg = dense_model.config
    dim = cfg.hidden_size
    ffn = cfg.intermediate_size

    model = MistralForCausalLM(cfg)
    d_sd = dense_model.state_dict()

    # 先加载完整权重（含 attention/embed/norm/lm_head + 原 mlp）
    model.load_state_dict(d_sd, strict=True)

    # 逐层替换 mlp 为 DenseMoEHybrid，稠密旁路权重从原 mlp 拷贝
    for L in model.model.layers:
        old_mlp = L.mlp
        hybrid = DenseMoEHybrid(dim, ffn, n_expert, top_k, ffn_e)
        with torch.no_grad():
            hybrid.dense_gate.weight.copy_(old_mlp.gate_proj.weight)
            hybrid.dense_up.weight.copy_(old_mlp.up_proj.weight)
            hybrid.dense_down.weight.copy_(old_mlp.down_proj.weight)
        L.mlp = hybrid

    # 冻结稠密旁路 + attention/embed/norm/lm_head，只训练 MoE 旁路（router + 专家）
    for name, p in model.named_parameters():
        if "moe" in name:          # MoE 旁路（router + experts）
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[init] 可训练参数（仅 MoE 旁路）= {n_train/1e6:.2f} M")
    return model


# ---------------------------------------------------------------------------
# λ_s 退火调度（floor → 1，保证 MoE 旁路从 step 0 起就有梯度）
# ---------------------------------------------------------------------------
def lambda_s_at(step, total, floor=0.5):
    """λ_s 从 floor 平滑增到 1（结构过渡）。

    floor>0 是关键：λ_s=0 起步会让 MoE 旁路输出被 λ_s 乘子压成 ≈0，
    router 梯度 ∝ λ_s ≈ 0，早期完全学不到；等到 λ_s→1 时 lr 已烧干。
    从 floor 起步，MoE 从 step 0 起就有 floor 比例的梯度信号。
    """
    if step >= total:
        return 1.0
    p = step / total
    return floor + (1.0 - floor) * 0.5 * (1.0 - math.cos(math.pi * p))


def router_lambda_at(step, total, lam_start=1.0, lam_min=0.05):
    """router 内部 λ：soft 全专家梯度 → hard top-k 梯度（与 train_moe.py 一致，1→0）。"""
    if step >= total:
        return lam_min
    p = step / total
    return lam_min + (lam_start - lam_min) * 0.5 * (1.0 + math.cos(math.pi * p))


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
    ap.add_argument("--dense", default=r"E:\models\full_ft_probe", help="阶段0稠密模型目录")
    ap.add_argument("--out_dir", default=r"E:\models\moe_lambda_s")
    ap.add_argument("--data_file", default=r"E:\models\tinystories\train_full.txt")
    ap.add_argument("--valid", default=r"E:\models\tinystories\data\validation-00000-of-00001-869c898b519ad725.parquet")
    ap.add_argument("--n_expert", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--ffn_e", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_floor", type=float, default=0.1,
                    help="cosine 衰减的 lr 下限比例（不衰减到 0）")
    ap.add_argument("--lam_s_floor", type=float, default=0.5,
                    help="λ_s 结构过渡的起步下限，保证 MoE 旁路从 step 0 起有梯度")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--aux_w", type=float, default=0.0,
                    help="负载均衡 loss 权重（0=关闭。结构蒸馏中 aux 鼓励均匀，与分化目标相反，默认关闭）")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ckpt_every", type=int, default=2000,
                    help="每隔多少步保存一次 checkpoint（0=不保存中间 checkpoint）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(r"E:\models")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # 阶段 0 稠密模型（teacher）
    print(f"[dense] loading stage-0 dense model from {args.dense} ...")
    dense = MistralForCausalLM.from_pretrained(args.dense, torch_dtype=torch.float32).to(device)
    dense.eval()

    # 基线 ppl（稠密 teacher 在 validation 上的表现）
    import pyarrow.parquet as pq
    tbl = pq.read_table(args.valid)
    val_texts = [t for t in tbl.column("text").to_pylist() if t and len(t.strip()) > 20]
    base_ppl = val_ppl(dense, tok, val_texts, device)
    print(f"[dense] baseline validation perplexity = {base_ppl:.2f}")

    # 构建 hybrid（稠密旁路冻结 + MoE 训练）
    model = build_hybrid(dense, args.n_expert, args.top_k, args.ffn_e).to(device)
    model.train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        # cosine 衰减到 lr_floor（不衰减到 0），保证 λ_s→1 时 router 仍能分化
        return args.lr * (args.lr_floor + (1.0 - args.lr_floor) * 0.5 * (1.0 + math.cos(math.pi * p)))

    it = make_data_iterator(tok, args.block_size, args.data_file)
    block_buf = []

    print(f"[train] λ_s: {args.lam_s_floor} -> 1 (结构过渡), max_steps={args.max_steps}, "
          f"batch={args.batch_size}, lr={args.lr} lr_floor={args.lr_floor}", flush=True)
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        while len(block_buf) < args.batch_size:
            try:
                block_buf.append(next(it))
            except StopIteration:
                # 数据耗尽：单 epoch 直接退出（避免反复重读文件导致假死）
                print(f"[train] data exhausted at step {step}, stopping (single-epoch).", flush=True)
                step = args.max_steps
                break
        if step >= args.max_steps:
            break
        blocks = block_buf[: args.batch_size]
        block_buf = block_buf[args.batch_size:]
        x, y = collate(blocks, device)

        # λ_s 结构过渡（floor → 1）+ MoE 内部 router λ 退火（soft → hard，1→0）
        lam_s = lambda_s_at(step, args.max_steps, floor=args.lam_s_floor)
        lam_r = router_lambda_at(step, args.max_steps)
        for L in model.model.layers:
            L.mlp.set_lambda_s(lam_s)
            L.mlp.moe.set_lambda(lam_r)   # router 内部 λ：soft 全专家梯度 → hard top-k

        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        # 负载均衡 loss（防止 hard 路由坍缩）
        if args.aux_w > 0:
            aux = sum(L.mlp.moe._aux for L in model.model.layers)
            loss = loss + args.aux_w * aux

        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == args.max_steps - 1:
            ents = [L.mlp.moe.router_entropy() for L in model.model.layers]
            ent_avg = sum(ents) / len(ents)
            print(f"step {step:5d}/{args.max_steps}  ce={loss.item():.4f} "
                  f"λ_s={lam_s:.3f} entropy={ent_avg:.3f} lr={lr:.2e}  "
                  f"{(time.time()-t0)/max(1,step+1):.3f} s/step", flush=True)

        if (step + 1) % args.eval_every == 0:
            ppl = val_ppl(model, tok, val_texts, device)
            ent = model.model.layers[-1].mlp.moe.router_entropy()
            print(f"[eval] step {step+1}  λ_s={lam_s:.3f}  ppl={ppl:.2f}  "
                  f"(dense基线={base_ppl:.2f}, 涨幅={ppl/base_ppl-1:.1%})  L9熵={ent:.3f}", flush=True)
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
