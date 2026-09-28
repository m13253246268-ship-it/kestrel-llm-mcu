#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_moe.py — Kestrel-MCU 自研 MoE 训练工具（λ 退火路由 + 知识蒸馏）。

设计依据：公理库 axiom_registry.json
  · MoE 叠加路由公理（shs_5faxiom_5fstatement_8d696040）：
      R = (1-λ)·R_classic ⊕ λ·R_new
      R_classic = 经典唯一 top-k 硬选择；R_new = 集合态多专家 soft 激活（各带置信度）。
  · 连续可还原律 SHS-3：λ 平滑收敛，训练早期 λ=1（soft 全专家，梯度足），
    随训练退火到 λ→0（hard top-k，逼近稀疏推理态）。

关键设计（区别于上一轮 transformers MixtralForCausalLM 失败点）：
  1. 自研 LambdaMoE 模块：router 输出混合权重 w = (1-λ)·hard_topk + λ·softmax。
     λ=1 时等价稠密（soft 全专家加权），λ→0 时收敛到 top-k 稀疏。
  2. 不再用 transformers 的 load-balancing loss（它把 router 压向均匀，aux 恒 2.0），
     改用 λ 退火让 router 自然分化。
  3. 复用 Mistral 骨架（attention/embed/norm 从 teacher 拷贝），仅替换 mlp 为 LambdaMoE，
     大幅降低从头训练难度。

用法：
  python train_moe.py                                   # 默认 TinyStories 流式（需 HF_ENDPOINT 镜像）
  python train_moe.py --data_file corpus.txt --max_steps 5000
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM


# ---------------------------------------------------------------------------
# 自研 MoE 核心
# ---------------------------------------------------------------------------
class TernaryLinear(nn.Linear):
    """W1.58 三值权重线性层（BitNet b1.58 风格 STE）。

    forward 用 {-1,0,+1} × absmean(γ) 量化权重做 matmul，反向通过
    straight-through estimator 把梯度直通给 fp32 master 权重。
    参考 PFor (cyfrit/p-for-llm) 的 ternary_ste_linear 实现。
    """

    def forward(self, x):
        gamma = self.weight.float().abs().mean()
        codes = torch.clamp(torch.round(self.weight / gamma), -1.0, 1.0)
        qw = codes.to(self.weight.dtype) * gamma.to(self.weight.dtype)
        ste_w = self.weight + (qw - self.weight).detach()
        return F.linear(x, ste_w, self.bias)


class ExpertFFN(nn.Module):
    """单个专家的完整 SwiGLU FFN（gate/up/down 三矩阵，非聚类碎片）。"""

    def __init__(self, dim: int, ffn_e: int, ternary: bool = False):
        super().__init__()
        lin = TernaryLinear if ternary else nn.Linear
        self.gate = lin(dim, ffn_e, bias=False)
        self.up = lin(dim, ffn_e, bias=False)
        self.down = lin(ffn_e, dim, bias=False)

    def forward(self, h):
        # h: [B, S, dim]
        g = F.silu(self.gate(h))
        u = self.up(h)
        return self.down(g * u)


class LambdaMoE(nn.Module):
    """λ 退火 MoE 层，实现公理 R = (1-λ)·R_classic ⊕ λ·R_new。

    前向始终 hard top-k（R_classic，稀疏、等价推理态、打破专家对称性），
    λ 只退火控制反向梯度软化程度（早期 λ=1 全专家 soft 梯度一起学，后期 λ→0 只有
    选中专家有梯度，分工锐化）。这是 straight-through estimator。

    forward(h) 返回 [B, S, dim]，与 MistralMLP 接口一致，可直接替换 layer.mlp。
    """

    def __init__(self, dim: int, n_expert: int, top_k: int, ffn_e: int, ternary: bool = False):
        super().__init__()
        self.dim = dim
        self.n_expert = n_expert
        self.top_k = top_k
        lin = TernaryLinear if ternary else nn.Linear
        self.router = lin(dim, n_expert, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(dim, ffn_e, ternary) for _ in range(n_expert)])
        # λ：由训练循环退火更新（1.0 = soft 梯度，0.0 = hard 梯度）
        self.register_buffer("lam", torch.tensor(1.0))
        self._aux = 0.0   # 最近一次 forward 的负载均衡 loss（供主循环累加）
        self._ent = 0.0   # 最近一次 forward 的真实 router 熵（真实输入，非 embed）

    def set_lambda(self, lam: float):
        self.lam.fill_(lam)

    def forward(self, h):
        logits = self.router(h)                     # [B, S, N]
        p_soft = F.softmax(logits, dim=-1)          # R_new：soft 全专家置信度

        # 监控：真实 router 熵（h 是 post_attention_layernorm 后的真实输入）
        self._ent = -(p_soft * torch.log(p_soft + 1e-9)).sum(dim=-1).mean().item()

        # R_classic：hard top-k 选择
        topk_val, topk_idx = torch.topk(p_soft, self.top_k, dim=-1)
        p_hard = torch.zeros_like(p_soft)
        p_hard.scatter_(-1, topk_idx, topk_val)
        p_hard = p_hard / p_hard.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # straight-through：前向 = p_hard（稀疏），反向梯度 = λ·soft + (1-λ)·hard
        w_mix = self.lam * p_soft + (1.0 - self.lam) * p_hard
        w = (w_mix - w_mix.detach()) + p_hard.detach()   # [B, S, N]

        out = torch.zeros_like(h)
        for e in range(self.n_expert):
            out = out + w[..., e:e + 1] * self.experts[e](h)

        self._aux = self._load_balancing_loss(p_soft, topk_idx)
        return out

    def _load_balancing_loss(self, p_soft, topk_idx):
        """Switch Transformer 负载均衡 loss（轻量，防止 hard 路由坍缩到单专家）。

        f_e = 每个专家被选中的 hard 比例；P_e = soft 平均概率。均匀时 loss = top_k，
        完全坍缩到单专家时 loss 最大。用于对抗 hard 路由的正反馈过度分化。
        """
        onehot = F.one_hot(topk_idx, num_classes=self.n_expert).float()   # [B,S,K,N]
        f_e = onehot.sum(dim=(0, 1, 2)) / (p_soft.shape[0] * p_soft.shape[1] * self.top_k)
        P_e = p_soft.mean(dim=(0, 1))
        return (self.n_expert * (f_e * P_e).sum()).item()

    def router_entropy(self, h=None):
        """监控指标：真实 router 熵。log(N)≈均匀（未分化），越小越分化。

        若 h 为 None，返回最近一次 forward 记录的真实熵 self._ent。
        """
        if h is None:
            return self._ent
        p = F.softmax(self.router(h), dim=-1)
        return -(p * torch.log(p + 1e-9)).sum(dim=-1).mean().item()


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------
def build_student(teacher, n_expert, top_k, ffn_e, ternary=False):
    """复用 teacher 的 Mistral 骨架，替换每层 mlp 为 LambdaMoE。"""
    cfg = teacher.config
    student = MistralForCausalLM(cfg)

    dim = cfg.hidden_size
    for L in student.model.layers:
        L.mlp = LambdaMoE(dim, n_expert, top_k, ffn_e, ternary)

    # 拷贝 teacher 的 embed/attention/norm/lm_head（同名同 shape），mlp 已被替换不会命中
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    copied, skipped = 0, 0
    with torch.no_grad():
        for name, p in s_sd.items():
            if name in t_sd and t_sd[name].shape == p.shape:
                p.copy_(t_sd[name])
                copied += 1
            else:
                skipped += 1
    print(f"[init] copied {copied} params from teacher, left {skipped} from scratch (MoE)")
    return student


# ---------------------------------------------------------------------------
# λ 退火调度
# ---------------------------------------------------------------------------
def lambda_at(step, total, lam_start=1.0, lam_min=0.05):
    """cosine 从 lam_start 退火到 lam_min。lam_min>0 保留微弱 soft 分量，防止专家坍缩。"""
    if step >= total:
        return lam_min
    p = step / total
    return lam_min + (lam_start - lam_min) * 0.5 * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def make_data_iterator(tokenizer, block_size, data_file=None,
                       dataset_name="roneneldan/TinyStories", epochs=1):
    def gen_texts():
        for _ in range(epochs):
            if data_file:
                with open(data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
            else:
                from datasets import load_dataset
                ds = load_dataset(dataset_name, split="train", streaming=True)
                for ex in ds:
                    t = ex.get("text", "").strip()
                    if t:
                        yield t

    buf = []
    for text in gen_texts():
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(buf) >= block_size + 1:
            block = buf[: block_size + 1]
            buf = buf[block_size:]
            yield block


def collate(blocks, device):
    xs = [b[:-1] for b in blocks]
    ys = [b[1:] for b in blocks]
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


def kd_loss_fn(student_logits, teacher_logits, labels, T, alpha):
    s = student_logits[..., :-1, :].reshape(-1, student_logits.shape[-1])
    t = teacher_logits[..., :-1, :].reshape(-1, teacher_logits.shape[-1])
    lab = labels[..., 1:].reshape(-1)
    kd = F.kl_div(F.log_softmax(s / T, dim=-1), F.softmax(t / T, dim=-1),
                  reduction="batchmean") * (T * T)
    ce = F.cross_entropy(s, lab)
    return (1.0 - alpha) * ce + alpha * kd, {"kd": kd.item(), "ce": ce.item()}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=r"E:\models")
    ap.add_argument("--out_dir", default=r"E:\models\moe_student")
    ap.add_argument("--data_file", default=None)
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--n_expert", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--ffn_e", type=int, default=128)
    ap.add_argument("--ternary", action="store_true",
                    help="MoE 专家与 router 用 W1.58 三值权重（BitNet STE）训练")
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=1, help="数据遍历轮数（epoch），max_steps 仅作安全上限")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--llrd_gamma", type=float, default=None,
                    help="逐层学习率衰减 γ（None=全局 lr；如 0.95 表示顶层 lr、每向下一层 ×0.95，embed 最低）")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lam_start", type=float, default=1.0, help="λ 初始值（1.0=soft 梯度预热，0.0=纯 hard 从头）")
    ap.add_argument("--lam_min", type=float, default=0.05)
    ap.add_argument("--loss", type=str, default="ce", choices=["ce", "kd"],
                    help="ce=纯交叉熵（全量训练范式，推荐）；kd=知识蒸馏（旧，能力不足）")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--aux_w", type=float, default=0.01, help="负载均衡 loss 权重")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=1000,
                    help="每隔多少步保存一次 checkpoint（0=不保存中间 checkpoint）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    print("[teacher] loading ...")
    teacher = MistralForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"[student] building LambdaMoE (N={args.n_expert}, top_k={args.top_k}, "
          f"ffn_e={args.ffn_e}) ...")
    student = build_student(teacher, args.n_expert, args.top_k, args.ffn_e, args.ternary).to(device)
    student.train()
    print(f"[student] params = {sum(p.numel() for p in student.parameters())/1e6:.2f} M")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def mult_at(step):
        """全局 lr 乘子（warmup + cosine），范围 0..1。逐层 lr = base_lr × 乘子。"""
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    llrd_groups = None
    if args.llrd_gamma is not None:
        # LLRD：顶层 lr 最高（=base），每向下一层 ×γ，embed 最低
        gamma = args.llrd_gamma
        nl = len(student.model.layers)
        llrd_groups = []
        for i, layer in enumerate(student.model.layers):
            base_i = args.lr * (gamma ** (nl - 1 - i))
            llrd_groups.append({"params": list(layer.parameters()), "base_lr": base_i})
        named = dict(student.named_parameters())
        embed_p = [p for n, p in named.items() if "embed_tokens" in n]
        rest_p = [p for n, p in named.items() if "embed_tokens" not in n and ".layers." not in n]
        llrd_groups.append({"params": embed_p, "base_lr": args.lr * (gamma ** nl)})
        llrd_groups.append({"params": rest_p, "base_lr": args.lr})
        opt = torch.optim.AdamW(
            [{"params": g["params"], "lr": g["base_lr"]} for g in llrd_groups],
            betas=(0.9, 0.95), weight_decay=0.1)
        print(f"[opt] LLRD γ={gamma} nl={nl}（embed lr={args.lr * (gamma ** nl):.2e}，顶层 lr={args.lr:.2e}）")
    else:
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    it = make_data_iterator(tok, args.block_size, args.data_file, args.dataset, args.epochs)
    block_buf = []

    print(f"[train] max_steps={args.max_steps} λ: 1.0 -> {args.lam_min} "
          f"T={args.T} alpha={args.alpha} lr={args.lr}")
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        while len(block_buf) < args.batch_size:
            try:
                block_buf.append(next(it))
            except StopIteration:
                # 数据耗尽：单 epoch 直接退出（避免反复重读文件导致假死）
                print(f"[train] data exhausted at step {step} (epochs={args.epochs}), stopping.", flush=True)
                step = args.max_steps
                break
        if step >= args.max_steps:
            break
        blocks = block_buf[: args.batch_size]
        block_buf = block_buf[args.batch_size:]
        x, y = collate(blocks, device)

        # λ 退火：soft(lam_start) -> hard(lam_min)
        lam = lambda_at(step, args.max_steps, args.lam_start, args.lam_min)
        for L in student.model.layers:
            L.mlp.set_lambda(lam)

        if args.loss == "kd":
            with torch.no_grad():
                t_out = teacher(input_ids=x).logits
        else:
            t_out = None

        s_out = student(input_ids=x).logits
        if args.loss == "kd":
            loss, logs = kd_loss_fn(s_out, t_out, y, args.T, args.alpha)
        else:
            loss = F.cross_entropy(s_out.reshape(-1, s_out.shape[-1]), y.reshape(-1))
            logs = {"ce": loss.item()}

        # 轻量负载均衡 loss（防止 hard 路由坍缩到单专家）
        aux = sum(L.mlp._aux for L in student.model.layers)
        if args.aux_w > 0:
            loss = loss + args.aux_w * aux
            logs["aux"] = aux

        m = mult_at(step)
        lr = args.lr * m
        if llrd_groups is not None:
            for g, grp in zip(opt.param_groups, llrd_groups):
                g["lr"] = grp["base_lr"] * m
        else:
            for g in opt.param_groups:
                g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == args.max_steps - 1:
            # 真实 router 熵：取各层 forward 时记录的真实输入的熵
            ents = [L.mlp.router_entropy() for L in student.model.layers]
            ent_avg = sum(ents) / len(ents)
            kd_s = f" kd={logs['kd']:.4f}" if "kd" in logs else ""
            aux_s = f" aux={logs['aux']:.3f}" if "aux" in logs else ""
            print(f"step {step:5d}/{args.max_steps}  loss={loss.item():.4f} "
                  f"ce={logs['ce']:.4f}{kd_s}{aux_s} "
                  f"λ={lam:.3f} entropy={ent_avg:.3f} lr={lr:.2e}  "
                  f"{(time.time()-t0)/max(1,step+1):.3f} s/step", flush=True)

        if (step + 1) % args.eval_every == 0:
            # eval 时 λ=0（hard top-k 推理态）
            for L in student.model.layers:
                L.mlp.set_lambda(0.0)
            match = eval_match(student, teacher, tok, device)
            print(f"[eval] step {step+1} argmax match = {match:.3f}", flush=True)
            student.train()

        step += 1

        # 周期 checkpoint：防止进程被外部终止时进度全丢
        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            ckpt_dir = os.path.join(args.out_dir, f"ckpt_step{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            student.save_pretrained(ckpt_dir)
            print(f"[ckpt] saved {ckpt_dir}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    student.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[done] saved to {args.out_dir}  total={time.time()-t0:.1f}s", flush=True)


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
