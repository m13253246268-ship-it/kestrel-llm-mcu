#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_moe_kd.py — 从头训练 MoE（8 专家 top-2），用稠密 Minueza 作 teacher 蒸馏。

依据 docs/moe_framework.md 第 5 节：
  Student = MixtralForCausalLM（num_local_experts=8, num_experts_per_tok=2, ffn_e=128）
  Teacher = 现有稠密 Minueza（MistralForCausalLM，冻结，只出软标签）
  L = (1-α)·CE(student, hard) + α·T²·KL(softmax(student/T) || softmax(teacher/T)) + β·aux

降低训练难度的两个关键手段：
  1. 知识蒸馏：student 模仿 teacher 的 logits 分布，而非从零学语言建模；
  2. 结构复用（--init_from_teacher，默认开）：embed / attention / norm 直接复用 teacher
     已训练权重，只有 MoE FFN（router + 8 专家）从零初始化。FFN 是唯一要学的新东西。

参数名映射（Mixtral → KMCU v2）：
  block_sparse_moe.gate.weight          -> L{i}.router        [N, dim]
  block_sparse_moe.experts.{e}.w1.weight -> L{i}.expert{e}.gate [ffn_e, dim]
  block_sparse_moe.experts.{e}.w3.weight -> L{i}.expert{e}.up   [ffn_e, dim]
  block_sparse_moe.experts.{e}.w2.weight -> L{i}.expert{e}.down [dim, ffn_e]
  （w1/w3 为 SwiGLU 的 gate/up，w2 为 down）

用法：
  python train_moe_kd.py                          # 默认 TinyStories 流式 + 全默认参数
  python train_moe_kd.py --max_steps 200 --block_size 256 --eval_every 50
  python train_moe_kd.py --data_file corpus.txt   # 用本地文本
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    MixtralConfig,
    MixtralForCausalLM,
    MistralForCausalLM,
)


def _rope_theta(cfg):
    """兼容 transformers 5.x 的 RoPE 参数迁移（rope_theta -> rope_parameters['rope_theta']）。"""
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return rp["rope_theta"]
    return getattr(cfg, "default_theta", 10000.0)


def build_student_config(teacher_cfg, n_expert=8, top_k=2, ffn_e=128):
    """从 teacher 的 Mistral 配置推导等价的 Mixtral（MoE）配置。"""
    c = MixtralConfig(
        vocab_size=teacher_cfg.vocab_size,
        hidden_size=teacher_cfg.hidden_size,
        intermediate_size=ffn_e,              # 每个专家的 ffn 维度（对齐 32）
        num_hidden_layers=teacher_cfg.num_hidden_layers,
        num_attention_heads=teacher_cfg.num_attention_heads,
        num_key_value_heads=teacher_cfg.num_key_value_heads,
        num_local_experts=n_expert,
        num_experts_per_tok=top_k,
        rms_norm_eps=teacher_cfg.rms_norm_eps,
        rope_theta=_rope_theta(teacher_cfg),
        max_position_embeddings=teacher_cfg.max_position_embeddings,
        bos_token_id=teacher_cfg.bos_token_id,
        eos_token_id=teacher_cfg.eos_token_id,
        tie_word_embeddings=teacher_cfg.tie_word_embeddings,
        torch_dtype="float32",
    )
    return c


def copy_teacher_weights(student, teacher):
    """把 teacher 里与 student 同名、同 shape 的参数拷进 student。

    MoE FFN（block_sparse_moe）在 teacher 中不存在同名参数，自然跳过，
    因此只有 embed / attention / norm / lm_head 被复用，FFN 与 router 从零。
    """
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
    print(f"[init] copied {copied} params from teacher, left {skipped} from scratch")
    return student


def make_data_iterator(tokenizer, block_size, data_file=None, dataset_name="roneneldan/TinyStories",
                       batch_size=16):
    """流式文本 → 固定长度 token 块 → batch。"""
    def gen_texts():
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
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= block_size + 1:
            block = buf[: block_size + 1]
            buf = buf[block_size:]
            yield block

    # 不足一个 block 的末尾丢弃（不做 padding，避免训练到 pad 噪声）


def collate(blocks, batch_size, device, pad_id):
    xs, ys = [], []
    for b in blocks:
        xs.append(b[:-1])
        ys.append(b[1:])
    if not xs:
        return None, None
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


def kd_loss_fn(student_logits, teacher_logits, labels, T, alpha):
    """Hinton 软标签蒸馏 + 可选 hard label CE。

    student_logits / teacher_logits: [B, S, V]
    labels: [B, S]  (已 shift 后的下一个 token)
    """
    s = student_logits[..., :-1, :].reshape(-1, student_logits.shape[-1])
    t = teacher_logits[..., :-1, :].reshape(-1, teacher_logits.shape[-1])
    lab = labels[..., 1:].reshape(-1)

    # 软标签 KL：放大温度，让 student 学 token 间相对关系
    kd = F.kl_div(
        F.log_softmax(s / T, dim=-1),
        F.softmax(t / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

    # hard label CE（可加权，alpha=1 时纯蒸馏）
    ce = F.cross_entropy(s, lab)

    return (1.0 - alpha) * ce + alpha * kd, {"kd": kd.item(), "ce": ce.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=r"E:\models", help="teacher 权重目录（含 config.json + safetensors）")
    ap.add_argument("--out_dir", default=r"E:\models\moe_student", help="student 保存目录")
    ap.add_argument("--data_file", default=None, help="本地文本文件（可选，默认用 TinyStories 流式）")
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--n_expert", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--ffn_e", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--T", type=float, default=4.0, help="蒸馏温度")
    ap.add_argument("--alpha", type=float, default=0.9, help="KD 权重（1=纯蒸馏）")
    ap.add_argument("--aux_w", type=float, default=0.01, help="router 负载均衡 loss 权重")
    ap.add_argument("--init_from_teacher", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    # ---- teacher ----
    print("[teacher] loading ...")
    teacher = MistralForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # ---- student ----
    print(f"[student] building MoE (N={args.n_expert}, top_k={args.top_k}, ffn_e={args.ffn_e}) ...")
    scfg = build_student_config(teacher.config, args.n_expert, args.top_k, args.ffn_e)
    student = MixtralForCausalLM(scfg).to(device)
    if args.init_from_teacher:
        copy_teacher_weights(student, teacher)
    student.train()

    n_param = sum(p.numel() for p in student.parameters())
    print(f"[student] total params = {n_param/1e6:.2f} M")

    # 只训练需要训练的（全量 student 都 requires_grad=True，但 teacher 已冻结）
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * p))

    # ---- 数据流 ----
    it = make_data_iterator(tok, args.block_size, args.data_file, args.dataset, args.batch_size)
    block_buf = []
    pad_id = tok.pad_token_id

    print(f"[train] max_steps={args.max_steps} block={args.block_size} batch={args.batch_size} "
          f"lr={args.lr} T={args.T} alpha={args.alpha}")
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        # 攒够一个 batch 的 block
        while len(block_buf) < args.batch_size:
            try:
                block_buf.append(next(it))
            except StopIteration:
                # 数据耗尽，重置迭代器（流式数据集每次会重新请求）
                it = make_data_iterator(tok, args.block_size, args.data_file, args.dataset, args.batch_size)
                if not block_buf:
                    print("[train] data exhausted, restarting stream ...")

        blocks = block_buf[: args.batch_size]
        block_buf = block_buf[args.batch_size:]
        x, y = collate(blocks, args.batch_size, device, pad_id)
        if x is None:
            continue

        # ---- teacher 软标签（no_grad）----
        with torch.no_grad():
            t_out = teacher(input_ids=x).logits

        # ---- student 前向 ----
        s_out = student(input_ids=x, output_router_logits=True)
        loss, logs = kd_loss_fn(s_out.logits, t_out, y, args.T, args.alpha)
        if args.aux_w > 0 and getattr(s_out, "aux_loss", None) is not None:
            loss = loss + args.aux_w * s_out.aux_loss
            logs["aux"] = s_out.aux_loss.item()

        # ---- 反向 ----
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == args.max_steps - 1:
            dt = time.time() - t0
            sps = dt / max(1, step + 1)
            aux_s = f" aux={logs['aux']:.4f}" if "aux" in logs else ""
            print(f"step {step:5d}/{args.max_steps}  loss={loss.item():.4f}  "
                  f"kd={logs['kd']:.4f} ce={logs['ce']:.4f}{aux_s} lr={lr:.2e}  "
                  f"{sps:.3f} s/step")

        # ---- 定期 eval：student vs teacher 贪心 argmax 匹配率 ----
        if (step + 1) % args.eval_every == 0:
            match = eval_match(student, teacher, tok, device, n_prompt=4, n_gen=16)
            print(f"[eval] step {step+1} argmax match = {match:.2f} / 1.0")

        step += 1

    # ---- 保存 ----
    os.makedirs(args.out_dir, exist_ok=True)
    student.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[done] student saved to {args.out_dir}  total_time={time.time()-t0:.1f}s")


@torch.no_grad()
def eval_match(student, teacher, tok, device, n_prompt=4, n_gen=16):
    """贪心 argmax 序列匹配率：同一 prompt 下 student 与 teacher 逐 token 一致的比例。"""
    prompts = [
        "Once upon a time",
        "The little cat",
        "A boy named",
        "In a faraway",
        "One day, a",
        "The old man",
    ][:n_prompt]
    student.eval()
    total, hit = 0, 0
    for p in prompts:
        ids = tok.encode(p, return_tensors="pt").to(device)
        attn = torch.ones_like(ids)
        s_gen = student.generate(ids, attention_mask=attn, max_new_tokens=n_gen,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        t_gen = teacher.generate(ids, attention_mask=attn, max_new_tokens=n_gen,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        s_tok = s_gen[0].tolist()
        t_tok = t_gen[0].tolist()
        L = min(len(s_tok), len(t_tok))
        total += L
        hit += sum(1 for i in range(L) if s_tok[i] == t_tok[i])
    student.train()
    return hit / max(1, total)


if __name__ == "__main__":
    main()
