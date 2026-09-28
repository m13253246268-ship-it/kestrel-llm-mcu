#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mistral_ple.py — Mistral + MoE + PLE（Per-Layer Embeddings）+ 三值查表 + tied head。

这是「正式训练 + 部署」的**唯一权威架构源**：训练、convert_minueza 导出、
ref_forward 对拍、kmcu.c 都以此处定义的张量命名与数值口径为准。

架构 = kestrel_mcu 现有 Minueza(Mistral) 骨架（GQA 注意力 + SwiGLU + RMSNorm）
        + MoE（router softmax → top-k 归一化 → 专家 SwiGLU 加权）
        + PLE（per-layer embedding 查表，三值量化）
        + tied head（lm_head 复用 embed_tokens）

PLE 数值口径照 esp32-ai model.py 逐式搬运：
    x0 = embed_tokens(idx)
    ple = ple_model_proj(x0) * dim^-0.5                    # [seq, L*ple_dim]
    ple = RMSNorm(ple.view(seq, L, ple_dim))               # 每 ple_dim 切片归一
    table = ple_table(idx)                                 # [seq, L*ple_dim] 三值
    ple = (ple + table * ple_dim^0.5) * 2^-0.5
    每层 i（attention + ffn 之后）：
        g = gelu(ple_gate_i(x))                            # [seq, ple_dim]
        x = x + RMSNorm(ple_proj_i(g * ple[:, :, i]))      # ple_proj: ple_dim -> dim

张量命名与 HF Mistral 对齐（core 部分复用 convert_minueza.py 的读取），PLE 为新键。
"""
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class MistralPLEConfig:
    vocab_size: int = 32002
    dim: int = 312
    n_layers: int = 10
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_e: int = 128          # MoE 专家 hidden
    n_expert: int = 8
    top_k: int = 2
    ple_dim: int = 128        # PLE 每层查表宽度
    seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tied: bool = True         # tied head（复用 embed_tokens 作 lm_head）
    has_ple: bool = True
    n_cls: int = 0            # 决策头类别数（0=无决策头；>0 挂 cls_head 做分类）
    # 训练用
    ternary_table: bool = True  # PLE 表三值 STE 训练

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    @property
    def kv_rep(self):
        return self.n_heads // self.n_kv_heads

    @property
    def table_width(self):
        return self.n_layers * self.ple_dim

    def to_hf_config(self):
        return {
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": self.dim,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.ffn_e,
            "hidden_act": "silu",
            "max_position_embeddings": self.seq_len,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.norm_eps,
            "rope_theta": self.rope_theta,
            "tie_word_embeddings": self.tied,
            "n_expert": self.n_expert,
            "top_k": self.top_k,
            "ple_dim": self.ple_dim,
            "has_ple": self.has_ple,
            "n_cls": self.n_cls,
        }


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def build_rope(seq_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                     # [seq, half]
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    # x: [seq, n_heads, head_dim]
    return x * cos[None, :, None, :] + rotate_half(x) * sin[None, :, None, :]


# ---------------------------------------------------------------------------
# 三值嵌入表（STE）
# ---------------------------------------------------------------------------
class TernaryEmbedding(nn.Module):
    """PLE 查表三值化：forward 用 {-1,0,+1} × absmean 量化，反向 STE 直通 fp32 master。"""

    def __init__(self, num, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, idx):
        w = self.weight.float()
        gamma = w.abs().mean()
        codes = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        qw = codes.to(self.weight.dtype) * gamma.to(self.weight.dtype)
        ste_w = self.weight + (qw - self.weight).detach()
        return F.embedding(idx, ste_w)


# ---------------------------------------------------------------------------
# 注意力（GQA）
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: MistralPLEConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        nh, nkv, hd = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)      # [B, nh, T, hd]
        k = self.k_proj(x).view(B, T, nkv, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, nkv, hd).transpose(1, 2)

        # RoPE（cos/sin: [seq, half]）
        q = self._rope(q, cos[:T], sin[:T])
        k = self._rope(k, cos[:T], sin[:T])

        k = k.repeat_interleave(self.cfg.kv_rep, dim=1)            # GQA 广播
        v = v.repeat_interleave(self.cfg.kv_rep, dim=1)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=hd ** -0.5)
        o = o.transpose(1, 2).contiguous().view(B, T, nh * hd)
        return self.o_proj(o)

    def _rope(self, x, cos, sin):
        # x: [B, n_heads, T, hd]; cos/sin: [T, half]
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos = cos[None, None, :, :]          # [1, 1, T, half]
        sin = sin[None, None, :, :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# MoE FFN（top-k 归一化，与 ref_forward/kmcu.c 一致）
# ---------------------------------------------------------------------------
class ExpertFFN(nn.Module):
    def __init__(self, dim, ffn_e):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_e, bias=False)
        self.up = nn.Linear(dim, ffn_e, bias=False)
        self.down = nn.Linear(ffn_e, dim, bias=False)

    def forward(self, h):
        return self.down(F.silu(self.gate(h)) * self.up(h))


class MoEFFN(nn.Module):
    def __init__(self, cfg: MistralPLEConfig):
        super().__init__()
        self.n_expert = cfg.n_expert
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.dim, cfg.n_expert, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(cfg.dim, cfg.ffn_e) for _ in range(cfg.n_expert)])

    def forward(self, x):
        B, T, D = x.shape
        p = F.softmax(self.router(x), dim=-1)                     # [B, T, N]
        topk_val, topk_idx = torch.topk(p, self.top_k, dim=-1)
        topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True)  # 归一化（与部署一致）
        xf = x.view(B * T, D)
        out = torch.zeros(B * T, D, device=x.device, dtype=x.dtype)
        idx_f = topk_idx.view(B * T, self.top_k)
        val_f = topk_val.view(B * T, self.top_k)
        for k in range(self.top_k):
            for e in range(self.n_expert):
                sel = idx_f[:, k] == e
                if sel.any():
                    out[sel] += val_f[sel, k].unsqueeze(-1) * self.experts[e](xf[sel])
        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# 层
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, cfg: MistralPLEConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MoEFFN(cfg) if cfg.n_expert > 0 else nn.Identity()
        if cfg.has_ple:
            self.ple_gate = nn.Linear(cfg.dim, cfg.ple_dim, bias=False)
            self.ple_proj = nn.Linear(cfg.ple_dim, cfg.dim, bias=False)
            self.ple_norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, x, cos, sin, ple=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        if ple is not None:
            g = F.gelu(self.ple_gate(x))                         # [B, T, ple_dim]
            x = x + self.ple_norm(self.ple_proj(g * ple))
        return x


# ---------------------------------------------------------------------------
# 完整模型
# ---------------------------------------------------------------------------
class MistralPLE(nn.Module):
    def __init__(self, cfg: MistralPLEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)

        if cfg.has_ple:
            self.ple_model_proj = nn.Linear(cfg.dim, cfg.table_width, bias=False)
            self.ple_proj_norm = RMSNorm(cfg.ple_dim, cfg.norm_eps)
            self.ple_table = (TernaryEmbedding(cfg.vocab_size, cfg.table_width)
                              if cfg.ternary_table
                              else nn.Embedding(cfg.vocab_size, cfg.table_width))

        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)

        # 决策头（可选分支，n_cls=0 时不存在，不破坏现有生成架构）
        self.cls_head = (nn.Linear(cfg.dim, cfg.n_cls, bias=False)
                         if cfg.n_cls > 0 else None)

        self.apply(self._init)
        # GPT-2 风格：residual 写入投影缩小初始化
        for n, p in self.named_parameters():
            if n.endswith("o_proj.weight") or n.endswith("down.weight") or n.endswith("ple_proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))
        # PLE 分支 RMSNorm 从零增益起步（与 esp32-ai 一致，保证分支初始为 no-op）
        if cfg.has_ple:
            for layer in self.layers:
                nn.init.zeros_(layer.ple_norm.weight)

        cos, sin = build_rope(cfg.seq_len, cfg.head_dim, cfg.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        cfg = self.cfg
        B, T = idx.shape
        x = self.embed_tokens(idx)

        ple = None
        if cfg.has_ple:
            ple = self.ple_model_proj(x) * (cfg.dim ** -0.5)
            ple = self.ple_proj_norm(ple.view(B, T, cfg.n_layers, cfg.ple_dim))
            table = self.ple_table(idx).view(B, T, cfg.n_layers, cfg.ple_dim)
            ple = (ple + table * (cfg.ple_dim ** 0.5)) * (2 ** -0.5)

        cos = self.cos[:T]
        sin = self.sin[:T]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if ple is None else ple[:, :, i])

        x = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight) if cfg.tied else self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
                                   ignore_index=-1)
        return logits, loss

    def forward_features(self, idx):
        """返回 last hidden [B, T, dim]（决策头分支用，不动生成路径）。"""
        cfg = self.cfg
        B, T = idx.shape
        x = self.embed_tokens(idx)
        ple = None
        if cfg.has_ple:
            ple = self.ple_model_proj(x) * (cfg.dim ** -0.5)
            ple = self.ple_proj_norm(ple.view(B, T, cfg.n_layers, cfg.ple_dim))
            table = self.ple_table(idx).view(B, T, cfg.n_layers, cfg.ple_dim)
            ple = (ple + table * (cfg.ple_dim ** 0.5)) * (2 ** -0.5)
        cos = self.cos[:T]
        sin = self.sin[:T]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if ple is None else ple[:, :, i])
        return self.norm(x)

    def cls_forward(self, idx):
        """决策头前向：mean pooling last hidden → cls_head → [B, n_cls]。"""
        x = self.forward_features(idx).mean(dim=1)   # [B, dim]
        return self.cls_head(x)                       # [B, n_cls]

    # ---- 参数记账（core/stream/table 分层） ----
    def param_budget(self):
        cfg = self.cfg
        table = 0
        if cfg.has_ple:
            table += self.ple_table.weight.numel()
        stream = 0 if cfg.tied else self.embed_tokens.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        return {"core": total - table - stream, "stream": stream,
                "table": table, "total": total}

    # ---- 导出为 HF safetensors + config.json（key 名与 convert 对齐） ----
    def export_state_dict(self):
        cfg = self.cfg
        sd = {}
        sd["model.embed_tokens.weight"] = self.embed_tokens.weight.detach()
        sd["model.norm.weight"] = self.norm.weight.detach()
        if cfg.has_ple:
            sd["model.ple_model_proj.weight"] = self.ple_model_proj.weight.detach()
            sd["model.ple_proj_norm.weight"] = self.ple_proj_norm.weight.detach()
            sd["model.ple_table.weight"] = self.ple_table.weight.detach()
        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}."
            sd[p + "input_layernorm.weight"] = layer.input_layernorm.weight.detach()
            sd[p + "self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight.detach()
            sd[p + "self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight.detach()
            sd[p + "self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight.detach()
            sd[p + "self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight.detach()
            sd[p + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.detach()
            if cfg.n_expert > 0:
                sd[p + "mlp.router.weight"] = layer.mlp.router.weight.detach()
                for e, expert in enumerate(layer.mlp.experts):
                    ep = p + f"mlp.experts.{e}."
                    sd[ep + "gate.weight"] = expert.gate.weight.detach()
                    sd[ep + "up.weight"] = expert.up.weight.detach()
                    sd[ep + "down.weight"] = expert.down.weight.detach()
            if cfg.has_ple:
                sd[p + "ple_gate.weight"] = layer.ple_gate.weight.detach()
                sd[p + "ple_proj.weight"] = layer.ple_proj.weight.detach()
                sd[p + "ple_norm.weight"] = layer.ple_norm.weight.detach()
        if self.cls_head is not None:
            sd["cls_head.weight"] = self.cls_head.weight.detach()
        return sd

    def save_pretrained(self, out_dir):
        import json
        from safetensors.torch import save_file
        os.makedirs(out_dir, exist_ok=True)
        save_file(self.export_state_dict(), os.path.join(out_dir, "model.safetensors"))
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.to_hf_config(), f, indent=2)
