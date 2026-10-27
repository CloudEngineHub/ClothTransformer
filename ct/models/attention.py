# Pre-norm Transformer encoder used by the temporal model.
#
# Adapted from RenderFormer (https://github.com/microsoft/renderformer),
# Copyright (c) Microsoft Corporation, MIT License. Reduced to the self-attention
# encoder path with PyTorch SDPA; parameter names are unchanged.
from typing import Literal

import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x))))


class FeedForwardGeLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w2(self.dropout(F.gelu(self.w1(x)))))


def _norm_module(norm_type: str):
    if norm_type == 'layer_norm':
        return nn.LayerNorm
    if norm_type == 'rms_norm':
        return nn.RMSNorm
    raise ValueError("Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'.")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = True, qk_norm: bool = False,
                 norm_type: str = 'layer_norm'):
        super().__init__()
        self.num_heads = num_heads
        self.in_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        if qk_norm:
            norm = _norm_module(norm_type)
            self.q_norm = norm(dim, eps=EPS)
            self.k_norm = norm(dim, eps=EPS)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, src_key_padding_mask=None, attn_mask=None, is_causal=False):
        """
        Args:
            x: (B, N, D)
            src_key_padding_mask: (B, N) bool, True = attend to this token
            attn_mask: (N, N) or (B, H, N, N) bool, True = allowed
        """
        bs, seq_len = x.shape[:2]
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = self.q_norm(q).type(v.dtype)
        k = self.k_norm(k).type(v.dtype)

        q = q.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)  # (B, H, N, head_dim)
        k = k.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)

        if src_key_padding_mask is not None:
            key_mask = src_key_padding_mask.view(bs, 1, 1, seq_len).bool()
            if attn_mask is not None:
                if attn_mask.ndim == 2:
                    attn_mask = attn_mask.view(1, 1, seq_len, seq_len)
                attn_mask = attn_mask.bool() & key_mask
            else:
                attn_mask = key_mask.expand(-1, self.num_heads, -1, -1)
        elif attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = attn_mask.view(1, 1, seq_len, seq_len).expand(bs, self.num_heads, -1, -1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(out)


class AttentionLayer(nn.Module):
    """Pre-norm self-attention + feed-forward block."""

    def __init__(self, dim: int, num_heads: int, ffn_hidden_dim: int, dropout: float = 0.1,
                 bias: bool = True, activation: str = 'swiglu',
                 norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
                 qk_norm: bool = False):
        super().__init__()
        self.multihead_attn = MultiHeadSelfAttention(dim, num_heads, bias=bias, qk_norm=qk_norm,
                                                     norm_type=norm_type)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        norm = _norm_module(norm_type)
        self.query_norm = norm(dim, eps=EPS)
        if activation == 'swiglu':
            self.ffn = FeedForwardSwiGLU(dim, ffn_hidden_dim, dropout=dropout, bias=bias)
        elif activation == 'gelu':
            self.ffn = FeedForwardGeLU(dim, ffn_hidden_dim, dropout=dropout, bias=bias)
        else:
            raise ValueError("Unsupported activation function. Choose from 'gelu' and 'swiglu'.")
        self.ffn_norm = norm(dim, eps=EPS)

    def forward(self, x, src_key_padding_mask=None, attn_mask=None, is_causal=False):
        x = x + self.dropout(self.multihead_attn(self.query_norm(x), src_key_padding_mask,
                                                 attn_mask, is_causal))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, hidden_dim: int, ffn_hidden_dim: int,
                 dropout: float = 0.1, bias: bool = True, activation: str = 'swiglu',
                 norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
                 qk_norm: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionLayer(hidden_dim, num_heads, ffn_hidden_dim, dropout=dropout, bias=bias,
                           activation=activation, norm_type=norm_type, qk_norm=qk_norm)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_key_padding_mask=None, attn_mask=None, is_causal=False):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, attn_mask=attn_mask,
                      is_causal=is_causal)
        return x
