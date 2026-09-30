"""Checkpoint-compatible attention and tensor-native kinship masks.

ResidualAttentionBlock derives from the Apache-2.0 TiTok implementation.
Copyright (2024) Bytedance Ltd. and/or its affiliates. See LICENSE and NOTICE.
"""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio=4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        # Preserve packed QKV parameter names for strict checkpoint loading.
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio
        if mlp_ratio > 0:
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(d_model, int(d_model * mlp_ratio))),
                        ("gelu", nn.GELU()),
                        ("c_proj", nn.Linear(int(d_model * mlp_ratio), d_model)),
                    ]
                )
            )

    def forward(self, x, block_mask=None):
        h = self.ln_1(x)
        length, batch, width = h.shape
        qkv = F.linear(h, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = [
            t.reshape(length, batch, self.n_head, width // self.n_head)
            .permute(1, 2, 0, 3)
            .contiguous()
            for t in qkv.chunk(3, -1)
        ]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)
        out = out.permute(2, 0, 1, 3).reshape(length, batch, width)
        x = x + F.linear(out, self.attn.out_proj.weight, self.attn.out_proj.bias)
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x


def parent_indices_vec(lod_t, patch_t, num_patch_side_list, min_lod):
    sides = torch.tensor(num_patch_side_list, device=lod_t.device)
    side = sides[lod_t.clamp(min=0)]
    parent = (patch_t // side // 2) * sides[(lod_t - 1).clamp(min=0)] + patch_t % side // 2
    return torch.where(lod_t > min_lod, parent, -1)


def _mask(lod, parent, lengths, min_lod, device, num_latent=0, backend="auto"):
    batch, size = lod.shape
    total = size + num_latent

    def mask_mod(b, h, q, k):
        qt = (q - num_latent).clamp(0, size - 1)
        kt = (k - num_latent).clamp(0, size - 1)
        lq, lk = lod[b, qt], lod[b, kt]
        kinship = (lk < lq) | ((lk == lq) & ((lq == min_lod) | (parent[b, qt] == parent[b, kt])))
        tree = (kt < lengths[b]) & kinship
        allow = torch.where(q < num_latent, k < num_latent, (k < num_latent) | tree)
        # Padded queries are discarded, but let them see a real key to avoid NaNs.
        allow = allow | ((q >= lengths[b] + num_latent) & (k == 0))
        return allow & (q < total) & (k < total)

    if backend not in ("auto", "sdpa"):
        raise ValueError(f"Only SDPA is supported; got {backend!r}")
    b = torch.arange(batch, device=device)[:, None, None]
    q = torch.arange(total, device=device)[None, :, None]
    k = torch.arange(total, device=device)[None, None, :]
    return mask_mod(b, 0, q, k).unsqueeze(1)


def build_batched_kinship_block_mask(lod, parent, lengths, min_lod, device, backend="auto"):
    return _mask(lod, parent, lengths, min_lod, device, backend=backend)


def build_batched_selector_block_mask(
    num_latent, lod, parent, lengths, min_lod, device, backend="auto"
):
    return _mask(lod, parent, lengths, min_lod, device, num_latent, backend)
