"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""

import torch
from torch import nn
from einops.layers.torch import Rearrange
from .attention import (
    ResidualAttentionBlock,
    parent_indices_vec,
    build_batched_kinship_block_mask,
    build_batched_selector_block_mask,
)


def scatter_patches(
    canvas: torch.Tensor,
    patches: torch.Tensor,
    batch_coords: torch.Tensor,
    y_starts: torch.Tensor,
    x_starts: torch.Tensor,
    patch_size: int,
):
    N, C, _, _ = patches.shape
    device = patches.device
    patch_y_offsets = torch.arange(patch_size, device=device).view(patch_size, 1)
    patch_x_offsets = torch.arange(patch_size, device=device).view(1, patch_size)

    y_dest = y_starts.view(N, 1, 1) + patch_y_offsets
    x_dest = x_starts.view(N, 1, 1) + patch_x_offsets

    canvas[batch_coords[:, None, None], :, y_dest, x_dest] = patches.permute(
        0, 2, 3, 1
    ).contiguous()

    return canvas


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class QuadTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_backend = config.model.get("attention_backend", "auto")
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size

        self.width = {
            "small": 512,
            "base": 768,
            "large": 1024,
        }[self.model_size]
        self.num_layers = {
            "small": 8,
            "base": 12,
            "large": 24,
        }[self.model_size]
        self.num_heads = {
            "small": 8,
            "base": 12,
            "large": 16,
        }[self.model_size]

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        scale = self.width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size**2 + 1, self.width)
        )

        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            )
        self.ln_post = nn.LayerNorm(self.width)
        self.out_proj = nn.Linear(self.width, self.width)

    def forward(self, pixel_values):
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)  # shape = [*, grid ** 2 + 1, width]

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 1:]
        x = self.ln_post(x)
        x = self.out_proj(x)

        return x


class QuadTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_backend = config.model.get("attention_backend", "auto")
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.token_size = config.model.selector.token_size

        self.num_patch_side_list = config.model.selector.num_patch_side_list
        self.patch_size_list = config.model.selector.patch_size_list
        self.num_lod = len(config.model.selector.num_patch_side_list)

        self.width = {
            "small": 512,
            "base": 768,
            "large": 512,
        }[self.model_size]
        self.num_layers = {
            "small": 8,
            "base": 12,
            "large": 24,
        }[self.model_size]
        self.num_heads = {
            "small": 8,
            "base": 12,
            "large": 16,
        }[self.model_size]

        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        self.token_incides_embedding_dict = nn.ModuleDict()
        self.max_seq_len = 0
        for lod_idx, num_patches in enumerate(self.num_patch_side_list):
            total_patches = num_patches**2
            self.max_seq_len += total_patches
            if lod_idx >= 3:
                self.token_incides_embedding_dict[str(lod_idx)] = nn.Embedding(
                    total_patches, self.width
                )

        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            )
        self.ln_post = nn.LayerNorm(self.width)

        self.decoder_channels = [self.width // (2**i) for i in range(self.num_lod)]

        self.latent_unpatchers = nn.ModuleDict()
        for i in range(self.num_lod):
            if i >= 3:
                patch_size = self.patch_size_list[i]
                out_channels = self.decoder_channels[i]
                self.latent_unpatchers[str(i)] = nn.Sequential(
                    nn.Linear(self.width, out_channels * patch_size * patch_size),
                    Rearrange("b (c p1 p2) -> b c p1 p2", p1=patch_size, p2=patch_size),
                )

        self.upsamplers = nn.ModuleDict()
        for i in range(self.num_lod - 1):
            if i >= 3:
                if i < 4:
                    in_channels = self.decoder_channels[i]
                    out_channels = self.decoder_channels[i + 1]
                    self.upsamplers[str(i)] = nn.Sequential(  #
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=32, num_channels=in_channels),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    )
                else:
                    in_channels = self.decoder_channels[i]
                    out_channels = self.decoder_channels[i + 1]
                    self.upsamplers[str(i)] = nn.Sequential(  #
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=32, num_channels=in_channels),
                        nn.GELU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    )

        self.conv_out = nn.Conv2d(self.decoder_channels[-1], 3, 3, padding=1, bias=True)

    def hierarchical_latent_decode_vec_batched(self, features, lod_pad, pat_pad):
        """Fully-batched per-image hierarchical decode (no python loop over images).
        features: (B,S,D) transformer output; lod_pad/pat_pad: (B,S) long, padding rows lod=-1.
        One scatter_patches per LOD across the whole batch; matches the node-based
        hierarchical_latent_decode applied per image."""
        device = features.device
        dtype = features.dtype
        B = features.shape[0]
        previous_feature_map = None
        for lod_idx in range(self.num_lod):
            if lod_idx < 3:
                continue
            channels = self.decoder_channels[lod_idx]
            patch_size = self.patch_size_list[lod_idx]
            nps = self.num_patch_side_list[lod_idx]
            if lod_idx == 3:
                upsampled_map = torch.zeros(
                    B, channels, patch_size * nps, patch_size * nps, device=device, dtype=dtype
                )
            else:
                upsampled_map = self.upsamplers[str(lod_idx - 1)](previous_feature_map)
            canvas = torch.zeros_like(upsampled_map)
            lod_mask = lod_pad == lod_idx  # (B,S); padding (-1) excluded
            if lod_mask.any():
                batch_coords = torch.nonzero(lod_mask, as_tuple=True)[0]  # (N,)
                feats = features[lod_mask]  # (N,D)
                unpatched = self.latent_unpatchers[str(lod_idx)](feats)  # (N,C,p,p)
                p_idx = pat_pad[lod_mask]  # (N,)
                y_starts = (p_idx // nps) * patch_size
                x_starts = (p_idx % nps) * patch_size
                canvas = scatter_patches(
                    canvas, unpatched.to(canvas.dtype), batch_coords, y_starts, x_starts, patch_size
                )
            previous_feature_map = upsampled_map + canvas
        return previous_feature_map

    def _decode_optimize_core(self, z_quantized, lod_pad, pat_pad, seqlens):
        """Tensor core of the batched multi-tree decode (replaces the node-based _forward_optimize).
        lod_pad/pat_pad: (B,S) long (padding lod=-1); z_quantized: (B,>=S,D). batched
        kinship attention mask + batched hierarchical decode."""
        device = lod_pad.device
        B, max_seq = lod_pad.shape
        z_emb = self.decoder_embed(z_quantized)
        if z_emb.shape[1] < max_seq:
            z_emb = torch.nn.functional.pad(z_emb, (0, 0, 0, max_seq - z_emb.shape[1]))
        elif z_emb.shape[1] > max_seq:
            z_emb = z_emb[:, :max_seq]
        tok = torch.zeros(B, max_seq, self.width, device=device, dtype=z_emb.dtype)
        for lod_str, emb_layer in self.token_incides_embedding_dict.items():
            m = lod_pad == int(lod_str)
            if m.any():
                tok[m] = emb_layer(pat_pad[m]).to(z_emb.dtype)
        valid = torch.arange(max_seq, device=device)[None, :] < seqlens[:, None]
        x = (z_emb + tok) * valid.unsqueeze(-1)
        min_lod = 3
        parent = parent_indices_vec(
            lod_pad.reshape(-1), pat_pad.reshape(-1), self.num_patch_side_list, min_lod
        ).reshape(B, max_seq)
        block_mask = build_batched_kinship_block_mask(
            lod_pad, parent, seqlens, min_lod, device, backend=self.attention_backend
        )
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i in range(self.num_layers):
            x = self.transformer[i](x, block_mask=block_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        upsampled_latent = self.hierarchical_latent_decode_vec_batched(x, lod_pad, pat_pad)
        return self.conv_out(upsampled_latent)


class QuadTokSelector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.attention_backend = config.model.get("attention_backend", "auto")

        self.num_patch_side_list = config.model.selector.num_patch_side_list
        self.image_size = config.dataset.preprocessing.crop_size
        self.num_lod = len(config.model.selector.num_patch_side_list)

        self.model_size = config.model.vq_model.vit_enc_model_size
        self.token_size = config.model.selector.token_size

        self.width = {
            "small": 512,
            "base": 768,
            "large": 1024,
        }[self.model_size]
        self.num_layers = {
            "small": 8,
            "base": 12,
            "large": 24,
        }[self.model_size]
        self.num_heads = {
            "small": 8,
            "base": 12,
            "large": 16,
        }[self.model_size]

        self.token_incides_embedding_dict = nn.ModuleDict()
        self.max_seq_len = 256
        for lod_idx, num_patches in enumerate(self.num_patch_side_list):
            total_patches = num_patches**2
            self.max_seq_len += total_patches
            if lod_idx >= 3:
                self.token_incides_embedding_dict[str(lod_idx)] = nn.Embedding(
                    total_patches, self.width
                )

        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            )

        self.ln_post = nn.LayerNorm(self.width)
        self.out_proj = nn.Linear(self.width, self.token_size)

    def _select_optimize_core(self, latent_feats, lod_pad, pat_pad, seqlens):
        """Tensor core of the batched multi-tree selection (replaces node-based _forward_optimize).
        latent_feats: (B,num_latent,D); lod_pad/pat_pad: (B,S) long (padding lod=-1).
        Batched selector kinship attention mask."""
        device = latent_feats.device
        B, max_seq = lod_pad.shape
        num_latent = latent_feats.shape[1]
        tok = torch.zeros(B, max_seq, self.width, device=device, dtype=latent_feats.dtype)
        for lod_str, emb_layer in self.token_incides_embedding_dict.items():
            m = lod_pad == int(lod_str)
            if m.any():
                tok[m] = emb_layer(pat_pad[m]).to(latent_feats.dtype)
        valid = torch.arange(max_seq, device=device)[None, :] < seqlens[:, None]
        tok = tok * valid.unsqueeze(-1)
        x = torch.cat([latent_feats, tok], dim=1)
        min_lod = 3
        parent = parent_indices_vec(
            lod_pad.reshape(-1), pat_pad.reshape(-1), self.num_patch_side_list, min_lod
        ).reshape(B, max_seq)
        block_mask = build_batched_selector_block_mask(
            num_latent, lod_pad, parent, seqlens, min_lod, device, backend=self.attention_backend
        )
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i in range(self.num_layers):
            x = self.transformer[i](x, block_mask=block_mask)
        x = x.permute(1, 0, 2)
        x = x[:, num_latent:]
        x = self.ln_post(x)
        x = self.out_proj(x)
        x = x.permute(0, 2, 1).unsqueeze(2).contiguous()
        return x
