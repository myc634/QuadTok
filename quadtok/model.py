"""Two-level 256px QuadTok VQ tokenizer, compatible with the released checkpoint.

Architecture adapted from QuadTok/TiTok (Apache-2.0).
Copyright (2024) Bytedance Ltd. and/or its affiliates. See LICENSE and NOTICE.
"""

import torch
from torch import nn
from omegaconf import OmegaConf

from .blocks import QuadTokEncoder, QuadTokDecoder, QuadTokSelector
from .quantizer import VectorQuantizer


class QuadTok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = OmegaConf.create(config) if isinstance(config, dict) else config
        cfg = self.config
        if (
            list(cfg.model.selector.num_patch_side_list) != [1, 2, 4, 8, 16]
            or list(cfg.model.selector.patch_size_list) != [16] * 5
            or cfg.dataset.preprocessing.crop_size != 256
            or cfg.model.vq_model.quantize_mode != "vq"
        ):
            raise ValueError("This release supports the two-level 256px VQ tokenizer only.")
        self.encoder = QuadTokEncoder(cfg)
        self.decoder = QuadTokDecoder(cfg)
        self.selector = QuadTokSelector(cfg)
        self.apply(self._init_weights)
        vq = cfg.model.vq_model
        if vq.token_size != cfg.model.selector.token_size:
            raise ValueError("VQ and selector token_size must agree.")
        self.quantize = VectorQuantizer(
            vq.codebook_size, vq.token_size, vq.commitment_cost, vq.use_l2_norm
        )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def encode(self, images):
        return self.encoder(images)

    def select(self, latent, lod, patch, lengths):
        return self.selector._select_optimize_core(latent, lod, patch, lengths)

    def decode(self, embeddings, lod, patch, lengths):
        return self.decoder._decode_optimize_core(embeddings, lod, patch, lengths)

    def decode_codes(self, codes, lod, patch, lengths):
        embeddings = self.quantize.get_codebook_entry(codes.long().reshape(-1))
        embeddings = embeddings.reshape(*codes.shape, -1)
        return self.decode(embeddings, lod, patch, lengths)

    def forward(self, images, active=None):
        from .topology import build_slot_maps, active_to_padded, random_active

        maps = build_slot_maps(images.device)
        if active is None:
            # Original training samples one shared tree for the whole batch.
            active = random_active(
                images.shape[0],
                images.device,
                self.config.training.get("expansion_probability", 0.75),
                shared=True,
            )
        lod, patch, lengths = active_to_padded(active, *maps[:2])
        z = self.select(self.encode(images), lod, patch, lengths)
        quantized, result = self.quantize(z)
        embeddings = quantized.squeeze(2).transpose(1, 2).contiguous()
        return self.decode(embeddings, lod, patch, lengths), result


def load_model(config_path, checkpoint, device="cpu", attention="auto"):
    config = OmegaConf.load(config_path)
    config.model.attention_backend = attention
    model = QuadTok(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)
