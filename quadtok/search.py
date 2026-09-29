"""Batched LPIPS A/B search, adapted from the source tensor-native implementation."""

import torch
from torch.nn import functional as F
from .topology import active_to_padded


@torch.no_grad()
def extract_codes(model, latent, active, maps):
    lod, patch, lengths = active_to_padded(active, *maps[:2])
    z = model.select(latent, lod, patch, lengths)
    _, result = model.quantize(z)
    codes = result["min_encoding_indices"].reshape(active.shape[0], -1)
    return codes, lod, patch, lengths


@torch.no_grad()
def reconstruct(model, latent, active, maps):
    return model.decode_codes(*extract_codes(model, latent, active, maps)).clamp(0, 1)


@torch.no_grad()
def guided_search(model, lpips_fn, images, tau, maps, rng):
    latent = model.encode(images)
    batch = images.shape[0]
    split = torch.zeros(64, dtype=torch.bool, device=images.device)
    split[rng.sample(range(64), 32)] = True
    coarse = torch.ones(batch, 64, dtype=torch.bool, device=images.device)
    active_a = torch.cat([coarse, split[maps[2]][None].expand(batch, -1)], 1)
    active_b = torch.cat([coarse, (~split)[maps[2]][None].expand(batch, -1)], 1)
    ra = reconstruct(model, latent, active_a, maps).float()
    rb = reconstruct(model, latent, active_b, maps).float()
    # Spatial VGG LPIPS is always evaluated in fp32, including under bf16 inference.
    with torch.autocast(images.device.type, enabled=False):
        la = lpips_fn(images.float(), ra, normalize=True).sum(1)
        lb = lpips_fn(images.float(), rb, normalize=True).sum(1)
    sign = torch.where(split, 1.0, -1.0).reshape(8, 8)
    gain = F.avg_pool2d((lb - la).unsqueeze(1), 32)[:, 0] * sign
    expanded = gain.reshape(batch, 64) >= tau
    return latent, torch.cat([coarse, expanded[:, maps[2]]], 1)
