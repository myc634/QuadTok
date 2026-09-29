"""Fixed-slot topology and canonical breadth-first token ordering."""

from functools import lru_cache
import torch


@lru_cache(maxsize=8)
def build_slot_maps(device):
    lod = torch.cat([torch.full((64,), 3), torch.full((256,), 4)]).to(device)
    patch = torch.cat([torch.arange(64), torch.arange(256)]).to(device)
    child = torch.arange(256, device=device)
    parent = child // 16 // 2 * 8 + child % 16 // 2
    return lod, patch, parent


@lru_cache(maxsize=8)
def slot_rank(device):
    # Enumerate the same TL/TR/BL/BR children as the original tree builder.
    patches = [0]
    ordered = []
    for level in range(1, 5):
        side = 2**level
        patches = [
            child
            for p in patches
            for child in (
                (p // (side // 2) * 2) * side + (p % (side // 2) * 2),
                (p // (side // 2) * 2) * side + (p % (side // 2) * 2) + 1,
                (p // (side // 2) * 2 + 1) * side + (p % (side // 2) * 2),
                (p // (side // 2) * 2 + 1) * side + (p % (side // 2) * 2) + 1,
            )
        ]
        if level >= 3:
            ordered.extend(p + (64 if level == 4 else 0) for p in patches)
    rank = torch.empty(320, dtype=torch.long, device=device)
    rank[torch.tensor(ordered, device=device)] = torch.arange(320, device=device)
    return rank


def active_to_padded(active, slot_lod, slot_patch):
    lengths = active.sum(1).long()
    size = int(lengths.max())
    rank = slot_rank(active.device)
    order = torch.argsort(torch.where(active, rank[None], 1 << 30), dim=1, stable=True)[:, :size]
    valid = torch.arange(size, device=active.device)[None] < lengths[:, None]
    return (
        torch.where(valid, slot_lod[order], -1),
        torch.where(valid, slot_patch[order], 0),
        lengths,
    )


def random_active(batch, device, probability=0.75, shared=False):
    if not 0 <= probability <= 1:
        raise ValueError("Expansion probability must be in [0,1].")
    parent = build_slot_maps(device)[2]
    split = torch.rand(1 if shared else batch, 64, device=device) < probability
    return torch.cat(
        [
            torch.ones(batch, 64, device=device, dtype=torch.bool),
            split.expand(batch, -1)[:, parent],
        ],
        1,
    )
