"""Shared CLI runtime and reproducibility metadata."""

import argparse
import hashlib
from contextlib import nullcontext
from pathlib import Path
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attention", choices=["auto", "flex", "sdpa"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--tree", choices=["guided", "coarse", "full"], default="guided")
    parser.add_argument("--prefetch-batches", type=int, default=4)
    return parser


def autocast(device, precision):
    if device.type == "cuda" and precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bf16; pass --precision fp32.")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_lpips(device):
    import lpips

    return lpips.LPIPS(net="vgg", spatial=True).eval().requires_grad_(False).to(device)


def select_tree(model, perceptual, images, args, maps, rng):
    from .search import guided_search

    if args.tree == "guided":
        return guided_search(model, perceptual, images, args.tau, maps, rng)
    active = torch.ones(images.shape[0], 320, dtype=torch.bool, device=images.device)
    if args.tree == "coarse":
        active[:, 64:] = False
    return model.encode(images), active


def metadata(args, accelerator):
    if args.batch_size < 1 or getattr(args, "workers", 0) < 0 or args.prefetch_batches < 1:
        raise ValueError("batch-size must be positive and workers nonnegative.")
    return {
        "config_sha256": sha256(args.config),
        "checkpoint_sha256": sha256(args.checkpoint),
        "torch": torch.__version__,
        "device": str(accelerator.device),
        "attention": "sdpa" if args.attention == "auto" else args.attention,
        "precision": args.precision if accelerator.device.type == "cuda" else "fp32",
        "seed": args.seed,
        "tau": args.tau,
        "tree": args.tree,
        "batch_size": args.batch_size,
        "world_size": accelerator.num_processes,
        "checkpoint": Path(args.checkpoint).name,
    }
