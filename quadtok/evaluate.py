"""Distributed reconstruction evaluation without duplicated tail samples."""

import json
import random
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset

from .data import FolderImages, batches, expand_shards, tar_samples, prefetch
from .model import load_model
from .runtime import autocast, inference_parser, make_lpips, metadata, select_tree
from .search import reconstruct
from .topology import build_slot_maps


def main():
    parser = inference_parser(__doc__)
    parser.add_argument("--workers", type=int, default=4)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data")
    source.add_argument("--shards")
    parser.add_argument("--output", default="results/reconstruction.json")
    parser.add_argument("--limit", type=int, help="Global number of images, folder input only")
    parser.add_argument(
        "--skip-fid", action="store_true", help="Skip Inception download and rFID/IS computation"
    )
    parser.add_argument(
        "--skip-lpips", action="store_true", help="Only available with coarse/full topology"
    )
    args = parser.parse_args()
    if args.skip_lpips and args.tree == "guided":
        parser.error("Guided search requires LPIPS")
    if args.limit is not None and (args.limit < 1 or args.shards):
        parser.error("--limit must be positive and requires --data")
    acc = Accelerator()
    info = metadata(args, acc)
    model = load_model(args.config, args.checkpoint, acc.device, args.attention)
    perceptual = None if args.skip_lpips else make_lpips(acc.device)
    metrics = None
    if not args.skip_fid:
        from .metrics import ReconstructionMetrics

        metrics = ReconstructionMetrics(acc.device)
    maps = build_slot_maps(acc.device)
    rng = random.Random(args.seed + acc.process_index)
    if args.data:
        dataset = FolderImages(args.data)
        count = min(args.limit or len(dataset), len(dataset))
        dataset = Subset(dataset, range(acc.process_index, count, acc.num_processes))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=acc.device.type == "cuda",
        )
    else:
        paths = expand_shards(args.shards)[acc.process_index :: acc.num_processes]
        loader = batches((sample for p in paths for sample in tar_samples(p)), args.batch_size)
    totals = torch.zeros(4, device=acc.device, dtype=torch.float64)
    started = time.perf_counter()
    with torch.no_grad():
        for images, _, _ in prefetch(loader, args.prefetch_batches):
            images = images.to(acc.device, non_blocking=True)
            with autocast(acc.device, args.precision):
                latent, active = select_tree(model, perceptual, images, args, maps, rng)
                reconstruction = reconstruct(model, latent, active, maps).float()
            totals[0] += images.shape[0]
            totals[1] += (
                -10 * ((images - reconstruction).square().mean((1, 2, 3)) + 1e-10).log10()
            ).sum()
            totals[2] += active.sum()
            if perceptual is not None:
                totals[3] += (
                    perceptual(images.float(), reconstruction, normalize=True).mean((1, 2, 3)).sum()
                )
            if metrics is not None:
                metrics.update(images, reconstruction)
    totals = acc.reduce(totals, reduction="sum")
    if metrics is not None:
        metrics.synchronize(acc)
    if acc.is_main_process:
        n = int(totals[0])
        if n == 0:
            raise ValueError("No evaluation images were read.")
        result = {
            "num_images": n,
            "PSNR": float(totals[1] / n),
            "mean_tokens": float(totals[2] / n),
        }
        if perceptual is not None:
            result["LPIPS_vgg"] = float(totals[3] / n)
        if metrics is not None:
            result.update(metrics.result())
        result.update(runtime=info, elapsed_seconds=time.perf_counter() - started)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
