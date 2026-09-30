"""CUDA SDPA kernel profiling and steady-state tokenizer throughput.

Run from the repository root: python -m benchmarks.gpu --checkpoint FILE --output results/gpu.json
Timing excludes checkpoint loading, image decoding and compilation; includes tree masks,
encoder, selector, quantizer and decoder (and LPIPS search for guided pretokenization).
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from quadtok.model import load_model
from quadtok.topology import build_slot_maps, random_active
from quadtok.search import guided_search, extract_codes
from quadtok.runtime import make_lpips, sha256


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return {
        "median_ms": statistics.median(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "repeat_seconds": times,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/tokenizer.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--guided", action="store_true")
    p.add_argument("--data", help="Optional image folder; otherwise seeded synthetic images")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and never falls back to CPU.")
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "input": "image_folder" if args.data else "seeded_uniform_noise",
        "rows": [],
    }
    model = load_model(args.config, args.checkpoint, "cuda", "sdpa")
    maps = build_slot_maps("cuda")
    if args.data:
        from quadtok.data import FolderImages

        ds = FolderImages(args.data)
        base = torch.stack([ds[i % len(ds)][0] for i in range(max(args.batch_sizes))]).cuda()
    else:
        base = torch.rand(max(args.batch_sizes), 3, 256, 256, device="cuda")
    with torch.no_grad():
        image = base[:1]
        active = random_active(1, "cuda", 0.75)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(image, active)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                model(image, active)
                torch.cuda.synchronize()
        report["attention_kernel_events"] = sorted(
            {
                e.name
                for e in prof.events()
                if any(k in e.name.lower() for k in ["attention", "flash", "fmha"])
            }
        )
        for backend in ["sdpa"]:
            for batch in args.batch_sizes:
                images = base[:batch]
                torch.manual_seed(1234 + batch)
                active = random_active(batch, "cuda", 0.75)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = measure(lambda: model(images, active), args.warmup, args.repeats)
                row = {
                    "backend": backend,
                    "operation": "reconstruction",
                    "batch_size": batch,
                    "mean_tokens": active.sum(1).float().mean().item(),
                    "precision": "bf16",
                    **result,
                }
                row["images_per_second"] = batch * 1000 / row["median_ms"]
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
        if args.guided:
            perceptual = make_lpips("cuda")
            for backend in ["sdpa"]:
                for batch in args.batch_sizes:
                    images = base[:batch]

                    def run():
                        # Same A/B assignment across timing repeats and backends.
                        latent, active = guided_search(
                            model, perceptual, images, 0.05, maps, random.Random(1234)
                        )
                        return extract_codes(model, latent, active, maps)

                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = measure(run, args.warmup, args.repeats)
                        codes, lod, patch, lengths = run()
                    row = {
                        "backend": backend,
                        "operation": "guided_pretokenize",
                        "batch_size": batch,
                        "mean_tokens": lengths.float().mean().item(),
                        "precision": "bf16",
                        **result,
                    }
                    row["images_per_second"] = batch * 1000 / row["median_ms"]
                    report["rows"].append(row)
                    print(json.dumps(row), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("GPU_VALIDATION_PASS", flush=True)


if __name__ == "__main__":
    main()
