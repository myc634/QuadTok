"""Shard-parallel pretokenization with atomic output and verified resumability."""

import json
import os
import random
import tarfile
from pathlib import Path

import torch
from accelerate import Accelerator

from .data import batches, expand_shards, tar_samples, write_token_sample, prefetch
from .model import load_model
from .runtime import autocast, inference_parser, make_lpips, metadata, select_tree, sha256
from .search import extract_codes
from .topology import build_slot_maps


def main():
    parser = inference_parser(__doc__)
    parser.add_argument("--shards", required=True, help="Quoted local .tar glob or brace range")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--augmentation", choices=["center", "center_hflip"], default="center_hflip"
    )
    args = parser.parse_args()
    acc = Accelerator()
    info = metadata(args, acc)
    paths = expand_shards(args.shards)
    names = [Path(p).name for p in paths]
    if len(names) != len(set(names)):
        parser.error("Input shard basenames must be unique")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(Path(p).parent == output for p in paths):
        parser.error("Output must be different from input directories")
    model = load_model(args.config, args.checkpoint, acc.device, args.attention)
    perceptual = make_lpips(acc.device) if args.tree == "guided" else None
    maps = build_slot_maps(acc.device)
    for index in range(acc.process_index, len(paths), acc.num_processes):
        source = Path(paths[index])
        destination = output / source.name
        manifest = destination.with_suffix(".json")
        signature = {
            **info,
            "input_sha256": sha256(source),
            "augmentation": args.augmentation,
            "shard_seed": args.seed + index,
            "format_version": 1,
        }
        # Assignment can change across ranks without changing this shard's deterministic seed.
        signature.pop("world_size")
        signature.pop("device")
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if (
                previous["signature"] != signature
                or not destination.exists()
                or sha256(destination) != previous["output_sha256"]
            ):
                raise ValueError(
                    f"Existing output does not match this run: {destination}; choose a new output directory"
                )
            print(f"[rank {acc.process_index}] verified skip: {destination.name}", flush=True)
            continue
        lock = destination.with_suffix(".lock")
        # An interrupted output without a manifest is incomplete and is recomputed.
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        temporary = destination.with_suffix(".tar.tmp")
        manifest_tmp = manifest.with_suffix(".json.tmp")
        count = tokens = 0
        rng = random.Random(args.seed + index)
        try:
            with torch.no_grad(), tarfile.open(temporary, "w") as writer:
                seen = set()
                for images, labels, keys in prefetch(
                    batches(tar_samples(source), args.batch_size, args.augmentation),
                    args.prefetch_batches,
                ):
                    images = images.to(acc.device)
                    with autocast(acc.device, args.precision):
                        latent, active = select_tree(model, perceptual, images, args, maps, rng)
                        codes, lod, patch, lengths = extract_codes(model, latent, active, maps)
                    codes, lod, patch, lengths = [
                        x.cpu().numpy() for x in (codes, lod, patch, lengths)
                    ]
                    for b, key in enumerate(keys):
                        if key in seen:
                            raise ValueError(f"Duplicate sample key in shard: {key}")
                        seen.add(key)
                        length = int(lengths[b])
                        write_token_sample(
                            writer,
                            key,
                            codes[b, :length],
                            lod[b, :length],
                            patch[b, :length],
                            labels[b],
                        )
                        count += 1
                        tokens += length
            if not count:
                raise ValueError(f"Empty input shard: {source}")
            os.replace(temporary, destination)
            manifest_tmp.write_text(
                json.dumps(
                    {
                        "signature": signature,
                        "num_samples": count,
                        "mean_tokens": tokens / count,
                        "output_sha256": sha256(destination),
                    },
                    indent=2,
                )
                + "\n"
            )
            os.replace(manifest_tmp, manifest)
            print(
                f"[rank {acc.process_index}] {destination.name}: {count} samples, {tokens / count:.2f} tokens/image",
                flush=True,
            )
        finally:
            temporary.unlink(missing_ok=True)
            manifest_tmp.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
