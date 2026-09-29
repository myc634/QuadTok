"""Local image folders and streaming WebDataset shards; failures are explicit."""

import glob
import io
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision.transforms.functional import to_tensor
from torchvision import transforms

from .augmentation import center_crop_arr

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


def image_tensor(image, training=False):
    image = image.convert("RGB")
    if training:
        return _TRAIN_TRANSFORM(image)
    return to_tensor(center_crop_arr(image, 256))


class FolderImages(Dataset):
    def __init__(self, root, training=False):
        self.root = Path(root).resolve()
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )
        if not self.paths:
            raise ValueError(f"No images found under {root}")
        classes = sorted(
            {
                p.relative_to(self.root).parts[0]
                for p in self.paths
                if len(p.relative_to(self.root).parts) > 1
            }
        )
        self.classes = {name: i for i, name in enumerate(classes)}
        self.training = training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as im:
            image = image_tensor(im, self.training)
        rel = path.relative_to(self.root)
        return image, self.classes.get(rel.parts[0], 0), f"{index:09d}"


def expand_shards(pattern):
    from braceexpand import braceexpand

    paths = sorted(
        {str(Path(p).resolve()) for expr in braceexpand(pattern) for p in glob.glob(expr)}
    )
    if not paths or any(not Path(p).is_file() or not p.endswith(".tar") for p in paths):
        raise ValueError(f"Expected local .tar shards matching {pattern!r}")
    return paths


def tar_samples(path, training=False):
    # WebDataset standard: members of each sample must be consecutive.
    def decode(key, sample):
        image_keys = [name for name in sample if "." + name.lower() in IMAGE_EXTENSIONS]
        if len(image_keys) != 1:
            raise ValueError(f"{path}: sample {key} needs exactly one supported image")
        with Image.open(io.BytesIO(sample[image_keys[0]])) as image:
            tensor = image_tensor(image, training)
        label = int(sample["cls"]) if "cls" in sample else 0
        return tensor, label, key

    key, sample = None, {}
    seen_keys = set()
    with tarfile.open(path, "r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            base, dot, ext = member.name.rpartition(".")
            if not dot:
                continue
            if key is not None and key != base:
                yield decode(key, sample)
                seen_keys.add(key)
                sample = {}
            if base in seen_keys:
                raise ValueError(f"{path}: nonconsecutive duplicate sample key {base}")
            key = base
            if ext in sample:
                raise ValueError(f"{path}: duplicate member {member.name}")
            sample[ext] = archive.extractfile(member).read()
    if sample:
        yield decode(key, sample)


class TrainingShards(IterableDataset):
    """Shard-partitioned, repeating training stream; no hidden dataset service."""

    def __init__(self, paths, rank, world_size, seed=42, shuffle_buffer=1000):
        self.paths, self.rank, self.world_size = paths, rank, world_size
        self.seed, self.shuffle_buffer = seed, shuffle_buffer

    def __iter__(self):
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        partitions = self.world_size * workers
        identity = self.rank * workers + wid
        paths = self.paths[identity::partitions]
        if not paths:
            raise ValueError("Use at least world_size * num_workers shards, or reduce workers.")
        rng = random.Random(self.seed + identity)
        while True:
            rng.shuffle(paths)
            buffer = []
            count = 0
            for path in paths:
                for sample in tar_samples(path, training=True):
                    count += 1
                    if len(buffer) >= self.shuffle_buffer:
                        slot = rng.randrange(len(buffer))
                        yield buffer[slot]
                        buffer[slot] = sample
                    else:
                        buffer.append(sample)
            if not count:
                raise ValueError("Assigned training shards contain no images.")
            rng.shuffle(buffer)
            yield from buffer


def batches(samples, batch_size, augmentation="center"):
    images, labels, keys = [], [], []
    for image, label, key in samples:
        views = [image] if augmentation == "center" else [image, image.flip(-1)]
        for view_id, view in enumerate(views):
            images.append(view)
            labels.append(label)
            keys.append(f"{key}_{view_id}" if len(views) > 1 else key)
            if len(images) == batch_size:
                yield torch.stack(images), labels, keys
                images, labels, keys = [], [], []
    if images:
        yield torch.stack(images), labels, keys


def write_token_sample(archive, key, codes, lod, patch, label):
    # No extraction is performed by readers; reject ambiguous output member names.
    if not key or key.startswith("/") or ".." in Path(key).parts:
        raise ValueError(f"Unsafe sample key: {key}")
    arrays = {
        "code_indices.npy": np.asarray(codes, dtype=np.int32),
        "lod_indices.npy": np.asarray(lod, dtype=np.int16),
        "patch_indices.npy": np.asarray(patch, dtype=np.int16),
    }
    for suffix, value in arrays.items():
        buf = io.BytesIO()
        np.save(buf, value, allow_pickle=False)
        payload = buf.getvalue()
        info = tarfile.TarInfo(f"{key}.{suffix}")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    payload = str(int(label)).encode()
    info = tarfile.TarInfo(f"{key}.cls")
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def prefetch(iterable, depth=4):
    """Overlap CPU image decoding with GPU work, propagating worker exceptions."""
    import queue
    import threading

    pending = queue.Queue(maxsize=depth)
    stopped = threading.Event()
    sentinel = object()

    def put(item):
        while not stopped.is_set():
            try:
                pending.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def worker():
        try:
            for sample in iterable:
                if stopped.is_set():
                    break
                put((True, sample))
        except BaseException as error:
            put((False, error))
        finally:
            put((True, sentinel))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            ok, item = pending.get()
            if not ok:
                raise item
            if item is sentinel:
                break
            yield item
    finally:
        stopped.set()
        thread.join(timeout=1)
