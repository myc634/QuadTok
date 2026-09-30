# QuadTok

Content-adaptive discrete image tokenization with a quadtree representation.
This repository contains **tokenizer training, reconstruction evaluation, and pretokenization**.

The released model tokenizes 256 × 256 RGB images into an 8 × 8 coarse level and an optional
16 × 16 fine level. Every image retains 64 coarse tokens; each expanded coarse region adds
four fine tokens, giving **64–320 tokens per image**. The VQ codebook has **16,384 entries of
8 dimensions**. Image generation models are outside the scope of this release.

## Installation

Use Python 3.10 or newer and a matching PyTorch/torchvision installation. CUDA is recommended.
The GPU validation environment uses PyTorch 2.7.1 with CUDA 12.8 and NVIDIA A100 80GB.

```bash
python -m venv ~/venvs/quadtok
source ~/venvs/quadtok/bin/activate
python -m pip install --upgrade pip
# Choose a CUDA wheel matching your driver and platform.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
```

Run commands below from the repository root. No cluster tooling, cloud credentials,
W&B account, or private dataset service is required. LPIPS/VGG, ConvNeXt (training), and
Inception (rFID/IS) download their public weights on first use. Set `TORCH_HOME` to a
writable cache directory, or prepopulate it on machines without Internet access.

## Checkpoint

Download the [released tokenizer weights](https://drive.google.com/file/d/1ZTX97n5WEHYfkzs8GZA7jB3RxhKBH-Sp/view).
Place them at `checkpoints/pytorch_model.bin`:

```bash
python -m pip install gdown
mkdir -p checkpoints
gdown 1ZTX97n5WEHYfkzs8GZA7jB3RxhKBH-Sp -O checkpoints/pytorch_model.bin
```

Expected size: **560,230,566 bytes**. SHA256:

```text
3f63776cde2e21cddb91095551758022bc415a97d0f82ab7835af2ee6d041195
```

The model has 140,011,083 parameters. Checkpoint loading is strict; missing or unexpected
keys are errors. Use `configs/tokenizer.yaml` with these weights. This release supports
this two-level, 256px VQ architecture; other resolutions/depths and VAE/policy variants
are not included.

## Data

Images are RGB floats in `[0,1]`. Evaluation uses ADM-style resize and center crop to
256 × 256. Training resizes the shorter edge to 256 with bicubic interpolation, then applies
a random 256px crop and horizontal flip. Supply your own licensed data.

Two input layouts are supported:

- An image directory, optionally with class subdirectories. Class IDs follow sorted
  subdirectory names; flat-directory images get label 0.
- Local WebDataset `.tar` shards with consecutive members for each sample:
  `sample.jpg` (or `.png`, `.jpeg`, `.webp`, `.bmp`) and optional `sample.cls` containing
  an integer class ID. Missing labels become 0. Duplicate keys or corrupt images fail
  explicitly rather than silently reducing the dataset.

Quote shard globs and brace ranges so the application expands them:
`'/datasets/imagenet/train-{000000..000127}.tar'`.

## Training

The configuration retains the single-stage recipe: L2 reconstruction, VQ commitment and
codebook losses, VGG LPIPS + ConvNeXt perceptual loss, and a discriminator with LeCam
regularization starting at step 200,000. Training samples a shared tree per batch with
fine-level expansion probability 0.75. EMA uses the source decay ramp capped at 0.999.
EMA, optimizer, scheduler, step and RNG state are
saved in each checkpoint.

The default launcher uses **8 GPUs, 32 images per GPU and 1 accumulation step**
(global batch **256**, matching the original VQ recipe). Training uses eager SDPA;
`torch.compile` is not enabled.

```bash
# Default 8-GPU training, image directory.
bash scripts/train.sh --data /datasets/imagenet/train --output outputs/tokenizer

# Default 8-GPU training, streaming tar shards.
bash scripts/train.sh \
  --shards '/datasets/imagenet/train-*.tar' \
  --output outputs/tokenizer --workers 4

# Resume with the same batch settings.
bash scripts/train.sh \
  --data /datasets/imagenet/train --output outputs/tokenizer \
  --resume outputs/tokenizer/checkpoint-00025000

# Lower microbatch while preserving global batch 256.
PER_GPU_BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=2 \
  bash scripts/train.sh --data /datasets/imagenet/train --output outputs/tokenizer
```

`GPUS_PER_NODE` overrides the default 8 GPUs; `PYTHON` selects the Python executable.
You can also pass `--per-gpu-batch-size` and `--gradient-accumulation-steps` directly.
The global batch is `per_gpu_batch_size × processes × accumulation` and is logged at
startup. Reducing the GPU count changes the global batch unless you adjust accumulation.
Use `--init-checkpoint checkpoints/pytorch_model.bin` for weight-only initialization.
Other config values can be changed with `--set key=value`.
For streaming training, provide at least `processes × max(workers,1)` input shards.
Resume restores training state but restarts the data stream/epoch; it does not promise
bit-exact sample replay. Keep the same configuration when resuming.

Each `checkpoint-XXXXXXXX/` contains a raw tokenizer `pytorch_model.bin`, an
`ema_model.bin`, `config.yaml`, and Accelerate training state. Both standalone weight
files can be used by evaluation/pretokenization. Training does not delete old checkpoints.

## Reconstruction evaluation

```bash
python -m quadtok.evaluate \
  --checkpoint checkpoints/pytorch_model.bin \
  --data /datasets/imagenet/val \
  --batch-size 16 --output results/imagenet.json

accelerate launch --num_processes 4 --multi_gpu -m quadtok.evaluate \
  --checkpoint checkpoints/pytorch_model.bin \
  --shards '/datasets/imagenet/val-*.tar' \
  --batch-size 16 --output results/imagenet.json
```

Default evaluation performs content-adaptive LPIPS A/B search with `--tau 0.05` and
reports PSNR, spatial VGG LPIPS, mean token count, reconstruction FID, and Inception
Score. FID statistics are accumulated over the actual input/reconstruction pairs and
merged across ranks. No samples are repeated to fill distributed batches. IS uses the
whole evaluated population, not an average over arbitrary splits. Small-subset FID is
not comparable to full-dataset results.

For a quick check without perceptual/Inception downloads:

```bash
python -m quadtok.evaluate \
  --checkpoint checkpoints/pytorch_model.bin --data /path/to/images \
  --limit 2 --batch-size 1 --workers 0 \
  --tree coarse --skip-lpips --skip-fid --precision fp32
```

`--tree coarse` fixes 64 tokens; `--tree full` fixes 320. These are diagnostic modes,
not the content-adaptive operating point. `--limit` is a global folder-input limit.

## Pretokenization

```bash
accelerate launch --num_processes 4 --multi_gpu -m quadtok.pretokenize \
  --checkpoint checkpoints/pytorch_model.bin \
  --shards '/datasets/imagenet/train-*.tar' \
  --output outputs/tokens --batch-size 32 --augmentation center_hflip
```

Each input shard becomes one output shard with the same filename. `center_hflip` emits
two samples per source image; use `--augmentation center` for one. Shards are assigned
across ranks without duplication. Finished output is renamed atomically and accompanied
by a JSON manifest recording sample count, token count, input/checkpoint/config hashes,
search settings, runtime and output checksum. Re-running verifies and skips matching
completed shards. A settings/checksum mismatch fails; choose a new output directory.
After a hard process crash, remove only that run's stale `.lock` file before resuming.

Each sample contains:

| Member | Shape / dtype | Meaning |
| --- | --- | --- |
| `code_indices.npy` | `[L]`, int32 | VQ code indices |
| `lod_indices.npy` | `[L]`, int16 | Coarse level 3 or fine level 4 |
| `patch_indices.npy` | `[L]`, int16 | Raster patch index within that level |
| `cls` | integer text | Class ID |

All three arrays share **breadth-first tree order**, with TL/TR/BL/BR child order and
coarse tokens before fine tokens. This is not flat raster order. All active parents
remain in the token sequence. Padding is excluded from files. Load NumPy arrays with
`allow_pickle=False`.

The A/B split is random and reproducible for a fixed seed, batch composition and shard
list. Changing batch size can change the selected trees. Each shard has an independent
seed so changing process count does not change its search sequence.

## Runtime

Only PyTorch SDPA is supported; `--attention auto` and `--attention sdpa` are aliases.
The selector/decoder retain the kinship mask. This release does not use packed varlen
FlashAttention or `torch.compile`. Inference defaults to bf16 on CUDA and fp32 on CPU;
use `--precision fp32` to disable bf16 inference. Run `python -m pytest -q` for core tests.

## License and acknowledgments

Code is released under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for the retained
TiTok, One-D-Piece, OpenCLIP, ADM and Inception attribution. Dataset and downloaded
third-party weight licenses remain with their respective providers.
