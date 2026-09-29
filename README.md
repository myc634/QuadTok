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

```bash
# Single GPU, image directory.
python -m quadtok.train \
  --data /datasets/imagenet/train \
  --output outputs/tokenizer

# Multiple GPUs, streaming tar shards.
accelerate launch --num_processes 4 --multi_gpu -m quadtok.train \
  --shards '/datasets/imagenet/train-*.tar' \
  --output outputs/tokenizer --workers 4

# Resume a training checkpoint.
python -m quadtok.train \
  --data /datasets/imagenet/train --output outputs/tokenizer \
  --resume outputs/tokenizer/checkpoint-00025000
```

Use `--init-checkpoint checkpoints/pytorch_model.bin` for weight-only initialization.
Use `--set training.per_gpu_batch_size=8 training.gradient_accumulation_steps=4` to
adjust memory use. The global batch is `per_gpu_batch_size × processes × accumulation`.
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

## Attention and performance

The shared implementation batches topology selection and hierarchical decoding as
tensors. `--attention auto` selects **PyTorch SDPA**, which was faster for this short two-level
sequence in the A100 measurements. Use `--attention flex` to explicitly select compiled
**FlexAttention** for kinship-masked selector/decoder attention; ordinary encoder
attention always uses SDPA. CPU supports `auto`/`sdpa`. The first CUDA calls compile kernels;
exclude this startup cost from steady-state throughput measurements.

Inference defaults to bf16 on CUDA and fp32 on CPU. There is no silent Flex-to-SDPA
fallback when FlexAttention compilation fails. Runtime metadata records the selected
backend and precision. `--precision fp32` disables bf16 inference.

```bash
python -m pytest -q
python -m benchmarks.gpu \
  --checkpoint checkpoints/pytorch_model.bin \
  --data /datasets/imagenet/val --guided \
  --batch-sizes 1 8 32 --warmup 3 --repeats 10 \
  --output results/gpu.json
```

The benchmark validates fp32 reconstruction parity, captures Flex/Triton profiler events,
and records synchronized warmed bf16 timing and peak allocated GPU memory. Reconstruction
timing includes the full model with a supplied random tree. Guided pretokenization timing
includes image encoding, both candidate reconstructions, LPIPS selection and final code
extraction; disk decoding/writing is excluded. Without `--data`, it uses seeded synthetic
images and labels that explicitly in the report. See `docs/validation.md` for measured results.

## License and acknowledgments

Code is released under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for the retained
TiTok, One-D-Piece, OpenCLIP, ADM and Inception attribution. Dataset and downloaded
third-party weight licenses remain with their respective providers.
