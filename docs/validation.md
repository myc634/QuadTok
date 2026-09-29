# Release validation

Measured on 2026-09-29 with the released two-level 256px checkpoint on an
NVIDIA A100-SXM4-80GB, PyTorch 2.7.1+cu128, CUDA 12.8, Python 3.10.
The implementation was extracted from source revision
`e16c53ba40c5183b02ef09d031def138848eb3d9`.

## Checkpoint and attention correctness

Strict checkpoint loading passed for all 519 state tensors (140,011,083 parameters).
The checkpoint SHA256 is recorded in the README and [raw benchmark](gpu_benchmark.json).
FP32 SDPA/Flex reconstruction comparison passed: maximum absolute error
0.00011873, mean absolute error 0.00000343, output-to-output PSNR 102.92 dB.
Instrumentation recorded 32 compiled Flex calls on CUDA (8 selector layers and
24 decoder layers), with `triton_tem_fused_0` among the captured profiler events.
CUDA tests also compare masked attention against an independent dense reference
and check finite backward gradients. Explicit Flex does not silently fall back.

## Steady-state throughput

First 32 sorted COCO val2017 images, bf16, 3 warmups and 10 synchronized repetitions;
values below use median latency. Reconstruction uses identical supplied random trees
for both backends. Guided pretokenization includes encoding, both candidate decodes,
LPIPS selection and final code extraction. Disk decoding/writing and initial compilation
are excluded. Peak memory is PyTorch allocated memory, not total device occupancy.

| Operation | Batch | SDPA images/s | Flex images/s | SDPA peak MiB | Flex peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| reconstruction | 1 | 20.96 | 14.54 | 631 | 631 |
| reconstruction | 8 | 153.19 | 143.93 | 996 | 996 |
| reconstruction | 32 | 500.85 | 448.11 | 2263 | 2263 |
| guided pretokenize | 1 | 10.71 | 7.88 | 799 | 799 |
| guided pretokenize | 8 | 69.64 | 47.01 | 1911 | 1911 |
| guided pretokenize | 32 | 143.83 | 122.11 | 5728 | 5728 |

SDPA was faster for this measured short two-level workload and is the `auto` default.
These measurements do not establish a universal ranking across hardware or tree depths.
Flex batch-1 latency was variable; all individual timings are retained in the raw report.
BF16 backends are not bit-identical: threshold decisions and VQ assignments can differ
near boundaries. For batch 32, guided mean token counts were 239.875 (SDPA) and
239.000 (Flex); these adaptive workloads are consequently not exactly identical.

## End-to-end checks

The release validation runs the supplied checkpoint on real COCO images. Full-loss
Flex training includes perceptual and adversarial losses, checkpoint creation and
resume. Reconstruction evaluation exercises PSNR, LPIPS, reconstruction FID and
Inception Score on 32 images. Pretokenization checks output values/dtypes, class labels,
checksums and verified resume. These small runs validate execution, not paper-quality
metrics or training convergence.

The final CUDA test suite passed **8/8 tests**, including EMA ramp and state restoration.
The full-loss Flex run completed step 1 and restored all training state to step 2 with
finite losses. The first EMA update exactly matched the trained raw weights.
Two-GPU DDP training passed with gradient accumulation of 2. Distributed evaluation
counted exactly 5 images across uneven rank tails. Two-rank pretokenization produced
4 samples across 2 shards; rerunning with one rank verified and skipped both outputs.
The end-to-end suite exited successfully. CPU-only tests pass with the CUDA test skipped.

The repository also passes Ruff lint/format checks and builds a wheel with all three
CLI entry points. Validation does not cover longer training convergence, other GPU
architectures, or checkpoints outside the released architecture.
