#!/usr/bin/env bash
# Single-node training: 8 GPUs x 32 images x 1 accumulation = global batch 256.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE:-8}" \
  -m quadtok.train \
  --per-gpu-batch-size "${PER_GPU_BATCH_SIZE:-32}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  "$@"
