# Training performance

This release uses SDPA only. It does **not** call `flash_attn_varlen_func`, use packed
`cu_seqlens`, or remove padding through a varlen kernel. Training samples one shared
tree per batch, so samples already have equal token lengths within that batch.
Inference can have different trees per image and still uses padded dense tensors.

On the measured PyTorch 2.7.1 / CUDA 12.8 / A100 configuration, the training profiler
recorded 8 encoder calls to `_scaled_dot_product_flash_attention` and 32 masked
selector/decoder calls to `_scaled_dot_product_efficient_attention`, with corresponding
backward kernels. SDPA is a dispatcher; it is not synonymous with the FlashAttention
varlen API. A plain causal or sliding-window mask cannot replace the kinship relation.
Packed varlen support would require a topology-aware reformulation and gradient checks.

## Measurement

The final measurements below use Accelerate bf16 wrappers for both the tokenizer and
discriminator, matching the training CLI. The checkpoint, losses, AdamW parameter groups,
gradient clipping and EMA follow the released configuration. Batch size is 8, with 3
warmups and 5 timed steps on 8 real COCO images resident on one A100 80GB. The two phases
exercise the configured loss before and after discriminator activation.

The diagnostic includes forward, backward, optimizer and EMA work. It excludes dataset
loading/augmentation, checkpoint IO, scheduler bookkeeping and multi-GPU communication.
CUDA-event stage times include host submission gaps; independently computed medians
need not sum exactly to the median whole step. These are short diagnostic measurements,
not sustained distributed training throughput or convergence evidence.

| Phase | Median step | Images/s | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| before discriminator | 183.87 ms | 43.51 | 6814 |
| with discriminator | 227.41 ms | 35.18 | 7661 |

See the [raw training profile](training_sdpa_profile.json) for stage medians and actual kernel records.

## Optimization priorities

1. **Fused AdamW and batched EMA updates.** The current optimizers do not explicitly
   enable `fused=True`, and EMA loops over parameter tensors. Compare fused AdamW and
   foreach EMA at identical batch size and precision; validate updates and resume.
2. **Remove host synchronization and dynamic indexing overhead.** `active_to_padded`
   converts a CUDA maximum to a Python integer. Selector/decoder construction also uses
   `if mask.any()` and dynamic boolean indexing. A shared-tree training path can compute
   topology once and reuse it across samples, then compile stable transformer blocks.
   Preserve tree distribution and the exact kinship mask.
3. **Convolution layout and loss execution.** Decoder, VGG/ConvNeXt perceptual losses
   and discriminator have substantial convolution work. Benchmark `channels_last` and
   the input pipeline before changing numerical precision. Perceptual loss precision
   changes require reconstruction/gradient checks and training-quality evaluation.
4. **Batch size and communication.** Sweep per-GPU batch size while keeping effective
   global batch constant. For DDP, profile communication and unused-parameter discovery;
   do not disable discovery until every supported topology has been checked for unused
   branches. The single-GPU diagnostic provides no DDP speedup evidence.

These are optimization candidates, not implemented or measured speedup claims.
The prior Flex/SDPA report measured reconstruction and guided pretokenization only;
it does not establish a ranking for training backward performance. Short sequences,
block-mask construction and partially occupied sparse blocks are plausible contributors
to its observed Flex overhead, but that report did not isolate their individual costs.
