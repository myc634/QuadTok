"""Single-stage tokenizer training with Accelerate, EMA and resumable optimizer state."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

from .data import FolderImages, TrainingShards, expand_shards
from .losses import ReconstructionLoss
from .model import QuadTok


class TrainingState:
    def __init__(self, model, objective, decay):
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)
        self.objective, self.decay, self.step = objective, decay, 0

    @torch.no_grad()
    def update(self, model):
        # Match the source EMA ramp; the first update copies trained weights.
        decay = 0.0 if self.step == 0 else min(self.decay, (1 + self.step) / (10 + self.step))
        for avg, value in zip(self.ema.parameters(), model.parameters()):
            avg.lerp_(value.detach(), 1 - decay)
        for avg, value in zip(self.ema.buffers(), model.buffers()):
            avg.copy_(value)
        self.step += 1

    def state_dict(self):
        return {
            "ema": self.ema.state_dict(),
            "step": self.step,
            "decay": self.decay,
            "loss_buffers": {
                k: v for k, v in self.objective.named_buffers() if k.startswith("ema_")
            },
        }

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["ema"], strict=True)
        self.step, self.decay = state["step"], state["decay"]
        for k, v in state["loss_buffers"].items():
            getattr(self.objective, k).copy_(v)


def parameter_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        (no_decay if value.ndim < 2 or "embedding" in name or "norm" in name else decay).append(
            value
        )
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--data", help="Image folder, optionally class subdirectories")
    data.add_argument("--shards", help="Quoted local WebDataset tar glob or brace range")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume", help="Training checkpoint directory; includes optimizer/EMA/RNG state"
    )
    parser.add_argument(
        "--init-checkpoint", help="Tokenizer weights only, strict architecture match"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--set", nargs="*", default=[], help="OmegaConf dotted overrides (key=value)"
    )
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    t = cfg.training
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if (
        args.workers < 0
        or min(
            t.max_train_steps,
            t.per_gpu_batch_size,
            t.gradient_accumulation_steps,
            t.save_every,
            t.log_every,
        )
        < 1
    ):
        parser.error("Worker count must be nonnegative; training counts must be positive")
    precision = t.mixed_precision if torch.cuda.is_available() else "no"
    acc = Accelerator(
        mixed_precision=precision,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(t.seed, device_specific=True)
    if acc.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = QuadTok(cfg).to(acc.device)
    if args.init_checkpoint:
        model.load_state_dict(
            torch.load(args.init_checkpoint, map_location="cpu", weights_only=True), strict=True
        )
    objective = ReconstructionLoss(cfg).to(acc.device)
    disc = objective.discriminator
    kwargs = dict(betas=(t.beta1, t.beta2))
    optimizer = torch.optim.AdamW(
        parameter_groups(model, t.weight_decay), lr=t.learning_rate, **kwargs
    )
    d_optimizer = torch.optim.AdamW(
        parameter_groups(disc, t.weight_decay), lr=t.discriminator_learning_rate, **kwargs
    )

    def lr_factor(step, total=t.max_train_steps, base_lr=t.learning_rate):
        if step < t.warmup_steps:
            return step / max(t.warmup_steps, 1)
        progress = min((step - t.warmup_steps) / max(total - t.warmup_steps, 1), 1.0)
        return (
            t.end_lr / base_lr + (1 - t.end_lr / base_lr) * (1 + math.cos(math.pi * progress)) / 2
        )

    # Schedulers advance once per actual optimizer update, independent of world size.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(
        d_optimizer,
        lambda step: lr_factor(
            step,
            max(t.max_train_steps - cfg.losses.discriminator_start, 1),
            t.discriminator_learning_rate,
        ),
    )
    model, disc, optimizer, d_optimizer = acc.prepare(model, disc, optimizer, d_optimizer)
    state = TrainingState(acc.unwrap_model(model), objective, t.ema_decay)
    acc.register_for_checkpointing(state, scheduler, d_scheduler)
    sampler = None
    if args.shards:
        paths = expand_shards(args.shards)
        if len(paths) < acc.num_processes * max(args.workers, 1):
            parser.error("Need at least world_size * max(workers,1) shards")
        dataset = TrainingShards(paths, acc.process_index, acc.num_processes, t.seed)
    else:
        dataset = FolderImages(args.data, training=True)
        sampler = DistributedSampler(
            dataset, num_replicas=acc.num_processes, rank=acc.process_index, seed=t.seed
        )
        if len(sampler) < t.per_gpu_batch_size:
            parser.error("Dataset is too small for one full batch on each process")
    loader = DataLoader(
        dataset,
        batch_size=t.per_gpu_batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=acc.device.type == "cuda",
        drop_last=True,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if acc.is_main_process:
        OmegaConf.save(cfg, output / "config.yaml")
        print(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "device": str(acc.device),
                    "attention": cfg.model.attention_backend,
                    "precision": precision,
                    "world_size": acc.num_processes,
                }
            ),
            flush=True,
        )
    if args.resume:
        acc.load_state(args.resume)
    model.train()
    epoch = 0

    def save():
        directory = output / f"checkpoint-{state.step:08d}"
        acc.save_state(str(directory))
        if acc.is_main_process:
            torch.save(acc.unwrap_model(model).state_dict(), directory / "pytorch_model.bin")
            torch.save(state.ema.state_dict(), directory / "ema_model.bin")
            OmegaConf.save(cfg, directory / "config.yaml")
        acc.wait_for_everyone()

    while state.step < t.max_train_steps:
        if sampler:
            sampler.set_epoch(epoch)
        for images, _, _ in loader:
            images = images.to(acc.device, non_blocking=True)
            d_active = objective.should_discriminator_be_trained(state.step)
            with acc.accumulate(model, disc):
                reconstructed, quantizer = model(images)
                # G sees frozen discriminator weights; only D's own phase needs DDP reduction.
                objective.discriminator = acc.unwrap_model(disc)
                loss, logs = objective(images, reconstructed, quantizer, state.step)
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), t.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if d_active:
                    disc.requires_grad_(True)
                    objective.discriminator = disc
                    d_loss, _ = objective(
                        images, reconstructed.detach(), quantizer, state.step, mode="discriminator"
                    )
                    acc.backward(d_loss)
                    if acc.sync_gradients:
                        acc.clip_grad_norm_(disc.parameters(), t.max_grad_norm)
                    d_optimizer.step()
                    d_optimizer.zero_grad(set_to_none=True)
                if acc.sync_gradients and not acc.optimizer_step_was_skipped:
                    scheduler.step()
                    if d_active:
                        d_scheduler.step()
                    for name, value in objective.named_buffers():
                        if name.startswith("ema_"):
                            value.copy_(acc.reduce(value, reduction="mean"))
                    state.update(acc.unwrap_model(model))
            if acc.sync_gradients:
                if state.step % t.log_every == 0:
                    value = acc.reduce(loss.detach(), reduction="mean").item()
                    acc.print(
                        f"step={state.step} loss={value:.6f} lr={scheduler.get_last_lr()[0]:.6g}"
                    )
                if state.step % t.save_every == 0 or state.step >= t.max_train_steps:
                    save()
                if state.step >= t.max_train_steps:
                    break
        epoch += 1
    acc.end_training()


if __name__ == "__main__":
    main()
