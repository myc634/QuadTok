"""Streaming reconstruction FID/IS using the source Inception preprocessing.

Adapted from the Apache-2.0 TiTok evaluator, Copyright (2024) Bytedance Ltd.
and/or its affiliates. Vectorized sufficient statistics replace per-image outer products.
"""

import numpy as np
import torch
from scipy import linalg


class ReconstructionMetrics:
    def __init__(self, device):
        from .inception import get_inception_model

        self.network = get_inception_model().to(device).eval()
        self.count = torch.zeros((), device=device, dtype=torch.float64)
        self.real_sum = torch.zeros(2048, device=device, dtype=torch.float64)
        self.fake_sum = torch.zeros_like(self.real_sum)
        self.real_outer = torch.zeros(2048, 2048, device=device, dtype=torch.float64)
        self.fake_outer = torch.zeros_like(self.real_outer)
        self.prob_sum = torch.zeros(1008, device=device, dtype=torch.float64)
        self.prob_log_sum = torch.zeros_like(self.prob_sum)

    @torch.no_grad()
    def update(self, real, fake):
        with torch.autocast(real.device.type, enabled=False):
            rf = self.network((real.clamp(0, 1) * 255).to(torch.uint8))["2048"].double()
            out = self.network((fake.clamp(0, 1) * 255).to(torch.uint8))
            ff = out["2048"].double()
            prob = out["logits_unbiased"].double().softmax(-1)
            self.count += len(real)
            self.real_sum += rf.sum(0)
            self.fake_sum += ff.sum(0)
            self.real_outer += rf.T @ rf
            self.fake_outer += ff.T @ ff
            self.prob_sum += prob.sum(0)
            self.prob_log_sum += (prob * (prob + 1e-16).log()).sum(0)

    def synchronize(self, accelerator):
        for name in (
            "count",
            "real_sum",
            "fake_sum",
            "real_outer",
            "fake_outer",
            "prob_sum",
            "prob_log_sum",
        ):
            setattr(self, name, accelerator.reduce(getattr(self, name), reduction="sum"))

    def result(self):
        n = int(self.count)
        if n < 2:
            raise ValueError(
                "rFID needs at least two images; use --skip-fid for a one-image check."
            )
        real_mean, fake_mean = self.real_sum / n, self.fake_sum / n
        real_cov = (self.real_outer - self.real_sum[:, None] * self.real_sum[None, :] / n) / (n - 1)
        fake_cov = (self.fake_outer - self.fake_sum[:, None] * self.fake_sum[None, :] / n) / (n - 1)
        a, b = real_cov.cpu().numpy(), fake_cov.cpu().numpy()
        product_root = linalg.sqrtm(a @ b)
        if not np.isfinite(product_root).all():
            eye = np.eye(a.shape[0]) * 1e-6
            product_root = linalg.sqrtm((a + eye) @ (b + eye))
        if np.iscomplexobj(product_root):
            if not np.allclose(np.diag(product_root).imag, 0, atol=1e-3):
                raise ValueError("Unstable covariance product; evaluate a larger sample.")
            product_root = product_root.real
        fid = (
            float((real_mean - fake_mean).square().sum().cpu())
            + np.trace(a)
            + np.trace(b)
            - 2 * np.trace(product_root)
        )
        p = self.prob_sum / n
        score = ((self.prob_log_sum / n - p * (p + 1e-16).log()).sum()).exp().item()
        return {"rFID": float(fid), "inception_score": score}
