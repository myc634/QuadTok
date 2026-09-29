import io
import tarfile

import numpy as np
import pytest
import torch

from quadtok.attention import (
    ResidualAttentionBlock,
    build_batched_selector_block_mask,
    parent_indices_vec,
)
from quadtok.topology import active_to_padded, build_slot_maps, random_active, slot_rank
from quadtok.quantizer import VectorQuantizer
from quadtok.data import write_token_sample, tar_samples


def test_bfs_order_and_parent_groups():
    # Independent recursive reference uses original TL/TR/BL/BR traversal.
    levels = {3: [], 4: []}

    def visit(level, row, col):
        if level >= 3:
            levels[level].append(row * (2**level) + col)
        if level < 4:
            for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                visit(level + 1, 2 * row + dr, 2 * col + dc)

    visit(0, 0, 0)
    expected = levels[3] + [64 + p for p in levels[4]]
    assert torch.argsort(slot_rank("cpu")).tolist() == expected
    maps = build_slot_maps("cpu")
    torch.manual_seed(5)
    active = random_active(3, "cpu", 0.3)
    lod, patch, lengths = active_to_padded(active, *maps[:2])
    for b in range(3):
        slots = [s for s in expected if active[b, s]]
        n = lengths[b]
        assert patch[b, :n].tolist() == [s if s < 64 else s - 64 for s in slots]
        assert torch.all(lod[b, n:] == -1)
        for p in range(64):
            assert active[b, 64:][maps[2] == p].unique().numel() == 1


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
def test_attention_reference_and_backward(backend):
    if backend == "flex" and not torch.cuda.is_available():
        pytest.skip("CUDA required for compiled FlexAttention")
    device = "cuda" if backend == "flex" else "cpu"
    torch.manual_seed(7)
    layer = ResidualAttentionBlock(128, 4).to(device)
    lod = torch.tensor([[3, 3, 4, 4, 4], [3, 3, 4, -1, -1]], device=device)
    patch = torch.tensor([[0, 1, 0, 1, 4], [0, 1, 0, 0, 0]], device=device)
    lengths = torch.tensor([5, 3], device=device)
    parent = parent_indices_vec(lod, patch, [1, 2, 4, 8, 16], 3)
    mask = build_batched_selector_block_mask(2, lod, parent, lengths, 3, device, backend)
    # Independent dense mask, including the deliberately discarded padded query rows.
    expected = torch.zeros(2, 7, 7, dtype=torch.bool, device=device)
    for b in range(2):
        for q in range(7):
            for k in range(7):
                if q < 2:
                    allow = k < 2
                elif k < 2:
                    allow = True
                elif k - 2 >= lengths[b]:
                    allow = False
                else:
                    ql, kl = int(lod[b, q - 2]), int(lod[b, k - 2])
                    allow = kl < ql or (
                        kl == ql and (ql == 3 or parent[b, q - 2] == parent[b, k - 2])
                    )
                if q >= lengths[b] + 2 and k == 0:
                    allow = True
                expected[b, q, k] = allow
    x = torch.randn(7, 2, 128, device=device, requires_grad=True)
    actual = layer(x, mask)
    h = layer.ln_1(x)
    dense = ~expected.repeat_interleave(4, 0)
    ref = x + layer.attn(h, h, h, attn_mask=dense, need_weights=False)[0]
    ref = ref + layer.mlp(layer.ln_2(ref))
    torch.testing.assert_close(actual, ref, atol=3e-5, rtol=3e-5)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.attn.in_proj_weight.grad is not None


def test_quantizer_roundtrip_and_gradients():
    torch.manual_seed(9)
    q = VectorQuantizer(32, 8, use_l2_norm=True)
    z = torch.randn(2, 8, 1, 9, requires_grad=True)
    quantized, result = q(z)
    embeddings = q.get_codebook_entry(result["min_encoding_indices"].flatten()).reshape(2, 1, 9, 8)
    torch.testing.assert_close(quantized.permute(0, 2, 3, 1), embeddings, atol=1e-6, rtol=1e-6)
    (quantized.square().mean() + result["quantizer_loss"]).backward()
    assert torch.isfinite(z.grad).all()
    assert q.embedding.weight.grad.abs().sum() > 0


def test_token_archive_format(tmp_path):
    path = tmp_path / "codes.tar"
    with tarfile.open(path, "w") as archive:
        write_token_sample(archive, "sample", [1, 7], [3, 4], [0, 1], 42)
    with tarfile.open(path) as archive:
        assert archive.extractfile("sample.cls").read() == b"42"
        for suffix, expected, dtype in [
            ("code_indices", [1, 7], np.int32),
            ("lod_indices", [3, 4], np.int16),
            ("patch_indices", [0, 1], np.int16),
        ]:
            array = np.load(
                io.BytesIO(archive.extractfile(f"sample.{suffix}.npy").read()), allow_pickle=False
            )
            assert array.tolist() == expected and array.dtype == dtype


def test_corrupt_image_fails_explicitly(tmp_path):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as archive:
        data = b"not an image"
        info = tarfile.TarInfo("sample.jpg")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with pytest.raises(Exception):
        list(tar_samples(path))


def test_prefetch_propagates_error():
    from quadtok.data import prefetch

    def broken():
        yield 1
        raise RuntimeError("decode failed")

    stream = prefetch(broken(), depth=1)
    assert next(stream) == 1
    with pytest.raises(RuntimeError, match="decode failed"):
        next(stream)


def test_ema_source_ramp_and_resume():
    from quadtok.train import TrainingState

    model = torch.nn.Linear(1, 1, bias=False)
    objective = torch.nn.Module()
    objective.register_buffer("ema_real_logits_mean", torch.ones(1))
    state = TrainingState(model, objective, 0.999)
    with torch.no_grad():
        model.weight.fill_(2)
    state.update(model)
    torch.testing.assert_close(state.ema.weight, torch.full_like(model.weight, 2))
    with torch.no_grad():
        model.weight.fill_(4)
    state.update(model)
    torch.testing.assert_close(
        state.ema.weight, torch.full_like(model.weight, 2 * (2 / 11) + 4 * (9 / 11))
    )
    restored = TrainingState(model, objective, 0.5)
    restored.load_state_dict(state.state_dict())
    assert restored.step == 2 and restored.decay == 0.999
    torch.testing.assert_close(restored.ema.weight, state.ema.weight)
