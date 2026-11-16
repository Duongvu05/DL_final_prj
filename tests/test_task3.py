"""Fast CPU tests for Task 3. They use fake digit pools, so no MNIST download is needed."""

import numpy as np
import pytest
import torch

from dl_final_prj.data.image_sequences import (
    BOS,
    EOS,
    PAD,
    VOCAB_SIZE,
    DigitPool,
    MultiDigitDataset,
    MultiDigitSpec,
    decode,
    encode,
)
from dl_final_prj.models.cnn_attention_decoder import AttentionRecognizer
from dl_final_prj.models.ctc_recognizer import CTCRecognizer, ctc_collapse

SPEC = MultiDigitSpec(seed=1)


def fake_pool(start: int, n: int) -> DigitPool:
    rng = np.random.default_rng(start)
    return DigitPool(
        images=rng.integers(0, 256, (n, 28, 28), dtype=np.uint8),
        labels=rng.integers(0, 10, n),
        src_index=np.arange(start, start + n),
    )


def test_dataset_shapes() -> None:
    ds = MultiDigitDataset(fake_pool(0, 50), SPEC, size=8, split="train")
    img, tokens, n = ds[0]
    assert img.shape == (1, SPEC.height, SPEC.width)
    assert img.dtype == torch.float32 and img.min() >= 0.0 and img.max() <= 1.0
    assert tokens.shape == (SPEC.max_target_len,)
    assert tokens[0] == BOS and SPEC.min_digits <= n <= SPEC.max_digits
    assert tokens[n + 1] == EOS
    assert (tokens[n + 2 :] == PAD).all()


def test_dataset_is_fixed_by_seed() -> None:
    a = MultiDigitDataset(fake_pool(0, 50), SPEC, size=4, split="val")
    b = MultiDigitDataset(fake_pool(0, 50), SPEC, size=4, split="val")
    assert torch.equal(a[3][0], b[3][0])


def test_no_digit_shared_across_splits() -> None:
    pools = {"train": fake_pool(0, 40), "val": fake_pool(40, 10), "test": fake_pool(50, 20)}
    used: dict[str, set[int]] = {}
    for split, pool in pools.items():
        ds = MultiDigitDataset(pool, SPEC, size=30, split=split)
        used[split] = {int(i) for k in range(30) for i in ds.sample(k)[2]}
        assert used[split] <= set(pool.src_index.tolist())
    assert not used["train"] & used["val"]
    assert not used["train"] & used["test"]
    assert not used["val"] & used["test"]


def test_encode_decode_round_trip() -> None:
    for text in ["0", "123", "9087", "000000"]:
        ids = encode(text, max_len=8)
        assert len(ids) == 8 and ids[0] == BOS
        assert decode(ids) == text
    assert decode([BOS, 5, 6, EOS, 7, 8]) == "23"  # stops at EOS
    assert decode([PAD, PAD]) == ""
    with pytest.raises(ValueError):
        encode("1234567", max_len=8)


def test_decoder_output_shape_and_greedy() -> None:
    model = AttentionRecognizer(hidden=32, channels=(8, 16, 16), emb_dim=8, attn_dim=16).eval()
    images = torch.rand(2, 1, SPEC.height, SPEC.width)
    tokens = torch.tensor([encode("123", 8), encode("45678", 8)])
    logits = model(images, tokens[:, :-1])
    assert logits.shape == (2, 7, VOCAB_SIZE)
    assert model.loss(images, tokens).ndim == 0
    ids, alphas = model.greedy_decode(images, max_len=8)
    assert ids.shape == (2, 8)
    assert alphas.shape == (2, 8, SPEC.width // 2)


def test_attention_weights_sum_to_one() -> None:
    model = AttentionRecognizer(hidden=32, channels=(8, 16, 16), emb_dim=8, attn_dim=16).eval()
    _, alphas = model.greedy_decode(torch.rand(3, 1, SPEC.height, SPEC.width), max_len=5)
    assert torch.allclose(alphas.sum(dim=-1), torch.ones(3, 5), atol=1e-5)
    assert (alphas >= 0).all()


def test_no_attention_uses_uniform_weights() -> None:
    model = AttentionRecognizer(
        hidden=32, channels=(8, 16, 16), emb_dim=8, attn_dim=16, use_attention=False
    ).eval()
    _, alphas = model.greedy_decode(torch.rand(1, 1, SPEC.height, SPEC.width), max_len=3)
    assert torch.allclose(alphas, torch.full_like(alphas, 1.0 / alphas.shape[-1]))


def test_ctc_collapse_repeats_and_blanks() -> None:
    a, b = 3, 4  # digit ids
    assert ctc_collapse([a, a, 0, a, b, b, 0, 0, b]) == [a, a, b, b]
    assert ctc_collapse([0, 0, 0]) == []
    assert ctc_collapse([a, a, a]) == [a]


def test_ctc_model_loss_and_decode() -> None:
    model = CTCRecognizer(hidden=32, channels=(8, 16, 16)).eval()
    images = torch.rand(2, 1, SPEC.height, SPEC.width)
    tokens = torch.tensor([encode("123", 8), encode("45678", 8)])
    assert model(images).shape == (SPEC.width // 2, 2, VOCAB_SIZE)
    loss = model.loss(images, tokens, torch.tensor([3, 5]))
    assert loss.ndim == 0 and torch.isfinite(loss)
    preds = model.greedy_decode(images)
    assert len(preds) == 2 and all(p.isdigit() or p == "" for p in preds)
