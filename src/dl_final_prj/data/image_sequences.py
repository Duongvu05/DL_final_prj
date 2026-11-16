"""Synthetic multi-digit MNIST for Task 3 (image to sequence).

Each sample concatenates 3 to 6 MNIST digits into one grayscale strip of fixed height.
Split hygiene. The 60k official MNIST train images are split once into disjoint train and
validation digit pools. The test set draws only from the 10k official MNIST test images.
So no digit image is shared across splits. Every sample is generated from an RNG seeded by
(seed, split, index), so each split is a fixed set that can be rebuilt exactly.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms.functional import rotate

PAD, BOS, EOS = 0, 1, 2
DIGIT_OFFSET = 3
VOCAB_SIZE = DIGIT_OFFSET + 10
DIGIT_SIZE = 28
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


@dataclass
class MultiDigitSpec:
    """Controls how digit strips are rendered."""

    min_digits: int = 3
    max_digits: int = 6
    height: int = 32
    width: int = 192
    min_gap: int = 0
    max_gap: int = 6
    overlap_prob: float = 0.1
    max_overlap: int = 4
    noise_std: float = 0.05
    max_rotation: float = 10.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.min_digits <= self.max_digits:
            raise ValueError("need 1 <= min_digits <= max_digits")
        if self.max_digits * DIGIT_SIZE > self.width:
            raise ValueError("width too small for max_digits without gaps")
        if self.height < DIGIT_SIZE:
            raise ValueError("height must be at least 28")

    @property
    def max_target_len(self) -> int:
        """Longest token sequence, BOS + digits + EOS."""
        return self.max_digits + 2


@dataclass
class DigitPool:
    """MNIST digits available to one split. src_index points into the official MNIST set."""

    images: np.ndarray
    labels: np.ndarray
    src_index: np.ndarray


def load_mnist_pools(
    root: str | Path, seed: int = 0, val_frac: float = 0.1
) -> dict[str, DigitPool]:
    """Disjoint digit pools. Train and val come from MNIST train, test from MNIST test."""
    train_set = MNIST(str(root), train=True, download=True)
    test_set = MNIST(str(root), train=False, download=True)
    perm = np.random.default_rng(seed).permutation(len(train_set))
    n_val = int(len(train_set) * val_frac)
    x_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()
    pools = {
        "train": perm[n_val:],
        "val": perm[:n_val],
    }
    out = {k: DigitPool(x_train[i], y_train[i], i) for k, i in pools.items()}
    x_test, y_test = test_set.data.numpy(), test_set.targets.numpy()
    out["test"] = DigitPool(x_test, y_test, np.arange(len(x_test)))
    return out


def encode(digits: str, max_len: int) -> list[int]:
    """Digit string to BOS d1 ... dn EOS PAD ... of length max_len."""
    ids = [BOS, *(DIGIT_OFFSET + int(c) for c in digits), EOS]
    if len(ids) > max_len:
        raise ValueError("sequence longer than max_len")
    return ids + [PAD] * (max_len - len(ids))


def decode(ids: list[int]) -> str:
    """Token ids to a digit string. Drops PAD and BOS, stops at EOS."""
    chars = []
    for i in ids:
        if i == EOS:
            break
        if i >= DIGIT_OFFSET:
            chars.append(str(i - DIGIT_OFFSET))
    return "".join(chars)


class MultiDigitDataset(Dataset):
    """Fixed set of synthetic digit strips. Item is (image, tokens, n_digits).

    image is float32 (1, height, width) in [0, 1]. tokens is int64 of length max_target_len.
    With augment=True a mild random contrast and shift is applied at load time (train only).
    """

    def __init__(
        self,
        pool: DigitPool,
        spec: MultiDigitSpec,
        size: int,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        self.pool, self.spec, self.size = pool, spec, size
        self.split_code = SPLIT_CODES[split]
        self.augment = augment

    def __len__(self) -> int:
        return self.size

    def sample(self, idx: int) -> tuple[np.ndarray, str, np.ndarray]:
        """Render sample idx. Returns image, digit string and the pool rows that were used."""
        s = self.spec
        rng = np.random.default_rng([s.seed, self.split_code, idx])
        n = int(rng.integers(s.min_digits, s.max_digits + 1))
        rows = rng.integers(0, len(self.pool.labels), size=n)
        gaps = []
        for _ in range(n - 1):
            if rng.random() < s.overlap_prob:
                gaps.append(-int(rng.integers(1, s.max_overlap + 1)))
            else:
                gaps.append(int(rng.integers(s.min_gap, s.max_gap + 1)))
        if n * DIGIT_SIZE + sum(gaps) > s.width:
            gaps = [min(g, 0) for g in gaps]
        total = n * DIGIT_SIZE + sum(gaps)
        x = int(rng.integers(0, s.width - total + 1))
        canvas = np.zeros((s.height, s.width), dtype=np.float32)
        for k, row in enumerate(rows):
            digit = torch.from_numpy(self.pool.images[row].astype(np.float32) / 255.0)[None]
            angle = float(rng.uniform(-s.max_rotation, s.max_rotation))
            digit = rotate(digit, angle)[0].numpy()
            y = int(rng.integers(0, s.height - DIGIT_SIZE + 1))
            region = canvas[y : y + DIGIT_SIZE, x : x + DIGIT_SIZE]
            np.maximum(region, digit, out=region)
            x += DIGIT_SIZE + (gaps[k] if k < n - 1 else 0)
        canvas += rng.normal(0.0, s.noise_std, canvas.shape).astype(np.float32)
        text = "".join(str(int(self.pool.labels[r])) for r in rows)
        return np.clip(canvas, 0.0, 1.0), text, self.pool.src_index[rows]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, text, _ = self.sample(idx)
        img = torch.from_numpy(image)[None]
        if self.augment:
            img = img * float(torch.empty(1).uniform_(0.8, 1.2))
            img = torch.roll(img, int(torch.randint(-3, 4, (1,))), dims=2).clamp(0.0, 1.0)
        tokens = torch.tensor(encode(text, self.spec.max_target_len), dtype=torch.long)
        return img, tokens, torch.tensor(len(text), dtype=torch.long)
