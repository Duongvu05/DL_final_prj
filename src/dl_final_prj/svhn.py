"""SVHN Format 1 full numbers for Task 3 (plan Sections 1.2 and 4.1).

prepare() parses digitStruct.mat, makes the fixed hash-based 90/10 train/validation split
stratified by sequence length, and letterboxes every image once into a uint8 cache with the
Rust extension. Training then reads only the cache, so no image is decoded during training.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from dl_final_prj.fast import letterbox_images, mean_rgb

PAD, SOS, EOS = 0, 1, 2
DIGIT0 = 3
VOCAB = DIGIT0 + 10
URL = "https://ufldl.stanford.edu/housenumbers/"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_digit_struct(mat_path: Path) -> list[dict]:
    """[{name, label (str, left to right), boxes [[left, top, width, height], ...]}] in file order."""
    import h5py

    out = []
    with h5py.File(mat_path, "r") as handle:
        f: Any = handle  # h5py objects are untyped
        bbox, names = f["digitStruct/bbox"], f["digitStruct/name"]

        def field(group, key: str) -> list[float]:
            ds = group[key]
            if ds.shape[0] == 1:
                return [float(ds[()].item())]
            return [float(f[ds[j][0]][()].item()) for j in range(ds.shape[0])]

        for i in range(len(names)):
            name = "".join(chr(int(c[0])) for c in f[names[i][0]][()])
            g = f[bbox[i][0]]
            lab = field(g, "label")
            left, top = field(g, "left"), field(g, "top")
            width, height = field(g, "width"), field(g, "height")
            order = np.argsort(left, kind="stable")
            digits = "".join(str(int(lab[k]) % 10) for k in order)  # MATLAB label 10 is digit 0
            boxes = [[left[k], top[k], width[k], height[k]] for k in order]
            out.append({"name": name, "label": digits, "boxes": boxes})
    return out


def hash_split(
    records: list[dict], val_frac: float = 0.1, salt: str = "group12"
) -> tuple[list[int], list[int]]:
    """Deterministic split stratified by length. Within each length, the val_frac of images
    with the smallest sha1(salt + name) go to validation."""
    by_len: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        by_len.setdefault(len(r["label"]), []).append(i)
    train, val = [], []
    for idx in by_len.values():
        idx = sorted(
            idx, key=lambda i: hashlib.sha1((salt + records[i]["name"]).encode()).hexdigest()
        )
        k = int(len(idx) * val_frac)
        val += idx[:k]
        train += idx[k:]
    return sorted(train), sorted(val)


def length_hist(labels: list[str]) -> dict[str, int]:
    h: dict[str, int] = {}
    for s in labels:
        h[str(len(s))] = h.get(str(len(s)), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


def prepare(raw_dir: Path, out_dir: Path, height: int = 64, width: int = 256) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_recs = parse_digit_struct(raw_dir / "train" / "digitStruct.mat")
    test_recs = parse_digit_struct(raw_dir / "test" / "digitStruct.mat")
    tr_idx, va_idx = hash_split(train_recs)
    splits = {
        "train": [train_recs[i] | {"dir": "train"} for i in tr_idx],
        "val": [train_recs[i] | {"dir": "train"} for i in va_idx],
        "test": [r | {"dir": "test"} for r in test_recs],
    }
    path = lambda r: str(raw_dir / r["dir"] / r["name"])  # noqa: E731
    pad = np.rint(mean_rgb([path(r) for r in splits["train"]])).astype(int)
    summary = {}
    for name, recs in splits.items():
        imgs, geom = letterbox_images([path(r) for r in recs], height, width, pad.tolist())
        np.save(out_dir / f"{name}_images.npy", imgs)
        np.save(out_dir / f"{name}_geometry.npy", geom)
        (out_dir / f"{name}_labels.json").write_text(
            json.dumps(
                [
                    {"name": r["name"], "dir": r["dir"], "label": r["label"], "boxes": r["boxes"]}
                    for r in recs
                ]
            )
        )
        summary[name] = {
            "n_images": len(recs),
            "length_hist": length_hist([r["label"] for r in recs]),
            "source_width_pct_5_50_95": np.percentile(geom[:, 0], [5, 50, 95]).tolist()
            if len(recs)
            else None,
            "source_height_pct_5_50_95": np.percentile(geom[:, 1], [5, 50, 95]).tolist()
            if len(recs)
            else None,
        }
    audit = {
        "dataset": "SVHN Format 1 (full numbers)",
        "url": URL,
        "raw_files": ["train.tar.gz", "test.tar.gz"],
        "image_format": "PNG, RGB, variable size",
        "content": "house-number digits cropped from Google Street View photos",
        "label_rule": "digits of each image ordered by increasing bounding-box left, MATLAB label 10 -> 0",
        "canvas": [height, width, 3],
        "resize_rule": "aspect ratio kept, bilinear (triangle) resize to fit, centered, padded with train mean RGB",
        "pad_rgb": pad.tolist(),
        "split_rule": "official test untouched, official train split 90/10 by sha1(salt+name) within each length",
        "vocabulary": {"PAD": PAD, "SOS": SOS, "EOS": EOS, "digits": "token = 3 + digit"},
        "max_len": max(len(r["label"]) for r in train_recs),
        "splits": summary,
    }
    (out_dir / "svhn_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def encode(label: str, max_len: int) -> list[int]:
    """Decoder targets: d1 .. dn EOS PAD .., length max_len + 1."""
    ids = [DIGIT0 + int(c) for c in label] + [EOS]
    return ids + [PAD] * (max_len + 1 - len(ids))


def decode(ids: list[int]) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        if i >= DIGIT0:
            out.append(str(i - DIGIT0))
    return "".join(out)


@dataclass
class OcrBatch:
    images: torch.Tensor  # (B, 3, H, W) normalized float
    tokens: torch.Tensor  # (B, max_len + 1) decoder targets
    index: torch.Tensor  # (B,) row in the split


class SvhnSplit:
    """One cached split held on the device as uint8, with GPU-side augmentation."""

    def __init__(self, cache_dir: Path, split: str, max_len: int, device: str = "cpu") -> None:
        imgs = np.load(cache_dir / f"{split}_images.npy", mmap_mode="r")
        self.records = json.loads((cache_dir / f"{split}_labels.json").read_text())
        self.labels = [r["label"] for r in self.records]
        self.images = torch.from_numpy(np.array(imgs)).to(device).permute(0, 3, 1, 2)
        self.tokens = torch.tensor([encode(s, max_len) for s in self.labels], device=device)
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def batch(
        self, idx: torch.Tensor, augment: dict | None = None, g: torch.Generator | None = None
    ) -> OcrBatch:
        idx = idx.to(self.device)
        x = self.images[idx].float() / 255.0
        if augment:
            x = random_affine_color(x, augment, g)
        x = (x - self.mean) / self.std
        return OcrBatch(x, self.tokens[idx], idx)

    def iterate(
        self, batch_size: int, augment: dict | None = None, g: torch.Generator | None = None
    ):
        order = torch.randperm(len(self), generator=g) if g is not None else torch.arange(len(self))
        for i in range(0, len(self), batch_size):
            yield self.batch(order[i : i + batch_size], augment, g)


def random_affine_color(x: torch.Tensor, aug: dict, g: torch.Generator | None) -> torch.Tensor:
    """Label-preserving augmentation on the device. Shift, scale, small rotation (no flips),
    then per-image brightness and contrast. Random numbers come from the CPU generator `g`
    so a resumed epoch sees the same augmentation."""
    b = x.shape[0]
    r = torch.rand(b, 6, generator=g).to(x.device)
    ang = (r[:, 0] * 2 - 1) * np.deg2rad(aug["max_rotation"])
    lo, hi = aug["scale"]
    s = lo + r[:, 1] * (hi - lo)
    tx = (r[:, 2] * 2 - 1) * aug["max_shift"] * 2
    ty = (r[:, 3] * 2 - 1) * aug["max_shift"] * 2
    h, w = x.shape[-2:]
    aspect = w / h
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack(
        [
            torch.stack([cos, -sin / aspect, tx], -1),
            torch.stack([sin * aspect, cos, ty], -1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    bright = 1 + (r[:, 4] * 2 - 1) * aug["brightness"]
    contrast = 1 + (r[:, 5] * 2 - 1) * aug["contrast"]
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast.view(-1, 1, 1, 1) + mean
    return (x * bright.view(-1, 1, 1, 1)).clamp(0, 1)
