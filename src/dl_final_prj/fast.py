"""Thin wrappers over the Rust extension `dlfast`, each with a NumPy/Python reference.

The Rust versions are used by default. The `*_ref` functions are the specification. The
tests check that both give the same result.
"""

from collections.abc import Sequence

import numpy as np

try:
    import dlfast as _rs
except ImportError:  # pragma: no cover - only hit when the extension is not built
    _rs = None


def has_rust() -> bool:
    return _rs is not None


# --- semicolon table -------------------------------------------------------


def parse_semicolon_table(path: str) -> tuple[list[str], list[str], np.ndarray]:
    """Column names, timestamps and an (n, k) float64 matrix from a `;` table with decimal commas."""
    if _rs is None:
        return parse_semicolon_table_ref(path)
    names, stamps, values = _rs.parse_semicolon_table(str(path))
    return list(names), list(stamps), values


def parse_semicolon_table_ref(path: str) -> tuple[list[str], list[str], np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path, sep=";", decimal=",", index_col=0)
    return [str(c) for c in df.columns], [str(s) for s in df.index], df.to_numpy(np.float64)


# --- block bootstrap -------------------------------------------------------


def block_bootstrap_means(columns: np.ndarray, block: int, reps: int, seed: int) -> np.ndarray:
    """Moving-block bootstrap of column means. columns (n, k) -> (reps, k).

    Each replicate concatenates ceil(n/block) blocks with uniform starts in [0, n-block]
    and truncates the last block so it has exactly n rows. block=1 is the ordinary
    i.i.d. bootstrap used for per-image comparisons.
    """
    cols = np.ascontiguousarray(columns, dtype=np.float64)
    if cols.ndim == 1:
        cols = cols[:, None]
    if _rs is None:
        n = len(cols)
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, n - block + 1, size=(reps, -(-n // block)))
        return block_means_from_starts_ref(cols, block, starts)
    return _rs.block_bootstrap_means(cols, int(block), int(reps), int(seed))


def block_means_from_starts(columns: np.ndarray, block: int, starts: np.ndarray) -> np.ndarray:
    cols = np.ascontiguousarray(columns, dtype=np.float64)
    st = np.ascontiguousarray(starts, dtype=np.int64)
    if _rs is None:
        return block_means_from_starts_ref(cols, block, st)
    return _rs.block_means_from_starts(cols, int(block), st)


def block_means_from_starts_ref(cols: np.ndarray, block: int, starts: np.ndarray) -> np.ndarray:
    n = len(cols)
    out = np.empty((len(starts), cols.shape[1]))
    for r, row in enumerate(starts):
        idx = np.concatenate([np.arange(s, s + block) for s in row])[:n]
        out[r] = cols[idx].mean(axis=0)
    return out


# --- edit distance ---------------------------------------------------------


def levenshtein_batch(a: Sequence[str], b: Sequence[str]) -> np.ndarray:
    if _rs is None:
        return np.array([levenshtein_ref(x, y) for x, y in zip(a, b, strict=True)])
    return np.asarray(_rs.levenshtein_batch(list(a), list(b)), dtype=np.int64)


def levenshtein_ref(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --- images ----------------------------------------------------------------


def letterbox_images(
    paths: Sequence[str],
    height: int,
    width: int,
    pad: Sequence[int],
    crops: np.ndarray | None = None,
    *,
    stretch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Images (N, H, W, 3) uint8 and geometry (N, 5) [src_w, src_h, scale, x_off, y_off].

    Each image is optionally cropped to [left, top, right, bottom] (clamped to the image,
    see `clamp_box`). By default it is then resized with its aspect ratio kept so it fits
    inside (height, width) and centered on a canvas filled with pad. With stretch=True it is
    resized to exactly (width, height), the offsets are 0 and scale is width / src_w.
    """
    pad3 = [int(v) for v in pad]
    c = None if crops is None else np.ascontiguousarray(crops, dtype=np.float32)
    if _rs is None:
        return letterbox_images_ref(paths, height, width, pad3, c, stretch=stretch)
    return _rs.letterbox_images([str(p) for p in paths], height, width, pad3, c, stretch=stretch)


def clamp_box(box: Sequence[float], w: int, h: int) -> tuple[int, int, int, int]:
    """Integer crop [left, top, right, bottom] inside a w x h image, at least 1 pixel wide.

    Same rule as the Rust code. The float box is floored/ceiled outward, then clamped.
    The float32 cast matches the precision the Rust extension receives.
    """
    left, top, right, bottom = (float(np.float32(v)) for v in box)
    li = int(np.clip(np.floor(left), 0, w - 1))
    ti = int(np.clip(np.floor(top), 0, h - 1))
    ri = int(np.clip(np.ceil(right), li + 1, w))
    bi = int(np.clip(np.ceil(bottom), ti + 1, h))
    return li, ti, ri, bi


def letterbox_images_ref(
    paths: Sequence[str],
    height: int,
    width: int,
    pad: list[int],
    crops: np.ndarray | None,
    *,
    stretch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    imgs = np.empty((len(paths), height, width, 3), np.uint8)
    geom = np.empty((len(paths), 5), np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        if crops is not None:
            im = im.crop(clamp_box(crops[i], *im.size))
        w, h = im.size
        if stretch:
            imgs[i] = np.asarray(im.resize((width, height), Image.Resampling.BILINEAR))
            geom[i] = (w, h, np.float32(width) / np.float32(w), 0, 0)
            continue
        scale = min(height / h, width / w)
        nw = int(np.clip(round(w * scale), 1, width))
        nh = int(np.clip(round(h * scale), 1, height))
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        x0, y0 = (width - nw) // 2, (height - nh) // 2
        imgs[i] = np.array(pad, np.uint8)
        imgs[i, y0 : y0 + nh, x0 : x0 + nw] = np.asarray(im)
        geom[i] = (w, h, scale, x0, y0)
    return imgs, geom


def mean_rgb(paths: Sequence[str]) -> np.ndarray:
    """Pixel-weighted RGB mean over all images, in [0, 255]."""
    if _rs is None:
        return mean_rgb_ref(paths)
    return np.asarray(_rs.mean_rgb([str(p) for p in paths]))


def mean_rgb_ref(paths: Sequence[str]) -> np.ndarray:
    from PIL import Image

    total, count = np.zeros(3), 0
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB"), np.float64)
        total += a.reshape(-1, 3).sum(0)
        count += a.shape[0] * a.shape[1]
    return total / count
