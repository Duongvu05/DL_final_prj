"""SVHN Format 1 full numbers for Task 3, cropped protocol (plan Sections 1.2, 2.3 and 4.1).

prepare() parses digitStruct.mat of train, extra and test, drops train/extra numbers longer
than MAX_LEN, merges train+extra and makes the fixed hash-based 90/10 train/validation split
stratified by length. Every image is cropped once to its expanded ground-truth union box and
resized to 64x64 (aspect ratio ignored) into a uint8 cache with the Rust extension. The whole
test images are also letterboxed into 64x64 as a stress-test cache. Training then reads only
the cache, so no image is decoded during training.

prepare_whole() adds the whole-image setting (He et al. 2019). It reuses the record lists of
the cropped cache (same images, same order, same train/val/test split) and resizes each WHOLE
original image to 64x64 (aspect ratio ignored) without any crop, so no ground-truth digit box
enters its data path. Its files carry the suffix "_full" and are read with
SvhnSplit(cache, split, suffix="_full").

whole_border_audit() is a separate, diagnostic-only step. It reads the ground-truth boxes of the
cropped cache once and measures how often the 54x54 crops of SvhnSplit cut digits off a
stretched whole image. Nothing it computes feeds training or evaluation.
"""

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from dl_final_prj.fast import letterbox_images

PAD, SOS, EOS = 0, 1, 2
DIGIT0 = 3
VOCAB = DIGIT0 + 10
MAX_LEN = 5
URL = "https://ufldl.stanford.edu/housenumbers/"
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
CROP_EXPAND = 1.3
SOURCES = ("train", "extra", "test")
SPLITS = ("train", "val", "test")
WHOLE_SUFFIX = "_full"
# suffixes prepare_whole() must never write: "" is the cropped cache itself and "_whole" would
# clash with the letterbox stress cache test_whole_* of prepare()
_RESERVED_SUFFIXES = ("", "_whole")
_BOX_KEYS = ("label", "left", "top", "width", "height")


# --- digitStruct.mat ----------------------------------------------------------------


def _read_values(f: Any, ds: Any) -> list[float]:
    """One bbox field. Single-digit images store the value, others store object references."""
    import h5py

    arr = ds[()]
    if h5py.check_dtype(ref=ds.dtype) is not None:
        return [float(f[r][()].item()) for r in np.ravel(arr)]
    return [float(v) for v in np.ravel(arr)]


def _parse_range(mat_path: str, start: int, stop: int) -> list[dict]:
    """Records start..stop-1 of one digitStruct.mat. Top-level so worker processes can run it."""
    import h5py

    out = []
    with h5py.File(mat_path, "r") as handle:
        f: Any = handle  # h5py objects are untyped
        name_refs = f["digitStruct/name"][start:stop, 0]  # one read for the whole range
        bbox_refs = f["digitStruct/bbox"][start:stop, 0]
        for nref, bref in zip(name_refs, bbox_refs, strict=True):
            name = np.ravel(f[nref][()]).astype(np.uint8).tobytes().decode("ascii")
            g = f[bref]
            lab, left, top, width, height = (_read_values(f, g[k]) for k in _BOX_KEYS)
            order = np.argsort(left, kind="stable")
            digits = "".join(str(int(lab[k]) % 10) for k in order)  # MATLAB label 10 is digit 0
            boxes = [[left[k], top[k], width[k], height[k]] for k in order]
            out.append({"name": name, "label": digits, "boxes": boxes})
    return out


def parse_digit_struct(
    mat_path: Path, workers: int | None = None, chunk: int | None = None
) -> list[dict]:
    """[{name, label (str, left to right), boxes [[left, top, width, height], ...]}] in file order.

    Large files (extra has 202,353 images) are parsed by `workers` processes over index
    chunks. workers=None uses up to 16 processes for files with more than 4,000 images.
    """
    import h5py

    with h5py.File(mat_path, "r") as handle:
        f: Any = handle
        n = int(f["digitStruct/name"].shape[0])
    if workers is None:
        workers = min(16, os.cpu_count() or 1) if n > 4000 else 1
    if workers <= 1:
        return _parse_range(str(mat_path), 0, n)
    chunk = chunk or max(1, math.ceil(n / (workers * 8)))
    ranges = [(s, min(n, s + chunk)) for s in range(0, n, chunk)]
    parts: dict[int, list[dict]] = {}
    t0, done = time.time(), 0
    # spawn, not fork: the parent may hold torch threads, and h5py handles must not be shared
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(_parse_range, str(mat_path), s, e): s for s, e in ranges}
        for fut in as_completed(futs):
            parts[futs[fut]] = fut.result()
            done += len(parts[futs[fut]])
            if len(parts) % max(1, len(ranges) // 10) == 0 or done == n:
                logger.info(f"{mat_path.parent.name}: parsed {done}/{n} ({time.time() - t0:.0f}s)")
    return [r for s, _ in ranges for r in parts[s]]


# --- split and crop rules -----------------------------------------------------------


def record_key(r: dict) -> str:
    """Unique image id. Names repeat across folders (train/1.png and extra/1.png)."""
    return f"{r['dir']}/{r['name']}" if "dir" in r else r["name"]


def hash_split(
    records: list[dict], val_frac: float = 0.1, salt: str = "group12"
) -> tuple[list[int], list[int]]:
    """Deterministic split stratified by length. Within each length, the val_frac of images
    with the smallest sha1(salt + dir/name) go to validation."""
    by_len: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        by_len.setdefault(len(r["label"]), []).append(i)
    train, val = [], []
    for idx in by_len.values():
        idx = sorted(
            idx, key=lambda i: hashlib.sha1((salt + record_key(records[i])).encode()).hexdigest()
        )
        k = int(len(idx) * val_frac)
        val += idx[:k]
        train += idx[k:]
    return sorted(train), sorted(val)


def drop_long(records: list[dict], max_len: int = MAX_LEN) -> tuple[list[dict], list[dict]]:
    """(kept, dropped) where dropped have more than max_len digits."""
    kept = [r for r in records if len(r["label"]) <= max_len]
    return kept, [r for r in records if len(r["label"]) > max_len]


def crop_box(boxes: Sequence[Sequence[float]], expand: float = CROP_EXPAND) -> list[float]:
    """[left, top, right, bottom] of the square around the digit-box union, scaled by expand.

    The union box is squared around its center with side max(width, height), then grown by
    (expand - 1) / 2 * side on each side. Clamping to the image happens in the Rust crop.
    """
    b = np.asarray(boxes, np.float64)
    left, top = b[:, 0].min(), b[:, 1].min()
    right, bottom = (b[:, 0] + b[:, 2]).max(), (b[:, 1] + b[:, 3]).max()
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half = max(float(right - left), float(bottom - top)) * expand / 2
    return [float(cx - half), float(cy - half), float(cx + half), float(cy + half)]


def length_hist(labels: list[str]) -> dict[str, int]:
    h: dict[str, int] = {}
    for s in labels:
        h[str(len(s))] = h.get(str(len(s)), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


# --- cache ------------------------------------------------------------------------


def _write_cache(
    path: Path,
    paths: list[str],
    size: int,
    crops: np.ndarray | None,
    stretch: bool,
    pad: list[int],
    chunk: int,
) -> np.ndarray:
    """Decode/resize in chunks straight into a .npy memmap. Returns geometry (N, 5)."""
    n = len(paths)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    geom = np.zeros((n, 5), np.float32)
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        c = None if crops is None else crops[s:e]
        imgs, g = letterbox_images(paths[s:e], size, size, pad, c, stretch=stretch)
        arr[s:e], geom[s:e] = imgs, g
        logger.info(f"{path.name}: {e}/{n} images ({time.time() - t0:.0f}s)")
    arr.flush()
    del arr
    return geom


def _channel_mean(path: Path, chunk: int = 20000) -> np.ndarray:
    """Per-channel RGB mean of a cached (N, H, W, 3) uint8 array, in [0, 255]."""
    arr = np.load(path, mmap_mode="r")
    total = np.zeros(3)
    for s in range(0, len(arr), chunk):
        total += np.asarray(arr[s : s + chunk]).reshape(-1, 3).sum(0, dtype=np.float64)
    return total / max(1, len(arr) * arr.shape[1] * arr.shape[2])


def _labels_json(recs: list[dict], geom: np.ndarray) -> str:
    return json.dumps(
        [
            {
                "name": r["name"],
                "dir": r["dir"],
                "label": r["label"],
                "boxes": r["boxes"],
                "crop": [round(v, 3) for v in r["crop"]],
                "geometry": [round(float(v), 5) for v in g],
            }
            for r, g in zip(recs, geom, strict=True)
        ]
    )


def prepare(
    raw_dir: Path,
    out_dir: Path,
    size: int = 64,
    max_len: int = MAX_LEN,
    val_frac: float = 0.1,
    workers: int | None = None,
    chunk: int = 20000,
) -> dict:
    """Build the cropped cache from raw_dir/{train,extra,test}. Returns the audit dict."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs: dict[str, list[dict]] = {}
    for src in SOURCES:
        t0 = time.time()
        recs[src] = parse_digit_struct(raw_dir / src / "digitStruct.mat", workers)
        for r in recs[src]:
            r["dir"] = src
            r["crop"] = crop_box(r["boxes"])
        logger.info(f"{src}: {len(recs[src])} images parsed in {time.time() - t0:.1f}s")

    pool, dropped = [], []
    for src in ("train", "extra"):
        kept, drop = drop_long(recs[src], max_len)
        pool += kept
        dropped += drop
    long_test = [record_key(r) for r in recs["test"] if len(r["label"]) > max_len]
    if long_test:
        raise ValueError(f"test labels longer than {max_len} digits: {long_test[:5]}")
    tr_idx, va_idx = hash_split(pool, val_frac)
    splits = {
        "train": [pool[i] for i in tr_idx],
        "val": [pool[i] for i in va_idx],
        "test": recs["test"],
    }
    logger.info(
        f"dropped {len(dropped)} train/extra images longer than {max_len} digits, "
        + ", ".join(f"{k} {len(v)}" for k, v in splits.items())
    )

    def path(r: dict) -> str:
        return str(raw_dir / r["dir"] / r["name"])

    summary = {}
    for name, rs in splits.items():
        crops = np.array([r["crop"] for r in rs], np.float32).reshape(-1, 4)
        geom = _write_cache(
            out_dir / f"{name}_images.npy", [path(r) for r in rs], size, crops, True, [0] * 3, chunk
        )
        (out_dir / f"{name}_labels.json").write_text(_labels_json(rs, geom))
        summary[name] = {
            "n_images": len(rs),
            "by_source": {s: c for s in SOURCES if (c := sum(r["dir"] == s for r in rs))},
            "length_hist": length_hist([r["label"] for r in rs]),
            "crop_width_px_pct_5_50_95": np.percentile(geom[:, 0], [5, 50, 95]).tolist()
            if rs
            else None,
        }
    pad = np.rint(_channel_mean(out_dir / "train_images.npy")).astype(int).tolist()
    test = splits["test"]
    geom = _write_cache(
        out_dir / "test_whole_images.npy", [path(r) for r in test], size, None, False, pad, chunk
    )
    (out_dir / "test_whole_labels.json").write_text(_labels_json(test, geom))
    summary["test_whole"] = {"n_images": len(test), "length_hist": summary["test"]["length_hist"]}

    audit = {
        "dataset": "SVHN Format 1 (full numbers), cropped protocol",
        "url": URL,
        "raw_files": [f"{s}.tar.gz" for s in SOURCES],
        "image_format": "PNG, RGB, variable size",
        "content": "house-number digits from Google Street View photos",
        "counts_per_source": {s: len(recs[s]) for s in SOURCES},
        "length_hist_per_source": {s: length_hist([r["label"] for r in recs[s]]) for s in SOURCES},
        "dropped_long_labels": {
            "rule": f"train/extra images with more than {max_len} digits are removed",
            "count": len(dropped),
            "by_source": {s: sum(r["dir"] == s for r in dropped) for s in ("train", "extra")},
            "images": [{"key": record_key(r), "label": r["label"]} for r in dropped],
        },
        "pool_size_before_split": len(pool),
        "label_rule": "digits ordered by increasing bounding-box left, MATLAB label 10 -> 0",
        "split_rule": (
            f"train+extra merged, {1 - val_frac:.0%}/{val_frac:.0%} train/val by "
            "sha1(salt + dir/name) within each length; official test untouched"
        ),
        "crop_rule": (
            "union of the digit boxes, squared around its center with side max(w, h), expanded "
            f"{CROP_EXPAND}x (0.15*side each way), clamped to the image, resized to "
            f"{size}x{size} RGB ignoring the aspect ratio (bilinear/triangle). Test crops use "
            "the ground-truth boxes (oracle localization)"
        ),
        "train_crop": "random 54x54 crop of the 64x64 cache plus ColorJitter (training only)",
        "eval_crop": "center 54x54 crop",
        "test_whole_rule": (
            f"whole test image letterboxed into {size}x{size}, aspect kept, padded with the "
            "mean RGB of the train crop cache (stress test, not the primary protocol)"
        ),
        "whole_pad_rgb": pad,
        "normalization": {"mean": list(NORM_MEAN), "std": list(NORM_STD), "scale": "x / 255"},
        "canvas": [size, size, 3],
        "vocabulary": {"PAD": PAD, "SOS": SOS, "EOS": EOS, "digits": f"token = {DIGIT0} + digit"},
        "max_len": max_len,
        "splits": summary,
    }
    (out_dir / "svhn_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


# --- whole-image setting (no ground-truth box) ----------------------------------------


def _label_only(records: list[dict]) -> list[dict]:
    """Name, folder and digit string of each record. Boxes, crops and geometry are dropped."""
    return [{"name": r["name"], "dir": r["dir"], "label": r["label"]} for r in records]


def whole_manifest_entry(audit: dict, border: dict | None = None) -> dict:
    """The split_manifest.json entry of the whole-image cache, from its full_audit.json.

    With the matching full_border_audit.json its crop-clipping rates are added under
    "border_diagnostic" (diagnostic only).
    """
    entry = {
        "rule": audit["whole_rule"],
        "records": audit["records_rule"],
        "suffix": audit["suffix"],
        "n_images": {k: v["n_images"] for k, v in audit["splits"].items()},
        "length_hist": {k: v["length_hist"] for k, v in audit["splits"].items()},
    }
    if border is not None:
        keys = ("eval_union_clipped_frac", "eval_digit_center_out_frac", "train_union_clipped_prob")
        entry["border_diagnostic"] = {
            "file": "full_border_audit.json",
            "eval_window_px": border["eval_window_px"],
            "splits": {k: {m: v[m] for m in keys} for k, v in border["splits"].items()},
        }
    return entry


def whole_cache_matches(cache_dir: Path, audit: dict) -> bool:
    """True when the whole-image cache was built from the current cropped-cache records."""
    for name, v in audit["splits"].items():
        src = Path(cache_dir) / f"{name}_labels.json"
        if not src.exists() or hashlib.sha256(src.read_bytes()).hexdigest() != v.get(
            "source_labels_sha256"
        ):
            return False
    return True


def check_whole_suffix(suffix: object) -> str:
    """The suffix of the whole-image cache. Refuses names of the cropped or stress caches.

    With suffix "" prepare_whole() would overwrite {split}_images.npy and {split}_labels.json,
    the oracle-crop cache, and "_whole" would overwrite the stress cache test_whole_*.
    """
    if not isinstance(suffix, str) or suffix in _RESERVED_SUFFIXES:
        raise ValueError(
            f"whole-image suffix {suffix!r} would overwrite the cropped or stress cache, "
            f"use e.g. {WHOLE_SUFFIX!r}"
        )
    if not re.fullmatch(r"_[A-Za-z0-9]+", suffix):
        raise ValueError(f"whole-image suffix {suffix!r} must look like '_full'")
    return suffix


def prepare_whole(
    raw_dir: Path,
    cache_dir: Path,
    size: int = 64,
    chunk: int = 20000,
    suffix: str = WHOLE_SUFFIX,
) -> dict:
    """Build the whole-image cache next to the cropped cache. Returns the audit dict.

    The records come from cache_dir/{train,val,test}_labels.json written by prepare(), so
    both settings use the same images, order and split. Only name, folder and label are kept
    from them. Each whole original image is resized to size x size RGB with the aspect ratio
    ignored (stretch), with no crop. Writes {split}{suffix}_images.npy,
    {split}{suffix}_labels.json and full_audit.json.
    """
    check_whole_suffix(suffix)
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    summary = {}
    t_all = time.time()
    for name in SPLITS:
        src = cache_dir / f"{name}_labels.json"
        if not src.exists():
            raise FileNotFoundError(f"{src} missing: build the cropped cache with prepare() first")
        recs = _label_only(json.loads(src.read_text()))
        logger.info(f"whole {name}: {len(recs)} images from {src.name}")
        paths = [str(raw_dir / r["dir"] / r["name"]) for r in recs]
        out = cache_dir / f"{name}{suffix}_images.npy"
        geom = _write_cache(out, paths, size, None, True, [0, 0, 0], chunk)
        rows = [
            r | {"geometry": [round(float(v), 5) for v in g]}
            for r, g in zip(recs, geom, strict=True)
        ]
        (cache_dir / f"{name}{suffix}_labels.json").write_text(json.dumps(rows))
        summary[name] = {
            "n_images": len(recs),
            "by_source": {s: c for s in SOURCES if (c := sum(r["dir"] == s for r in recs))},
            "length_hist": length_hist([r["label"] for r in recs]),
            "source_labels_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "source_width_px_pct_5_50_95": np.percentile(geom[:, 0], [5, 50, 95]).tolist()
            if recs
            else None,
            "source_height_px_pct_5_50_95": np.percentile(geom[:, 1], [5, 50, 95]).tolist()
            if recs
            else None,
        }
    audit = {
        "setting": "whole image, no ground-truth box (He et al. 2019 style)",
        "suffix": suffix,
        "records_rule": (
            "records of the cropped cache {split}_labels.json, same images, order and "
            "train/val/test assignment. Only name, folder and digit string are read"
        ),
        "whole_rule": (
            f"whole original image decoded as RGB and resized to {size}x{size} ignoring the "
            "aspect ratio (stretch, bilinear/triangle). No crop and no digit box is used, "
            "for training, validation and test alike"
        ),
        "boxes_used": False,
        "train_crop": "random 54x54 crop of the 64x64 cache plus ColorJitter (training only)",
        "eval_crop": (
            "center 54x54 crop. A stretched whole image has no margin around the digits, so "
            "this crop can cut digits near the border. full_border_audit.json measures how often"
        ),
        "normalization": {"mean": list(NORM_MEAN), "std": list(NORM_STD), "scale": "x / 255"},
        "canvas": [size, size, 3],
        "files": {k: [f"{k}{suffix}_images.npy", f"{k}{suffix}_labels.json"] for k in SPLITS},
        "splits": summary,
    }
    (cache_dir / "full_audit.json").write_text(json.dumps(audit, indent=2))
    logger.info(f"whole-image cache written in {time.time() - t_all:.0f}s")
    return audit


def _union_ltrb(boxes: Sequence[Sequence[float]], w: float, h: float) -> np.ndarray:
    """Digit boxes [[left, top, width, height]] clamped to the image -> (n, 4) as l, t, r, b."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    ltrb = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)
    return np.clip(ltrb, 0, [w, h, w, h])


def _fits(lo: float, hi: float, crop: int, margin: int) -> int:
    """Number of integer offsets o in 0..margin with o <= lo and hi <= o + crop."""
    first, last = max(0, math.ceil(hi - crop)), min(margin, math.floor(lo))
    return max(0, last - first + 1)


def whole_border_audit(cache_dir: Path, crop: int = 54, suffix: str = WHOLE_SUFFIX) -> dict:
    """DIAGNOSTIC ONLY: how often the 54x54 crops of SvhnSplit cut digits off whole images.

    Not part of the whole-image data path: it reads the ground-truth digit boxes of the
    cropped cache ({split}_labels.json) once, maps them through the stretch geometry of
    {split}{suffix}_labels.json onto the size x size canvas and writes full_border_audit.json.
    Its numbers explain part of the gap between the two settings and must never be used to
    choose a model, a margin or a decoding on test.

    Per split it reports
      eval_union_clipped_frac      union box not inside the center window [m, m + crop)
      eval_digit_center_out_frac   some digit's center outside that window (digit mostly lost)
      train_union_clipped_prob     mean probability that the random training crop clips the
                                   union box (uniform integer offsets 0..size-crop per axis)
    where m = (size - crop) // 2, as in SvhnSplit.batch.
    """
    check_whole_suffix(suffix)
    cache_dir = Path(cache_dir)
    full_audit = json.loads((cache_dir / "full_audit.json").read_text())
    size = int(full_audit["canvas"][0])
    margin = size - crop
    if margin < 0:
        raise ValueError(f"crop {crop} does not fit the {size}px cache")
    lo, hi = margin // 2, margin // 2 + crop
    out: dict[str, dict] = {}
    for name in SPLITS:
        src = cache_dir / f"{name}_labels.json"
        crop_recs = json.loads(src.read_text())
        full_recs = json.loads((cache_dir / f"{name}{suffix}_labels.json").read_text())
        if [(r["dir"], r["name"]) for r in crop_recs] != [(r["dir"], r["name"]) for r in full_recs]:
            raise ValueError(f"{name}: whole-image cache does not follow {src.name}, rebuild it")
        clipped, center_out, train_p, overhang = [], [], [], []
        for cr, fr in zip(crop_recs, full_recs, strict=True):
            w, h = float(fr["geometry"][0]), float(fr["geometry"][1])
            d = _union_ltrb(cr["boxes"], w, h) * [size / w, size / h, size / w, size / h]
            u = [d[:, 0].min(), d[:, 1].min(), d[:, 2].max(), d[:, 3].max()]
            over = max(lo - u[0], lo - u[1], u[2] - hi, u[3] - hi, 0.0)
            clipped.append(over > 0)
            overhang.append(over)
            cx, cy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
            center_out.append(bool(((cx < lo) | (cx >= hi) | (cy < lo) | (cy >= hi)).any()))
            ok = _fits(u[0], u[2], crop, margin) * _fits(u[1], u[3], crop, margin)
            train_p.append(1.0 - ok / (margin + 1) ** 2)
        n = len(crop_recs)
        over_c = [o for o, c in zip(overhang, clipped, strict=True) if c]
        out[name] = {
            "n_images": n,
            "eval_union_clipped_frac": float(np.mean(clipped)) if n else None,
            "eval_digit_center_out_frac": float(np.mean(center_out)) if n else None,
            "train_union_clipped_prob": float(np.mean(train_p)) if n else None,
            "overhang_px_pct_50_95_of_clipped": np.percentile(over_c, [50, 95]).tolist()
            if over_c
            else None,
            "source_labels_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        }
        logger.info(
            f"border diagnostic {name}: union clipped by the eval crop in "
            f"{out[name]['eval_union_clipped_frac']}, a digit center outside in "
            f"{out[name]['eval_digit_center_out_frac']}"
        )
    audit = {
        "purpose": (
            "diagnostic only, not part of the whole-image data path. Ground-truth boxes of the "
            "cropped cache are read here once to measure how often the fixed 54x54 crops cut "
            "digits off the stretched whole images. Never used for training, selection or test "
            "decisions"
        ),
        "suffix": suffix,
        "canvas": size,
        "crop": crop,
        "eval_window_px": [lo, hi],
        "box_rule": (
            "digit boxes clamped to the original image, scaled by (size / width, size / height) "
            "as the stretch resize, union over the digits of a number"
        ),
        "splits": out,
    }
    (cache_dir / "full_border_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def load_border_audit(cache_dir: Path, audit: dict) -> dict | None:
    """full_border_audit.json when it matches the whole-image cache of `audit` and the
    current cropped records, else None."""
    path = Path(cache_dir) / "full_border_audit.json"
    if not path.exists() or not whole_cache_matches(cache_dir, audit):
        return None
    border = json.loads(path.read_text())
    same = border.get("suffix") == audit["suffix"] and all(
        border["splits"].get(k, {}).get("source_labels_sha256") == v["source_labels_sha256"]
        for k, v in audit["splits"].items()
    )
    return border if same else None


# --- training-time access -----------------------------------------------------------


def encode(label: str, max_len: int = MAX_LEN) -> list[int]:
    """Decoder targets: d1 .. dn EOS PAD .., length max_len + 1."""
    if len(label) > max_len:
        raise ValueError(f"label {label!r} longer than max_len={max_len}")
    ids = [DIGIT0 + int(c) for c in label] + [EOS]
    return ids + [PAD] * (max_len + 1 - len(ids))


def decode(ids: Sequence[int]) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        if i >= DIGIT0:
            out.append(str(i - DIGIT0))
    return "".join(out)


@dataclass
class OcrBatch:
    images: torch.Tensor  # (B, 3, crop, crop) normalized float
    tokens: torch.Tensor  # (B, max_len + 1) decoder targets d1..dn EOS PAD..
    lengths: torch.Tensor  # (B,) length class, number of digits - 1
    index: torch.Tensor  # (B,) row in the split


def crop_batch(x: torch.Tensor, offsets: torch.Tensor, crop: int) -> torch.Tensor:
    """x (B, C, H, W), offsets (B, 2) as (top, left) -> (B, C, crop, crop)."""
    ar = torch.arange(crop, device=x.device)
    rows = offsets[:, 0, None] + ar
    cols = offsets[:, 1, None] + ar
    bi = torch.arange(x.shape[0], device=x.device)[:, None, None]
    # advanced indices around a slice put the broadcast dims first: (B, crop, crop, C)
    return x[bi, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)


def color_jitter(x: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    """Per-image brightness, contrast and saturation (torchvision formulas, fixed order).

    x (B, 3, H, W) in [0, 1], factors (B, 3). No hue change.
    """
    f = factors.to(x.dtype).view(-1, 3, 1, 1, 1).unbind(1)
    x = (x * f[0]).clamp(0, 1)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    mean = gray.mean(dim=(1, 2, 3), keepdim=True)
    x = (f[1] * x + (1 - f[1]) * mean).clamp(0, 1)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    return (f[2] * x + (1 - f[2]) * gray).clamp(0, 1)


class SvhnSplit:
    """One cached split held on the device as (N, 3, size, size) uint8.

    Reads {split}{suffix}_images.npy and {split}{suffix}_labels.json. The default suffix ""
    is the cropped cache, suffix "_full" the whole-image cache of prepare_whole().

    Training batches take a random crop per image and apply ColorJitter on the device,
    evaluation batches take the center crop. All random numbers come from the CPU
    generator passed in, so a resumed epoch sees exactly the same augmentation.
    """

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        max_len: int = MAX_LEN,
        device: str = "cpu",
        crop: int = 54,
        jitter: float = 0.3,
        limit: int | None = None,
        suffix: str = "",
    ) -> None:
        imgs = np.load(Path(cache_dir) / f"{split}{suffix}_images.npy", mmap_mode="r")
        records = json.loads((Path(cache_dir) / f"{split}{suffix}_labels.json").read_text())
        if limit is not None:
            imgs, records = imgs[:limit], records[:limit]
        self.split, self.records = split, records
        self.labels = [r["label"] for r in records]
        x = torch.from_numpy(np.array(imgs)).to(device)  # copy: the memmap is read-only
        self.images = x.permute(0, 3, 1, 2).contiguous()
        self.tokens = torch.tensor(
            [encode(s, max_len) for s in self.labels], dtype=torch.long, device=device
        ).reshape(-1, max_len + 1)
        self.lengths = torch.tensor(
            [len(s) - 1 for s in self.labels], dtype=torch.long, device=device
        )
        self.device, self.crop, self.jitter = device, crop, jitter
        self.size = int(self.images.shape[-1])
        if not 0 < crop <= self.size:
            raise ValueError(f"crop {crop} does not fit the {self.size}px cache")
        self.mean = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def batch(
        self, idx: torch.Tensor, train: bool = False, g: torch.Generator | None = None
    ) -> OcrBatch:
        idx = idx.to(self.device)
        b, margin = len(idx), self.size - self.crop
        if train:
            off = torch.randint(0, margin + 1, (b, 2), generator=g)
            factors = 1 + (torch.rand(b, 3, generator=g) * 2 - 1) * self.jitter
        else:
            off = torch.full((b, 2), margin // 2, dtype=torch.long)
            factors = None
        x = crop_batch(self.images[idx], off.to(self.device), self.crop).float() / 255.0
        if factors is not None and self.jitter > 0:
            x = color_jitter(x, factors.to(self.device))
        x = (x - self.mean) / self.std
        return OcrBatch(x, self.tokens[idx], self.lengths[idx], idx)

    def iterate(
        self, batch_size: int, train: bool = False, g: torch.Generator | None = None
    ) -> Iterator[OcrBatch]:
        """Shuffled augmented batches when train, else the split in order with center crops."""
        n = len(self)
        order = torch.randperm(n, generator=g) if train else torch.arange(n)
        for i in range(0, n, batch_size):
            yield self.batch(order[i : i + batch_size], train, g)
