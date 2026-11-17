"""Duplicate-content audit for SVHN train/extra/test (leakage check before results are final).

Exact duplicates: sha1 of the raw PNG bytes. Near duplicates: a 64-bit difference hash (dHash)
of each cached 64x64 crop (grayscale, resized to 9x8). Pairs are reported per split pair, so a
duplicate that crosses train/val/test is visible. Writes results/manifests/svhn_duplicate_audit.json.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from loguru import logger

SPLITS = ("train", "val", "test")


def dhash(images: np.ndarray, batch: int = 8192) -> np.ndarray:
    """64-bit dHash per image of a (N, H, W, 3) uint8 array, as uint64."""
    out = np.empty(len(images), np.uint64)
    weights = torch.tensor([1 << k for k in range(63)] + [-(1 << 63)], dtype=torch.int64)
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.asarray(images[i : i + batch])).float().mean(-1, keepdim=True)
        x = F.interpolate(x.permute(0, 3, 1, 2), size=(8, 9), mode="area")[:, 0]
        bits = (x[:, :, 1:] > x[:, :, :-1]).reshape(len(x), 64).to(torch.int64)
        out[i : i + len(x)] = (bits * weights).sum(1).numpy().view(np.uint64)
    return out


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def cross_pairs(groups: dict, owner: list[tuple[str, str]]) -> dict:
    """Count groups (same hash) whose members span several splits."""
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list] = defaultdict(list)
    for members in groups.values():
        if len(members) < 2:
            continue
        splits = sorted({owner[m][0] for m in members})
        key = "+".join(splits)
        counts[key] += 1
        if len(examples[key]) < 5:
            examples[key].append([owner[m][1] for m in members][:4])
    return {"groups_by_split_combination": dict(counts), "examples": dict(examples)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/processed/svhn")
    ap.add_argument("--raw", default="data/raw/svhn")
    ap.add_argument("--out", default="results/manifests/svhn_duplicate_audit.json")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    cache, raw = Path(a.cache), Path(a.raw)

    owner: list[tuple[str, str]] = []
    hashes, files = [], []
    for split in SPLITS:
        recs = json.loads((cache / f"{split}_labels.json").read_text())
        imgs = np.load(cache / f"{split}_images.npy", mmap_mode="r")
        hashes.append(dhash(imgs))
        for r in recs:
            owner.append((split, f"{r['dir']}/{r['name']}"))
            files.append(raw / r["dir"] / r["name"])
        logger.info(f"{split}: {len(recs)} images hashed")
    dh = np.concatenate(hashes)

    with ThreadPoolExecutor(a.workers) as ex:
        shas = list(ex.map(sha1, files, chunksize=512))
    logger.info("sha1 of all raw files done")

    exact: dict[str, list[int]] = defaultdict(list)
    near: dict[int, list[int]] = defaultdict(list)
    for i, (s, h) in enumerate(zip(shas, dh, strict=True)):
        exact[s].append(i)
        near[int(h)].append(i)

    report = {
        "n_images": {s: sum(o[0] == s for o in owner) for s in SPLITS},
        "exact_duplicates_sha1": cross_pairs(exact, owner),
        "near_duplicates_dhash64_on_64x64_crops": cross_pairs(near, owner),
        "note": "A group is a set of images with the same hash. Combinations with more than one "
        "split name cross a split boundary. dHash collisions can include visually different "
        "but low-texture crops, so cross-split near-duplicate groups need manual inspection.",
    }
    Path(a.out).write_text(json.dumps(report, indent=2))
    logger.info(
        json.dumps(
            {
                k: v["groups_by_split_combination"]
                for k, v in report.items()
                if isinstance(v, dict) and "groups_by_split_combination" in v
            }
        )
    )


if __name__ == "__main__":
    main()
