"""Whole-image SVHN setting (svhn.prepare_whole, no ground-truth box)."""

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from dl_final_prj import fast, svhn
from dl_final_prj.svhn import (
    SPLITS,
    SvhnSplit,
    load_border_audit,
    prepare,
    prepare_whole,
    whole_border_audit,
    whole_cache_matches,
)
from test_svhn import synthetic_items, write_digit_struct

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"train": 30, "extra": 30, "test": 9}


def write_raw_blocks(root: Path, sizes: dict[str, int]) -> Path:
    """Like test_svhn.write_raw, but images of varied aspect ratio made of 6x6 color blocks,
    so a bilinear resize is well defined and the Rust and Pillow resizers agree closely."""
    rng = np.random.default_rng(5)
    for split, n in sizes.items():
        d = root / split
        d.mkdir(parents=True)
        items = synthetic_items(n, rng, long_label=split == "train")
        write_digit_struct(d / "digitStruct.mat", items)
        for i, (name, _) in enumerate(items):
            h, w = 40 + 7 * (i % 4), 64 + 23 * (i % 5)  # 40-61 high, 64-156 wide
            arr = rng.integers(0, 255, (h // 6 + 1, w // 6 + 1, 3), dtype=np.uint8)
            arr = np.kron(arr, np.ones((6, 6, 1), np.uint8))[:h, :w]
            Image.fromarray(arr).save(d / name)
    return root


@pytest.fixture
def caches(tmp_path):
    raw = write_raw_blocks(tmp_path / "raw", SIZES)
    cache = tmp_path / "cache"
    prepare(raw, cache, size=64, workers=1, chunk=7)
    audit = prepare_whole(raw, cache, size=64, chunk=7)
    return raw, cache, audit


def load(cache: Path, name: str) -> tuple[np.ndarray, list[dict]]:
    recs = json.loads((cache / f"{name}_labels.json").read_text())
    return np.load(cache / f"{name}_images.npy"), recs


def test_prepare_whole_keeps_records_order_and_split(caches):
    _, cache, audit = caches
    crop_audit = json.loads((cache / "svhn_audit.json").read_text())
    assert audit["boxes_used"] is False and audit["suffix"] == "_full"
    for split in SPLITS:
        _, crop = load(cache, split)
        imgs, full = load(cache, f"{split}_full")
        assert imgs.shape == (len(crop), 64, 64, 3) and imgs.dtype == np.uint8
        key = [(r["dir"], r["name"], r["label"]) for r in crop]
        assert [(r["dir"], r["name"], r["label"]) for r in full] == key
        assert all(set(r) == {"name", "dir", "label", "geometry"} for r in full)
        sp, csp = audit["splits"][split], crop_audit["splits"][split]
        assert sp["n_images"] == csp["n_images"] and sp["length_hist"] == csp["length_hist"]
        assert sp["by_source"] == csp["by_source"]
    assert {r["dir"] for r in load(cache, "test_full")[1]} == {"test"}
    assert json.loads((cache / "full_audit.json").read_text()) == audit
    assert whole_cache_matches(cache, audit)
    labels = cache / "val_labels.json"
    labels.write_text(json.dumps(json.loads(labels.read_text())[:-1]))  # crop cache changed
    assert not whole_cache_matches(cache, audit)
    entry = svhn.whole_manifest_entry(audit)
    assert entry["n_images"] == {k: audit["splits"][k]["n_images"] for k in SPLITS}


def test_full_images_are_stretched_whole_images(caches):
    raw, cache, _ = caches
    errs_full, errs_crop = [], []
    for split in SPLITS:
        imgs, full = load(cache, f"{split}_full")
        crop_imgs, _ = load(cache, split)
        paths = [str(raw / r["dir"] / r["name"]) for r in full]
        ref, ref_geom = fast.letterbox_images_ref(paths, 64, 64, [0, 0, 0], None, stretch=True)
        np.testing.assert_allclose([r["geometry"] for r in full], ref_geom, rtol=1e-5)
        for i, p in enumerate(paths):
            im = Image.open(p).convert("RGB")
            w, h = im.size
            assert full[i]["geometry"] == pytest.approx([w, h, 64 / w, 0, 0], rel=1e-5)
            pil = np.asarray(im.resize((64, 64), Image.Resampling.BILINEAR)).astype(int)
            np.testing.assert_array_equal(pil, ref[i])
            errs_full.append(np.abs(imgs[i].astype(int) - pil).mean())
            errs_crop.append(np.abs(crop_imgs[i].astype(int) - pil).mean())
    assert max(errs_full) < 12  # Rust triangle filter vs Pillow bilinear
    assert np.mean(errs_full) < np.mean(errs_crop) / 3  # and clearly not the GT-box crop


def test_prepare_whole_never_reads_boxes(caches, monkeypatch):
    raw, cache, _ = caches
    before = {s: load(cache, f"{s}_full") for s in SPLITS}

    def forbidden(*args, **kwargs):
        raise AssertionError("a digit box was used")

    calls = []
    real = svhn.letterbox_images

    def spy(paths, height, width, pad, crops=None, *, stretch=False):
        assert crops is None and stretch
        calls.append(len(paths))
        return real(paths, height, width, pad, crops, stretch=stretch)

    monkeypatch.setattr(svhn, "crop_box", forbidden)
    monkeypatch.setattr(fast, "clamp_box", forbidden)
    monkeypatch.setattr(svhn, "letterbox_images", spy)
    for s in SPLITS:  # labels/strings only: drop every box-derived field from the inputs
        p = cache / f"{s}_labels.json"
        recs = json.loads(p.read_text())
        p.write_text(json.dumps([{k: r[k] for k in ("name", "dir", "label")} for r in recs]))
    prepare_whole(raw, cache, size=64, chunk=7)
    assert sum(calls) == sum(len(before[s][1]) for s in SPLITS)
    for s in SPLITS:
        imgs, recs = load(cache, f"{s}_full")
        np.testing.assert_array_equal(imgs, before[s][0])
        assert recs == before[s][1]


def test_prepare_whole_needs_the_crop_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare"):
        prepare_whole(tmp_path / "raw", tmp_path / "empty")


def test_svhn_split_suffix(caches):
    _, cache, _ = caches
    assert inspect.signature(SvhnSplit).parameters["suffix"].default == ""
    imgs, full = load(cache, "test_full")
    s = SvhnSplit(cache, "test", suffix="_full")
    assert s.labels == [r["label"] for r in full] and s.records == full
    assert torch.equal(s.images, torch.from_numpy(imgs).permute(0, 3, 1, 2))
    b = s.batch(torch.arange(len(s)))
    assert b.images.shape == (len(s), 3, 54, 54) and b.lengths.tolist() == [
        len(r["label"]) - 1 for r in full
    ]
    crop_imgs, crop = load(cache, "test")
    d = SvhnSplit(cache, "test")  # default: the cropped cache, unchanged
    assert d.records == crop
    assert torch.equal(d.images, torch.from_numpy(crop_imgs).permute(0, 3, 1, 2))
    assert not torch.equal(d.images, s.images)


def test_whole_config_mirrors_the_crop_config():
    crop = yaml.safe_load((ROOT / "configs/phase1_task3.yaml").read_text())
    whole = yaml.safe_load((ROOT / "configs/phase1_task3_whole.yaml").read_text())
    assert whole["cache_suffix"] == "_full" and whole["stress_split"] is None
    assert set(whole) - set(crop) <= {"cache_suffix", "stress_split"}
    for k, v in crop.items():
        if k == "models":
            assert whole["models"] == {f"{name}_whole": m for name, m in v.items()}
        elif k not in ("title", "cache_suffix", "stress_split"):
            assert whole[k] == v, k


def test_prepare_data_with_whole_writes_manifest(tmp_path, synthetic_npz):
    raw = write_raw_blocks(tmp_path / "raw" / "svhn", SIZES).parent
    proc = tmp_path / "processed"
    proc.mkdir()
    synthetic_npz.rename(proc / "electricity_hourly.npz")
    (proc / "electricity_audit.json").write_text("{}")
    cfg = yaml.safe_load((ROOT / "configs/phase1.yaml").read_text())
    cfg["results_root"] = str(tmp_path / "results")
    (tmp_path / "phase1.yaml").write_text(yaml.safe_dump(cfg))
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "scripts/prepare_data.py", "--raw", str(raw), "--processed", str(proc)]
    cmd += ["--config", str(tmp_path / "phase1.yaml"), "--skip", "electricity", "--with-whole"]
    cmd += ["--workers", "1"]
    subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
    man = json.loads((tmp_path / "results" / "manifests" / "split_manifest.json").read_text())
    crop_counts = man["task3"]["n_images"]
    assert man["task3_whole"]["n_images"] == {k: crop_counts[k] for k in SPLITS}
    assert man["task3_whole"]["suffix"] == "_full"

    # the standalone script rebuilds the same cache and refreshes the manifest entry
    border = man["task3_whole"]["border_diagnostic"]
    assert border["eval_window_px"] == [5, 59] and set(border["splits"]) == set(SPLITS)
    assert (proc / "svhn" / "full_border_audit.json").exists()

    # the standalone script rebuilds the same cache and refreshes the manifest entry. A config
    # with cache_suffix "" falls back to "_full" and never touches the cropped cache
    man_path = tmp_path / "results" / "manifests" / "split_manifest.json"
    before = np.load(proc / "svhn" / "train_full_images.npy")
    crop_sha = {f: sha(proc / "svhn" / f) for f in crop_files()}
    wcfg = yaml.safe_load((ROOT / "configs/phase1_task3_whole.yaml").read_text())
    (tmp_path / "whole.yaml").write_text(yaml.safe_dump(wcfg | {"cache_suffix": ""}))
    cmd = [sys.executable, "scripts/prepare_svhn_whole.py", "--raw", str(raw / "svhn")]
    cmd += ["--cache", str(proc / "svhn"), "--manifest", str(man_path), "--threads", "2"]
    cmd += ["--config", str(tmp_path / "whole.yaml")]
    out = subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
    assert "threads: 2" in out.stderr
    np.testing.assert_array_equal(np.load(proc / "svhn" / "train_full_images.npy"), before)
    assert {f: sha(proc / "svhn" / f) for f in crop_files()} == crop_sha
    assert json.loads(man_path.read_text())["task3_whole"] == man["task3_whole"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop_files() -> list[str]:
    """Files of the cropped setting: train/val/test crops and the test_whole stress cache."""
    names = [*SPLITS, "test_whole"]
    return [f"{n}_{k}" for n in names for k in ("images.npy", "labels.json")]


@pytest.mark.parametrize("suffix", ["", None, "_whole", "full", "_a/b", "_"])
def test_prepare_whole_refuses_crop_cache_suffixes(caches, suffix):
    raw, cache, _ = caches
    before = {f: sha(cache / f) for f in crop_files()}
    with pytest.raises(ValueError, match="suffix"):
        prepare_whole(raw, cache, size=64, chunk=7, suffix=suffix)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="suffix"):
        whole_border_audit(cache, suffix=suffix)  # type: ignore[arg-type]
    assert {f: sha(cache / f) for f in crop_files()} == before


def write_border_case(cache: Path, size: int = 64) -> None:
    """A hand-made 100x50 image per record with known digit boxes, in every split."""
    boxes = [
        [[40, 10, 10, 30]],  # x 25.6..32, y 12.8..51.2: inside the window [5, 59)
        [[2, 10, 10, 30]],  # x from 1.28, digit center x 4.48: clipped, center out
        [[-5, 0, 20, 50]],  # clamped to x 0..9.6, y 0..64: clipped, center out, never fits
        [[30, 20, 8, 10], [80, 20, 8, 10]],  # x 19.2..56.32, y 25.6..38.4: inside
        [[30, 20, 8, 10], [88, 20, 8, 10]],  # second digit x 56.32..61.44: clipped, center in
    ]
    crop, full = [], []
    for i, b in enumerate(boxes):
        r = {"name": f"{i}.png", "dir": "train", "label": "1" * len(b)}
        crop.append(r | {"boxes": b})
        full.append(r | {"geometry": [100, 50, size / 100, 0, 0]})
    for name in SPLITS:
        (cache / f"{name}_labels.json").write_text(json.dumps(crop))
        (cache / f"{name}_full_labels.json").write_text(json.dumps(full))
    (cache / "full_audit.json").write_text(json.dumps({"canvas": [size, size, 3]}))


def test_border_audit_hand_cases(tmp_path):
    write_border_case(tmp_path)
    audit = whole_border_audit(tmp_path, crop=54)
    assert audit["eval_window_px"] == [5, 59] and audit["crop"] == 54
    for name in SPLITS:
        sp = audit["splits"][name]
        assert sp["n_images"] == 5
        assert sp["eval_union_clipped_frac"] == pytest.approx(3 / 5)
        assert sp["eval_digit_center_out_frac"] == pytest.approx(2 / 5)
        # x offsets in 0..10 that keep the box: case 1 all, case 2 {0, 1}, case 3 none (y
        # needs 0 and 10), case 4 o >= 56.32 - 54 -> 3..10, case 5 o >= 61.44 - 54 -> 8..10.
        # y always fits except in case 3
        p = [0, 1 - 2 / 11, 1.0, 1 - 8 / 11, 1 - 3 / 11]
        assert sp["train_union_clipped_prob"] == pytest.approx(np.mean(p))
        assert sp["overhang_px_pct_50_95_of_clipped"][0] == pytest.approx(3.72)  # 5 - 1.28
    assert json.loads((tmp_path / "full_border_audit.json").read_text()) == audit
    # a crop of the whole canvas never clips
    whole = whole_border_audit(tmp_path, crop=64)["splits"]["val"]
    assert whole["eval_union_clipped_frac"] == 0 and whole["train_union_clipped_prob"] == 0
    # records out of step with the whole-image cache are refused
    (tmp_path / "test_full_labels.json").write_text("[]")
    with pytest.raises(ValueError, match="rebuild"):
        whole_border_audit(tmp_path)


def test_border_audit_is_separate_from_the_data_path(caches):
    _, cache, audit = caches
    assert not (cache / "full_border_audit.json").exists()  # prepare_whole never writes it
    border = whole_border_audit(cache, crop=54)
    assert load_border_audit(cache, audit) == border
    for sp in border["splits"].values():
        for k in ("eval_union_clipped_frac", "eval_digit_center_out_frac"):
            assert 0 <= sp[k] <= 1
        assert sp["eval_digit_center_out_frac"] <= sp["eval_union_clipped_frac"]
    entry = svhn.whole_manifest_entry(audit, border)["border_diagnostic"]["splits"]["test"]
    assert entry["eval_union_clipped_frac"] == border["splits"]["test"]["eval_union_clipped_frac"]
    labels = cache / "val_labels.json"
    labels.write_text(json.dumps(json.loads(labels.read_text())[:-1]))
    assert load_border_audit(cache, audit) is None  # stale: built from other crop records


def test_run_ocr_whole_config_reads_only_the_full_cache(tmp_path):
    """run_ocr.py with configs/phase1_task3_whole.yaml trains and tests on {split}_full_* only."""
    raw = write_raw_blocks(tmp_path / "raw", SIZES)
    cache = tmp_path / "cache"
    prepare(raw, cache, size=64, workers=1, chunk=7)
    prepare_whole(raw, cache, size=64, chunk=7)
    for f in crop_files():  # the cropped and stress caches must never be read
        (cache / f).unlink()
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    model = "cnn_attention_gru_whole"
    cmd = [sys.executable, "scripts/run_ocr.py", "--model", model, "--seed", "0", "--smoke"]
    cmd += ["--cache", str(cache), "--out", str(tmp_path / "results")]
    cmd += ["--config", str(ROOT / "configs/phase1_task3_whole.yaml")]
    subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
    run = tmp_path / "results" / "smoke" / "runs" / "task3" / model / "seed0"
    spec = json.loads((run / "spec.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    fp = json.loads((run / "fingerprint.json").read_text())
    assert spec["setting"] == "whole_image" and spec["cache_suffix"] == "_full"
    assert set(spec["n_images"]) == set(SPLITS)  # no stress split
    assert not any(k.startswith("test_whole") for k in metrics)
    assert {"val", "test"} <= set(metrics)
    assert Path(fp["data"]["path"]).name == "train_full_images.npy"
    for split in ("val", "test"):
        full = json.loads((cache / f"{split}_full_labels.json").read_text())
        pred = json.loads((run / f"pred_{split}.json").read_text())
        assert [(p["dir"], p["name"], p["target"]) for p in pred] == [
            (r["dir"], r["name"], r["label"]) for r in full
        ]
