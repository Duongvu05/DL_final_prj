import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from dl_final_prj import fast
from dl_final_prj.metrics import per_sample_losses, sequence_report
from dl_final_prj.ocr_models import (
    BEAM_WIDTHS,
    AttentionGRURecognizer,
    CTCRecognizer,
    MultiHeadCNN,
    PooledGRURecognizer,
    build,
    ctc_greedy,
    ctc_loss,
    ctc_prefix_beam,
    ctc_prefix_beam_one,
    multihead_map,
    primary_mode,
    select_width,
    tokens_to_ctc_targets,
)
from dl_final_prj.svhn import (
    EOS,
    MAX_LEN,
    PAD,
    OcrBatch,
    SvhnSplit,
    crop_box,
    decode,
    drop_long,
    encode,
    hash_split,
    parse_digit_struct,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]


def write_digit_struct(path, items):
    """Mimic the MATLAB v7.3 layout. items = [(name, [(label, left, top, w, h), ...])]."""
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        names, boxes = [], []
        for i, (name, digits) in enumerate(items):
            n = refs.create_dataset(f"n{i}", data=np.array([[ord(c)] for c in name], np.uint16))
            names.append(n.ref)
            g = refs.create_group(f"b{i}")
            for k, key in enumerate(("label", "left", "top", "width", "height")):
                vals = [d[k] for d in digits]
                if len(vals) == 1:
                    g.create_dataset(key, data=np.array([[vals[0]]], np.float64))
                else:
                    r = [
                        refs.create_dataset(f"v{i}_{key}_{j}", data=np.array([[v]], np.float64)).ref
                        for j, v in enumerate(vals)
                    ]
                    g.create_dataset(key, data=np.array(r, dtype=h5py.ref_dtype).reshape(-1, 1))
            boxes.append(g.ref)
        ds = f.create_group("digitStruct")
        ds.create_dataset("name", data=np.array(names, dtype=h5py.ref_dtype).reshape(-1, 1))
        ds.create_dataset("bbox", data=np.array(boxes, dtype=h5py.ref_dtype).reshape(-1, 1))


ITEMS = [
    ("1.png", [(1, 10, 5, 8, 20), (9, 20, 5, 8, 20)]),
    ("2.png", [(10, 3, 2, 8, 20)]),  # MATLAB 10 is digit 0
    ("3.png", [(5, 30, 1, 8, 20), (2, 4, 1, 8, 20), (10, 17, 1, 8, 20)]),  # unsorted boxes
]


def synthetic_items(n: int, rng: np.random.Generator, long_label: bool = False):
    """n images named 1.png.. with 1-3 digits inside a 40x80 image, plus one 6-digit image."""
    items = []
    for i in range(n):
        k = 1 + i % 3
        digits = [(int(rng.integers(1, 11)), 8 + 12 * j, 8, 10, 22) for j in range(k)]
        items.append((f"{i + 1}.png", digits))
    if long_label:
        items.append((f"{n + 1}.png", [(j + 1, 2 + 12 * j, 8, 10, 22) for j in range(6)]))
    return items


def write_raw(root: Path, sizes: dict[str, int], long_in: tuple[str, ...] = ("train",)) -> Path:
    rng = np.random.default_rng(0)
    for split, n in sizes.items():
        d = root / split
        d.mkdir(parents=True)
        items = synthetic_items(n, rng, long_label=split in long_in)
        write_digit_struct(d / "digitStruct.mat", items)
        for name, _ in items:
            arr = rng.integers(0, 255, (40, 80, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / name)
    return root


@pytest.fixture
def raw_svhn(tmp_path):
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        d = tmp_path / "raw" / split
        d.mkdir(parents=True)
        write_digit_struct(d / "digitStruct.mat", ITEMS)
        for name, _ in ITEMS:
            Image.fromarray(rng.integers(0, 255, (30, 50, 3), dtype=np.uint8)).save(d / name)
    return tmp_path / "raw"


# --- annotations, split and crop rules ----------------------------------------------


def test_parse_digit_struct_orders_by_left_and_maps_ten_to_zero(raw_svhn):
    recs = parse_digit_struct(raw_svhn / "train" / "digitStruct.mat")
    assert [r["name"] for r in recs] == ["1.png", "2.png", "3.png"]
    assert [r["label"] for r in recs] == ["19", "0", "205"]
    assert recs[2]["boxes"][0] == [4, 1, 8, 20]


def test_parse_digit_struct_worker_chunks_match_serial(tmp_path):
    path = tmp_path / "digitStruct.mat"
    write_digit_struct(path, synthetic_items(11, np.random.default_rng(3), long_label=True))
    serial = parse_digit_struct(path, workers=1)
    assert parse_digit_struct(path, workers=2, chunk=3) == serial
    assert len(serial) == 12 and len(serial[-1]["label"]) == 6


def test_crop_box_squares_expands_and_clamps(tmp_path):
    # union [10, 5, 28, 25]: 18 wide, 20 high, center (19, 15), side 20, expanded to 26
    assert crop_box([[10, 5, 8, 20], [20, 5, 8, 20]]) == pytest.approx([6, 2, 32, 28])
    # union [1, 2, 11, 32]: side 30 -> 39, center (6, 17), so it leaves the image at the top left
    box = crop_box([[1, 2, 10, 30]])
    assert box == pytest.approx([-13.5, -2.5, 25.5, 36.5])
    assert fast.clamp_box(box, 40, 50) == (0, 0, 26, 37)
    assert fast.clamp_box(box, 20, 30) == (0, 0, 20, 30)
    p = tmp_path / "a.png"
    Image.fromarray(np.zeros((50, 40, 3), np.uint8)).save(p)
    _, g = fast.letterbox_images([str(p)], 64, 64, [0, 0, 0], np.array([box]), stretch=True)
    np.testing.assert_allclose(g[0], [26, 37, 64 / 26, 0, 0], rtol=1e-6)


def test_hash_split_is_deterministic_stratified_and_disjoint():
    recs = [{"name": f"{i}.png", "label": "1" * (1 + i % 3)} for i in range(300)]
    tr, va = hash_split(recs)
    assert hash_split(recs) == (tr, va)
    assert not set(tr) & set(va) and len(tr) + len(va) == 300
    assert len(va) == 30
    for L in (1, 2, 3):
        assert sum(len(recs[i]["label"]) == L for i in va) == 10


def test_hash_split_keys_include_the_source_folder():
    recs = [
        {"name": f"{i}.png", "dir": d, "label": "1" * (1 + i % 2)}
        for d in ("train", "extra")
        for i in range(400)
    ]
    tr, va = hash_split(recs)
    assert hash_split(list(recs)) == (tr, va)
    assert len(va) == 80 and not set(tr) & set(va)
    val_names = {
        d: {recs[i]["name"] for i in va if recs[i]["dir"] == d} for d in ("train", "extra")
    }
    # if the key were the bare name, the same names would land in validation for both folders
    assert val_names["train"] != val_names["extra"]


def test_drop_long_labels():
    recs = [{"name": "a", "label": "12345"}, {"name": "b", "label": "123456"}]
    kept, dropped = drop_long(recs)
    assert [r["name"] for r in kept] == ["a"] and [r["name"] for r in dropped] == ["b"]


def test_encode_decode_roundtrip_max_len_5():
    assert MAX_LEN == 5
    ids = encode("907")
    assert len(ids) == 6 and ids[3] == EOS and ids[4:] == [PAD] * 2
    assert decode(ids) == "907"
    assert encode("12345")[-1] == EOS and decode(encode("12345")) == "12345"
    with pytest.raises(ValueError):
        encode("123456")


# --- cache and batches ----------------------------------------------------------------


def write_cache(d: Path, labels: list[str], size: int = 64) -> Path:
    """Images whose red channel is the row and green channel is the column index."""
    d.mkdir(parents=True, exist_ok=True)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.stack([yy, xx, np.full_like(yy, 100)], -1).astype(np.uint8)
    imgs = np.stack([img + i for i in range(len(labels))])
    np.save(d / "train_images.npy", imgs)
    recs = [{"name": f"{i}.png", "dir": "train", "label": s} for i, s in enumerate(labels)]
    (d / "train_labels.json").write_text(json.dumps(recs))
    return d


def test_svhn_split_center_crop_for_eval(tmp_path):
    s = SvhnSplit(write_cache(tmp_path, ["1", "23", "45678"]), "train", jitter=0.3)
    b = s.batch(torch.arange(3))
    assert b.images.shape == (3, 3, 54, 54) and s.images.dtype == torch.uint8
    raw = torch.round((b.images * 0.5 + 0.5) * 255)
    assert raw[0, 0, 0, 0] == 5 and raw[0, 1, 0, 0] == 5  # offset (5, 5)
    assert raw[0, 0, 53, 0] == 58 and raw[0, 1, 0, 53] == 58
    assert raw[2, 2, 0, 0] == 102  # no jitter at evaluation
    assert b.tokens.shape == (3, 6) and b.lengths.tolist() == [0, 1, 4]
    assert b.tokens[2].tolist() == encode("45678")


def test_svhn_split_train_crop_and_jitter_are_reproducible(tmp_path):
    s = SvhnSplit(write_cache(tmp_path, ["1"] * 16), "train", jitter=0.3)
    b1 = s.batch(torch.arange(16), train=True, g=torch.Generator().manual_seed(0))
    b2 = s.batch(torch.arange(16), train=True, g=torch.Generator().manual_seed(0))
    b3 = s.batch(torch.arange(16), train=True, g=torch.Generator().manual_seed(1))
    torch.testing.assert_close(b1.images, b2.images)
    assert not torch.equal(b1.images, b3.images)
    assert b1.images.min() >= -1 and b1.images.max() <= 1

    s.jitter = 0.0  # crop only: the top-left pixel gives the random offset
    b = s.batch(torch.arange(16), train=True, g=torch.Generator().manual_seed(0))
    raw = torch.round((b.images * 0.5 + 0.5) * 255) - torch.arange(16).view(-1, 1, 1, 1)
    top, left = raw[:, 0, 0, 0], raw[:, 1, 0, 0]
    assert top.min() >= 0 and top.max() <= 10 and left.min() >= 0 and left.max() <= 10
    assert len(set(top.tolist())) > 1  # crops differ between images
    torch.testing.assert_close(raw[:, 0, :, 0], top[:, None] + torch.arange(54.0))

    order1 = [
        b.index.tolist() for b in s.iterate(5, train=True, g=torch.Generator().manual_seed(4))
    ]
    order2 = [
        b.index.tolist() for b in s.iterate(5, train=True, g=torch.Generator().manual_seed(4))
    ]
    assert order1 == order2 and sorted(i for o in order1 for i in o) == list(range(16))


def test_prepare_builds_cropped_cache(tmp_path):
    raw = write_raw(tmp_path / "raw", {"train": 30, "extra": 30, "test": 9})
    audit = prepare(raw, tmp_path / "cache", size=64, workers=1, chunk=7)
    sp = audit["splits"]
    assert audit["counts_per_source"] == {"train": 31, "extra": 30, "test": 9}
    assert audit["dropped_long_labels"]["count"] == 1
    assert audit["dropped_long_labels"]["images"][0]["key"] == "train/31.png"
    assert sp["train"]["n_images"] + sp["val"]["n_images"] == 60 == audit["pool_size_before_split"]
    assert sp["val"]["n_images"] == 6 and sp["test"]["n_images"] == 9
    assert audit["max_len"] == 5 and audit["normalization"]["mean"] == [0.5] * 3
    cache = tmp_path / "cache"
    for split in ("train", "val", "test", "test_whole"):
        imgs = np.load(cache / f"{split}_images.npy")
        labels = json.loads((cache / f"{split}_labels.json").read_text())
        assert imgs.shape == (len(labels), 64, 64, 3) and imgs.dtype == np.uint8
    labels = json.loads((cache / "test_labels.json").read_text())
    rec = labels[0]
    assert set(rec) == {"name", "dir", "label", "boxes", "crop", "geometry"}
    left, top, right, bottom = fast.clamp_box(rec["crop"], 80, 40)
    assert rec["geometry"][:2] == [right - left, bottom - top] and rec["geometry"][3:] == [0, 0]
    whole = json.loads((cache / "test_whole_labels.json").read_text())
    assert whole[0]["geometry"] == [80, 40, 0.8, 0, 16]  # 80x40 letterboxed into 64x32
    w = np.load(cache / "test_whole_images.npy")
    assert (w[0, 0] == audit["whole_pad_rgb"]).all()
    s = SvhnSplit(cache, "test_whole")
    assert s.labels == [x["label"] for x in labels]


# --- metrics --------------------------------------------------------------------------


def test_sequence_report_hand_example():
    targets = ["12", "345", "7"]
    preds = ["12", "34", "79"]  # exact, early EOS, extra digit
    r = sequence_report(targets, preds)
    yt = np.array([1, 2, 3, 4, 5, 7.0])
    yp = np.array([1, 2, 3, 4, 0, 7.0])
    assert r["MAE"] == pytest.approx(np.abs(yp - yt).mean())
    assert r["RMSE"] == pytest.approx(np.sqrt(25 / 6))
    assert r["R2"] == pytest.approx(1 - 25 / np.sum((yt - yt.mean()) ** 2))
    assert r["exact_match"] == pytest.approx(1 / 3)
    assert r["length_accuracy"] == pytest.approx(1 / 3)
    assert r["char_accuracy"] == pytest.approx(5 / 6)
    assert r["norm_edit_distance"] == pytest.approx((0 + 1 / 3 + 1 / 2) / 3)
    assert r["CER"] == pytest.approx(2 / 6)
    per = per_sample_losses(targets, preds)
    np.testing.assert_allclose(per["sq_err_sum"], [0, 25, 0])
    np.testing.assert_allclose(per["exact"], [1, 0, 0])


# --- models ---------------------------------------------------------------------------


def test_ctc_helpers():
    tokens = torch.tensor([encode("10"), encode("7")])
    tgt, lens = tokens_to_ctc_targets(tokens)
    assert tgt.tolist() == [2, 1, 8] and lens.tolist() == [2, 1]
    logits = torch.full((1, 6, 11), -9.0)
    for t, c in enumerate([2, 2, 0, 2, 1, 1]):  # "1", "1" after blank, "0"
        logits[0, t, c] = 0
    assert ctc_greedy(logits) == ["110"]
    good = torch.full((2, 14, 11), -9.0)
    for t, c in enumerate([2, 0, 1] + [0] * 11):
        good[0, t, c] = 0
    for t, c in enumerate([8] + [0] * 13):
        good[1, t, c] = 0
    assert ctc_loss(good, tokens) < ctc_loss(torch.zeros(2, 14, 11), tokens)


def ocr_batch(labels: list[str], seed: int = 0) -> OcrBatch:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.tensor([encode(s) for s in labels])
    lengths = torch.tensor([len(s) - 1 for s in labels])
    return OcrBatch(torch.randn(len(labels), 3, 54, 54, generator=g), tokens, lengths, tokens[:, 0])


@pytest.mark.parametrize("kind", ["attention", "pooled", "multihead", "ctc"])
def test_recognizer_shapes_and_backward(kind):
    torch.manual_seed(0)
    b = ocr_batch(["123", "4", "56789"])
    m = build(kind)
    if kind in ("attention", "pooled"):
        assert isinstance(m, AttentionGRURecognizer | PooledGRURecognizer)
        logits, length_logits, att = m(b.images, b.tokens)
        assert logits.shape == (3, 6, 13) and length_logits.shape == (3, 5)
        assert (att.shape == (3, 6, 49)) if kind == "attention" else att is None
    elif kind == "multihead":
        assert isinstance(m, MultiHeadCNN)
        length_logits, digit_logits = m(b.images)
        assert length_logits.shape == (3, 5) and digit_logits.shape == (3, 5, 10)
    else:
        assert isinstance(m, CTCRecognizer)
        assert m(b.images).shape == (3, 14, 11)
    loss = m.loss(b)
    loss.backward()
    assert torch.isfinite(loss) and loss.dtype == torch.float32
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    m.eval()
    with torch.no_grad():
        for mode in m.modes:
            preds, _ = m.predict(b.images, mode)
            assert len(preds) == 3 and all(p.isdigit() or p == "" for p in preds)


def test_multihead_loss_counts_only_true_positions():
    torch.manual_seed(0)
    m = MultiHeadCNN().eval()
    b = ocr_batch(["7", "12"])
    length_logits, digit_logits = m(b.images)
    ce = torch.nn.functional.cross_entropy
    want = (
        ce(length_logits, b.lengths, reduction="sum")
        + ce(digit_logits[0, :1], torch.tensor([7]), reduction="sum")
        + ce(digit_logits[1, :2], torch.tensor([1, 2]), reduction="sum")
    ) / 2
    torch.testing.assert_close(m.loss(b), want)


@pytest.mark.parametrize("kind", ["attention", "pooled"])
def test_length_constrained_decoding_gives_requested_lengths(kind):
    torch.manual_seed(1)
    m = build(kind).eval()
    b = ocr_batch(["1"] * 5, seed=2)
    with torch.no_grad():
        want = torch.tensor([1, 2, 3, 4, 5])
        ids, _ = m.greedy(b.images, constrain=True, lengths=want)
        assert [len(decode(s)) for s in ids.tolist()] == want.tolist()
        heads = m.length_head(m.encode(b.images)[1]).argmax(-1) + 1
        preds, _ = m.predict(b.images, "length")
        assert [len(p) for p in preds] == heads.tolist()


@pytest.mark.parametrize("kind", ["attention", "pooled"])
def test_beam_width_one_equals_greedy(kind):
    torch.manual_seed(2)
    m = build(kind).eval()
    b = ocr_batch(["1"] * 8, seed=3)
    with torch.no_grad():
        greedy, aux = m.predict(b.images, "greedy")
        ids1, att1 = m.beam(b.images, 1)
        assert [decode(s) for s in ids1.tolist()] == greedy
        assert m.predict(b.images, "beam1")[0] == greedy
        for w in BEAM_WIDTHS:
            preds, aux_w = m.predict(b.images, f"beam{w}")
            assert len(preds) == 8 and all(p.isdigit() and 1 <= len(p) <= MAX_LEN for p in preds)
            assert ("attention" in aux_w) == (kind == "attention")
    if kind == "attention":
        att = aux["attention"]
        assert att.shape == (8, 6, 49)
        torch.testing.assert_close(att.sum(-1), torch.ones(8, 6))
        torch.testing.assert_close(att1, att)  # the best beam's weights, here the greedy path
        att10 = aux_w["attention"]
        assert att10.shape == (8, 6, 49)
        torch.testing.assert_close(att10.sum(-1), torch.ones(8, 6))
    else:
        assert att1 is None and "attention" not in aux


def test_beam_attention_follows_the_chosen_beam():
    """The attention returned by beam search is the one recomputed by a teacher-forced pass
    over the chosen sequence (the decoder is deterministic in eval mode)."""
    torch.manual_seed(4)
    m = build("attention").eval()
    b = ocr_batch(["1"] * 6, seed=5)
    with torch.no_grad():
        ids, att = m.beam(b.images, 5)
        _, _, forced = m(b.images, ids)
    torch.testing.assert_close(att, forced, atol=1e-5, rtol=1e-4)


def test_beam_search_finds_a_better_sequence_than_greedy():
    """A hand-made decoder: step 0 prefers digit 1 slightly, but after 1 the model is unsure,
    while after 2 it is certain about EOS. Greedy reads "13", beam search finds "2"."""
    m = build("pooled").eval()
    table = {  # previous token -> log-probabilities of next token
        "start": {4: np.log(0.5), 5: np.log(0.45)},
        4: {EOS: np.log(0.3), 3: np.log(0.3), 6: np.log(0.4)},
        5: {EOS: 0.0},
    }

    def step(prev, h, feats, kp, mean):
        logits = torch.full((len(prev), 13), -30.0)
        for i, p in enumerate(prev.tolist()):
            row = table.get(p, table["start"] if p == 1 else {EOS: 0.0})
            for tok, lp in row.items():
                logits[i, tok] = float(lp)
        return logits, h, None

    m.step = step  # type: ignore[method-assign]
    x = torch.zeros(1, 3, 54, 54)
    with torch.no_grad():
        assert m.predict(x, "greedy")[0] == ["13"]
        assert m.predict(x, "beam3")[0] == ["2"]


@pytest.mark.parametrize("mode", ["greedy", "beam1", "beam3", "beam5", "beam10"])
def test_unconstrained_decoding_stops_at_max_len(mode):
    """A decoder that never prefers EOS still outputs at most MAX_LEN digits: EOS is forced
    at the last step."""
    m = build("pooled").eval()

    def step(prev, h, feats, kp, mean):
        logits = torch.full((len(prev), 13), -30.0)
        logits[:, 4] = 0.0  # digit 1 is always the most likely token
        logits[:, 5] = -1.0
        logits[:, EOS] = -5.0
        return logits, h, None

    m.step = step  # type: ignore[method-assign]
    with torch.no_grad():
        preds, _ = m.predict(torch.zeros(2, 3, 54, 54), mode)
    assert all(len(p) == MAX_LEN for p in preds)
    if mode in ("greedy", "beam1"):
        assert preds == ["1" * MAX_LEN] * 2


def test_beam_beats_greedy_for_every_width():
    """The hand-made decoder of test_beam_search_finds_a_better_sequence_than_greedy, read
    through the primary-mode helper for every candidate width."""
    m = build("pooled").eval()

    def step(prev, h, feats, kp, mean):
        logits = torch.full((len(prev), 13), -30.0)
        for i, p in enumerate(prev.tolist()):
            if p == 1:  # SOS: digit 1 (0.5) slightly ahead of digit 2 (0.45)
                logits[i, 4], logits[i, 5] = math.log(0.5), math.log(0.45)
            elif p == 4:  # after 1 the model is unsure
                logits[i, EOS], logits[i, 3], logits[i, 6] = (math.log(x) for x in (0.3, 0.3, 0.4))
            else:  # after 2 (and after EOS) it is sure about EOS
                logits[i, EOS] = 0.0
        return logits, h, None

    m.step = step  # type: ignore[method-assign]
    x = torch.zeros(2, 3, 54, 54)
    with torch.no_grad():
        assert m.predict(x, "greedy")[0] == ["13"] * 2
        for w in BEAM_WIDTHS:
            assert m.predict(x, primary_mode(m, w))[0] == ["2"] * 2


def ctc_logits(probs: list[dict[int, float]]) -> torch.Tensor:
    """(1, T, 11) logits whose softmax is the given class probabilities (others ~0)."""
    x = torch.full((1, len(probs), 11), -50.0)
    for t, row in enumerate(probs):
        for c, p in row.items():
            x[0, t, c] = math.log(p)
    return x


def test_ctc_prefix_beam_merges_repeats_only_without_a_blank():
    one_one = ctc_logits([{2: 0.9, 0: 0.1}, {0: 0.9, 2: 0.1}, {2: 0.9, 0: 0.1}])  # 1 _ 1
    one = ctc_logits([{2: 0.9, 0: 0.1}, {2: 0.9, 0: 0.1}, {0: 0.9, 2: 0.1}])  # 1 1 _
    for w in (1, *BEAM_WIDTHS):
        assert ctc_prefix_beam(one_one, w) == ["11"]
        assert ctc_prefix_beam(one, w) == ["1"]
    # exact prefix probabilities of "1 1 _": "1" collects 1-1-_, 1-_-_, _-1-_, 1-1-1, ...
    ranked = dict(ctc_prefix_beam_one(one.log_softmax(-1)[0].tolist(), 10))
    p_one = sum(
        math.prod(row[c] for row, c in zip([[0.1, 0, 0.9]] * 2 + [[0.9, 0, 0.1]], path))
        for path in itertools.product((0, 2), repeat=3)
        if ctc_greedy(torch.tensor([[[0.0 if k == c else -9.0 for k in range(11)] for c in path]]))
        == ["1"]
    )
    assert math.exp(ranked[(2,)]) == pytest.approx(p_one, rel=1e-4)


def test_ctc_prefix_beam_sums_paths_and_beats_best_path():
    """Per step: blank 0.4, digit 1 0.35, digit 2 0.25. The best path is blank-blank (""),
    but "1" collects 1-1, 1-_ and _-1 (0.4025)."""
    logits = ctc_logits([{0: 0.4, 2: 0.35, 3: 0.25}] * 2)
    assert ctc_greedy(logits) == [""]
    for w in BEAM_WIDTHS:
        assert ctc_prefix_beam(logits, w) == ["1"]
    ranked = ctc_prefix_beam_one(logits.log_softmax(-1)[0].tolist(), 10)
    probs = {k: math.exp(v) for k, v in ranked}
    assert probs[(2,)] == pytest.approx(0.35**2 + 2 * 0.35 * 0.4, rel=1e-4)
    assert probs[()] == pytest.approx(0.16, rel=1e-4)
    assert sum(probs.values()) == pytest.approx(1.0, rel=1e-4)  # T=2: all prefixes survive


def test_ctc_prefix_beam_equals_best_path_when_one_path_dominates():
    g = torch.Generator().manual_seed(0)
    checked = 0
    for _ in range(60):
        path = torch.where(
            torch.rand(14, generator=g) < 0.6, 0, torch.randint(1, 11, (14,), generator=g)
        )
        logits = torch.full((1, 14, 11), -12.0)
        logits[0, torch.arange(14), path] = 0.0
        best = ctc_greedy(logits)
        if len(best[0]) > MAX_LEN:
            continue
        checked += 1
        for w in (1, *BEAM_WIDTHS):
            assert ctc_prefix_beam(logits, w) == best
    assert checked > 20


def test_ctc_prefix_beam_returns_at_most_max_len_digits():
    path = [2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]  # best path reads 7 digits
    logits = torch.full((1, 14, 11), -6.0)
    logits[0, torch.arange(14), torch.tensor(path)] = 0.0
    assert len(ctc_greedy(logits)[0]) == 7
    logp = logits.log_softmax(-1)[0].tolist()
    for w in BEAM_WIDTHS:
        out = ctc_prefix_beam(logits, w)[0]
        assert out.isdigit() and len(out) <= MAX_LEN
        assert out != "12345"  # not the best path cut to MAX_LEN, a scored prefix instead
        ranked = ctc_prefix_beam_one(logp, w, MAX_LEN)
        assert all(len(k) <= MAX_LEN for k, _ in ranked)  # long prefixes take no slot
        assert out == "".join(str(c - 1) for c in ranked[0][0])
    m = CTCRecognizer().eval()
    with torch.no_grad():
        for mode in m.modes:
            preds, _ = m.predict(torch.randn(3, 3, 54, 54), mode)
            if mode != "greedy":
                assert all(len(p) <= MAX_LEN for p in preds)


def test_ctc_prefix_beam_with_max_len_is_the_brute_force_best_short_prefix():
    """With a wide beam the result is the most probable prefix of at most max_len labels,
    computed by summing every path (exact, since longer prefixes never feed shorter ones)."""
    g = torch.Generator().manual_seed(3)
    for _ in range(60):
        t = int(torch.randint(2, 7, (1,), generator=g))
        c = int(torch.randint(2, 5, (1,), generator=g))
        max_len = int(torch.randint(1, 3, (1,), generator=g))
        logp = torch.randn(t, c, generator=g).mul(2).log_softmax(-1)
        total: dict[tuple[int, ...], float] = {}
        for path in itertools.product(range(c), repeat=t):
            labels = tuple(k for i, k in enumerate(path) if k and (i == 0 or path[i - 1] != k))
            p = math.exp(sum(logp[i, k].item() for i, k in enumerate(path)))
            total[labels] = total.get(labels, 0.0) + p
        short = {k: v for k, v in total.items() if len(k) <= max_len}
        want = max(short, key=lambda k: short[k])
        ranked = ctc_prefix_beam_one(logp.tolist(), 10_000, max_len)
        assert ranked[0][0] == want
        assert math.isclose(math.exp(ranked[0][1]), short[want], rel_tol=1e-9)
        assert all(len(k) <= max_len for k, _ in ranked)


def test_multihead_map_differs_from_the_length_argmax():
    """p(L) = (0.35, 0.40, 0.25), head 0 is sure (0.99), head 1 is not (0.5). The length
    argmax reads 2 digits (0.40 * 0.99 * 0.5 = 0.198), the joint MAP reads 1 (0.3465)."""
    length = torch.log(torch.tensor([[0.35, 0.40, 0.25, 1e-9, 1e-9]]))
    digits = torch.full((1, 5, 10), -30.0)
    digits[0, 0, 7] = math.log(0.99)
    digits[0, 0, 1] = math.log(0.01)
    digits[0, 1, 3], digits[0, 1, 4] = math.log(0.5), math.log(0.5) - 1e-4
    for i in range(2, 5):
        digits[0, i, 0] = 0.0
    lengths, best = multihead_map(length, digits)
    assert lengths.tolist() == [1] and best[0, 0].item() == 7
    m = MultiHeadCNN().eval()
    m.forward = lambda images: (length, digits)  # type: ignore[method-assign]
    assert m.predict(torch.zeros(1), "greedy")[0] == ["73"]
    assert m.predict(torch.zeros(1), "map")[0] == ["7"]


def test_multihead_map_is_the_brute_force_joint_argmax():
    g = torch.Generator().manual_seed(1)
    length = torch.randn(4, 3, generator=g) * 2
    digits = torch.randn(4, 3, 10, generator=g) * 2
    lengths, best = multihead_map(length, digits)
    lp_len, lp_dig = length.log_softmax(-1), digits.log_softmax(-1)
    for b in range(4):
        score = {
            seq: lp_len[b, len(seq) - 1].item()
            + sum(lp_dig[b, i, d].item() for i, d in enumerate(seq))
            for n in range(1, 4)
            for seq in itertools.product(range(10), repeat=n)
        }
        want = max(score, key=lambda k: score[k])
        assert (lengths[b].item(), tuple(best[b, : lengths[b]].tolist())) == (len(want), want)


def test_select_width_uses_exact_match_and_prefers_the_smaller_width():
    assert select_width({3: 0.5, 5: 0.6, 10: 0.6}) == 5
    assert select_width({10: 0.7, 3: 0.7, 5: 0.6}) == 3
    assert select_width({3: 0.1, 5: 0.2, 10: 0.3}) == 10
    with pytest.raises(ValueError):
        select_width({})


def load_run_ocr():
    spec = importlib.util.spec_from_file_location("run_ocr", ROOT / "scripts" / "run_ocr.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeSplit:
    def __init__(self, name: str, labels: list[str]) -> None:
        self.name, self.labels = name, labels
        self.records = [
            {"name": f"{i}.png", "dir": "test", "label": s} for i, s in enumerate(labels)
        ]

    def iterate(self, bs: int):
        yield SimpleNamespace(images=self.name)


def test_width_is_selected_on_validation_and_never_on_test():
    """val prefers width 5 (tie with 10, smaller wins), test would prefer 10. The reported
    test result must use width 5, and every val decode must come before the first test one."""
    run_ocr = load_run_ocr()
    labels = ["12", "345", "6", "78"]
    right = {"val": labels, "test": labels, "test_whole": labels}
    wrong = {k: ["9"] * 4 for k in right}
    table = {
        ("val", "prefix3"): [*labels[:2], "9", "9"],
        ("val", "prefix5"): labels,
        ("val", "prefix10"): labels,
        ("test", "prefix10"): labels,
    }
    calls = []
    model = CTCRecognizer()

    def fake_predict(images, mode):
        calls.append((images, mode))
        return table.get((images, mode), (right if mode == "greedy" else wrong)[images]), {}

    model.predict = fake_predict  # type: ignore[method-assign]
    data = {s: FakeSplit(s, labels) for s in ("val", "test", "test_whole")}
    metrics, _, att = run_ocr.evaluate(model, data, "test_whole", 8, False, keep_attention=False)
    dec = metrics["decoding"]
    assert dec["primary"] == "ctc_prefix_beam" and dec["width"] == 5 and dec["mode"] == "prefix5"
    assert dec["val_by_width"] == {"3": 0.5, "5": 1.0, "10": 1.0}
    assert dec["stress"] == {"split": "test_whole", "mode": "prefix5", "width": 5}
    assert metrics["val"]["exact_match"] == 1.0 and metrics["test"]["exact_match"] == 0.0
    assert metrics["test_greedy"]["exact_match"] == 1.0
    first_test = next(i for i, c in enumerate(calls) if c[0] != "val")
    assert all(c[0] == "val" for c in calls[:first_test])
    assert {m for s, m in calls if s != "val"} == {"prefix5", "greedy"}
    assert set(metrics) == {
        "val", "val_greedy", "test", "test_greedy", "test_whole", "test_whole_greedy", "decoding"
    }  # fmt: skip
    assert att is None


GRU_MODELS = ("cnn_attention_gru", "cnn_pool_gru")
PRIMARY = {
    "cnn_attention_gru": "beam",
    "cnn_pool_gru": "beam",
    "multihead_cnn": "map",
    "crnn_ctc": "ctc_prefix_beam",
    "crnn_ctc_whole": "ctc_prefix_beam",
}


def smoke_cmd(model: str, cache: Path, out: Path, *extra: str, config: Path | None = None):
    cmd = [sys.executable, "scripts/run_ocr.py", "--model", model, "--seed", "0", "--smoke"]
    cmd += ["--cache", str(cache), "--out", str(out), *extra]
    return cmd + (["--config", str(config)] if config else [])


def check_run(run: Path, model: str, stress: str | None = "test_whole") -> dict:
    """Shared checks of a Task 3 run directory written by the current decoding."""
    metrics = json.loads((run / "metrics.json").read_text())
    spec = json.loads((run / "spec.json").read_text())
    gru = model in GRU_MODELS
    reports = {"val", "test", "val_greedy", "test_greedy"}
    reports |= {"test_length_constrained"} if gru else set()
    reports |= {stress, f"{stress}_greedy"} if stress else set()
    assert set(metrics) == reports | {"decoding"}
    dec = metrics["decoding"]
    assert dec["primary"] == PRIMARY[model] and spec["decoding"]["primary"] == PRIMARY[model]
    if PRIMARY[model] == "map":
        assert dec["width"] is None and dec["val_by_width"] == {} and dec["mode"] == "map"
    else:
        assert set(dec["val_by_width"]) == {"3", "5", "10"} and dec["width"] in (3, 5, 10)
        best = max(dec["val_by_width"].values())
        assert dec["width"] == min(w for w in (3, 5, 10) if dec["val_by_width"][str(w)] == best)
        assert metrics["val"]["exact_match"] == dec["val_by_width"][str(dec["width"])]
        assert dec["mode"].endswith(str(dec["width"]))
    assert dec["stress"] == (
        {"split": stress, "mode": dec["mode"], "width": dec["width"]} if stress else None
    )
    assert set(spec["decoding_modes"]) == reports
    assert "greedy" in spec["selection"] and "training-time" in spec["selection"]
    for key in reports:
        preds = json.loads((run / f"pred_{key}.json").read_text())
        split = (
            "val"
            if key.startswith("val")
            else (stress if stress and key.startswith(stress) else "test")
        )
        assert len(preds) == spec["n_images"][split]
        assert (
            sequence_report([r["target"] for r in preds], [r["pred"] for r in preds])
            == metrics[key]
        )
    if PRIMARY[model] != "map":  # beam outputs never exceed MAX_LEN
        assert all(
            len(r["pred"]) <= MAX_LEN for r in json.loads((run / "pred_test.json").read_text())
        )
    assert (run / "attention_test.npz").exists() == (model == "cnn_attention_gru")
    if model == "cnn_attention_gru":
        att = np.load(run / "attention_test.npz")
        assert (
            att["attention"].shape
            == att["attention_greedy"].shape
            == (spec["n_images"]["test"], 6, 49)
        )
        assert str(att["mode"]) == dec["mode"]
    return metrics


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_seconds(m: dict) -> dict:
    return m | {"decoding": {k: v for k, v in m["decoding"].items() if k != "seconds"}}


def test_end_to_end_prepare_and_smoke_runs(tmp_path):
    raw = write_raw(tmp_path / "raw", {"train": 40, "extra": 40, "test": 12})
    cache = tmp_path / "cache"
    prepare(raw, cache, size=64, workers=1)
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    for model in ("cnn_attention_gru", "cnn_pool_gru", "multihead_cnn", "crnn_ctc"):
        cmd = smoke_cmd(model, cache, tmp_path / "results")
        subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
        run = tmp_path / "results" / "smoke" / "runs" / "task3" / model / "seed0"
        spec = json.loads((run / "spec.json").read_text())
        history = json.loads((run / "history.json").read_text())
        gru = model in GRU_MODELS
        metrics = check_run(run, model)
        assert len(history) == 2 and spec["epochs_run"] == 2 and spec["n_params"] > 0
        assert spec["n_images"] == {"train": 74, "val": 6, "test": 12, "test_whole": 12}
        assert spec["setting"] == "oracle_crop" and spec["cache_suffix"] == ""
        by_src = spec["n_images_by_source"]
        assert {s: sum(c.values()) for s, c in by_src.items()} == spec["n_images"]
        assert by_src["test"] == {"train": 0, "extra": 0, "test": 12}
        # the synthetic train folder adds one 6-digit label, which prepare drops
        assert spec["source_counts_raw"] == {"train": 41, "extra": 40, "test": 12}
        assert spec["dropped_long_labels"]["by_source"] == {"train": 1, "extra": 0}
        assert (spec["vocabulary"].get("EOS") == EOS) == gru
        assert (spec["length_head"] is None) == (model == "crnn_ctc")
        # a finished run is skipped unless --force
        out = subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
        assert "already done" in out.stderr
        # --eval-only reproduces the evaluation from ckpt.pt without touching it
        ck = sha(run / "ckpt.pt")
        subprocess.run([*cmd, "--eval-only"], check=True, cwd=ROOT, env=env, capture_output=True)
        assert sha(run / "ckpt.pt") == ck
        assert strip_seconds(check_run(run, model)) == strip_seconds(metrics)
        assert not (run / "metrics_v1_greedy.json").exists()  # already non-greedy


def test_eval_only_rewrites_a_greedy_run_and_keeps_the_checkpoint(tmp_path):
    raw = write_raw(tmp_path / "raw", {"train": 40, "extra": 40, "test": 12})
    cache = tmp_path / "cache"
    prepare(raw, cache, size=64, workers=1)
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    model = "cnn_attention_gru"
    cmd = smoke_cmd(model, cache, tmp_path / "results")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
    run = tmp_path / "results" / "smoke" / "runs" / "task3" / model / "seed0"
    new = json.loads((run / "metrics.json").read_text())
    # turn it into a run written by the previous (greedy primary) code
    old_keys = {  # v1 key -> the current key holding the same evaluation
        "val": "val_greedy",
        "test": "test_greedy",
        "test_length_constrained": "test_length_constrained",
        "test_beam3": "test",  # beam 3 or another width, only the format matters here
        "test_whole": "test_whole_greedy",
    }
    v1 = {k: new[v] for k, v in old_keys.items()}
    (run / "metrics.json").write_text(json.dumps(v1))
    shutil.copy(run / "pred_test_greedy.json", run / "pred_test.json")
    (run / "pred_test_greedy.json").unlink()
    spec = json.loads((run / "spec.json").read_text())
    for k in ("decoding", "decoding_modes", "selection", "setting", "cache_suffix"):
        spec.pop(k)  # the previous code wrote none of these
    spec["crop_protocol"].pop("stress_split")
    (run / "spec.json").write_text(json.dumps(spec))
    fp = json.loads((run / "fingerprint.json").read_text())
    fp["config"].pop("cache_suffix", None)  # old fingerprints have no cache_suffix
    (run / "fingerprint.json").write_text(json.dumps(fp))
    keep = {f: sha(run / f) for f in ("ckpt.pt", "fingerprint.json", "history.json")}

    out = subprocess.run(
        [*cmd, "--eval-only"], check=True, cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert "backed up the greedy metrics" in out.stderr
    assert {f: sha(run / f) for f in keep} == keep
    assert json.loads((run / "metrics_v1_greedy.json").read_text()) == v1
    metrics = check_run(run, model)
    assert strip_seconds(metrics) == strip_seconds(new)
    assert metrics["test_greedy"] == v1["test"]  # the old primary is now the greedy diagnostic
    spec = json.loads((run / "spec.json").read_text())
    assert spec["eval_only"]["previous_metrics"] == "metrics_v1_greedy.json"
    assert set(spec["eval_only"]["data"]) == {"val", "test"}
    assert spec["epochs_run"] == 2  # training entries are kept
    # the old run is labelled as the oracle crop setting, like newly trained runs
    assert spec["setting"] == "oracle_crop" and spec["cache_suffix"] == ""
    assert "training-time" in spec["selection"]
    assert spec["crop_protocol"]["stress_split"] == "test_whole"
    # a second --eval-only keeps the first backup
    subprocess.run([*cmd, "--eval-only"], check=True, cwd=ROOT, env=env, capture_output=True)
    assert json.loads((run / "metrics_v1_greedy.json").read_text()) == v1
    assert {f: sha(run / f) for f in keep} == keep
    # --eval-only refuses a config that changes the evaluation data
    metrics_sha = sha(run / "metrics.json")
    base = yaml.safe_load((ROOT / "configs" / "phase1_task3.yaml").read_text())
    for i, change in enumerate(({"crop": 48}, {"cache_suffix": "_full"}, {"max_len": 4})):
        path = tmp_path / f"changed{i}.yaml"
        path.write_text(yaml.safe_dump(base | change))
        bad = smoke_cmd(model, cache, tmp_path / "results", "--eval-only", config=path)
        out = subprocess.run(bad, cwd=ROOT, env=env, capture_output=True, text=True)
        assert out.returncode != 0 and "differ from training" in out.stderr
    # ... and a cache whose training file changed size (rebuilt or another cache)
    fp["data"]["size"] += 1
    (run / "fingerprint.json").write_text(json.dumps(fp))
    out = subprocess.run([*cmd, "--eval-only"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "training saw" in out.stderr
    assert sha(run / "metrics.json") == metrics_sha
    # --eval-only refuses unfinished runs and --force
    other = smoke_cmd("cnn_pool_gru", cache, tmp_path / "results", "--eval-only")
    assert subprocess.run(other, cwd=ROOT, env=env, capture_output=True).returncode != 0
    assert (
        subprocess.run(
            [*cmd, "--eval-only", "--force"], cwd=ROOT, env=env, capture_output=True
        ).returncode
        != 0
    )
    assert not (
        tmp_path
        / "results"
        / "smoke"
        / "runs"
        / "task3"
        / "cnn_pool_gru"
        / "seed0"
        / "metrics.json"
    ).exists()


def test_cache_suffix_and_no_stress_split(tmp_path):
    """A whole-image style config: train/val/test read {split}_full_* and no stress split."""
    raw = write_raw(tmp_path / "raw", {"train": 40, "extra": 40, "test": 12})
    cache = tmp_path / "cache"
    prepare(raw, cache, size=64, workers=1)
    for split in ("train", "val", "test"):  # stand-in for prepare_whole: the same records
        for kind in ("images.npy", "labels.json"):
            shutil.copy(cache / f"{split}_{kind}", cache / f"{split}_full_{kind}")
    (cache / "test_whole_images.npy").unlink()  # must not be read
    cfg = yaml.safe_load((ROOT / "configs" / "phase1_task3.yaml").read_text())
    cfg |= {"cache_suffix": "_full", "stress_split": None}
    cfg["models"] = {"crnn_ctc_whole": cfg["models"]["crnn_ctc"]}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    cmd = smoke_cmd("crnn_ctc_whole", cache, tmp_path / "results", config=path)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
    run = tmp_path / "results" / "smoke" / "runs" / "task3" / "crnn_ctc_whole" / "seed0"
    check_run(run, "crnn_ctc_whole", stress=None)
    spec = json.loads((run / "spec.json").read_text())
    assert spec["setting"] == "whole_image" and spec["cache_suffix"] == "_full"
    assert set(spec["n_images"]) == {"train", "val", "test"}
    fp = json.loads((run / "fingerprint.json").read_text())
    assert fp["data"]["path"].endswith("train_full_images.npy")
    subprocess.run([*cmd, "--eval-only"], check=True, cwd=ROOT, env=env, capture_output=True)
    check_run(run, "crnn_ctc_whole", stress=None)
