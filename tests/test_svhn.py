import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from dl_final_prj import fast
from dl_final_prj.metrics import per_sample_losses, sequence_report
from dl_final_prj.ocr_models import (
    AttentionGRURecognizer,
    CTCRecognizer,
    MultiHeadCNN,
    PooledGRURecognizer,
    build,
    ctc_greedy,
    ctc_loss,
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
        assert [decode(s) for s in m.beam(b.images, 1).tolist()] == greedy
        beam3, _ = m.predict(b.images, "beam3")
    assert len(beam3) == 8 and all(len(p) <= MAX_LEN for p in beam3)
    if kind == "attention":
        att = aux["attention"]
        assert att.shape == (8, 6, 49)
        torch.testing.assert_close(att.sum(-1), torch.ones(8, 6))


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


@pytest.mark.parametrize("mode", ["greedy", "beam1", "beam3"])
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
    if mode != "beam3":
        assert preds == ["1" * MAX_LEN] * 2


def test_end_to_end_prepare_and_smoke_runs(tmp_path):
    raw = write_raw(tmp_path / "raw", {"train": 40, "extra": 40, "test": 12})
    cache = tmp_path / "cache"
    prepare(raw, cache, size=64, workers=1)
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    for model in ("cnn_attention_gru", "cnn_pool_gru", "multihead_cnn", "crnn_ctc"):
        cmd = [sys.executable, "scripts/run_ocr.py", "--model", model, "--seed", "0", "--smoke"]
        cmd += ["--cache", str(cache), "--out", str(tmp_path / "results")]
        subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
        run = tmp_path / "results" / "smoke" / "runs" / "task3" / model / "seed0"
        metrics = json.loads((run / "metrics.json").read_text())
        spec = json.loads((run / "spec.json").read_text())
        history = json.loads((run / "history.json").read_text())
        gru = model in ("cnn_attention_gru", "cnn_pool_gru")
        want = {"val", "test", "test_whole"} | (
            {"test_length_constrained", "test_beam3"} if gru else set()
        )
        assert set(metrics) == want
        assert len(history) == 2 and spec["epochs_run"] == 2 and spec["n_params"] > 0
        assert spec["n_images"] == {"train": 74, "val": 6, "test": 12, "test_whole": 12}
        by_src = spec["n_images_by_source"]
        assert {s: sum(c.values()) for s, c in by_src.items()} == spec["n_images"]
        assert by_src["test"] == {"train": 0, "extra": 0, "test": 12}
        # the synthetic train folder adds one 6-digit label, which prepare drops
        assert spec["source_counts_raw"] == {"train": 41, "extra": 40, "test": 12}
        assert spec["dropped_long_labels"]["by_source"] == {"train": 1, "extra": 0}
        assert (spec["vocabulary"].get("EOS") == EOS) == gru
        assert (spec["length_head"] is None) == (model == "crnn_ctc")
        for key in want:
            preds = json.loads((run / f"pred_{key}.json").read_text())
            assert len(preds) == spec["n_images"]["val" if key == "val" else "test"]
        assert (run / "attention_test.npz").exists() == (model == "cnn_attention_gru")
        if model == "cnn_attention_gru":
            att = np.load(run / "attention_test.npz")["attention"]
            assert att.shape == (12, 6, 49)
        # a finished run is skipped unless --force
        out = subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
        assert "already done" in out.stderr
