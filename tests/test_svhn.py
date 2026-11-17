import json

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from dl_final_prj.metrics import per_sample_losses, sequence_report
from dl_final_prj.ocr_models import (
    AttentionRecognizer,
    CTCRecognizer,
    PooledRecognizer,
    ctc_greedy,
    ctc_loss,
    greedy_strings,
    tokens_to_ctc_targets,
)
from dl_final_prj.svhn import (
    EOS,
    PAD,
    SvhnSplit,
    decode,
    encode,
    hash_split,
    parse_digit_struct,
    prepare,
)


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


def test_parse_digit_struct_orders_by_left_and_maps_ten_to_zero(raw_svhn):
    recs = parse_digit_struct(raw_svhn / "train" / "digitStruct.mat")
    assert [r["name"] for r in recs] == ["1.png", "2.png", "3.png"]
    assert [r["label"] for r in recs] == ["19", "0", "205"]
    assert recs[2]["boxes"][0][0] == 4


def test_hash_split_is_deterministic_stratified_and_disjoint():
    recs = [{"name": f"{i}.png", "label": "1" * (1 + i % 3)} for i in range(300)]
    tr, va = hash_split(recs)
    assert hash_split(recs) == (tr, va)
    assert not set(tr) & set(va) and len(tr) + len(va) == 300
    assert len(va) == 30
    for L in (1, 2, 3):
        assert sum(len(recs[i]["label"]) == L for i in va) == 10


def test_encode_decode_roundtrip():
    ids = encode("907", 6)
    assert len(ids) == 7 and ids[3] == EOS and ids[4:] == [PAD] * 3
    assert decode(ids) == "907"


def test_prepare_and_split_cache(raw_svhn, tmp_path):
    audit = prepare(raw_svhn, tmp_path / "cache", height=64, width=256)
    assert audit["splits"]["test"]["n_images"] == 3
    assert audit["splits"]["train"]["n_images"] + audit["splits"]["val"]["n_images"] == 3
    s = SvhnSplit(tmp_path / "cache", "test", max_len=6)
    assert s.images.shape == (3, 3, 64, 256) and s.images.dtype == torch.uint8
    assert s.labels == ["19", "0", "205"]
    labels = json.loads((tmp_path / "cache" / "test_labels.json").read_text())
    assert labels[0]["dir"] == "test"
    aug = {
        "max_shift": 0.06,
        "scale": [0.9, 1.1],
        "max_rotation": 5.0,
        "brightness": 0.2,
        "contrast": 0.2,
    }
    g = torch.Generator().manual_seed(0)
    b1 = s.batch(torch.arange(3), aug, g)
    b2 = s.batch(torch.arange(3), aug, torch.Generator().manual_seed(0))
    torch.testing.assert_close(
        b1.images, b2.images
    )  # augmentation is reproducible from the generator
    assert b1.tokens.shape == (3, 7)


def test_sequence_report_hand_example():
    targets = ["12", "345", "7"]
    preds = ["12", "34", "79"]  # exact, early EOS, extra digit
    r = sequence_report(targets, preds)
    # valid positions: 1,2 | 3,4,5 | 7 ; predictions: 1,2 | 3,4,0(missing) | 7
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


def test_ctc_helpers():
    tokens = torch.tensor([encode("10", 6), encode("7", 6)])
    tgt, lens = tokens_to_ctc_targets(tokens)
    assert tgt.tolist() == [2, 1, 8] and lens.tolist() == [2, 1]
    logits = torch.full((1, 6, 11), -9.0)
    for t, c in enumerate([2, 2, 0, 2, 1, 1]):  # "1", "1" after blank, "0"
        logits[0, t, c] = 0
    assert ctc_greedy(logits) == ["110"]
    assert greedy_strings(
        torch.nn.functional.one_hot(torch.tensor([[4, 3, EOS, 5]]), 13).float()
    ) == ["10"]


@pytest.mark.parametrize("cls", [AttentionRecognizer, PooledRecognizer, CTCRecognizer])
def test_recognizer_shapes_and_backward(cls):
    from dl_final_prj.svhn import OcrBatch

    torch.manual_seed(0)
    tokens = torch.tensor([encode("123", 6), encode("4", 6)])
    b = OcrBatch(torch.randn(2, 3, 64, 256), tokens, torch.arange(2))
    m = cls(pretrained=False) if cls is CTCRecognizer else cls(6, pretrained=False)
    logits, aux = m(b, teacher_forcing=1.0)
    if cls is CTCRecognizer:
        assert logits.shape == (2, 32, 11)
        loss = ctc_loss(logits, tokens)
    else:
        assert logits.shape == (2, 7, 13)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 13), tokens.reshape(-1), ignore_index=PAD
        )
        if cls is AttentionRecognizer:
            assert aux["attention"].shape == (2, 7, 64)
    loss.backward()
    assert torch.isfinite(loss)
