"""Secondary Task 3 decodings: n-best lists, MBR, length and forward-backward rescoring,
right-to-left targets, validation-only selection and scripts/rescore_task3.py end to end."""

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from dl_final_prj.ocr_models import (
    LAMBDA_GRID,
    AttentionGRURecognizer,
    GRUDecoderRecognizer,
    NBest,
    PooledGRURecognizer,
    bidir_rescore,
    build,
    exact_by_lambda,
    length_logp_of,
    length_rescore,
    mbr_edit,
    select_lambda,
)
from dl_final_prj.svhn import DIGIT0, EOS, MAX_LEN, VOCAB, SvhnSplit, decode, encode, prepare
from test_svhn import write_raw
from test_svhn_whole import write_raw_blocks

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = load_script("rescore_task3")


# --- a hand-made autoregressive model ------------------------------------------------------


class TableModel(GRUDecoderRecognizer):
    """A GRU-decoder interface whose next-token distribution is a table over the emitted
    prefix (in the model's own order). The image is ignored. "$" is EOS. A prefix missing from
    the table is uniform, a symbol missing from a listed prefix gets probability ~1e-6."""

    def __init__(self, table: dict[str, dict[str, float]], length_logits=None, reverse=False):
        nn.Module.__init__(self)
        self.max_len, self.attn, self.table = MAX_LEN, None, table
        self.reverse_target = reverse
        self.length_head = nn.Linear(1, MAX_LEN)
        with torch.no_grad():
            self.length_head.weight.zero_()
            self.length_head.bias.copy_(torch.tensor(length_logits or [0.0] * MAX_LEN))

    def encode(self, images):
        b = images.shape[0]
        return torch.zeros(b, 1, 1), torch.zeros(b, 1), None

    def init_state(self, mean):  # h[:, 0, 0] = step, h[:, 0, 1:] = emitted tokens
        return torch.zeros(mean.shape[0], 1, self.max_len + 2)

    def step(self, prev, h, feats, kp, mean):
        k = int(h[0, 0, 0])
        h = h.clone()
        if k > 0:
            h[:, 0, k] = prev.float()
        h[:, 0, 0] += 1
        logits = torch.full((h.shape[0], VOCAB), -30.0)
        for i in range(h.shape[0]):
            toks = [int(t) for t in h[i, 0, 1 : k + 1]]
            prefix = "".join(str(t - DIGIT0) for t in toks if t >= DIGIT0)
            if any(t < DIGIT0 for t in toks) or prefix not in self.table:
                logits[i, DIGIT0:] = 0.0
                logits[i, EOS] = 0.0
                continue
            for sym, p in self.table[prefix].items():
                logits[i, EOS if sym == "$" else DIGIT0 + int(sym)] = math.log(p)
        return logits, h, None


L2R_TABLE = {"": {"1": 1.0}, "1": {"2": 0.55, "3": 0.45}, "12": {"$": 1.0}, "13": {"$": 1.0}}
# right-to-left: emits the reversed string, strongly prefers "31" (read "13")
R2L_TABLE = {"": {"2": 0.1, "3": 0.9}, "2": {"1": 1.0}, "3": {"1": 1.0}}
R2L_TABLE |= {"21": {"$": 1.0}, "31": {"$": 1.0}}


def images(n: int = 1) -> torch.Tensor:
    return torch.zeros(n, 3, 54, 54)


def small_attention(seed: int = 0, **kw) -> AttentionGRURecognizer:
    torch.manual_seed(seed)
    m = AttentionGRURecognizer(hidden=32, attn_dim=16, emb_dim=8, **kw).eval()
    with torch.no_grad():  # a sharper output layer gives hypotheses of different lengths
        m.out.weight.mul_(8.0)
    return m


def nbest_of(model, x: torch.Tensor, width: int, split: str = "val") -> NBest:
    with torch.no_grad():
        hyps, lp = model.nbest(x, width)
    return NBest.from_hyps(split, width, hyps, lp.numpy())


# --- n-best lists ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", [AttentionGRURecognizer, PooledGRURecognizer])
@pytest.mark.parametrize("width", [1, 3, 5, 10])
def test_nbest_is_ordered_and_its_first_entry_is_the_beam(model_cls, width):
    torch.manual_seed(width)
    model = model_cls(hidden=32, emb_dim=8).eval()
    with torch.no_grad():
        model.out.weight.mul_(8.0)
    x = torch.randn(6, 3, 54, 54)
    with torch.no_grad():
        ids, _ = model.beam(x, width)
        hyps, len_lp = model.nbest(x, width)
        beam_preds, _ = model.predict(x, f"beam{width}")
    assert beam_preds == [decode(r) for r in ids.tolist()]
    assert [h[0][0] for h in hyps] == beam_preds
    assert len_lp.shape == (6, MAX_LEN)
    assert torch.allclose(len_lp.exp().sum(-1), torch.ones(6))
    for row in hyps:
        assert 1 <= len(row) <= width
        preds = [p for p, *_ in row]
        assert len(set(preds)) == len(preds)  # beams are distinct hypotheses
        norms = [n for _, _, n, _ in row]
        assert norms == sorted(norms, reverse=True)
        for p, lp, n, nt in row:
            assert len(p) <= MAX_LEN and nt == len(p) + 1
            assert n == pytest.approx(lp / nt, rel=1e-6)
    with torch.no_grad():  # teacher forcing reproduces the beam's summed log-probabilities
        scored = model.score_strings(x, [[p for p, *_ in row] for row in hyps])
    for row, s in zip(hyps, scored, strict=True):
        assert np.allclose([lp for _, lp, _, _ in row], s, atol=1e-4)


def test_nbest_from_hyps_pads_and_round_trips():
    nb = NBest.from_hyps("val", 3, [[("12", -1.0, -1 / 3, 3)], [("4", -0.5, -0.25, 2), ("", -3.0, -3.0, 1)]], np.zeros((2, 5)))  # fmt: skip
    assert nb.logp.shape == (2, 3) and nb.valid.sum() == 3
    assert nb.hyps() == [["12"], ["4", ""]] and nb.top1() == ["12", "4"]
    assert nb.pick(np.zeros((2, 3))) == ["12", "4"]
    with pytest.raises(ValueError):
        nb.r2l_matrix([[0.0], [0.0]])  # image 1 has two hypotheses


# --- MBR --------------------------------------------------------------------------------------


def hand_nbest(preds, logp, split="val", len_logp=None) -> NBest:
    hyps = [
        [(p, lp, lp / (len(p) + 1), len(p) + 1) for p, lp in zip(ps, ls, strict=True)]
        for ps, ls in zip(preds, logp, strict=True)
    ]
    ll = np.zeros((len(preds), MAX_LEN)) if len_logp is None else np.asarray(len_logp, float)
    return NBest.from_hyps(split, max(len(p) for p in preds), hyps, ll)


def test_mbr_picks_the_edit_distance_medoid():
    # posterior 0.4/0.3/0.3, distances 12-13:1, 12-93:2, 13-93:1 -> risks 0.9, 0.7, 1.1
    nb = hand_nbest([["12", "13", "93"]], [[math.log(0.4), math.log(0.3), math.log(0.3)]])
    assert nb.top1() == ["12"]
    assert mbr_edit(nb) == ["13"]


def test_mbr_equals_map_when_one_hypothesis_dominates():
    nb = hand_nbest(
        [["12", "13", "93"], ["5", "55", "555"]], [[0.0, -40.0, -45.0], [-0.1, -50.0, -60.0]]
    )
    assert mbr_edit(nb) == nb.top1() == ["12", "5"]


def test_mbr_ignores_padding():
    nb = NBest.from_hyps("val", 3, [[("7", -0.1, -0.05, 2)]], np.zeros((1, MAX_LEN)))
    assert mbr_edit(nb) == ["7"]


# --- length rescoring -------------------------------------------------------------------------


def test_length_rescore_lambda_zero_is_the_primary_beam():
    model = small_attention()
    x = torch.randn(8, 3, 54, 54)
    nb = nbest_of(model, x, 5)
    with torch.no_grad():
        primary, _ = model.predict(x, "beam5")
    assert length_rescore(nb, 0.0) == nb.top1() == primary


def test_length_rescore_uses_the_length_head():
    lens = [[-9.0, -3.0, -0.1, -9.0, -9.0]]  # the head is sure of 3 digits
    nb = hand_nbest([["12", "123"]], [[-1.5, -2.4]], len_logp=lens)
    assert length_rescore(nb, 0.0) == ["12"]  # -0.5 vs -0.6
    assert length_rescore(nb, 1.0) == ["123"]
    assert np.allclose(length_logp_of(nb)[0], [-3.0, -0.1])


def test_length_rescore_never_picks_an_empty_string_once_lambda_is_positive():
    nb = hand_nbest([["", "4"]], [[-0.1, -3.0]])
    assert length_rescore(nb, 0.0) == [""] and length_rescore(nb, 0.1) == ["4"]


def test_length_head_of_a_real_model_feeds_the_nbest():
    model = small_attention()
    x = torch.randn(3, 3, 54, 54)
    nb = nbest_of(model, x, 3)
    with torch.no_grad():
        _, length_logits, _ = model(x, torch.zeros(3, MAX_LEN + 1, dtype=torch.long))
    assert np.allclose(nb.len_logp, length_logits.log_softmax(-1).numpy(), atol=1e-6)


# --- right-to-left targets ------------------------------------------------------------------


def write_tiny_cache(d: Path, labels: list[str], split: str = "val") -> None:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.save(d / f"{split}_images.npy", rng.integers(0, 255, (len(labels), 64, 64, 3), np.uint8))
    recs = [{"name": f"{i}.png", "dir": "train", "label": s} for i, s in enumerate(labels)]
    (d / f"{split}_labels.json").write_text(json.dumps(recs))


def test_reverse_target_encodes_reversed_training_targets(tmp_path):
    labels = ["123", "4", "90", "56781"]
    write_tiny_cache(tmp_path, labels)
    fwd = SvhnSplit(tmp_path, "val")
    bwd = SvhnSplit(tmp_path, "val", reverse=True)
    assert fwd.labels == bwd.labels == labels  # labels (and metrics) stay left-to-right
    assert torch.equal(fwd.lengths, bwd.lengths)
    assert bwd.tokens.tolist() == [encode(s[::-1]) for s in labels]
    assert fwd.tokens.tolist() == [encode(s) for s in labels]
    b = bwd.batch(torch.arange(4))
    assert b.tokens.tolist() == [encode(s[::-1]) for s in labels]


def test_reverse_target_decodes_left_to_right():
    r2l = TableModel(R2L_TABLE, reverse=True)  # emits "3", "1", EOS
    for mode in ("greedy", "beam1", "beam3"):
        assert r2l.predict(images(), mode)[0] == ["13"]
    hyps, _ = r2l.nbest(images(), 3)
    assert hyps[0][0][0] == "13"
    # teacher forcing scores the reversed string: log P_R2L("31") for the reported "13"
    s = r2l.score_strings(images(), [["13", "12"]])[0]
    assert s[0] == pytest.approx(math.log(0.9), abs=1e-4)
    assert s[1] == pytest.approx(math.log(0.1), abs=1e-4)
    fwd = TableModel(L2R_TABLE)
    assert fwd.predict(images(), "greedy")[0] == ["12"]


def test_build_reverse_target():
    m = build("attention", {"hidden": 16, "emb_dim": 8}, reverse_target=True)
    assert isinstance(m, GRUDecoderRecognizer) and m.reverse_target
    assert not build("attention", {"hidden": 16, "emb_dim": 8}).reverse_target
    plain = build("attention", {"hidden": 16, "emb_dim": 8})
    assert set(m.state_dict()) == set(plain.state_dict())  # no extra parameter or buffer
    for kind in ("ctc", "multihead"):
        with pytest.raises(ValueError):
            build(kind, reverse_target=True)


# --- forward-backward rescoring -----------------------------------------------------------------


def test_bidir_lambda_zero_is_the_primary_beam():
    fwd, bwd = small_attention(0), small_attention(1)
    bwd.reverse_target = True
    x = torch.randn(8, 3, 54, 54)
    nb = nbest_of(fwd, x, 5)
    with torch.no_grad():
        r2l = nb.r2l_matrix(bwd.score_strings(x, nb.hyps()))
        primary, _ = fwd.predict(x, "beam5")
    assert np.isfinite(r2l[nb.valid]).all()
    assert bidir_rescore(nb, r2l, 0.0) == nb.top1() == primary


def test_bidir_r2l_model_flips_the_choice():
    fwd, bwd = TableModel(L2R_TABLE), TableModel(R2L_TABLE, reverse=True)
    nb = nbest_of(fwd, images(), 3)
    assert nb.top1() == ["12"] and set(nb.hyps()[0][:2]) == {"12", "13"}
    r2l = nb.r2l_matrix(bwd.score_strings(images(), nb.hyps()))
    assert bidir_rescore(nb, r2l, 0.0) == ["12"]
    assert bidir_rescore(nb, r2l, 1.0) == ["13"]
    # (ln .55 + lam ln .1) / 3 < (ln .45 + lam ln .9) / 3 once lam > 0.0911
    assert bidir_rescore(nb, r2l, 0.05) == ["12"] and bidir_rescore(nb, r2l, 0.1) == ["13"]


# --- validation-only selection --------------------------------------------------------------


def threshold_nbest(split: str, thresholds: list[float]) -> NBest:
    """Image i: hyps A="1" (primary score -1) and B="12" (score -1 - t_i), length head with
    log P(2) - log P(1) = 1, so length_rescore picks B exactly when lambda > t_i."""
    hyps, lens = [], []
    for t in thresholds:
        hyps.append([("1", -2.0, -1.0, 2), ("12", 3 * (-1.0 - t), -1.0 - t, 3)])
        lens.append([-2.0, -1.0, -9.0, -9.0, -9.0])
    return NBest.from_hyps(split, 2, hyps, np.array(lens))


def test_lambda_selection_uses_validation_only(monkeypatch):
    """val is best at lambda 0.5 (image 1 needs > 0.3, image 2 needs < 0.6), test only at
    lambda 2. The frozen validation lambda must be the one applied to test."""
    val_nb, val_labels = threshold_nbest("val", [0.3, 0.6]), ["12", "1"]
    test_nb, test_labels = threshold_nbest("test", [1.7]), ["12"]
    val = rs.SplitInputs(val_nb, val_labels)
    test = rs.SplitInputs(test_nb, test_labels)

    table = exact_by_lambda(val_nb, val_labels, lambda lam: length_rescore(val_nb, lam))
    assert table == {0.0: 0.5, 0.1: 0.5, 0.25: 0.5, 0.5: 1.0, 0.75: 0.5, 1.0: 0.5, 1.5: 0.5, 2.0: 0.5}  # fmt: skip
    test_table = {lam: float(length_rescore(test_nb, lam) == test_labels) for lam in LAMBDA_GRID}
    assert max(test_table, key=lambda k: test_table[k]) == 2.0  # test would choose 2.0

    calls = []
    real_stage = rs.validation_stage
    monkeypatch.setattr(
        rs, "validation_stage", lambda v, g: calls.append(v.nb.split) or real_stage(v, g)
    )

    def get_test(i):
        calls.append("test")
        return test

    out = rs.rescore_core([val], get_test)[0]
    assert calls == ["val", "test"]
    assert out["lambdas"]["length_rescore"] == 0.5
    assert out["preds"]["test"]["length_rescore"] == ["1"]  # wrong on test, as frozen at 0.5
    assert out["preds"]["test"]["beam_primary"] == ["1"]
    assert "bidir_rescore" not in out["preds"]["val"]

    with pytest.raises(ValueError):
        rs.validation_stage(test, LAMBDA_GRID)
    with pytest.raises(ValueError):
        exact_by_lambda(test_nb, test_labels, lambda lam: length_rescore(test_nb, lam))
    with pytest.raises(ValueError):
        select_lambda(test_table, split="test")
    with pytest.raises(ValueError):
        rs.rescore_core([val], lambda i: val)  # a "test" stage fed with validation data
    with pytest.raises(ValueError):
        rs.rescore_core([test], get_test)


def test_select_lambda_ties_to_the_smaller_lambda():
    assert select_lambda({0.0: 0.5, 0.5: 0.7, 1.0: 0.7}) == 0.5
    assert select_lambda({2.0: 0.1, 0.0: 0.1}) == 0.0
    with pytest.raises(ValueError):
        select_lambda({})


def test_pooled_lambda_is_the_mean_over_seeds():
    t1 = {"length_rescore": {0.0: 0.5, 1.0: 0.9, 2.0: 0.6}}
    t2 = {"length_rescore": {0.0: 0.5, 1.0: 0.4, 2.0: 0.9}}
    lam, mean = rs.pooled_lambdas([t1, t2])["length_rescore"]
    assert lam == 2.0 and mean == pytest.approx({0.0: 0.5, 1.0: 0.65, 2.0: 0.75})


def test_pooled_selection_through_the_core():
    a = rs.SplitInputs(threshold_nbest("val", [0.3, 0.3]), ["12", "12"])  # lambda >= 0.5
    b = rs.SplitInputs(threshold_nbest("val", [1.2, 1.2]), ["1", "1"])  # lambda <= 1.0
    t = rs.SplitInputs(threshold_nbest("test", [0.3]), ["12"])
    out = rs.rescore_core([a, b], lambda i: t)
    # each seed alone would pick a different lambda (kept as the per-run diagnostic)
    assert out[0]["per_run_lambdas"]["length_rescore"] == 0.5
    assert out[1]["per_run_lambdas"]["length_rescore"] == 0.0
    assert out[1]["per_run_preds"]["test"]["length_rescore"] == ["1"]
    # the reported lambda is the setting-level one, the same for every seed
    lam, _ = out[0]["setting"]["length_rescore"]
    assert lam == 0.5  # mean 1.0 at 0.5, 0.75 and 1.0
    for r in out:
        assert r["scope"] == "setting" and r["lambdas"]["length_rescore"] == 0.5
        assert r["preds"]["test"]["length_rescore"] == ["12"]
    assert out[1]["preds"]["val"]["length_rescore"] == ["1", "1"]  # thresholds 1.2 > 0.5


def test_single_run_scope_reports_the_run_lambda():
    a = rs.SplitInputs(threshold_nbest("val", [0.3, 0.3]), ["12", "12"])
    out = rs.rescore_core([a], lambda i: rs.SplitInputs(threshold_nbest("test", [0.3]), ["12"]))
    assert out[0]["scope"] == "single_run" and out[0]["setting"] is None
    assert out[0]["lambdas"] == out[0]["per_run_lambdas"] and "per_run_preds" not in out[0]


def test_setting_width_is_the_mean_over_seeds():
    """Seeds 0 and 1 select 3 and 10 on their own; the setting-level width is 5."""
    s0 = {"3": 0.80, "5": 0.79, "10": 0.70}
    s1 = {"3": 0.60, "5": 0.70, "10": 0.71}
    assert {rs.select_width({int(k): v for k, v in t.items()}) for t in (s0, s1)} == {3, 10}
    width, mean = rs.setting_width([s0, s1])
    assert width == 5 and mean == pytest.approx({3: 0.70, 5: 0.745, 10: 0.705})
    assert rs.setting_width([{"3": 0.5, "5": 0.5, "10": 0.4}])[0] == 3  # ties to the smaller
    with pytest.raises(ValueError):
        rs.setting_width([s0, {"3": 0.5, "5": 0.5}])
    with pytest.raises(ValueError):
        rs.setting_width([{}])


# --- configs ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r2l", "l2r", "model"),
    [
        ("phase1_task3_r2l.yaml", "phase1_task3.yaml", "cnn_attention_gru"),
        ("phase1_task3_whole_r2l.yaml", "phase1_task3_whole.yaml", "cnn_attention_gru_whole"),
    ],
)
def test_r2l_configs_mirror_the_l2r_configs(r2l, l2r, model):
    back = yaml.safe_load((ROOT / "configs" / r2l).read_text())
    fwd = yaml.safe_load((ROOT / "configs" / l2r).read_text())
    assert back.pop("reverse_target") is True and "reverse_target" not in fwd
    assert back.pop("models") == {f"{model}_r2l": fwd["models"][model]}
    fwd.pop("models")
    assert back == fwd


# --- end to end: run_ocr.py and rescore_task3.py ---------------------------------------------


@pytest.fixture(scope="module")
def crop_cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("svhn")
    raw = write_raw(tmp / "raw", {"train": 40, "extra": 40, "test": 12})
    prepare(raw, tmp / "cache", size=64, workers=1)
    return tmp / "cache"


ENV = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "4"}


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    out = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
    if check and out.returncode != 0:
        raise AssertionError(out.stderr[-3000:])
    return out


def train(model: str, config: str, cache: Path, out: Path, *extra: str, seed: int = 0):
    cmd = [sys.executable, "scripts/run_ocr.py", "--model", model, "--seed", str(seed), "--smoke"]
    cmd += ["--config", str(ROOT / "configs" / config), "--cache", str(cache), "--out", str(out)]
    return run([*cmd, *extra])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


REPORT_KEYS = {"exact_match", "char_accuracy", "length_accuracy", "norm_edit_distance", "RMSE"}


LAMBDA_KEYS = ("length_rescore", "bidir_rescore")


def check_rescoring(
    run_dir: Path, methods: set[str], cache: Path, suffix: str = "", scope: str = "single_run"
) -> dict:
    res = json.loads((run_dir / "rescoring.json").read_text())
    assert set(res["methods"]) == methods
    assert res["cache_suffix"] == suffix and res["smoke"] is True
    assert res["lambda_grid"] == list(LAMBDA_GRID)
    assert res["selection_scope"] == scope
    assert res["plan_conformant"] is (scope == "setting" and res["selection_seeds"] == [0, 1, 2])
    metrics = json.loads((run_dir / "metrics.json").read_text())
    nb = res["nbest"]
    assert nb["run_width"] == metrics["decoding"]["width"]
    assert nb["width"] == nb["primary_width"]
    if scope == "single_run":
        assert nb["primary_width"] == nb["run_width"] and nb["setting_width"] is None
    else:
        assert nb["setting_width"]["value"] == nb["primary_width"]
    _, meta = rs.ro.read_nbest(run_dir / "nbest_val.json.gz")
    assert meta["width"] == nb["width"]
    same = nb["width"] == nb["run_width"]
    expect = {"val": 1.0, "test": 1.0} if same else {"val": None, "test": None}
    assert nb["primary_agreement"] == expect
    labels = {
        s: [r["label"] for r in json.loads((cache / f"{s}{suffix}_labels.json").read_text())]
        for s in ("val", "test")
    }

    def check_preds(files: dict, e: dict) -> None:
        for s in ("val", "test"):
            rows = json.loads((run_dir / files[s]).read_text())
            assert [r["target"] for r in rows] == labels[s]
            exact = np.mean([r["pred"] == r["target"] for r in rows])
            assert exact == pytest.approx(e[s]["exact_match"])

    for m, e in res["methods"].items():
        assert set(e["val"]) >= REPORT_KEYS and set(e["test"]) >= REPORT_KEYS
        if m in LAMBDA_KEYS:
            hp = e["hyperparameter"]
            assert hp["selected_on"] == "val" and hp["value"] in LAMBDA_GRID
            assert set(hp["val_exact_by_lambda"]) == {str(x) for x in LAMBDA_GRID}
            own = hp["val_exact_by_lambda"]
            table = hp["mean_val_exact_by_lambda"] if scope == "setting" else own
            assert table[str(hp["value"])] == max(table.values())
            if scope == "setting":
                pr = e["per_run"]
                assert own[str(pr["value"])] == max(own.values())
                check_preds(pr["pred_files"], pr)
            else:
                assert "per_run" not in e and "mean_val_exact_by_lambda" not in hp
        else:
            assert e["hyperparameter"] is None
        if m != "beam_primary":
            assert e["pred_files"] == {s: f"pred_{s}_{m}.json" for s in ("val", "test")}
            check_preds(e["pred_files"], e)
    if same:
        for s in ("val", "test"):
            assert res["methods"]["beam_primary"][s] == pytest.approx(metrics[s], nan_ok=True)
    return res


def check_setting(run_dirs: list[Path]) -> None:
    """One width and one lambda per method for the whole setting (every seed)."""
    res = [json.loads((d / "rescoring.json").read_text()) for d in run_dirs]
    assert len({r["nbest"]["width"] for r in res}) == 1
    for m in LAMBDA_KEYS:
        if m in res[0]["methods"]:
            assert len({r["methods"][m]["hyperparameter"]["value"] for r in res}) == 1
            means = {json.dumps(r["methods"][m]["hyperparameter"]["mean_val_exact_by_lambda"]) for r in res}  # fmt: skip
            assert len(means) == 1


def test_end_to_end_l2r_r2l_rescoring(crop_cache, tmp_path):
    out = tmp_path / "results"
    train("cnn_attention_gru", "phase1_task3.yaml", crop_cache, out, "--nbest")
    train("cnn_attention_gru_r2l", "phase1_task3_r2l.yaml", crop_cache, out)
    root = out / "smoke"
    l2r = root / "runs" / "task3" / "cnn_attention_gru" / "seed0"
    r2l = root / "runs" / "task3" / "cnn_attention_gru_r2l" / "seed0"

    # the right-to-left run: reversed training, left-to-right reports
    spec = json.loads((r2l / "spec.json").read_text())
    assert spec["reverse_target"]["enabled"] is True
    assert "reverse_target" not in json.loads((l2r / "spec.json").read_text())
    assert json.loads((r2l / "fingerprint.json").read_text())["config"]["reverse_target"] is True
    test_labels = [r["label"] for r in json.loads((crop_cache / "test_labels.json").read_text())]
    assert [r["target"] for r in json.loads((r2l / "pred_test.json").read_text())] == test_labels

    # --nbest wrote the lists of the primary beam, whose rank 1 is pred_*.json
    rs_ro = rs.ro
    for s in ("val", "test"):
        nb, meta = rs_ro.read_nbest(l2r / f"nbest_{s}.json.gz")
        assert meta["width"] == json.loads((l2r / "metrics.json").read_text())["decoding"]["width"]
        assert nb.top1() == [r["pred"] for r in json.loads((l2r / f"pred_{s}.json").read_text())]
    assert not (r2l / "nbest_val.json.gz").exists()

    before = {
        f: digest(l2r / f) for f in ("metrics.json", "spec.json", "ckpt.pt", "pred_test.json")
    }
    before_r2l = {f.name: digest(f) for f in r2l.iterdir()}
    cmd = [sys.executable, "scripts/rescore_task3.py", "--root", str(root)]
    cmd += ["--model", "cnn_attention_gru", "--r2l-model", "cnn_attention_gru_r2l", "--seeds", "0"]
    single = run([*cmd, "--cache", str(crop_cache)])
    assert "single-run selection" in single.stderr  # warned: not the plan's rule
    all4 = {"beam_primary", "mbr_edit", "length_rescore", "bidir_rescore"}
    res = check_rescoring(l2r, all4, crop_cache)
    assert res["nbest"]["generated_now"] == {"val": False, "test": False}  # reused
    assert res["r2l_run_dir"] == str(r2l) and res["setting"] == "oracle_crop"
    assert before == {f: digest(l2r / f) for f in before}  # primary outputs untouched
    assert before_r2l == {f.name: digest(f) for f in r2l.iterdir()}  # R2L run untouched

    # the R2L run cannot be rescored as a left-to-right run, nor paired the wrong way round
    bad = run([sys.executable, "scripts/rescore_task3.py", "--l2r", str(r2l)], check=False)
    assert bad.returncode != 0 and "right-to-left" in bad.stderr
    bad = run(
        [sys.executable, "scripts/rescore_task3.py", "--l2r", str(l2r), "--r2l", str(l2r)],
        check=False,
    )
    assert bad.returncode != 0 and "reverse_target" in bad.stderr

    # two seeds: setting-level width and lambdas reported, per-run lambdas as diagnostics
    train("cnn_attention_gru", "phase1_task3.yaml", crop_cache, out, seed=1)
    train("cnn_attention_gru_r2l", "phase1_task3_r2l.yaml", crop_cache, out, seed=1)
    cmd[-1] = "0"
    run([*cmd, "1", "--cache", str(crop_cache)])
    dirs = [root / "runs" / "task3" / "cnn_attention_gru" / f"seed{s}" for s in (0, 1)]
    for s, run_dir in enumerate(dirs):
        res = check_rescoring(run_dir, all4, crop_cache, scope="setting")
        assert res["r2l_run_dir"].endswith(f"cnn_attention_gru_r2l/seed{s}")
        assert res["selection_seeds"] == [0, 1] and res["plan_conformant"] is False
        assert "per_run" not in res["methods"]["mbr_edit"]
    check_setting(dirs)


@pytest.fixture(scope="module")
def whole_cache(tmp_path_factory):
    from dl_final_prj.svhn import prepare_whole

    tmp = tmp_path_factory.mktemp("svhn_whole")
    raw = write_raw_blocks(tmp / "raw", {"train": 30, "extra": 30, "test": 9})
    prepare(raw, tmp / "cache", size=64, workers=1, chunk=7)
    prepare_whole(raw, tmp / "cache", size=64, chunk=7)
    return tmp / "cache"


def test_end_to_end_phase2_whole_image_without_r2l(whole_cache, tmp_path):
    """A Phase 2 token-mixer run in Setting B: n-best generated from ckpt.pt (no --nbest at
    training), MBR and length rescoring on the _full cache."""
    out = tmp_path / "results"
    train("t3_m0_whole", "phase2_task3_whole.yaml", whole_cache, out)
    run_dir = out / "smoke" / "runs" / "task3" / "t3_m0_whole" / "seed0"
    assert not (run_dir / "nbest_val.json.gz").exists()
    cmd = [sys.executable, "scripts/rescore_task3.py", "--l2r", str(run_dir)]
    run([*cmd, "--cache", str(whole_cache)])
    res = check_rescoring(
        run_dir, {"beam_primary", "mbr_edit", "length_rescore"}, whole_cache, "_full"
    )
    assert res["nbest"]["generated_now"] == {"val": True, "test": True}
    assert res["setting"] == "whole_image" and res["kind"] == "mamba_attention"
    assert (run_dir / "nbest_test.json.gz").exists()


def test_end_to_end_whole_image_forward_backward(whole_cache, tmp_path):
    """Setting B with R2L runs (the _full cache): two seeds through --root, then the same runs
    with forced per-run widths 3 and 10 whose setting-level width is 5."""
    out = tmp_path / "results"
    for seed in (0, 1):
        train("cnn_attention_gru_whole", "phase1_task3_whole.yaml", whole_cache, out, seed=seed)
        train(
            "cnn_attention_gru_whole_r2l",
            "phase1_task3_whole_r2l.yaml",
            whole_cache,
            out,
            seed=seed,
        )
    root = out / "smoke"
    task3 = root / "runs" / "task3"
    r2l = task3 / "cnn_attention_gru_whole_r2l" / "seed0"
    assert json.loads((r2l / "fingerprint.json").read_text())["config"]["cache_suffix"] == "_full"
    cmd = [sys.executable, "scripts/rescore_task3.py", "--root", str(root)]
    cmd += ["--model", "cnn_attention_gru_whole", "--r2l-model", "cnn_attention_gru_whole_r2l"]
    cmd += ["--seeds", "0", "1", "--cache", str(whole_cache)]
    run(cmd)
    all4 = {"beam_primary", "mbr_edit", "length_rescore", "bidir_rescore"}
    dirs = [task3 / "cnn_attention_gru_whole" / f"seed{s}" for s in (0, 1)]
    for s, run_dir in enumerate(dirs):
        res = check_rescoring(run_dir, all4, whole_cache, "_full", scope="setting")
        assert res["setting"] == "whole_image" and res["kind"] == "attention"
        assert res["r2l_run_dir"].endswith(f"cnn_attention_gru_whole_r2l/seed{s}")
    check_setting(dirs)

    # different per-run widths: every seed is rescored at the setting-level width
    forced = tmp_path / "forced"
    ignore = shutil.ignore_patterns("nbest_*", "rescoring.json")
    shutil.copytree(task3, forced / "runs" / "task3", ignore=ignore)
    tables = [{"3": 0.80, "5": 0.79, "10": 0.70}, {"3": 0.60, "5": 0.70, "10": 0.71}]
    for s, (table, w) in enumerate(zip(tables, (3, 10), strict=True)):
        path = forced / "runs" / "task3" / "cnn_attention_gru_whole" / f"seed{s}" / "metrics.json"
        m = json.loads(path.read_text())
        m["decoding"] |= {"val_by_width": table, "width": w}
        path.write_text(json.dumps(m))
    cmd[cmd.index("--root") + 1] = str(forced)
    run(cmd)
    for s in (0, 1):
        run_dir = forced / "runs" / "task3" / "cnn_attention_gru_whole" / f"seed{s}"
        res = check_rescoring(run_dir, all4, whole_cache, "_full", scope="setting")
        nb = res["nbest"]
        assert nb["width"] == nb["primary_width"] == 5 and nb["run_width"] == (3, 10)[s]
        assert nb["setting_width"]["mean_val_exact_by_width"] == pytest.approx(
            {"3": 0.70, "5": 0.745, "10": 0.705}
        )
        assert nb["primary_agreement"] == {"val": None, "test": None}
        assert nb["generated_now"] == {"val": True, "test": True}  # rebuilt at width 5
