import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dl_final_prj.baselines import naive_predictions, ridge_feature_names, ridge_predictions
from dl_final_prj.electricity import (
    CALENDAR,
    INPUT_CHANNELS,
    TARGET,
    TARGET_FEATURES,
    ElectricityData,
    calendar_features,
    dst_flags,
    prepare,
    ridge_design,
    ridge_window_design,
    split_bounds,
    stable_clients,
)
from dl_final_prj.metrics import horizon_report, regression_report
from dl_final_prj.ts_models import BiRNNForecaster, Seq2SeqForecaster

ROOT = Path(__file__).resolve().parents[1]
T2 = (1, 2, 3, 4, 5)


def ys(d: ElectricityData, rows) -> np.ndarray:
    """Train-standardized target at the given rows."""
    return (d.y_raw[np.asarray(rows)] - d.y_mean) / d.y_std


def test_regression_report_hand_example():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 3.0, 2.0, 4.0])
    r = regression_report(y, p)
    # residual SS 2, total SS 5
    assert r["R2"] == pytest.approx(1 - 2 / 5)
    assert r["NSE"] == r["R2"]
    assert r["MAE"] == pytest.approx(0.5)
    assert r["RMSE"] == pytest.approx(np.sqrt(0.5))


def test_horizon_report_pooled_and_macro():
    y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    p = y + np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 3.0]])
    r = horizon_report(y, p)
    assert r["pooled"]["MAE"] == pytest.approx(4 / 6)
    assert r["h1"]["MAE"] == pytest.approx(1 / 3)
    assert r["h2"]["MAE"] == pytest.approx(1.0)
    assert r["horizon_macro"]["MAE"] == pytest.approx((1 / 3 + 1) / 2)
    # the pooled R2 uses the mean of all six cells
    ss_tot = np.sum((y - y.mean()) ** 2)
    assert r["pooled"]["R2"] == pytest.approx(1 - 10 / ss_tot)


def test_stable_clients_uses_train_rows_only():
    v = np.ones((100, 3))
    v[:50, 1] = 0  # late start, excluded
    v[80:, 2] = 0  # zeros only after train end, so still stable
    assert stable_clients(v, ["a", "b", "c"], train_end=70) == ["a", "c"]


# --- calendar and DST ----------------------------------------------------------


def test_dst_flag_on_known_change_days():
    flagged = [
        "2011-03-27 01:00",
        "2011-03-27 02:00",
        "2011-10-30 02:00",
        "2012-03-25 01:00",
        "2014-10-26 02:00",
        "2014-10-26 01:00",
    ]
    not_flagged = [
        "2011-03-27 00:00",  # same day, other hours
        "2011-03-27 03:00",
        "2011-03-20 02:00",  # a Sunday, but not the last one of March
        "2011-10-29 02:00",  # the Saturday before the October change
        "2012-04-01 02:00",  # first Sunday of April
        "2013-06-30 02:00",  # last Sunday of a non-change month
    ]
    stamps = pd.DatetimeIndex(pd.to_datetime(flagged + not_flagged))
    np.testing.assert_array_equal(dst_flags(stamps), [True] * 6 + [False] * 6)
    cal = calendar_features(stamps)
    np.testing.assert_array_equal(cal[:, CALENDAR.index("dst_flag")], [1.0] * 6 + [0.0] * 6)
    # the full UCI hourly range has two change days per year, two flagged rows each
    full = pd.date_range("2011-01-01 01:00", "2015-01-01 00:00", freq="h")
    assert len(full) == 35064 and int(dst_flags(full).sum()) == 16


def test_calendar_features_known_values():
    stamps = pd.DatetimeIndex(pd.to_datetime(["2012-01-02 07:00", "2012-12-31 00:00"]))
    cal = calendar_features(stamps)
    assert cal.shape == (2, 7) and cal.dtype == np.float32
    # 2012-01-02 is a Monday (day_of_week 0) and day of year 2
    np.testing.assert_allclose(
        cal[0],
        [
            np.sin(2 * np.pi * 7 / 24),
            np.cos(2 * np.pi * 7 / 24),
            0.0,
            1.0,
            np.sin(2 * np.pi * 2 / 365.25),
            np.cos(2 * np.pi * 2 / 365.25),
            0.0,
        ],
        atol=1e-6,
    )
    # 2012-12-31 is a Monday and day 366 of the leap year
    np.testing.assert_allclose(cal[1, 4], np.sin(2 * np.pi * 366 / 365.25), atol=1e-6)


def test_hours_roundtrip_and_data_calendar(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, (2,))
    assert d.hours[0] == pd.Timestamp("2012-01-01 01:00")
    i = 30
    assert d.hours[i] == pd.Timestamp("2012-01-02 07:00")
    np.testing.assert_allclose(d.C.numpy()[i], calendar_features(d.hours[i : i + 1])[0])
    s = d.feature_summary()
    assert s["dst_flag_hours"] == ["2012-03-25 01:00:00", "2012-03-25 02:00:00"]
    assert s["dst_flag_rows"] == 2


def test_irregular_grid_is_rejected(synthetic_npz, tmp_path):
    z = dict(np.load(synthetic_npz))
    z["hours"] = np.delete(z["hours"], 100)
    z["values"] = np.delete(z["values"], 100, axis=0)
    np.savez(tmp_path / "gap.npz", **z)
    with pytest.raises(ValueError, match="regular"):
        ElectricityData(tmp_path / "gap.npz", 168, (2,))


# --- windows, target features and residuals -------------------------------------


@pytest.mark.parametrize("offsets", [(2,), T2])
def test_input_channel_alignment(synthetic_npz, offsets):
    d = ElectricityData(synthetic_npz, lookback=168, offsets=offsets)
    assert d.feature_names == list(INPUT_CHANNELS) and d.n_features == 11
    o = d.origins["val"][:7]
    b = d.batch(torch.as_tensor(o))
    assert b.x.shape == (7, 168, 11)
    x = b.x.numpy().astype(np.float64)
    for i, t in enumerate(o):
        np.testing.assert_allclose(x[i, -1, 0], ys(d, t), rtol=1e-5)
        np.testing.assert_allclose(x[i, -1, 1], ys(d, t - 24), rtol=1e-5)
        np.testing.assert_allclose(x[i, -1, 2], ys(d, t - 48), rtol=1e-5)
        np.testing.assert_allclose(x[i, -1, 3], ys(d, t - 168), rtol=1e-5)
        steps = np.arange(t - 167, t + 1)  # the whole window, oldest first
        np.testing.assert_allclose(x[i, :, 0], ys(d, steps), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(x[i, :, 3], ys(d, steps - 168), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(x[i, :, 4:], d.cal[steps], atol=1e-6)


@pytest.mark.parametrize("offsets", [(2,), T2])
def test_target_feature_alignment(synthetic_npz, offsets):
    d = ElectricityData(synthetic_npz, 168, offsets)
    assert d.n_target_feats == len(TARGET_FEATURES) == 10
    o = d.origins["test"][:6]
    b = d.batch(torch.as_tensor(o))
    assert b.target_feats.shape == (6, len(offsets), 10)
    assert b.cal_future.shape == (6, len(offsets), 7)
    tf = b.target_feats.numpy().astype(np.float64)
    for i, t in enumerate(o):
        for k, off in enumerate(offsets):
            tau = t + off
            np.testing.assert_allclose(tf[i, k, :7], d.cal[tau], atol=1e-6)
            np.testing.assert_allclose(
                tf[i, k, 7:], ys(d, [tau - 24, tau - 48, tau - 168]), rtol=1e-5
            )
            assert tau - 24 <= t  # target-side lags are observed at the origin
    torch.testing.assert_close(b.target_feats[..., :7], b.cal_future)


@pytest.mark.parametrize("offsets", [(2,), T2])
def test_residual_round_trip(synthetic_npz, offsets):
    d = ElectricityData(synthetic_npz, 168, offsets)
    for split in ("train", "val", "test"):
        o = d.origins[split]
        r = np.concatenate([b.y.numpy() for b in d.iterate(split, 500)])
        assert r.shape == (len(o), len(offsets))
        np.testing.assert_allclose(d.inverse(r, o), d.true_targets(split), rtol=1e-5)
    o = d.origins["val"][:5]
    b = d.batch(torch.as_tensor(o))
    rows = o[:, None] + np.array(offsets)
    raw_r = d.y_raw[rows] - d.y_raw[rows - 24]
    np.testing.assert_allclose(b.y.numpy(), (raw_r - d.r_mean) / d.r_std, rtol=1e-4, atol=1e-5)
    r0 = (d.y_raw[o] - d.y_raw[o - 24] - d.r_mean) / d.r_std
    np.testing.assert_allclose(b.r0.numpy(), r0, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(d.base_b24(o), d.y_raw[rows - 24])


def test_scalers_fitted_on_train_rows(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, (2,))
    lo, hi = d.bounds["train"]
    z = np.load(synthetic_npz)
    y = z["values"][:, list(z["names"]).index(TARGET)]
    assert d.target_scale()[0] == pytest.approx(y[lo:hi].mean())
    r = y[24:hi] - y[: hi - 24]
    assert d.residual_scale() == pytest.approx((r.mean(), r.std()), rel=1e-6)


@pytest.mark.parametrize("offsets", [(2,), T2])
def test_no_lookahead_and_split_membership(synthetic_npz, tmp_path, offsets):
    d = ElectricityData(synthetic_npz, 168, offsets)
    assert d.first_origin == 167 + 168
    for split, (lo, hi) in split_bounds(d.n).items():
        o = d.origins[split]
        assert (o >= 167 + 168).all()
        assert (o + min(offsets) >= lo).all() and (o + max(offsets) < hi).all()
    assert len(set(d.origins["val"]) & set(d.origins["test"])) == 0
    # Overwrite every row after a test origin. Inputs, target-side features and r(t) must not
    # change; the targets must.
    t = int(d.origins["test"][20])
    z = dict(np.load(synthetic_npz))
    z["values"] = z["values"].copy()
    z["values"][t + 1 :] += 1000.0  # test rows only, so the train scalers are unchanged
    np.savez(tmp_path / "future.npz", **z)
    d2 = ElectricityData(tmp_path / "future.npz", 168, offsets)
    b1, b2 = d.batch(torch.tensor([t])), d2.batch(torch.tensor([t]))
    torch.testing.assert_close(b1.x, b2.x)
    torch.testing.assert_close(b1.target_feats, b2.target_feats)
    torch.testing.assert_close(b1.r0, b2.r0)
    assert not torch.allclose(b1.y, b2.y)


def test_target_only_false_appends_clients(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, (2,), target_only=False)
    assert d.feature_names[:11] == list(INPUT_CHANNELS)
    assert d.feature_names[11:] == ["MT_000", "MT_001", "MT_002", "MT_003"]
    assert TARGET not in d.feature_names[11:] and "MT_999" not in d.feature_names
    tr = slice(*d.bounds["train"])
    np.testing.assert_allclose(d.X[tr, 11:].numpy().mean(0), 0.0, atol=1e-4)


# --- baselines ----------------------------------------------------------------


def test_naive_baselines(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, T2)
    p = naive_predictions(d, "test")
    o = d.origins["test"]
    np.testing.assert_allclose(p["persistence"][:, 4], d.y_raw[o])
    np.testing.assert_allclose(p["seasonal_daily"][:, 0], d.y_raw[o + 1 - 24])
    np.testing.assert_allclose(p["seasonal_weekly"][:, 4], d.y_raw[o + 5 - 168])


@pytest.mark.parametrize("kind", ["sparse", "window"])
def test_ridge_beats_persistence_on_periodic_series(synthetic_npz, kind):
    d = ElectricityData(synthetic_npz, 168, T2)
    preds, log = ridge_predictions(d, kind)
    y = d.true_targets("test")
    assert preds["test"].shape == y.shape
    rmse = lambda p: np.sqrt(np.mean((p - y) ** 2))  # noqa: E731
    assert rmse(preds["test"]) < rmse(naive_predictions(d, "test")["persistence"])
    assert log["alpha"] in [float(k) for k in log["alpha_grid_val_rmse"]]
    assert len(log["features"]) == log["n_features_per_horizon"]


def test_ridge_sparse_features_are_aligned_to_the_target_hour(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, T2)
    o = d.origins["test"][:10]
    assert len(ridge_feature_names("sparse")) == 13
    for h in d.offsets:
        x = ridge_design(d.y_raw, d.cal, o, h)
        assert x.shape == (10, 13)
        np.testing.assert_allclose(
            x[:, :3], np.column_stack([d.y_raw[o], d.y_raw[o - 1], d.y_raw[o - 2]])
        )
        np.testing.assert_allclose(x[:, 3], d.y_raw[o + h - 24])  # same hour yesterday
        np.testing.assert_allclose(x[:, 4], d.y_raw[o + h - 48])
        np.testing.assert_allclose(x[:, 5], d.y_raw[o + h - 168])  # same hour last week
        np.testing.assert_allclose(x[:, 6:], d.cal[o + h])
        assert (o + h - 24 <= o).all()  # only observed values


def test_ridge_window_features_are_aligned(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, T2)
    o = d.origins["val"][:8]
    names = ridge_feature_names("window")
    assert len(names) == 175 and names[0] == "y[t-167]" and names[167] == "y[t]"
    for h in d.offsets:
        x = ridge_window_design(d.y_raw, d.cal, o, h)
        assert x.shape == (8, 175)
        for i, t in enumerate(o):
            np.testing.assert_allclose(x[i, :168], d.y_raw[t - 167 : t + 1])
        np.testing.assert_allclose(x[:, 168:], d.cal[o + h])


# --- models -------------------------------------------------------------------


def test_models_shapes_and_attention(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, T2)
    b = d.batch(torch.as_tensor(d.origins["train"][:4]))
    m = Seq2SeqForecaster(d.n_features, horizon=5, hidden=16, attn_dim=8)
    p, aux = m(b)
    assert p.shape == (4, 5)
    w = aux["attention"]
    assert w.shape == (4, 5, 168)
    torch.testing.assert_close(w.sum(-1), torch.ones(4, 5))
    m2 = Seq2SeqForecaster(d.n_features, horizon=5, hidden=16, attention=False)
    assert m2(b)[0].shape == (4, 5) and m2(b)[1] == {}

    d1 = ElectricityData(synthetic_npz, 168, (2,))
    b1 = d1.batch(torch.as_tensor(d1.origins["train"][:3]))
    for att in (True, False):
        m1 = BiRNNForecaster(d1.n_features, hidden=8, attn_dim=8, head_dim=8, attention=att)
        p, aux = m1(b1)
        assert p.shape == (3, 1)
        assert ("attention" in aux) == att
        if att:
            assert aux["attention"].shape == (3, 168)
            torch.testing.assert_close(aux["attention"].sum(-1), torch.ones(3))
    # default sizes follow the plan: head input 128 + 128 + 10, final state 128
    m1 = BiRNNForecaster(11)
    assert m1.head[1].in_features == 266 and m1.head[1].out_features == 128
    assert BiRNNForecaster(11, attention=False).head[1].in_features == 138


def test_seq2seq_teacher_forcing_only_in_training(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, T2)
    b = d.batch(torch.as_tensor(d.origins["val"][:4]))
    b_other = d.batch(torch.as_tensor(d.origins["val"][:4]))
    b_other.y = b_other.y + 5.0  # different "true" residuals
    m = Seq2SeqForecaster(d.n_features, horizon=5, hidden=8, attn_dim=8, dropout=0.0)
    m.eval()
    with torch.no_grad():
        a, _ = m(b, teacher_forcing=1.0)
        c, _ = m(b, teacher_forcing=0.0)
        e, _ = m(b_other, teacher_forcing=1.0)
    torch.testing.assert_close(a, c)
    torch.testing.assert_close(a, e)  # eval never reads Batch.y
    m.train()  # dropout 0, so train mode is deterministic
    with torch.no_grad():
        f, _ = m(b, teacher_forcing=1.0)
        g, _ = m(b_other, teacher_forcing=1.0)
    torch.testing.assert_close(f[:, 0], g[:, 0])  # step 1 is fed r(t), not Batch.y
    assert not torch.allclose(f[:, 1:], g[:, 1:])  # later steps are fed the true residual


# --- prepare and end-to-end smoke runs ------------------------------------------


def test_prepare_from_raw_ld_file(tmp_path):
    hours = 24 * 40
    stamps = pd.date_range("2011-01-01 00:15", periods=hours * 4, freq="15min")
    rng = np.random.default_rng(0)
    vals = rng.uniform(1, 9, (len(stamps), 3))
    vals[: 4 * 24 * 5, 2] = 0  # MT_003 starts late
    raw = tmp_path / "LD2011_2014.txt"
    _write_ld_file(raw, stamps, vals)
    audit = prepare(raw, tmp_path / "proc")
    assert audit["raw_rows"] == hours * 4 and audit["hourly_rows"] == hours
    assert audit["stable_clients"] == ["MT_001", "MT_235"]
    assert audit["hourly_first"] == "2011-01-01 01:00:00"
    assert audit["dst_flag_rows"] == 0  # January and February only
    z = np.load(tmp_path / "proc" / "electricity_hourly.npz")
    np.testing.assert_allclose(z["values"][0], vals[:4].mean(0), rtol=1e-6)
    assert pd.DatetimeIndex(z["hours"])[-1] == pd.Timestamp(stamps[-1])


def _write_ld_file(path: Path, stamps: pd.DatetimeIndex, vals: np.ndarray) -> None:
    lines = ['"";"MT_001";"MT_235";"MT_003"']
    for s, v in zip(stamps, vals):
        lines.append(
            f'"{s:%Y-%m-%d %H:%M:%S}";' + ";".join(f"{x:.6f}".replace(".", ",") for x in v)
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(("day", "anomaly"), [("2012-03-25", 4.5), ("2013-10-27", 900.0)])
def test_dst_flag_marks_both_affected_hourly_rows(tmp_path, day, anomaly):
    # UCI pattern: on a change day the readings stamped 01:00..01:45 are anomalous. The
    # right-closed hourly means put 1 of them in the 01:00 row and 3 in the 02:00 row.
    start = pd.Timestamp(day) - pd.Timedelta(days=2) + pd.Timedelta(minutes=15)
    stamps = pd.date_range(start, periods=4 * 24 * 4, freq="15min")
    vals = np.full((len(stamps), 3), 460.0)
    bad = (stamps >= pd.Timestamp(f"{day} 01:00")) & (stamps <= pd.Timestamp(f"{day} 01:45"))
    assert bad.sum() == 4
    vals[bad] = anomaly
    raw = tmp_path / "LD2011_2014.txt"
    _write_ld_file(raw, stamps, vals)
    audit = prepare(raw, tmp_path / "proc")
    z = np.load(tmp_path / "proc" / "electricity_hourly.npz")
    hours = pd.DatetimeIndex(z["hours"])
    off = hours[np.abs(z["values"][:, 1] - 460.0) > 1e-6]
    assert [str(h) for h in off] == [f"{day} 01:00:00", f"{day} 02:00:00"]
    assert audit["dst_flag_hours"] == [str(h) for h in off]
    np.testing.assert_allclose(
        audit["target_stats"]["dst_flag_values"],
        [(3 * 460 + anomaly) / 4, (460 + 3 * anomaly) / 4],
        rtol=1e-6,
    )


def test_window_rule_counts(synthetic_npz):
    for offsets in [(2,), T2]:
        d = ElectricityData(synthetic_npz, 168, offsets)
        w = d.split_summary()["all_windows"]
        assert w == d.feature_summary()["windows"]
        assert w["first_origin_row"] == 335
        n_all = d.n - 335 - max(offsets)
        assert w["n_all_windows"] == n_all and w["formula_all_windows"].endswith(f"= {n_all}")
        # every split origin is a candidate origin, boundary-straddling ones are dropped
        assert w["n_split_windows"] == sum(len(o) for o in d.origins.values()) <= n_all


def _run(script: str, *args: str) -> None:
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]


@pytest.mark.parametrize(("task", "model"), [(1, "bigru_attention"), (2, "seq2seq_attention")])
def test_smoke_end_to_end(synthetic_npz, tmp_path, task, model):
    import json

    common = ["--task", str(task), "--npz", str(synthetic_npz), "--out", str(tmp_path), "--smoke"]
    _run("run_baselines.py", *common)
    _run("run_ts.py", *common, "--model", model, "--seed", "0")
    runs = tmp_path / "smoke" / "runs" / f"task{task}"
    names = ["persistence", "seasonal_daily", "seasonal_weekly", "ridge", "ridge_window"]
    for name in names:
        for seed in (0, 1, 2):
            assert (runs / name / f"seed{seed}" / "metrics.json").exists()
    d = ElectricityData(synthetic_npz, 168, (2,) if task == 1 else T2)
    run = runs / model / "seed0"
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 2  # --smoke runs 2 epochs
    spec = json.loads((run / "spec.json").read_text())
    assert spec["features"]["input_channels"] == list(INPUT_CHANNELS)
    assert spec["features"]["target_features"] == list(TARGET_FEATURES)
    assert spec["features"]["residual_scaler"]["std"] == pytest.approx(d.r_std)
    metrics = json.loads((run / "metrics.json").read_text())
    test = metrics["test"] if task == 1 else metrics["test"]["pooled"]
    assert set(test) == {"R2", "NSE", "MAE", "RMSE"}
    for split in ("val", "test"):
        z = np.load(run / f"pred_{split}.npz")
        np.testing.assert_array_equal(z["origin"], d.origins[split])
        assert z["hour"].dtype == np.dtype("datetime64[s]")
        np.testing.assert_allclose(z["y_true"], d.true_targets(split))
        np.testing.assert_allclose(z["base_b24"], d.base_b24(d.origins[split]))
        assert z["y_pred"].shape == z["y_true"].shape and np.isfinite(z["y_pred"]).all()
        assert ("attention" in z.files) == (split == "test")
    att = np.load(run / "pred_test.npz")["attention"]
    assert att.shape[-1] == 168
    np.testing.assert_allclose(att.astype(np.float32).sum(-1), 1.0, atol=1e-2)
    ridge_spec = json.loads((runs / "ridge_window" / "seed1" / "spec.json").read_text())
    assert ridge_spec["ridge"]["n_features_per_horizon"] == 175
    # a finished run is skipped unless --force
    mtime = (run / "metrics.json").stat().st_mtime_ns
    _run("run_ts.py", *common, "--model", model, "--seed", "0")
    assert (run / "metrics.json").stat().st_mtime_ns == mtime
