import numpy as np
import pytest
import torch

from dl_final_prj.baselines import naive_predictions, ridge_predictions
from dl_final_prj.electricity import TARGET, ElectricityData, split_bounds, stable_clients
from dl_final_prj.metrics import horizon_report, regression_report
from dl_final_prj.ts_models import BiRNNForecaster, Seq2SeqForecaster


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


@pytest.mark.parametrize("offsets", [(2,), (1, 2, 3, 4, 5)])
def test_windows_have_no_lookahead_and_respect_splits(synthetic_npz, offsets):
    d = ElectricityData(synthetic_npz, lookback=168, offsets=offsets)
    assert d.feature_names[-4:] == ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    assert "MT_999" not in d.feature_names and TARGET in d.feature_names
    bounds = split_bounds(d.n)
    for split, (lo, hi) in bounds.items():
        o = d.origins[split]
        assert (o + min(offsets) >= lo).all() and (o + max(offsets) < hi).all()
    o = torch.as_tensor(d.origins["val"][:7])
    b = d.batch(o)
    for i, t in enumerate(o.tolist()):
        # last input row is the origin itself, targets are strictly after it
        assert torch.allclose(b.x[i, -1], d.X[t])
        assert torch.allclose(b.x[i, 0], d.X[t - 167])
        for k, off in enumerate(offsets):
            assert b.y[i, k] == d.Y[t + off]
    # inverse transform of the scaled target returns original units
    np.testing.assert_allclose(
        d.inverse(b.y.numpy()), d.y_raw[o.numpy()[:, None] + np.array(offsets)], rtol=1e-5
    )


def test_scaler_fitted_on_train_rows(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, (2,))
    lo, hi = d.bounds["train"]
    z = np.load(synthetic_npz)
    col = list(z["names"]).index(TARGET)
    assert d.mean[d.target_col] == pytest.approx(z["values"][lo:hi, col].mean())


def test_naive_baselines(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, (1, 2, 3, 4, 5))
    p = naive_predictions(d, "test")
    o = d.origins["test"]
    np.testing.assert_allclose(p["persistence"][:, 4], d.y_raw[o])
    np.testing.assert_allclose(p["seasonal_daily"][:, 0], d.y_raw[o + 1 - 24])
    np.testing.assert_allclose(p["seasonal_weekly"][:, 4], d.y_raw[o + 5 - 168])


def test_ridge_beats_persistence_on_periodic_series(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, (2,))
    preds, log = ridge_predictions(d)
    y = d.true_targets("test")
    rmse = lambda p: np.sqrt(np.mean((p - y) ** 2))  # noqa: E731
    assert rmse(preds["test"]) < rmse(naive_predictions(d, "test")["persistence"])
    assert log["alpha"] in [float(k) for k in log["alpha_grid_val_rmse"]]


def test_models_shapes_and_attention(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, (1, 2, 3, 4, 5))
    b = d.batch(torch.as_tensor(d.origins["train"][:4]))
    m = Seq2SeqForecaster(d.n_features, d.target_col, horizon=5, hidden=16, attn_dim=8)
    p, aux = m(b)
    assert p.shape == (4, 5)
    w = aux["attention"]
    assert w.shape == (4, 5, 168)
    assert torch.allclose(w.sum(-1), torch.ones(4, 5), atol=1e-5)
    m2 = Seq2SeqForecaster(d.n_features, d.target_col, horizon=5, hidden=16, attention=False)
    assert m2(b)[0].shape == (4, 5) and m2(b)[1] == {}

    d1 = ElectricityData(synthetic_npz, 168, (2,))
    b1 = d1.batch(torch.as_tensor(d1.origins["train"][:3]))
    for att in (True, False):
        p, aux = BiRNNForecaster(d1.n_features, hidden=8, attn_dim=8, head_dim=8, attention=att)(b1)
        assert p.shape == (3, 1)
        assert ("attention" in aux) == att


def test_seq2seq_eval_ignores_teacher_forcing(synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, (1, 2, 3, 4, 5))
    b = d.batch(torch.as_tensor(d.origins["val"][:4]))
    m = Seq2SeqForecaster(d.n_features, d.target_col, horizon=5, hidden=8, attn_dim=8).eval()
    with torch.no_grad():
        a, _ = m(b, teacher_forcing=1.0)
        c, _ = m(b, teacher_forcing=0.0)
    torch.testing.assert_close(a, c)
