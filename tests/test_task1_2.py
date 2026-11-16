import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from dl_final_prj.data.time_series import (
    TimeSeriesConfig,
    WindowDataset,
    build_time_series,
    load_frame,
    make_synthetic_series,
)
from dl_final_prj.evaluation.baselines import (
    LinearRegressionBaseline,
    baseline_predictions,
    persistence,
    seasonal_naive,
)
from dl_final_prj.models.rnn_forecaster import AdditiveAttention, RNNForecaster
from dl_final_prj.models.seq2seq_forecaster import Seq2SeqForecaster, teacher_forcing_ratio


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_synthetic_series(n=600, n_features=3, seed=1)


def _cfg(tmp_path, frame: pd.DataFrame, **kw) -> TimeSeriesConfig:
    path = tmp_path / "series.csv"
    frame.to_csv(path, index=False)
    return TimeSeriesConfig(path=str(path), pred_column="target", datetime_col="datetime", **kw)


def test_synthetic_is_seeded() -> None:
    a, b = make_synthetic_series(seed=3), make_synthetic_series(seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(make_synthetic_series(seed=4))


def test_build_shapes_and_split_order(tmp_path, frame: pd.DataFrame) -> None:
    cfg = _cfg(tmp_path, frame, lookback=24, offset=1, horizon=5)
    data = build_time_series(cfg)
    assert data.n_features == 4
    x, y = data.splits["train"]
    assert x.shape[1:] == (24, 4)
    assert y.shape[1:] == (5,)
    assert all(len(data.splits[k][0]) > 0 for k in ("val", "test"))


def test_scaler_fit_on_train_rows_only(tmp_path, frame: pd.DataFrame) -> None:
    cfg = _cfg(tmp_path, frame, lookback=24)
    data = build_time_series(cfg)
    n_train = int(len(frame) * cfg.train_frac)
    train_mean = frame.drop(columns="datetime").iloc[:n_train].mean().to_numpy()
    np.testing.assert_allclose(data.scaler.mean, train_mean)


def test_dataset_has_no_lookahead() -> None:
    n, lookback, offset, horizon = 200, 10, 2, 3
    steps = np.arange(n, dtype=float)
    df = pd.DataFrame({"target": steps})
    cfg = TimeSeriesConfig(
        path="", pred_column="target", lookback=lookback, offset=offset, horizon=horizon
    )
    data = build_time_series(cfg, df)
    for name, (x, y) in data.splits.items():
        ds = WindowDataset(x, y)
        for i in (0, len(ds) - 1):
            xi, yi = ds[i]
            x_raw = data.inverse_target(xi[:, 0].numpy())
            y_raw = data.inverse_target(yi.numpy())
            t = x_raw[-1]
            np.testing.assert_allclose(y_raw, t + offset + np.arange(horizon), atol=1e-4)
            assert x_raw.max() == t, f"{name} window sees the future"
    # test windows only use test rows, which come after validation rows
    assert data.inverse_target(data.splits["test"][0][0, 0, 0]) >= int(n * 0.85) - 1e-4


def test_resample_and_missing_values(tmp_path) -> None:
    idx = pd.date_range("2021-01-01", periods=240, freq="10min")
    df = pd.DataFrame({"datetime": idx, "a": np.arange(240.0), "target": np.arange(240.0)})
    df.loc[5, "a"] = np.nan
    path = tmp_path / "fine.csv"
    df.to_csv(path, index=False)
    cfg = TimeSeriesConfig(
        path=str(path), pred_column="target", datetime_col="datetime", resample="1h"
    )
    out = load_frame(cfg)
    assert len(out) == 40
    assert not out.isna().any().any()


def test_rnn_forecaster_shapes_and_attention_sums_to_one() -> None:
    x = torch.randn(4, 12, 5)
    for cell in ("lstm", "gru"):
        plain = RNNForecaster(5, hidden=16, cell=cell, attention=False)
        assert plain(x).shape == (4, 1)
        model = RNNForecaster(5, hidden=16, cell=cell, attention=True)
        pred, w = model(x, return_attention=True)
        assert pred.shape == (4, 1)
        assert w is not None
        assert w.shape == (4, 12)
        torch.testing.assert_close(w.sum(-1), torch.ones(4))


def test_additive_attention_weights_sum_to_one() -> None:
    attn = AdditiveAttention(8, 8)
    ctx, w = attn(torch.randn(3, 8), torch.randn(3, 7, 8))
    assert ctx.shape == (3, 8)
    torch.testing.assert_close(w.sum(-1), torch.ones(3))


@pytest.mark.parametrize("attention", [True, False])
def test_seq2seq_decode_shape(attention: bool) -> None:
    model = Seq2SeqForecaster(5, target_idx=2, horizon=5, hidden=16, attention=attention).eval()
    x = torch.randn(4, 12, 5)
    out, w = model(x, return_attention=True)
    assert out.shape == (4, 5, 1)
    if attention:
        assert w is not None
        assert w.shape == (4, 5, 12)
        torch.testing.assert_close(w.sum(-1), torch.ones(4, 5))
    else:
        assert w is None


def test_seq2seq_free_run_ignores_targets_and_teacher_forcing_uses_them() -> None:
    torch.manual_seed(0)
    model = Seq2SeqForecaster(3, target_idx=0, horizon=5, hidden=16, dropout=0.0).eval()
    x, y = torch.randn(4, 10, 3), torch.randn(4, 5)
    free = model(x)
    torch.testing.assert_close(model(x, y, teacher_forcing=0.0), free)
    assert not torch.allclose(model(x, y, teacher_forcing=1.0), free)
    # the first output cannot depend on y even with full teacher forcing
    torch.testing.assert_close(model(x, y, teacher_forcing=1.0)[:, 0], free[:, 0])


def test_teacher_forcing_decays() -> None:
    ratios = [teacher_forcing_ratio(e, 1.0, 0.5, 0.1) for e in range(1, 8)]
    assert ratios[0] == 1.0
    assert all(a >= b for a, b in itertools.pairwise(ratios))
    assert ratios[-1] == 0.1


def test_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = Seq2SeqForecaster(2, target_idx=0, horizon=5, hidden=16, dropout=0.0)
    x = torch.randn(32, 10, 2)
    y = x[:, -1, 0:1] + torch.arange(5.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = last = 0.0
    for step in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x, y, 0.5).squeeze(-1), y)
        loss.backward()
        opt.step()
        first, last = (loss.item(), last) if step == 0 else (first, loss.item())
    assert last < first


def test_baselines() -> None:
    x = np.arange(30, dtype=float).reshape(1, 30, 1)
    np.testing.assert_array_equal(persistence(x, 0, 3), [[29, 29, 29]])
    # period 24, steps t+1..t+2 map to rows 29+1-24=6 and 7
    np.testing.assert_array_equal(seasonal_naive(x, 0, 1, 2, 24), [[6, 7]])
    with pytest.raises(ValueError):
        seasonal_naive(x, 0, 1, 5, 3)


def test_linear_baseline_recovers_linear_map() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6, 2))
    y = np.stack([x[:, -1, 0] * 2 + 1, x[:, -2, 1] - 3], axis=1)
    model = LinearRegressionBaseline().fit(x, y)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-4)


def test_baseline_predictions_keys_and_shapes(tmp_path, frame: pd.DataFrame) -> None:
    cfg = _cfg(tmp_path, frame, lookback=48, offset=1, horizon=5)
    data = build_time_series(cfg)
    x_test = data.splits["test"][0]
    preds = baseline_predictions(data.splits["train"], x_test, data.target_idx, 1, 5, period=24)
    assert set(preds) == {"persistence", "linear_regression", "seasonal_naive"}
    assert all(p.shape == (len(x_test), 5) for p in preds.values())
    assert "seasonal_naive" not in baseline_predictions(
        data.splits["train"], x_test, data.target_idx, 1, 5
    )
