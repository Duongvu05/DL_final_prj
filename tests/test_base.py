import numpy as np

from dl_final_prj.data.splits import chronological_split, make_windows
from dl_final_prj.evaluation.regression_metrics import horizon_report, nse, r2
from dl_final_prj.evaluation.sequence_metrics import edit_distance, numeric_report, sequence_report


def test_windows_no_lookahead_task1() -> None:
    series = np.arange(20, dtype=float)
    x, y = make_windows(series[:, None], series, lookback=4, offset=2, horizon=1)
    assert x[0, -1, 0] == 3
    assert y[0, 0] == 5
    assert y[-1, 0] == 19


def test_windows_task2_shape() -> None:
    series = np.arange(30, dtype=float)
    _x, y = make_windows(series[:, None], series, lookback=6, offset=1, horizon=5)
    assert y.shape[1] == 5
    assert list(y[0]) == [6, 7, 8, 9, 10]
    assert y[-1, -1] == 29


def test_chronological_split_order() -> None:
    tr, va, te = chronological_split(100)
    assert tr.stop == va.start and va.stop == te.start and te.stop == 100


def test_r2_equals_nse() -> None:
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 1.9, 3.2, 3.7])
    assert r2(a, b) == nse(a, b)
    assert "mean_over_horizons" in horizon_report(np.stack([a, a], 1), np.stack([b, b], 1))


def test_sequence_metrics() -> None:
    assert edit_distance("1234", "1243") == 2
    rep = sequence_report(["123", "456"], ["123", "450"])
    assert rep["exact_match"] == 0.5
    assert numeric_report(["123", "456"], ["123", "x"])["invalid_rate"] == 0.5
