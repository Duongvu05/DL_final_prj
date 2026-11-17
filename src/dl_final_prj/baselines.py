"""Non-neural baselines for Tasks 1 and 2 (plan Section 2.1 items 1-4, Section 2.2).

All baselines are fitted and evaluated on the same origins as the neural models, in kW.
"""

from collections.abc import Callable

import numpy as np

from dl_final_prj.electricity import (
    CALENDAR,
    RECENT_LAGS,
    SEASONAL_LAGS,
    ElectricityData,
    ridge_design,
    ridge_window_design,
)

RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)

Design = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def naive_predictions(data: ElectricityData, split: str) -> dict[str, np.ndarray]:
    """Persistence y[t], daily seasonal y[t+h-24], weekly seasonal y[t+h-168]. Shape (n, H)."""
    o = data.origins[split][:, None]
    off = np.array(data.offsets)[None, :]
    y = data.y_raw
    return {
        "persistence": np.repeat(y[o], len(data.offsets), axis=1),
        "seasonal_daily": y[o + off - 24],
        "seasonal_weekly": y[o + off - 168],
    }


def ridge_feature_names(kind: str, lookback: int = 168) -> list[str]:
    """Column names of the ridge design matrix for one horizon."""
    cal = [f"target_{c}" for c in CALENDAR]
    if kind == "sparse":
        recent = ["y[t]"] + [f"y[t-{k - 1}]" for k in RECENT_LAGS[1:]]
        return recent + [f"y[tau-{k}]" for k in SEASONAL_LAGS] + cal
    if kind == "window":
        return [
            f"y[t-{lookback - 1 - i}]" if i < lookback - 1 else "y[t]" for i in range(lookback)
        ] + cal
    raise ValueError(kind)


def _design(kind: str, lookback: int) -> Design:
    if kind == "sparse":
        return ridge_design
    if kind == "window":
        return lambda y, cal, o, k: ridge_window_design(y, cal, o, k, lookback)
    raise ValueError(kind)


def ridge_predictions(
    data: ElectricityData, kind: str = "sparse"
) -> tuple[dict[str, np.ndarray], dict]:
    """Direct ridge per horizon, fitted on the train origins with kW targets.

    kind="sparse" is the corrected sparse ridge (y[t], y[t-1], y[t-2], target-aligned
    y[tau-24], y[tau-48], y[tau-168] and the calendar of tau, see electricity.ridge_design).
    kind="window" is the full-window target-only ridge (the raw lookback values y[t-167..t]
    and the calendar of tau, see electricity.ridge_window_design).

    Features are standardized with train statistics. The intercept is not penalized. One alpha
    is chosen for all horizons by the pooled validation RMSE. Returns predictions for val and
    test and the selection log.
    """
    design = _design(kind, data.lookback)
    splits = ("train", "val", "test")
    designs = {
        s: [design(data.y_raw, data.cal, data.origins[s], k) for k in data.offsets] for s in splits
    }
    targets = {s: data.true_targets(s) for s in splits}

    mu = [d.mean(0) for d in designs["train"]]
    sd = [d.std(0) + 1e-8 for d in designs["train"]]

    def augment(x: np.ndarray, h: int) -> np.ndarray:
        z = (x - mu[h]) / sd[h]
        return np.hstack([z, np.ones((len(z), 1))])

    xtr = [augment(designs["train"][h], h) for h in range(data.horizon)]
    gram = [x.T @ x for x in xtr]  # reused for every alpha
    xty = [x.T @ targets["train"][:, h] for h, x in enumerate(xtr)]

    def weights(alpha: float) -> list[np.ndarray]:
        out = []
        for g, b in zip(gram, xty):
            reg = alpha * np.eye(len(g))
            reg[-1, -1] = 0.0  # do not penalize the intercept
            out.append(np.linalg.solve(g + reg, b))
        return out

    def predict(split: str, w: list[np.ndarray]) -> np.ndarray:
        return np.column_stack([augment(designs[split][h], h) @ w[h] for h in range(data.horizon)])

    log = {}
    for a in RIDGE_ALPHAS:
        p = predict("val", weights(a))
        log[a] = float(np.sqrt(np.mean((p - targets["val"]) ** 2)))
    best = min(log, key=log.__getitem__)
    w = weights(best)
    preds = {"val": predict("val", w), "test": predict("test", w)}
    n_feat = designs["train"][0].shape[1]
    info = {
        "kind": kind,
        "alpha_grid_val_rmse": {str(k): v for k, v in log.items()},
        "alpha": best,
        "features": ridge_feature_names(kind, data.lookback),
        "n_features_per_horizon": n_feat,
        "n_params": data.horizon * (n_feat + 1),
        "standardization": "per feature, train-origin mean/std",
        "target": "kW, direct model per horizon, unpenalized intercept",
        "alpha_selection": "pooled validation RMSE over all horizons",
    }
    return preds, info
