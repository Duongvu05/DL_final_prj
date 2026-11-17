"""Non-neural baselines for Tasks 1 and 2 (plan Section 2.1 items 1-3, Section 2.2)."""

import numpy as np

from dl_final_prj.electricity import ElectricityData, calendar_features, ridge_design

RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


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


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xb = np.hstack([x, np.ones((len(x), 1))])
    reg = alpha * np.eye(xb.shape[1])
    reg[-1, -1] = 0.0  # do not penalize the intercept
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _apply(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((len(x), 1))]) @ w


def ridge_predictions(data: ElectricityData) -> tuple[dict[str, np.ndarray], dict]:
    """Direct ridge per horizon on lags {1,2,3,24,48,168} of the target plus target-hour calendar.

    Features are standardized with train statistics. One alpha is chosen for all horizons by
    the pooled validation RMSE. Returns predictions for val and test and the selection log.
    """
    cal = calendar_features(data.hours)
    designs = {
        s: [ridge_design(data.y_raw, cal, data.origins[s], k) for k in data.offsets]
        for s in ("train", "val", "test")
    }
    targets = {s: data.true_targets(s) for s in ("train", "val", "test")}
    mu = [d.mean(0) for d in designs["train"]]
    sd = [d.std(0) + 1e-8 for d in designs["train"]]

    def predict(split: str, alpha: float) -> np.ndarray:
        cols = []
        for h in range(len(data.offsets)):
            w = _fit_ridge((designs["train"][h] - mu[h]) / sd[h], targets["train"][:, h], alpha)
            cols.append(_apply((designs[split][h] - mu[h]) / sd[h], w))
        return np.column_stack(cols)

    log = {}
    for a in RIDGE_ALPHAS:
        p = predict("val", a)
        log[a] = float(np.sqrt(np.mean((p - targets["val"]) ** 2)))
    best = min(log, key=log.__getitem__)
    preds = {"val": predict("val", best), "test": predict("test", best)}
    return preds, {"alpha_grid_val_rmse": {str(k): v for k, v in log.items()}, "alpha": best}
