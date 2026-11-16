import numpy as np


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency. Equal to R2 under the standard definition."""
    return r2(y_true, y_pred)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": r2(y_true, y_pred),
        "NSE": nse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }


def horizon_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    """Metrics per horizon plus the average. Inputs have shape (n_samples, horizon)."""
    report = {
        f"h{h + 1}": regression_report(y_true[:, h], y_pred[:, h]) for h in range(y_true.shape[1])
    }
    report["flat"] = regression_report(y_true.ravel(), y_pred.ravel())
    report["mean_over_horizons"] = {
        key: float(np.mean([report[f"h{h + 1}"][key] for h in range(y_true.shape[1])]))
        for key in ("R2", "NSE", "MAE", "RMSE")
    }
    return report
