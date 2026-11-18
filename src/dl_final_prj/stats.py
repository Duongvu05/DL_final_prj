"""Paired tests from docs/experiment_plan.md Section 4.3.

Diebold-Mariano with a Newey-West (Bartlett) HAC variance, moving-block bootstrap
confidence intervals (Rust core in dlfast), McNemar for exact-string correctness and
Holm adjustment.
"""

import numpy as np
from scipy import stats as st

from dl_final_prj.fast import block_bootstrap_means


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int) -> dict[str, float]:
    """Diebold-Mariano on d = loss_a - loss_b. Negative mean means A is better.

    Uses the Bartlett-kernel long-run variance with truncation `lag` and the
    Harvey-Leybourne-Newbold small-sample correction with a t(n-1) reference.
    """
    d = np.asarray(loss_a, np.float64) - np.asarray(loss_b, np.float64)
    n = len(d)
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        lrv += 2.0 * w * float(dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return {
            "mean_diff": float(d.mean()),
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "lag": lag,
        }
    dm = d.mean() / np.sqrt(lrv / n)
    h = lag + 1
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * st.t.sf(abs(stat), df=n - 1))
    return {"mean_diff": float(d.mean()), "dm_stat": stat, "p_value": p, "lag": lag}


def bootstrap_loss_diff(
    y: np.ndarray,
    sq_a: np.ndarray,
    sq_b: np.ndarray,
    abs_a: np.ndarray,
    abs_b: np.ndarray,
    block: int,
    reps: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Moving-block bootstrap of A - B from per-origin losses (plan Section 4.3).

    sq_*/abs_* are per-origin squared/absolute errors, already averaged over horizons and over
    seeds when several seeds exist. y is (n,) or (n, H). Blocks of origins are resampled, and
    RMSE = sqrt(mean squared error), MAE = mean absolute error, NSE = 1 - MSE / var(y) are
    computed on each resample. Returns the point difference and the 2.5/97.5 percentiles.
    """
    y = np.asarray(y, np.float64).reshape(len(y), -1)
    cols = np.column_stack([sq_a, sq_b, abs_a, abs_b, y.mean(1), (y**2).mean(1)])

    def metrics(m: np.ndarray) -> dict[str, np.ndarray]:
        var = m[..., 5] - m[..., 4] ** 2
        return {
            "RMSE": np.sqrt(m[..., 0]) - np.sqrt(m[..., 1]),
            "MAE": m[..., 2] - m[..., 3],
            "NSE": (1 - m[..., 0] / var) - (1 - m[..., 1] / var),
        }

    point = metrics(cols.mean(0))
    boot = metrics(block_bootstrap_means(cols, block, reps, seed))
    return {
        k: {
            "diff": float(point[k]),
            "ci_low": float(np.percentile(boot[k], 2.5)),
            "ci_high": float(np.percentile(boot[k], 97.5)),
        }
        for k in point
    }


def bootstrap_forecast_diff(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    block: int,
    reps: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Single-seed convenience wrapper. y, pred_a, pred_b have shape (n,) or (n, H)."""
    y = np.asarray(y, np.float64).reshape(len(y), -1)
    a = np.asarray(pred_a, np.float64).reshape(y.shape)
    b = np.asarray(pred_b, np.float64).reshape(y.shape)
    return bootstrap_loss_diff(
        y,
        ((a - y) ** 2).mean(1),
        ((b - y) ** 2).mean(1),
        np.abs(a - y).mean(1),
        np.abs(b - y).mean(1),
        block,
        reps,
        seed,
    )


def bootstrap_image_diff(
    per_a: dict[str, np.ndarray], per_b: dict[str, np.ndarray], reps: int = 10_000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Paired i.i.d. bootstrap over test images of A - B, using metrics.per_sample_losses output.

    Returns, per metric, the point difference, the 2.5/97.5 percentiles and a two-sided
    percentile-bootstrap p-value 2 * min(P(boot <= 0), P(boot >= 0)) (floored at 1/reps).
    """
    cols = np.column_stack(
        [
            per_a["sq_err_sum"],
            per_b["sq_err_sum"],
            per_a["abs_err_sum"],
            per_b["abs_err_sum"],
            per_a["n_digits"],
            per_a["exact"],
            per_b["exact"],
            per_a["char_correct"],
            per_b["char_correct"],
            per_a["edit"],
            per_b["edit"],
        ]
    )

    def metrics(m: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "RMSE": np.sqrt(m[..., 0] / m[..., 4]) - np.sqrt(m[..., 1] / m[..., 4]),
            "MAE": (m[..., 2] - m[..., 3]) / m[..., 4],
            "exact_match": m[..., 5] - m[..., 6],
            "char_accuracy": (m[..., 7] - m[..., 8]) / m[..., 4],
            "mean_edit_distance": m[..., 9] - m[..., 10],
        }

    point = metrics(cols.mean(0))
    boot = metrics(block_bootstrap_means(cols, 1, reps, seed))
    return {
        k: {
            "diff": float(point[k]),
            "ci_low": float(np.percentile(boot[k], 2.5)),
            "ci_high": float(np.percentile(boot[k], 97.5)),
            "p_value": float(
                max(
                    1.0 / reps,
                    min(1.0, 2 * min(float(np.mean(boot[k] <= 0)), float(np.mean(boot[k] >= 0)))),
                )
            ),
        }
        for k in point
    }


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact two-sided McNemar test on paired binary correctness."""
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * float(st.binom.cdf(min(n01, n10), n, 0.5)))
    return {"a_only": n01, "b_only": n10, "p_value": p}


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, float((m - rank) * p[i])))
        adj[i] = running
    return adj.tolist()
