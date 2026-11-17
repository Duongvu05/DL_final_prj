"""Metric definitions from docs/experiment_plan.md Section 5.

All inputs are in original units. R2 and NSE share the same denominator (the mean of the
evaluated targets), so they are numerically equal. Both are reported because the course
asks for both.
"""

from collections.abc import Sequence

import numpy as np

from dl_final_prj.fast import levenshtein_batch

KEYS = ("R2", "NSE", "MAE", "RMSE")


def regression_report(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, np.float64).ravel()
    p = np.asarray(p, np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {p.shape}")
    err = p - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "R2": r2,
        "NSE": r2,
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
    }


def horizon_report(y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float]]:
    """Task 2. y, p have shape (n_origins, H).

    pooled is the primary result over all n*H cells. h1..hH use each horizon's own mean.
    horizon_macro is the plain average of the per-horizon metrics.
    """
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    out = {"pooled": regression_report(y, p)}
    per = [regression_report(y[:, h], p[:, h]) for h in range(y.shape[1])]
    for h, rep in enumerate(per, start=1):
        out[f"h{h}"] = rep
    out["horizon_macro"] = {k: float(np.mean([r[k] for r in per])) for k in KEYS}
    return out


# --- Task 3 -------------------------------------------------------------------


def align_digits(
    targets: Sequence[str], preds: Sequence[str]
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Flatten the valid target positions (plan 5.3 primary convention).

    Returns ((sample_idx, position, true_digit, predicted_digit), present). A missing
    predicted digit (early EOS) becomes 0. `present` tells whether the model actually
    emitted a digit at that position. Positions past the true length are dropped.
    """
    idx, pos, yt, yp, present = [], [], [], [], []
    for i, (t, p) in enumerate(zip(targets, preds, strict=True)):
        for j, c in enumerate(t):
            idx.append(i)
            pos.append(j)
            yt.append(int(c))
            ok = j < len(p) and p[j].isdigit()
            yp.append(int(p[j]) if ok else 0)
            present.append(ok)
    return (
        np.array(idx, np.int64),
        np.array(pos, np.int64),
        np.array(yt, np.float64),
        np.array(yp, np.float64),
    ), np.array(present, bool)


def sequence_report(targets: Sequence[str], preds: Sequence[str]) -> dict[str, float]:
    """Required numeric metrics (primary + two secondary conventions) and OCR metrics."""
    (idx, pos, yt, yp), present = align_digits(targets, preds)
    out = {k: v for k, v in regression_report(yt, yp).items()}

    pos_reps = []
    for j in np.unique(pos):
        m = pos == j
        pos_reps.append(regression_report(yt[m], yp[m]))
    for k in KEYS:
        out[f"pos_macro_{k}"] = float(np.nanmean([r[k] for r in pos_reps]))

    err = yp - yt
    n = len(targets)
    per_mae = np.bincount(idx, np.abs(err), minlength=n) / np.bincount(idx, minlength=n)
    per_mse = np.bincount(idx, err**2, minlength=n) / np.bincount(idx, minlength=n)
    out["sample_macro_MAE"] = float(per_mae.mean())
    out["sample_macro_RMSE"] = float(np.sqrt(per_mse).mean())

    dist = levenshtein_batch(list(targets), list(preds))
    lt = np.array([len(t) for t in targets])
    lp = np.array([len(p) for p in preds])
    out["exact_match"] = float(np.mean([t == p for t, p in zip(targets, preds, strict=True)]))
    out["length_accuracy"] = float(np.mean(lt == lp))
    out["char_accuracy"] = float(np.mean((yp == yt) & present))
    out["norm_edit_distance"] = float(np.mean(dist / np.maximum(np.maximum(lt, lp), 1)))
    out["CER"] = float(dist.sum() / lt.sum())
    return out


def per_sample_losses(targets: Sequence[str], preds: Sequence[str]) -> dict[str, np.ndarray]:
    """Per-image quantities for paired bootstrap and McNemar tests."""
    (idx, _, yt, yp), _ = align_digits(targets, preds)
    n = len(targets)
    err = yp - yt
    return {
        "sq_err_sum": np.bincount(idx, err**2, minlength=n),
        "abs_err_sum": np.bincount(idx, np.abs(err), minlength=n),
        "n_digits": np.bincount(idx, minlength=n).astype(np.float64),
        "exact": np.array([t == p for t, p in zip(targets, preds, strict=True)], np.float64),
        "edit": levenshtein_batch(list(targets), list(preds)).astype(np.float64),
    }
