import numpy as np

from dl_final_prj.evaluation.regression_metrics import regression_report


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sequence_report(targets: list[str], preds: list[str]) -> dict[str, float]:
    """Primary metrics for sequence outputs. Padding must be removed before calling."""
    n = len(targets)
    exact = sum(t == p for t, p in zip(targets, preds)) / n
    length_acc = sum(len(t) == len(p) for t, p in zip(targets, preds)) / n
    dist = [edit_distance(t, p) for t, p in zip(targets, preds)]
    cer = sum(dist) / max(1, sum(len(t) for t in targets))
    same_len = [(t, p) for t, p in zip(targets, preds) if len(t) == len(p)]
    char_total = sum(len(t) for t, _ in same_len)
    char_ok = sum(a == b for t, p in same_len for a, b in zip(t, p))
    return {
        "exact_match": exact,
        "length_accuracy": length_acc,
        "CER": cer,
        "mean_edit_distance": float(np.mean(dist)),
        "per_digit_accuracy_same_length": char_ok / max(1, char_total),
    }


def numeric_report(targets: list[str], preds: list[str]) -> dict[str, float]:
    """Secondary R2/NSE/MAE/RMSE on integer values of digit strings.

    Unparsable predictions count as failures and are reported as invalid_rate. The
    other metrics use only the parsable pairs. Value metrics are dominated by the
    leading digit, so always read them next to the primary metrics.
    """
    pairs = [(int(t), int(p)) for t, p in zip(targets, preds) if p.isdigit() and t.isdigit()]
    invalid_rate = 1.0 - len(pairs) / len(targets)
    if len(pairs) < 2:
        return {"invalid_rate": invalid_rate}
    y_true = np.array([t for t, _ in pairs], dtype=np.float64)
    y_pred = np.array([p for _, p in pairs], dtype=np.float64)
    return {**regression_report(y_true, y_pred), "invalid_rate": invalid_rate}
