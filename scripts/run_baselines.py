"""Task 1/2 non-neural baselines. Deterministic, but written once per seed as the plan requires.

Runs persistence, timestamp-aligned daily/weekly seasonal naive, the corrected sparse ridge
and the full-window target-only ridge on the same origins as the neural models.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from dl_final_prj.baselines import naive_predictions, ridge_predictions
from dl_final_prj.electricity import ElectricityData
from dl_final_prj.engine import write_json
from dl_final_prj.metrics import horizon_report, regression_report

DESCRIPTIONS = {
    "persistence": "y_hat(t+h) = y(t)",
    "seasonal_daily": "y_hat(tau) = y(tau - 24 h), timestamp aligned",
    "seasonal_weekly": "y_hat(tau) = y(tau - 168 h), timestamp aligned",
    "ridge": "corrected sparse ridge on y[t], y[t-1], y[t-2], y[tau-24], y[tau-48], "
    "y[tau-168] and the 7 calendar features of tau",
    "ridge_window": "full-window target-only ridge on the 168 raw values y[t-167..t] and the "
    "7 calendar features of tau",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, choices=(1, 2), required=True)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="written under smoke/")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    tcfg = cfg[f"task{a.task}"]
    root = Path(a.out or cfg["results_root"]) / ("smoke" if a.smoke else "phase1")
    run_root = root / "runs" / f"task{a.task}"
    todo = [
        (name, seed)
        for name in DESCRIPTIONS
        for seed in cfg["seeds"]
        if a.force or not (run_root / name / f"seed{seed}" / "metrics.json").exists()
    ]
    if not todo:
        logger.info(f"{run_root} baselines already done")
        return
    data = ElectricityData(
        Path(a.npz or cfg["electricity_npz"]), tcfg["lookback"], tuple(tcfg["offsets"])
    )
    report = horizon_report if a.task == 2 else regression_report

    preds = {s: naive_predictions(data, s) for s in ("val", "test")}
    info: dict[str, dict] = {}
    for name, kind in (("ridge", "sparse"), ("ridge_window", "window")):
        p, info[name] = ridge_predictions(data, kind)
        for s in ("val", "test"):
            preds[s][name] = p[s]

    for name, seed in todo:
        run_dir = run_root / name / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for s in ("val", "test"):
            y = data.true_targets(s)
            metrics[s] = report(y, preds[s][name])
            o = data.origins[s]
            np.savez_compressed(
                run_dir / f"pred_{s}.npz",
                origin=o,
                hour=data.hours[o].to_numpy("datetime64[s]"),
                y_true=y,
                y_pred=preds[s][name],
                base_b24=data.base_b24(o),
            )
        spec = {
            "task": a.task,
            "model": name,
            "seed": seed,
            "title": tcfg["title"],
            "kind": "baseline",
            "description": DESCRIPTIONS[name],
            "n_params": info[name]["n_params"] if name in info else 0,
            "offsets": list(data.offsets),
            "lookback": data.lookback,
            "deterministic": True,
            "splits": data.split_summary(),
        }
        if name in info:
            spec["ridge"] = info[name]
        write_json(spec, run_dir / "spec.json")
        write_json(metrics, run_dir / "metrics.json")
        if seed == cfg["seeds"][-1]:
            t = metrics["test"] if a.task == 1 else metrics["test"]["pooled"]
            logger.info(f"task{a.task} {name} test {t}")


if __name__ == "__main__":
    main()
