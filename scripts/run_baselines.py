"""Task 1/2 non-neural baselines. Deterministic, but written once per seed as the plan requires."""

import argparse
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from dl_final_prj.baselines import naive_predictions, ridge_predictions
from dl_final_prj.electricity import ElectricityData
from dl_final_prj.engine import write_json
from dl_final_prj.metrics import horizon_report, regression_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, choices=(1, 2), required=True)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    tcfg = cfg[f"task{a.task}"]
    root = Path(a.out or cfg["results_root"]) / ("smoke" if a.smoke else "phase1")
    data = ElectricityData(
        Path(a.npz or cfg["electricity_npz"]), tcfg["lookback"], tuple(tcfg["offsets"])
    )
    report = horizon_report if a.task == 2 else regression_report

    preds = {s: naive_predictions(data, s) for s in ("val", "test")}
    ridge, ridge_log = ridge_predictions(data)
    for s in ("val", "test"):
        preds[s]["ridge"] = ridge[s]

    for name in preds["test"]:
        for seed in cfg["seeds"]:
            run_dir = root / "runs" / f"task{a.task}" / name / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics = {}
            for s in ("val", "test"):
                y = data.true_targets(s)
                metrics[s] = report(y, preds[s][name])
                o = data.origins[s]
                np.savez_compressed(
                    run_dir / f"pred_{s}.npz",
                    origin=o,
                    hour=data.hours[o].asi8,
                    y_true=y,
                    y_pred=preds[s][name],
                )
            spec = {
                "task": a.task,
                "model": name,
                "seed": seed,
                "kind": "baseline",
                "n_params": 0,
                "offsets": list(data.offsets),
                "deterministic": True,
                "splits": data.split_summary(),
            }
            if name == "ridge":
                spec["ridge"] = ridge_log | {
                    "lags": [1, 2, 3, 24, 48, 168],
                    "features": "target lags + target-hour calendar",
                }
                spec["n_params"] = len(data.offsets) * (6 + 4 + 1)
            write_json(spec, run_dir / "spec.json")
            write_json(metrics, run_dir / "metrics.json")
        t = metrics["test"] if a.task == 1 else metrics["test"]["pooled"]
        logger.info(f"task{a.task} {name} test {t}")


if __name__ == "__main__":
    main()
