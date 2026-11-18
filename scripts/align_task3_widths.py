"""Apply the plan's setting-level decoding width to finished Task 3 runs.

For each model the width is the argmax of the mean validation exact match over seeds 0-2,
taken from the runs' own decoding.val_by_width tables (ties to the smaller width). Seeds whose
primary decoding used another width are re-evaluated with run_ocr.py --eval-only --width W.
No test number is used for the choice.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger


def setting_width(tables: list[dict[str, float]]) -> int:
    keys = sorted(tables[0], key=int)
    if any(sorted(t, key=int) != keys for t in tables):
        raise SystemExit("runs were evaluated on different candidate widths")
    mean = {k: sum(t[k] for t in tables) / len(tables) for k in keys}
    return int(max(keys, key=lambda k: (mean[k], -int(k))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    root = Path(cfg["results_root"]) / cfg.get("phase", "phase1") / "runs" / "task3"
    for model in a.models:
        mets = {
            s: json.loads((root / model / f"seed{s}" / "metrics.json").read_text()) for s in a.seeds
        }
        decs = {s: m["decoding"] for s, m in mets.items()}
        if not decs[a.seeds[0]].get("val_by_width"):
            logger.info(f"{model}: no width to select ({decs[a.seeds[0]]['primary']})")
            continue
        w = setting_width([d["val_by_width"] for d in decs.values()])
        todo = [s for s, d in decs.items() if d["width"] != w]
        logger.info(
            f"{model}: setting width {w}, per-run {[d['width'] for d in decs.values()]}, re-evaluate {todo}"
        )
        for s in todo:
            cmd = [sys.executable, "scripts/run_ocr.py", "--config", a.config, "--model", model,
                   "--seed", str(s), "--eval-only", "--width", str(w)]  # fmt: skip
            if not a.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
