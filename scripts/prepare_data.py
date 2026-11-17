"""Phase 0. Parse the raw UCI and SVHN files once, freeze splits, write the manifests.

Writes data/processed/{electricity_hourly.npz, electricity_audit.json, svhn/*} and
results/manifests/{dataset_manifest.json, split_manifest.json, environment.json, checksums.txt}.
The SVHN cache follows the cropped protocol of configs/phase1_task3.yaml (raw train, extra and
test folders under data/raw/svhn).
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml
from loguru import logger

from dl_final_prj import electricity, svhn
from dl_final_prj.electricity import ElectricityData
from dl_final_prj.engine import write_json
from dl_final_prj.fast import has_rust


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--task3-config", default="configs/phase1_task3.yaml")
    ap.add_argument("--workers", type=int, default=None, help="digitStruct.mat parse processes")
    ap.add_argument("--skip", choices=("electricity", "svhn"), action="append", default=[])
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    t3 = yaml.safe_load(Path(a.task3_config).read_text())
    raw, proc = Path(a.raw), Path(a.processed)
    man = Path(cfg["results_root"]) / "manifests"
    man.mkdir(parents=True, exist_ok=True)

    if "electricity" not in a.skip:
        t0 = time.time()
        audit = electricity.prepare(raw / "uci_electricity" / "LD2011_2014.txt", proc)
        logger.info(
            f"electricity prepared in {time.time() - t0:.1f}s, {audit['n_stable_clients']} stable clients"
        )
    if "svhn" not in a.skip:
        t0 = time.time()
        audit = svhn.prepare(
            raw / "svhn",
            proc / "svhn",
            size=t3["size"],
            max_len=t3["max_len"],
            val_frac=t3.get("val_frac", 0.1),
            workers=a.workers,
        )
        counts = {k: v["n_images"] for k, v in audit["splits"].items()}
        logger.info(f"svhn cached in {time.time() - t0:.1f}s: {counts}")

    e_audit = json.loads((proc / "electricity_audit.json").read_text())
    s_audit = json.loads((proc / "svhn" / "svhn_audit.json").read_text())
    write_json({"electricity": e_audit, "svhn": s_audit}, man / "dataset_manifest.json")

    splits = {
        "task3": {
            "rule": s_audit["split_rule"],
            "counts_per_source": s_audit["counts_per_source"],
            "dropped_long_labels": s_audit["dropped_long_labels"]["count"],
            "pool_size_before_split": s_audit["pool_size_before_split"],
            "n_images": {k: v["n_images"] for k, v in s_audit["splits"].items()},
            "splits": s_audit["splits"],
        }
    }
    npz = proc / "electricity_hourly.npz"
    for t in (1, 2):
        tc = cfg[f"task{t}"]
        d = ElectricityData(npz, tc["lookback"], tuple(tc["offsets"]))
        rule = d.window_rule()
        splits[f"task{t}"] = {
            "lookback": tc["lookback"],
            "offsets": tc["offsets"],
            "formula_all_windows": rule["formula_all_windows"],
            "n_all_windows": rule["n_all_windows"],
            "rule": "origin assigned to the split containing all its target hours, inputs may cross the boundary",
            "splits": d.split_summary(),
        }
    write_json(splits, man / "split_manifest.json")

    sums = []
    for p in sorted(
        list((raw / "uci_electricity").glob("*.zip")) + list((raw / "svhn").glob("*.tar.gz"))
    ):
        sums.append(f"{sha256(p)}  {p.name}")
    (man / "checksums.txt").write_text("\n".join(sums) + "\n")

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rust_extension": has_rust(),
        "git_commit": git_commit(),
        "host": platform.node(),
    }
    write_json(env, man / "environment.json")
    logger.info(f"manifests written to {man}")


if __name__ == "__main__":
    main()
