"""Build the whole-image SVHN cache of configs/phase1_task3_whole.yaml (no ground-truth box).

Needs the cropped cache of scripts/prepare_data.py in --cache. Reads its {train,val,test}_labels
.json (same images, order and split) and writes {split}_full_images.npy, {split}_full_labels.json
and full_audit.json next to it. Each whole original image is resized to 64x64 with the aspect
ratio ignored (He et al. 2019), no crop. Then, as a separate diagnostic that never feeds the data
path, full_border_audit.json measures how often the 54x54 crops cut digits off those images.
If --manifest exists, its "task3_whole" entry is updated.

    uv run python scripts/prepare_svhn_whole.py --threads 32
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/svhn")
    ap.add_argument("--cache", default=None, help="cropped cache dir (default: config svhn_dir)")
    ap.add_argument("--config", default="configs/phase1_task3_whole.yaml")
    ap.add_argument("--chunk", type=int, default=20000, help="images decoded per Rust call")
    ap.add_argument("--threads", type=int, default=None, help="Rust threads (default: all cores)")
    ap.add_argument("--manifest", default="results/manifests/split_manifest.json")
    a = ap.parse_args()
    if a.threads:
        os.environ["RAYON_NUM_THREADS"] = str(a.threads)  # read when the Rust pool starts
    from dl_final_prj import svhn  # after RAYON_NUM_THREADS is set
    from dl_final_prj.engine import write_json
    from dl_final_prj.fast import has_rust

    cfg = yaml.safe_load(Path(a.config).read_text())
    cache = Path(a.cache or cfg["svhn_dir"])
    threads = os.environ.get("RAYON_NUM_THREADS", f"all {os.cpu_count()} cores")
    logger.info(f"rust extension: {has_rust()}, threads: {threads}, cache: {cache}")
    t0 = time.time()
    audit = svhn.prepare_whole(
        Path(a.raw),
        cache,
        size=cfg["size"],
        chunk=a.chunk,
        suffix=cfg.get("cache_suffix") or svhn.WHOLE_SUFFIX,  # "" or null -> "_full"
    )
    counts = {k: v["n_images"] for k, v in audit["splits"].items()}
    logger.info(f"whole-image cache done in {time.time() - t0:.0f}s: {counts}")
    # diagnostic only, after the box-free cache is written: how often the 54x54 crops clip digits
    border = svhn.whole_border_audit(cache, crop=cfg["crop"], suffix=audit["suffix"])
    man = Path(a.manifest)
    if man.exists():
        splits = json.loads(man.read_text())
        splits["task3_whole"] = svhn.whole_manifest_entry(audit, border)
        write_json(splits, man)
        logger.info(f"updated task3_whole in {man}")


if __name__ == "__main__":
    main()
