"""Train and evaluate a clovaai-based recognizer on the Task 3 synthetic digit strips.

Usage:
    uv run python scripts/train_task3_sota.py --feature vgg --sequence bilstm --prediction ctc
    uv run python scripts/train_task3_sota.py --feature resnet --sequence bilstm --prediction attn

Protocol is the same as scripts/train_task3.py. The data splits come from
data/image_sequences.py (disjoint digit pools, fixed per split seed). Early stopping and
model choice use validation CER. The test split is decoded once, after training. Primary
metrics are the sequence metrics. Value metrics (R2, NSE, MAE, RMSE) are secondary.
The report also lists our from-scratch models from reports/task3_*.json when present.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from dl_final_prj.data.image_sequences import (
    MultiDigitDataset,
    MultiDigitSpec,
    decode,
    load_mnist_pools,
)
from dl_final_prj.evaluation.sequence_metrics import numeric_report, sequence_report
from dl_final_prj.models.sota_recognizers import ClovaRecognizer
from dl_final_prj.training.trainer import fit
from dl_final_prj.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
OWN_MODELS = ("attention", "no_attention", "ctc")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[list[str], list[str]]:
    """Greedy decode a split. Returns (targets, predictions) as digit strings."""
    model.eval()
    targets, preds = [], []
    for images, tokens, _ in loader:
        preds += model.predict(images.to(device))  # type: ignore[operator]
        targets += [decode(row) for row in tokens.tolist()]
    return targets, preds


def own_results(report_dir: Path) -> dict[str, dict]:
    """Test metrics of our from-scratch models, full runs only (smoke runs are skipped)."""
    out = {}
    for name in OWN_MODELS:
        path = report_dir / f"task3_{name}.json"
        if path.exists():
            r = json.loads(path.read_text())
            out[name] = {
                "params": r["params"],
                "test_sequence": r["test_sequence"],
                "test_numeric": r["test_numeric"],
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "sota_task3.yaml"))
    ap.add_argument("--feature", choices=["vgg", "resnet"], default=None)
    ap.add_argument("--sequence", choices=["none", "bilstm"], default=None)
    ap.add_argument("--prediction", choices=["ctc", "attn"], default=None)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a small slice")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    for key in ("feature", "sequence", "prediction"):
        if getattr(args, key) is not None:
            m[key] = getattr(args, key)
    if args.epochs is not None:
        t["epochs"] = args.epochs
    if args.smoke:
        t["epochs"] = 1
        d["train_size"], d["val_size"], d["test_size"] = 512, 128, 128
    variant = f"{m['feature']}_{m['sequence']}_{m['prediction']}"
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["gpu_mem_gb"] * 2**30 / total))
        torch.backends.cudnn.benchmark = True

    spec_keys = ("min_digits", "max_digits", "height", "width", "min_gap", "max_gap")
    spec_keys += ("overlap_prob", "max_overlap", "noise_std", "max_rotation")
    spec = MultiDigitSpec(**{k: d[k] for k in spec_keys}, seed=cfg["seed"])
    pools = load_mnist_pools(ROOT / cfg["data_root"], cfg["seed"], d["val_frac"])
    train_ds = MultiDigitDataset(pools["train"], spec, d["train_size"], "train", d["augment"])
    val_ds = MultiDigitDataset(pools["val"], spec, d["val_size"], "val")
    test_ds = MultiDigitDataset(pools["test"], spec, d["test_size"], "test")

    def loader(ds: MultiDigitDataset, shuffle: bool, bs: int) -> DataLoader:
        return DataLoader(ds, bs, shuffle=shuffle, num_workers=t["num_workers"])

    train_loader = loader(train_ds, True, t["batch_size"])
    val_loader = loader(val_ds, False, 256)
    test_loader = loader(test_ds, False, 256)

    model = ClovaRecognizer(
        m["feature"],
        m["sequence"],
        m["prediction"],
        output_channel=m["output_channel"],
        hidden=m["hidden"],
        max_len=spec.max_target_len,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model {variant} params {n_params / 1e6:.2f}M device {device}")

    def step_loss(net: nn.Module, batch: tuple, epoch: int) -> torch.Tensor:
        images, tokens, n_digits = batch
        return net.loss(images, tokens, n_digits)  # type: ignore[operator]

    def val_cer(net: nn.Module) -> float:
        targets, preds = predict(net, val_loader, device)
        return sequence_report(targets, preds)["CER"]

    start = time.time()
    model = fit(
        model,
        train_loader,
        step_loss,
        val_cer,
        epochs=t["epochs"],
        lr=t["lr"],
        clip=t["clip"],
        patience=t["patience"],
        device=device,
    )
    train_seconds = time.time() - start

    val_t, val_p = predict(model, val_loader, device)
    test_t, test_p = predict(model, test_loader, device)  # touched once
    out_dir = ROOT / cfg["report_dir"]
    result = {
        "variant": variant,
        "smoke": args.smoke,
        "source": "clovaai/deep-text-recognition-benchmark e2117f2 (Apache-2.0)",
        "params": n_params,
        "train_seconds": round(train_seconds, 1),
        "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0,
        "val_sequence": sequence_report(val_t, val_p),
        "test_sequence": sequence_report(test_t, test_p),  # primary
        "test_numeric": numeric_report(test_t, test_p),  # secondary
        "own_models_test": own_results(out_dir),
        "examples": [{"target": a, "pred": b} for a, b in list(zip(test_t, test_p))[:10]],
        "config": cfg,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out = out_dir / f"sota_task3_{variant}{suffix}.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info(f"test sequence {result['test_sequence']}")
    logger.info(f"test numeric {result['test_numeric']}")
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
