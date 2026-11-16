"""Train and evaluate a Task 3 image-to-sequence model on synthetic multi-digit MNIST.

Usage:
    uv run python scripts/train_task3.py --model attention
    uv run python scripts/train_task3.py --model ctc --smoke

Protocol. Early stopping and model choice use the validation split only. The test split
is decoded once, after training. Primary metrics are the sequence metrics (exact match,
per-digit accuracy, CER, edit distance, length accuracy). Value metrics (R2, NSE, MAE,
RMSE on the parsed integer) are secondary because the leading digit dominates them.
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
from dl_final_prj.models.cnn_attention_decoder import AttentionRecognizer
from dl_final_prj.models.ctc_recognizer import CTCRecognizer
from dl_final_prj.training.trainer import fit
from dl_final_prj.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]


def build_model(name: str, cfg: dict) -> nn.Module:
    m, height = cfg["model"], cfg["data"]["height"]
    common = {"height": height, "hidden": m["hidden"], "channels": tuple(m["channels"])}
    if name == "ctc":
        return CTCRecognizer(**common, dropout=m["dropout"])
    return AttentionRecognizer(
        **common,
        emb_dim=m["emb_dim"],
        attn_dim=m["attn_dim"],
        dropout=m["dropout"],
        use_attention=name == "attention",
    )


@torch.no_grad()
def predict(
    model: nn.Module, name: str, loader: DataLoader, max_len: int, device: str
) -> tuple[list[str], list[str]]:
    """Greedy decode a split. Returns (targets, predictions) as digit strings."""
    model.eval()
    targets, preds = [], []
    for images, tokens, _ in loader:
        images = images.to(device)
        if name == "ctc":
            preds += model.greedy_decode(images)  # type: ignore[operator]
        else:
            ids, _ = model.greedy_decode(images, max_len)  # type: ignore[operator]
            preds += [decode(row) for row in ids.tolist()]
        targets += [decode(row) for row in tokens.tolist()]
    return targets, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "task3.yaml"))
    ap.add_argument("--model", choices=["attention", "no_attention", "ctc"], required=True)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a small slice")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    d, t = cfg["data"], cfg["train"]
    if args.epochs is not None:
        t["epochs"] = args.epochs
    if args.smoke:
        t["epochs"] = 1
        d["train_size"], d["val_size"], d["test_size"] = 512, 128, 128
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
        return DataLoader(
            ds, bs, shuffle=shuffle, num_workers=t["num_workers"], persistent_workers=False
        )

    train_loader = loader(train_ds, True, t["batch_size"])
    val_loader = loader(val_ds, False, 256)
    test_loader = loader(test_ds, False, 256)

    model = build_model(args.model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model {args.model} params {n_params / 1e6:.2f}M device {device}")
    max_len = spec.max_target_len

    def step_loss(m: nn.Module, batch: tuple, epoch: int) -> torch.Tensor:
        images, tokens, n_digits = batch
        if args.model == "ctc":
            return m.loss(images, tokens, n_digits)  # type: ignore[operator]
        return m.loss(images, tokens)  # type: ignore[operator]

    def val_cer(m: nn.Module) -> float:
        targets, preds = predict(m, args.model, val_loader, max_len, device)
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

    val_t, val_p = predict(model, args.model, val_loader, max_len, device)
    test_t, test_p = predict(model, args.model, test_loader, max_len, device)  # touched once
    result = {
        "model": args.model,
        "smoke": args.smoke,
        "params": n_params,
        "train_seconds": round(train_seconds, 1),
        "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0,
        "val_sequence": sequence_report(val_t, val_p),
        "test_sequence": sequence_report(test_t, test_p),  # primary
        "test_numeric": numeric_report(test_t, test_p),  # secondary
        "examples": [{"target": a, "pred": b} for a, b in list(zip(test_t, test_p))[:10]],
        "config": cfg,
    }
    out_dir = ROOT / cfg["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out = out_dir / f"task3_{args.model}{suffix}.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info(f"test sequence {result['test_sequence']}")
    logger.info(f"test numeric {result['test_numeric']}")
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
