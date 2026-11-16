"""Task 2 reference. Train the TimeSeriesSeq2Seq based model and compare with our own seq2seq.

Uses the same data loader, windows, trainer, scheduled sampling and horizon_report as
train_task2.py. Evaluation is free-running (autoregressive, no teacher forcing).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from torch import nn

from dl_final_prj.data.time_series import TimeSeriesConfig, TimeSeriesData, build_time_series
from dl_final_prj.evaluation.regression_metrics import horizon_report, rmse
from dl_final_prj.models.seq2seq_forecaster import teacher_forcing_ratio
from dl_final_prj.models.sota_forecasters.ts_seq2seq import TSSeq2SeqForecaster
from dl_final_prj.training.trainer import fit
from dl_final_prj.utils.seed import set_seed

MAX_GPU_GB = 2.0
NAME = "ts_seq2seq_attention"
OWN_NAMES = ("seq2seq", "seq2seq_attention")


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, MAX_GPU_GB / total))
    return "cuda"


@torch.no_grad()
def predict(model: nn.Module, data: TimeSeriesData, split: str, device: str) -> np.ndarray:
    """Free-running predictions (no teacher forcing) in original units, shape (n, horizon)."""
    model.eval()
    out = [model(x.to(device)).squeeze(-1).cpu().numpy() for x, _ in data.loader(split, 512)]
    return data.inverse_target(np.concatenate(out))


def own_results(path: Path) -> dict:
    """Test metrics of our own seq2seq models, read from the Task 2 report if it exists."""
    if not path.exists():
        logger.warning(f"{path} not found, no side by side numbers")
        return {}
    report = json.loads(path.read_text())
    return {n: report["test"][n] for n in OWN_NAMES if n in report["test"]} | {
        "source": str(path),
        "train_config": report.get("config", {}).get("train"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task2.yaml")
    parser.add_argument("--own-report", default="reports/task2.json")
    parser.add_argument("--output", default="reports/task2_ref_seq2seq.json")
    parser.add_argument("--epochs", type=int, default=None, help="override the config epochs")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on a small slice")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tr = cfg["train"]
    out_path = Path(args.output)
    if args.smoke:
        cfg["data"]["max_rows"] = cfg["smoke"]["max_rows"]
        tr["epochs"] = cfg["smoke"]["epochs"]
        out_path = out_path.with_name(out_path.stem + "_smoke.json")
    if args.epochs:
        tr["epochs"] = args.epochs
    period = cfg["data"].pop("period", None)
    ts_cfg = TimeSeriesConfig(**cfg["data"])
    device = pick_device()
    data = build_time_series(ts_cfg)
    sizes = {k: len(v[0]) for k, v in data.splits.items()}
    logger.info(f"device {device} features {data.n_features} windows {sizes}")

    y_true = {s: data.inverse_target(data.splits[s][1]) for s in ("val", "test")}
    loss_fn = nn.HuberLoss() if tr["loss"] == "huber" else nn.MSELoss()
    train_loader = data.loader("train", tr["batch_size"], shuffle=True)

    def step_loss(model: nn.Module, batch: tuple, epoch: int) -> torch.Tensor:
        x, y = batch
        ratio = teacher_forcing_ratio(epoch, tr["tf_start"], tr["tf_decay"], tr["tf_floor"])
        return loss_fn(model(x, y, ratio).squeeze(-1), y)

    def val_metric(m: nn.Module) -> float:
        return rmse(y_true["val"], predict(m, data, "val", device))

    set_seed(cfg["seed"])
    model = TSSeq2SeqForecaster(data.n_features, data.target_idx, ts_cfg.horizon, **cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    model = fit(
        model, train_loader, step_loss, val_metric, epochs=tr["epochs"], lr=tr["lr"],
        clip=tr["clip"], patience=tr["patience"], device=device,
    )  # fmt: skip
    seconds = round(time.time() - t0, 1)

    # Validation guided the epoch choice above. Test is scored once.
    val = horizon_report(y_true["val"], predict(model, data, "val", device))
    test = horizon_report(y_true["test"], predict(model, data, "test", device))
    logger.info(f"{NAME} test mean {test['mean_over_horizons']}")

    results = {
        "model": NAME,
        "source": "https://github.com/leehyeonbeen/TimeSeriesSeq2Seq (MIT) commit 9b98154",
        "evaluation": "autoregressive, no teacher forcing, original units",
        "val": val,
        "test": test,
        "own_test": own_results(Path(args.own_report)),
        "train_seconds": seconds,
        "n_params": n_params,
        "config": cfg | {"period": period},
        "n_windows": sizes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
