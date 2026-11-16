"""SOTA forecasters (PatchTST, iTransformer) for Task 1 (y[t+2]) and Task 2 (y[t+1..t+5]).

Windows, scaler and metrics are the shared ones, so the numbers compare fairly with the RNN runs
and with the baselines written into the same report. Test is scored once after training.
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
from dl_final_prj.evaluation.baselines import baseline_predictions
from dl_final_prj.evaluation.regression_metrics import horizon_report, regression_report, rmse
from dl_final_prj.models.sota_forecasters import MODELS, SOTAForecaster
from dl_final_prj.training.trainer import fit
from dl_final_prj.utils.seed import set_seed

MAX_GPU_GB = 2.0


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, MAX_GPU_GB / total))
    return "cuda"


@torch.no_grad()
def predict(model: nn.Module, data: TimeSeriesData, split: str, device: str) -> np.ndarray:
    """Predictions in original units, shape (n, horizon)."""
    model.eval()
    out = [model(x.to(device)).cpu().numpy() for x, _ in data.loader(split, batch_size=512)]
    return data.inverse_target(np.concatenate(out))


def score(y_true: np.ndarray, y_pred: np.ndarray, last_only: bool) -> dict:
    """Task 1 scores the last step only (y[t+2]). Task 2 scores every step."""
    if last_only:
        return regression_report(y_true[:, -1], y_pred[:, -1])
    return horizon_report(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--config", default="configs/sota_forecast.yaml")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on a small slice")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tr, task = cfg["train"], cfg["tasks"][args.task]
    if args.smoke:
        cfg["data"]["max_rows"] = cfg["smoke"]["max_rows"]
        tr["epochs"] = cfg["smoke"]["epochs"]
    out_path = Path(cfg["output"].format(task=args.task, model=args.model))
    if args.smoke:
        out_path = out_path.with_name(out_path.stem + "_smoke.json")
    period = cfg["data"].pop("period", None)
    ts_cfg = TimeSeriesConfig(**cfg["data"], offset=task["offset"], horizon=task["horizon"])
    last_only = task["score_last_step_only"]
    device = pick_device()
    data = build_time_series(ts_cfg)
    sizes = {k: len(v[0]) for k, v in data.splits.items()}
    logger.info(f"task {args.task} {args.model} device {device} windows {sizes}")

    y_true = {s: data.inverse_target(data.splits[s][1]) for s in ("val", "test")}
    results: dict = {"val": {}, "test": {}, "train_seconds": {}}

    # Baselines on the same windows. For Task 1 the model windows end at t+2 with offset=1, so
    # the scored target y[t+2] is the last column and baselines use offset=2, horizon=1.
    base_off, base_hor = (ts_cfg.offset + ts_cfg.horizon - 1, 1) if last_only else (1, 5)
    base_train = data.splits["train"][0], data.splits["train"][1][:, -base_hor:]
    for split in ("val", "test"):
        preds = baseline_predictions(
            base_train, data.splits[split][0], data.target_idx, base_off, base_hor, period
        )
        for name, p in preds.items():
            truth = y_true[split][:, -base_hor:]
            results[split][name] = score(truth, data.inverse_target(p), last_only)

    set_seed(cfg["seed"])
    model = SOTAForecaster(
        args.model, data.n_features, data.target_idx, ts_cfg.lookback, ts_cfg.horizon,
        **cfg["models"][args.model],
    )  # fmt: skip
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = nn.HuberLoss() if tr["loss"] == "huber" else nn.MSELoss()
    train_loader = data.loader("train", tr["batch_size"], shuffle=True)

    def step_loss(m: nn.Module, batch: tuple, epoch: int) -> torch.Tensor:
        x, y = batch
        return loss_fn(m(x), y)

    def val_metric(m: nn.Module) -> float:
        pred, truth = predict(m, data, "val", device), y_true["val"]
        if last_only:
            pred, truth = pred[:, -1], truth[:, -1]
        return rmse(truth, pred)

    t0 = time.time()
    model = fit(
        model, train_loader, step_loss, val_metric, epochs=tr["epochs"], lr=tr["lr"],
        clip=tr["clip"], patience=tr["patience"], device=device,
    )  # fmt: skip
    results["train_seconds"][args.model] = round(time.time() - t0, 1)
    for split in ("val", "test"):
        pred = predict(model, data, split, device)
        results[split][args.model] = score(y_true[split], pred, last_only)
    logger.info(f"{args.model} test {results['test'][args.model]}")

    results["config"] = cfg | {"period": period, "task": args.task, "model": args.model}
    results["n_params"] = n_params
    results["n_windows"] = sizes
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
