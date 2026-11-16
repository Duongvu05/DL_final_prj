"""Task 2. Seq2seq with and without attention forecasting the next `horizon` steps."""

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
from dl_final_prj.evaluation.regression_metrics import horizon_report, rmse
from dl_final_prj.models.seq2seq_forecaster import Seq2SeqForecaster, teacher_forcing_ratio
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
    """Free-running predictions (no teacher forcing) in original units, shape (n, horizon)."""
    model.eval()
    out = [model(x.to(device)).squeeze(-1).cpu().numpy() for x, _ in data.loader(split, 512)]
    return data.inverse_target(np.concatenate(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task2.yaml")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on a small slice")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tr = cfg["train"]
    out_path = Path(cfg["output"])
    if args.smoke:
        cfg["data"]["max_rows"] = cfg["smoke"]["max_rows"]
        tr["epochs"] = cfg["smoke"]["epochs"]
        out_path = out_path.with_name(out_path.stem + "_smoke.json")
    period = cfg["data"].pop("period", None)
    ts_cfg = TimeSeriesConfig(**cfg["data"])
    device = pick_device()
    data = build_time_series(ts_cfg)
    sizes = {k: len(v[0]) for k, v in data.splits.items()}
    logger.info(f"device {device} features {data.n_features} windows {sizes}")

    y_true = {s: data.inverse_target(data.splits[s][1]) for s in ("val", "test")}
    results: dict[str, dict] = {"val": {}, "test": {}, "train_seconds": {}}

    # Baselines use the same windows. Test is scored once per method.
    for split in ("val", "test"):
        preds = baseline_predictions(
            data.splits["train"], data.splits[split][0], data.target_idx,
            ts_cfg.offset, ts_cfg.horizon, period,
        )  # fmt: skip
        for name, p in preds.items():
            results[split][name] = horizon_report(y_true[split], data.inverse_target(p))

    loss_fn = nn.HuberLoss() if tr["loss"] == "huber" else nn.MSELoss()
    train_loader = data.loader("train", tr["batch_size"], shuffle=True)

    def step_loss(model: nn.Module, batch: tuple, epoch: int) -> torch.Tensor:
        x, y = batch
        ratio = teacher_forcing_ratio(epoch, tr["tf_start"], tr["tf_decay"], tr["tf_floor"])
        return loss_fn(model(x, y, ratio).squeeze(-1), y)

    for var in cfg["variants"]:
        set_seed(cfg["seed"])
        logger.info(f"training {var['name']}")
        model = Seq2SeqForecaster(
            data.n_features, data.target_idx, ts_cfg.horizon, attention=var["attention"],
            **cfg["model"],
        )  # fmt: skip

        def val_metric(m: nn.Module) -> float:
            return rmse(y_true["val"], predict(m, data, "val", device))

        t0 = time.time()
        model = fit(
            model, train_loader, step_loss, val_metric, epochs=tr["epochs"], lr=tr["lr"],
            clip=tr["clip"], patience=tr["patience"], device=device,
        )  # fmt: skip
        results["train_seconds"][var["name"]] = round(time.time() - t0, 1)
        for split in ("val", "test"):
            results[split][var["name"]] = horizon_report(
                y_true[split], predict(model, data, split, device)
            )
        logger.info(f"{var['name']} test mean {results['test'][var['name']]['mean_over_horizons']}")

    results["config"] = cfg | {"period": period}
    results["n_windows"] = sizes
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
