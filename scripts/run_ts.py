"""Train (or resume) one Task 1/2 neural model for one seed and write its predictions and metrics.

The models predict the standardized seasonal residual r(tau) = y(tau) - y(tau-24) (plan
Sections 2.1 and 2.2). The loss is MSE on that standardized residual. Validation and all
reported metrics are in kW after adding back the daily base y(tau-24).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

from dl_final_prj.electricity import ElectricityData
from dl_final_prj.engine import (
    FitConfig,
    file_identity,
    fit,
    pick_device,
    prepare_run_dir,
    set_seed,
    write_json,
)
from dl_final_prj.metrics import horizon_report, regression_report
from dl_final_prj.ts_models import BiRNNForecaster, Seq2SeqForecaster, count_params


def build_model(kind: str, args: dict, data: ElectricityData) -> torch.nn.Module:
    common = {"horizon": data.horizon, "n_target_feats": data.n_target_feats}
    if kind == "birnn":
        return BiRNNForecaster(data.n_features, **common, **args)
    if kind == "seq2seq":
        return Seq2SeqForecaster(data.n_features, **common, **args)
    raise ValueError(kind)


def tf_ratio(tf: dict | None, epoch: int) -> float:
    """Linear decay from start (epoch 1) to end (epoch decay_epochs), then constant."""
    if not tf:
        return 0.0
    frac = min(1.0, (epoch - 1) / max(1, tf["decay_epochs"] - 1))
    return tf["start"] + frac * (tf["end"] - tf["start"])


@torch.no_grad()
def predict(model, data: ElectricityData, split: str, bs: int, keep_attention: bool = False):
    """kW predictions (n, H) for every origin of `split` in time order, plus test attention."""
    model.eval()
    preds, attn = [], []
    for b in data.iterate(split, bs):
        p, aux = model(b)
        preds.append(p.float().cpu().numpy())
        if keep_attention and "attention" in aux:
            attn.append(aux["attention"].float().cpu().numpy().astype(np.float16))
    pred = data.inverse(np.concatenate(preds), data.origins[split])
    return pred, (np.concatenate(attn) if attn else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, choices=(1, 2), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--npz", default=None, help="override the electricity npz path")
    ap.add_argument("--out", default=None, help="override the results root")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, written under smoke/")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    tcfg = cfg[f"task{a.task}"]
    mcfg = tcfg["models"][a.model]
    root = Path(a.out or cfg["results_root"]) / ("smoke" if a.smoke else "phase1")
    run_dir = root / "runs" / f"task{a.task}" / a.model / f"seed{a.seed}"
    if (run_dir / "metrics.json").exists() and not a.force:
        logger.info(f"{run_dir} already done")
        return
    npz = Path(a.npz or cfg["electricity_npz"])
    fingerprint = {
        "task": {k: v for k, v in tcfg.items() if k != "models"},
        "model": mcfg,
        "seed": a.seed,
        "smoke": a.smoke,
        "data": file_identity(npz),
    }
    prepare_run_dir(run_dir, fingerprint, a.force)
    set_seed(a.seed)  # seeds the weight initialization too, not only training
    device = pick_device()
    data = ElectricityData(
        Path(a.npz or cfg["electricity_npz"]),
        tcfg["lookback"],
        tuple(tcfg["offsets"]),
        device,
        target_only=tcfg.get("target_only", True),
    )
    model = build_model(mcfg["kind"], mcfg["args"], data).to(device)
    fcfg = FitConfig(**tcfg["fit"])
    if a.smoke:
        fcfg.epochs = 2
    tf = tcfg.get("teacher_forcing")
    y_val = data.true_targets("val")
    loss_fn = torch.nn.MSELoss()
    bs, ebs = tcfg["batch_size"], tcfg["eval_batch_size"]

    def train_batches(epoch: int, g: torch.Generator):
        return data.iterate("train", bs, g)

    def step_loss(m, b, epoch: int):
        p, _ = m(b, teacher_forcing=tf_ratio(tf, epoch))
        return loss_fn(p, b.y)

    def validate(m) -> float:
        p, _ = predict(m, data, "val", ebs)
        return float(np.sqrt(np.mean((p - y_val) ** 2)))

    t0 = time.time()
    model, history = fit(
        run_dir,
        model,
        [{"params": model.parameters()}],
        train_batches,
        step_loss,
        validate,
        fcfg,
        a.seed,
    )
    train_seconds = sum(h["seconds"] for h in history)

    report = horizon_report if a.task == 2 else (lambda y, p: regression_report(y, p))
    metrics = {}
    for split in ("val", "test"):
        pred, attn = predict(model, data, split, ebs, keep_attention=split == "test")
        y = data.true_targets(split)
        metrics[split] = report(y, pred)
        o = data.origins[split]
        arrays = {
            "origin": o,
            "hour": data.hours[o].to_numpy("datetime64[s]"),
            "y_true": y,
            "y_pred": pred,
            "base_b24": data.base_b24(o),
        }
        if attn is not None:
            arrays["attention"] = attn
        np.savez_compressed(run_dir / f"pred_{split}.npz", **arrays)

    best_epoch = int(np.argmin([h["val"] for h in history])) + 1
    spec = {
        "task": a.task,
        "model": a.model,
        "seed": a.seed,
        "title": tcfg["title"],
        "kind": mcfg["kind"],
        "architecture": mcfg["args"],
        "n_params": count_params(model),
        "input_shape": [tcfg["lookback"], data.n_features],
        "input_features": data.feature_names,
        "target_features_shape": [data.horizon, data.n_target_feats],
        "output_shape": [data.horizon],
        "offsets": list(data.offsets),
        "features": data.feature_summary(),
        "loss": "MSE on the train-standardized seasonal residual",
        "validation_metric": "RMSE in kW after adding back y(tau-24)",
        "optimizer": "AdamW",
        "fit": vars(fcfg),
        "batch_size": bs,
        "teacher_forcing": tf,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "wall_seconds_this_session": time.time() - t0,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "checkpoint": str(run_dir / "ckpt.pt"),
        "splits": data.split_summary(),
    }
    write_json(history, run_dir / "history.json")
    write_json(spec, run_dir / "spec.json")
    write_json(metrics, run_dir / "metrics.json")
    logger.info(f"{run_dir} test {metrics['test'] if a.task == 1 else metrics['test']['pooled']}")


if __name__ == "__main__":
    main()
