"""Train (or resume) one Task 3 SVHN recognizer for one seed and write predictions and metrics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

from dl_final_prj.engine import FitConfig, fit, pick_device, set_seed, write_json
from dl_final_prj.metrics import sequence_report
from dl_final_prj.ocr_models import (
    AttentionRecognizer,
    CTCRecognizer,
    PooledRecognizer,
    ctc_greedy,
    ctc_loss,
    greedy_strings,
    seq_loss,
)
from dl_final_prj.svhn import SvhnSplit
from dl_final_prj.ts_models import count_params


def build(kind: str, args: dict, max_len: int) -> torch.nn.Module:
    if kind == "attention":
        return AttentionRecognizer(max_len, **args)
    if kind == "pooled":
        return PooledRecognizer(max_len, **args)
    if kind == "ctc":
        return CTCRecognizer(**args)
    raise ValueError(kind)


def tf_ratio(tf: dict, epoch: int) -> float:
    """1.0 for the first hold_epochs, then linear to `end` at end_epoch, then constant."""
    if epoch <= tf["hold_epochs"]:
        return tf["start"]
    frac = min(1.0, (epoch - tf["hold_epochs"]) / max(1, tf["end_epoch"] - tf["hold_epochs"]))
    return tf["start"] + frac * (tf["end"] - tf["start"])


@torch.no_grad()
def predict(model, kind: str, split: SvhnSplit, bs: int, amp: bool, keep_attention: bool = False):
    model.eval()
    preds, attn = [], []
    for b in split.iterate(bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, aux = model(b, teacher_forcing=0.0)
        preds += ctc_greedy(logits) if kind == "ctc" else greedy_strings(logits)
        if keep_attention and "attention" in aux:
            attn.append(aux["attention"].float().cpu().numpy().astype(np.float16))
    return preds, (np.concatenate(attn) if attn else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--cache", default=None, help="override the SVHN cache directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 2,000 train images")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    tcfg = cfg["task3"]
    mcfg = tcfg["models"][a.model]
    root = Path(a.out or cfg["results_root"]) / ("smoke" if a.smoke else "phase1")
    run_dir = root / "runs" / "task3" / a.model / f"seed{a.seed}"
    if (run_dir / "metrics.json").exists() and not a.force:
        logger.info(f"{run_dir} already done")
        return
    set_seed(a.seed)  # seeds the weight initialization too, not only training
    device = pick_device()
    cache = Path(a.cache or cfg["svhn_dir"])
    max_len = tcfg["max_len"]
    data = {s: SvhnSplit(cache, s, max_len, device) for s in ("train", "val", "test")}
    if a.smoke:
        for s, n in (("train", 2000), ("val", 500), ("test", 500)):
            d = data[s]
            d.images, d.tokens, d.labels = d.images[:n], d.tokens[:n], d.labels[:n]
            d.records = d.records[:n]
    kind = mcfg["kind"]
    model = build(kind, mcfg["args"], max_len).to(device)
    fcfg = FitConfig(**tcfg["fit"])
    amp = fcfg.amp == "bf16" and device == "cuda"
    if not amp:
        fcfg.amp = "none"
    if a.smoke:
        fcfg.epochs = 2
    cnn_params = list(model.get_submodule("cnn").parameters())
    cnn_ids = {id(p) for p in cnn_params}
    groups = [
        {"params": cnn_params, "lr": fcfg.lr * tcfg["cnn_lr_mult"]},
        {"params": [p for p in model.parameters() if id(p) not in cnn_ids], "lr": fcfg.lr},
    ]
    bs, ebs = tcfg["batch_size"], tcfg["eval_batch_size"]
    aug = tcfg["augment"]
    tf = tcfg["teacher_forcing"]

    def on_epoch_start(m, epoch: int) -> None:
        m.cnn.set_stem_frozen(epoch <= tcfg["freeze_stem_epochs"])

    def train_batches(epoch: int, g: torch.Generator):
        return data["train"].iterate(bs, aug, g)

    def step_loss(m, b, epoch: int):
        logits, _ = m(b, teacher_forcing=tf_ratio(tf, epoch))
        if kind == "ctc":
            return ctc_loss(logits, b.tokens)
        return seq_loss(logits, b.tokens, tcfg["label_smoothing"])

    def validate(m) -> float:
        p, _ = predict(m, kind, data["val"], ebs, amp)
        return 1.0 - float(np.mean([x == y for x, y in zip(p, data["val"].labels)]))

    t0 = time.time()
    model, history = fit(
        run_dir, model, groups, train_batches, step_loss, validate, fcfg, a.seed, on_epoch_start
    )

    metrics = {}
    for s in ("val", "test"):
        preds, attn = predict(model, kind, data[s], ebs, amp, keep_attention=s == "test")
        metrics[s] = sequence_report(data[s].labels, preds)
        (run_dir / f"pred_{s}.json").write_text(
            json.dumps(
                [
                    {"name": r["name"], "target": r["label"], "pred": p}
                    for r, p in zip(data[s].records, preds)
                ]
            )
        )
        if attn is not None:
            np.savez_compressed(run_dir / f"attention_{s}.npz", attention=attn)

    best_epoch = int(np.argmin([h["val"] for h in history])) + 1
    spec = {
        "task": 3,
        "model": a.model,
        "seed": a.seed,
        "title": tcfg["title"],
        "kind": kind,
        "architecture": mcfg["args"]
        | {"backbone": "ResNet-18, layer3/layer4 first stride (2,1)", "feature_map": [512, 2, 32]},
        "n_params": count_params(model),
        "input_shape": [3, tcfg["height"], tcfg["width"]],
        "output": f"digit string, up to {max_len} digits"
        + (" (CTC, 32 steps)" if kind == "ctc" else " + EOS"),
        "loss": "CTC"
        if kind == "ctc"
        else f"cross-entropy, label smoothing {tcfg['label_smoothing']}",
        "optimizer": "AdamW",
        "fit": vars(fcfg),
        "cnn_lr_mult": tcfg["cnn_lr_mult"],
        "freeze_stem_epochs": tcfg["freeze_stem_epochs"],
        "teacher_forcing": None if kind == "ctc" else tf,
        "augment": aug,
        "decoding": "greedy" if kind != "ctc" else "CTC best path",
        "batch_size": bs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_seconds": sum(h["seconds"] for h in history),
        "wall_seconds_this_session": time.time() - t0,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "checkpoint": str(run_dir / "ckpt.pt"),
        "n_images": {s: len(d) for s, d in data.items()},
    }
    write_json(history, run_dir / "history.json")
    write_json(spec, run_dir / "spec.json")
    write_json(metrics, run_dir / "metrics.json")
    logger.info(
        f"{run_dir} test exact {metrics['test']['exact_match']:.4f} RMSE {metrics['test']['RMSE']:.4f}"
    )


if __name__ == "__main__":
    main()
