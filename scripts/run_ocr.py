"""Train (or resume) one Task 3 recognizer on cropped SVHN for one seed, then evaluate it.

Selection is early stopping on validation exact-match (greedy). The restored best checkpoint
is evaluated on val, test (primary, oracle GT-box crop), test_length_constrained and
test_beam3 (GRU decoders only) and test_whole (whole-image letterbox stress test).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

from dl_final_prj.engine import (
    FitConfig,
    file_identity,
    fit,
    pick_device,
    prepare_run_dir,
    set_seed,
    write_json,
)
from dl_final_prj.metrics import sequence_report
from dl_final_prj.ocr_models import build
from dl_final_prj.svhn import DIGIT0, EOS, PAD, SOS, VOCAB, SvhnSplit
from dl_final_prj.ts_models import count_params

SMOKE_SIZES = {"train": 2000, "val": 500, "test": 500, "test_whole": 500}
# metrics.json key -> (cached split, decoding mode)
EVALS = {
    "val": ("val", "greedy"),
    "test": ("test", "greedy"),
    "test_length_constrained": ("test", "length"),
    "test_beam3": ("test", "beam3"),
    "test_whole": ("test_whole", "greedy"),
}
MODE_TEXT = {
    "greedy": "greedy, unconstrained (primary)",
    "length": "greedy, constrained to the length-head prediction",
    "beam3": "beam search width 3, unconstrained, score / number of tokens",
}


@torch.no_grad()
def predict(model, split: SvhnSplit, mode: str, bs: int, amp: bool, keep_attention: bool = False):
    """Decoded strings for a split (center crops) and the stacked attention maps if asked."""
    model.eval()
    preds, attn = [], []
    for b in split.iterate(bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            p, aux = model.predict(b.images, mode)
        preds += p
        if keep_attention and "attention" in aux:
            attn.append(aux["attention"].float().cpu().numpy().astype(np.float16))
    return preds, (np.concatenate(attn) if attn else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/phase1_task3.yaml")
    ap.add_argument("--cache", default=None, help="override the SVHN cache directory")
    ap.add_argument("--out", default=None, help="override results_root")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 2,000/500/500 images")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    mcfg = cfg["models"][a.model]
    kind = mcfg["kind"]
    root = Path(a.out or cfg["results_root"]) / ("smoke" if a.smoke else "phase1")
    run_dir = root / "runs" / "task3" / a.model / f"seed{a.seed}"
    if (run_dir / "metrics.json").exists() and not a.force:
        logger.info(f"{run_dir} already done")
        return
    cache = Path(a.cache or cfg["svhn_dir"])
    fingerprint = {
        "model": mcfg,
        "config": {k: v for k, v in cfg.items() if k not in ("models", "seeds")},
        "seed": a.seed,
        "smoke": a.smoke,
        "data": file_identity(cache / "train_images.npy"),
    }
    prepare_run_dir(run_dir, fingerprint, a.force)
    device = pick_device()
    max_len, crop = cfg["max_len"], cfg["crop"]
    data = {
        s: SvhnSplit(
            cache,
            s,
            max_len,
            device,
            crop=crop,
            jitter=cfg["color_jitter"],
            limit=SMOKE_SIZES[s] if a.smoke else None,
        )
        for s in ("train", "val", "test", "test_whole")
    }
    logger.info(f"loaded {', '.join(f'{s} {len(d)}' for s, d in data.items())} on {device}")

    set_seed(a.seed)  # seeds the weight initialization too, not only training
    extra = (
        {"length_loss_weight": cfg["length_loss_weight"]} if kind in ("attention", "pooled") else {}
    )
    model = build(kind, mcfg.get("args"), max_len, **extra).to(device)
    fcfg = FitConfig(**cfg["fit"])
    amp = fcfg.amp == "bf16" and device == "cuda"
    if not amp:
        fcfg.amp = "none"
    if a.smoke:
        fcfg.epochs = 2
    bs, ebs = cfg["batch_size"], cfg["eval_batch_size"]
    modes = list(model.modes)  # type: ignore[arg-type]

    def train_batches(epoch: int, g: torch.Generator):
        return data["train"].iterate(bs, train=True, g=g)

    def step_loss(m, b, epoch: int) -> torch.Tensor:
        return m.loss(b)

    def validate(m) -> float:
        p, _ = predict(m, data["val"], "greedy", ebs, amp)
        return 1.0 - float(np.mean([x == y for x, y in zip(p, data["val"].labels)]))

    t0 = time.time()
    groups = [{"params": list(model.parameters())}]
    model, history = fit(run_dir, model, groups, train_batches, step_loss, validate, fcfg, a.seed)

    metrics = {}
    for key, (split, mode) in EVALS.items():
        if mode not in modes:
            continue
        keep = key == "test" and kind == "attention"
        preds, attn = predict(model, data[split], mode, ebs, amp, keep_attention=keep)
        metrics[key] = sequence_report(data[split].labels, preds)
        rows = [
            {"name": r["name"], "dir": r["dir"], "target": r["label"], "pred": p}
            for r, p in zip(data[split].records, preds)
        ]
        (run_dir / f"pred_{key}.json").write_text(json.dumps(rows))
        if attn is not None:
            np.savez_compressed(run_dir / "attention_test.npz", attention=attn)

    best_epoch = int(np.argmin([h["val"] for h in history])) + 1
    audit_path = cache / "svhn_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    dropped = audit.get("dropped_long_labels", {})
    w_len = cfg["length_loss_weight"]
    fc_dim = (mcfg.get("args") or {}).get("fc_dim", 1024)
    gru_vocab = {
        "PAD": PAD,
        "SOS": SOS,
        "EOS": EOS,
        "digits": f"token = {DIGIT0} + digit",
        "size": VOCAB,
    }
    vocab = {
        "attention": gru_vocab,
        "pooled": gru_vocab,
        "multihead": {"length_classes": f"1..{max_len}", "digit_classes": "0..9 per head"},
        "ctc": {"blank": 0, "digits": "class = 1 + digit", "size": 11},
    }[kind]
    grid = [512, 7, 14] if kind == "ctc" else [512, 7, 7]
    spec = {
        "task": 3,
        "model": a.model,
        "seed": a.seed,
        "title": cfg["title"],
        "kind": kind,
        "architecture": dict(mcfg.get("args") or {})
        | {
            "encoder": "VGG-BN, 4 stages (64,128,256,512) x 2 conv3x3+BN+ReLU, Dropout2d 0.1",
            "pooling": "(2,2),(2,2),(2,1)" if kind == "ctc" else "(2,2),(2,2),(2,2)",
            "encoder_grid": grid,
            "position_embeddings": "learned row/col (7 each)"
            if kind in ("attention", "pooled")
            else None,
        },
        "n_params": count_params(model),
        "input_shape": [3, crop, crop],
        "crop_protocol": {
            "cache": f"GT digit-box union squared, expanded 1.3x, clamped, resized to "
            f"{cfg['size']}x{cfg['size']} ignoring aspect (oracle localization at test)",
            "train": f"random {crop}x{crop} crop, ColorJitter brightness/contrast/saturation "
            f"factors in [{1 - cfg['color_jitter']:.1f}, {1 + cfg['color_jitter']:.1f}]",
            "eval": f"center {crop}x{crop} crop",
            "test_whole": f"whole image letterboxed to {cfg['size']}x{cfg['size']}, center crop",
            "normalization": "mean 0.5, std 0.5 per channel",
        },
        "output": f"digit string, 1 to {max_len} digits",
        "vocabulary": vocab,
        "length_head": {
            "attention": f"Linear(512, {max_len}) on the mean visual token, lengths 1..{max_len}, "
            f"CE weight {w_len}, used by length-constrained decoding",
            "pooled": f"Linear(512, {max_len}) on the mean visual token, lengths 1..{max_len}, "
            f"CE weight {w_len}, used by length-constrained decoding",
            "multihead": f"Linear({fc_dim}, {max_len}) on the FC trunk, lengths 1..{max_len}, "
            "CE weight 1, decides how many digit heads are read",
            "ctc": None,
        }[kind],
        "loss": {
            "attention": f"token CE (PAD ignored) + {cfg['length_loss_weight']} * length CE",
            "pooled": f"token CE (PAD ignored) + {cfg['length_loss_weight']} * length CE",
            "multihead": "length CE + sum of digit CE over the true positions",
            "ctc": "CTC (fp32)",
        }[kind],
        "teacher_forcing": "full" if kind in ("attention", "pooled") else None,
        "decoding_modes": {k: MODE_TEXT[m] for k, (_, m) in EVALS.items() if m in modes}
        if kind in ("attention", "pooled")
        else {
            "multihead": "argmax length, then the first L digit heads",
            "ctc": "CTC best path",
        }[kind],
        "optimizer": "AdamW",
        "fit": vars(fcfg),
        "batch_size": bs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "selection": "early stopping on validation exact-match (greedy)",
        "train_seconds": sum(h["seconds"] for h in history),
        "wall_seconds_this_session": time.time() - t0,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "checkpoint": str(run_dir / "ckpt.pt"),
        "n_images": {s: len(d) for s, d in data.items()},
        "n_images_by_source": {
            s: {src: sum(r["dir"] == src for r in d.records) for src in ("train", "extra", "test")}
            for s, d in data.items()
        },
        "source_counts_raw": audit.get("counts_per_source"),
        "dropped_long_labels": {k: dropped.get(k) for k in ("rule", "count", "by_source")}
        if dropped
        else None,
        "smoke": a.smoke,
    }
    write_json(history, run_dir / "history.json")
    write_json(spec, run_dir / "spec.json")
    write_json(metrics, run_dir / "metrics.json")
    logger.info(
        f"{run_dir} test exact {metrics['test']['exact_match']:.4f} "
        f"RMSE {metrics['test']['RMSE']:.4f}, whole exact {metrics['test_whole']['exact_match']:.4f}"
    )


if __name__ == "__main__":
    main()
