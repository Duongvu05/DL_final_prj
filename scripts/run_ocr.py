"""Train (or resume) one Task 3 recognizer on cropped SVHN for one seed, then evaluate it.

Selection is early stopping on validation exact-match (greedy). The restored best checkpoint
is evaluated on val, test (primary, oracle GT-box crop), test_length_constrained and
test_beam3 (GRU decoders only) and test_whole (whole-image letterbox stress test).

Phase 2 (configs/phase2_task3.yaml, `phase: phase2`) adds the token-mixer kinds of
ocr_mamba. Their runs go to results/phase2/... and Mamba runs also save the mean dt per
visual token on the test set (mamba_dt_test.npz).
"""

import argparse
import json
import math
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
from dl_final_prj.ocr_mamba import P0_PARAMS, MixerAttentionGRURecognizer
from dl_final_prj.ocr_models import build
from dl_final_prj.svhn import DIGIT0, EOS, PAD, SOS, VOCAB, OcrBatch, SvhnSplit, encode
from dl_final_prj.ts_models import count_params

MIXER_KINDS = ("mamba_attention", "gru_mixer_attention")  # Phase 2, ocr_mamba
GRU_DECODER_KINDS = ("attention", "pooled", *MIXER_KINDS)
ATTENTION_KINDS = ("attention", *MIXER_KINDS)
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


@torch.no_grad()
def mamba_diagnostics(model, split: SvhnSplit, bs: int, amp: bool) -> dict | None:
    """Mean dt per visual token (N, n_dirs, n_layers, 49) for a Mamba token mixer, else None."""
    if not isinstance(model, MixerAttentionGRURecognizer):
        return None
    model.eval()
    dts, diag = [], {}
    for b in split.iterate(bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            diag = model.mixer_diagnostics(b.images)
        if "dt_mean" not in diag:
            return None
        dts.append(diag["dt_mean"].float().cpu().numpy())
    return {
        "dt_mean": np.concatenate(dts),
        "directions": np.array(diag["directions"]),
        "static": np.array(diag["static"]),
        "notes": np.array(diag["notes"]),
    }


def mixer_architecture(model: MixerAttentionGRURecognizer, crop: int) -> dict:
    """Extra spec.json architecture keys of a Phase 2 token-mixer recognizer."""
    enc = model.encoder
    return {
        "encoder_grid": [512, 7, 7],
        "token_projection": f"Linear(512, {enc.dim}) per visual token (raster order)",
        "position_embeddings": f"learned row/col (7 each), {enc.dim}-d, added after the projection"
        if enc.row is not None
        else None,
        "token_mixer": enc.describe_mixer(),
        "token_mixer_precision": "fp32 (autocast off), CNN under autocast",
        "decoder_token_dim": enc.dim,
        "n_params_breakdown": model.param_breakdown(),
        "input_crop": crop,
    }


def param_match(model: MixerAttentionGRURecognizer) -> dict:
    """spec.json record of the plan 2.3 P0 budget deviation (see configs/phase2_task3.yaml)."""
    n = count_params(model)
    return {
        "matched_control": "t3_m1 (and t3_m2..t3_m4), all within 5% of t3_m0",
        "t3_p0_cnn_attention_gru_params": P0_PARAMS,
        "rel_diff_vs_t3_p0": n / P0_PARAMS - 1,
        "note": "plan 2.3 asks for T3-M0 within 5% of T3-P0. The plan's fixed Mamba settings "
        "(two residual bidirectional blocks, d_model 256, expand 2, d_state 16) make that "
        "impossible, so T3-M0 vs T3-P0 is not a budget-matched contrast",
    }


def bench(model, a, cfg: dict, device: str, amp: bool) -> None:
    """Times `a.bench` training steps (AdamW, clip, bf16 autocast on CUDA) on random images
    and labels at the config batch size. Logs ms/step, peak CUDA memory and the projected
    train time per epoch and per run (validation and evaluation excluded). Writes nothing."""
    bs, max_len, fcfg = cfg["batch_size"], cfg["max_len"], FitConfig(**cfg["fit"])
    if a.compile:
        enc = model.encoder
        enc.mixer = torch.compile(enc.mixer)  # type: ignore[assignment]
    g = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, max_len + 1, (bs,), generator=g)
    labels = [
        "".join(str(int(d)) for d in torch.randint(0, 10, (int(n),), generator=g)) for n in lengths
    ]
    tokens = torch.tensor([encode(s, max_len) for s in labels], device=device)
    images = torch.randn(bs, 3, cfg["crop"], cfg["crop"], generator=g).to(device)
    batch = OcrBatch(images, tokens, (lengths - 1).to(device), torch.arange(bs, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=fcfg.lr, weight_decay=fcfg.weight_decay)
    model.train()

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = model.loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), fcfg.clip)
        opt.step()

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(a.bench_warmup):
        step()
    sync()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(a.bench):
        step()
    sync()
    sec = (time.perf_counter() - t0) / a.bench
    labels_path = Path(a.cache or cfg["svhn_dir"]) / "train_labels.json"
    n_train = len(json.loads(labels_path.read_text())) if labels_path.exists() else None
    epoch_s = math.ceil(n_train / bs) * sec if n_train else None
    out = {
        "model": a.model,
        "scan": (cfg["models"][a.model].get("args") or {}).get("mamba", {}).get("scan"),
        "compile": a.compile,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "batch_size": bs,
        "ms_per_step": 1000 * sec,
        "peak_cuda_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
        "n_train": n_train,
        "train_min_per_epoch": epoch_s / 60 if epoch_s else None,
        "train_h_per_run": epoch_s * fcfg.epochs / 3600 if epoch_s else None,
    }
    logger.info(f"bench {json.dumps(out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/phase1_task3.yaml")
    ap.add_argument("--cache", default=None, help="override the SVHN cache directory")
    ap.add_argument("--out", default=None, help="override results_root")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 2,000/500/500 images")
    ap.add_argument("--force", action="store_true")
    bench_help = "time N training steps on random data and exit, no run directory (Phase 2)"
    ap.add_argument("--bench", type=int, default=0, metavar="N", help=bench_help)
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--scan", choices=["parallel", "loop"], help="--bench only: Mamba scan")
    ap.add_argument("--compile", action="store_true", help="--bench only: torch.compile mixer")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    mcfg = cfg["models"][a.model]
    kind = mcfg["kind"]
    if a.bench:
        if a.scan:
            args = mcfg.setdefault("args", {})
            args["mamba"] = dict(args.get("mamba") or {}) | {"scan": a.scan}
        device = pick_device()
        set_seed(a.seed)
        extra = (
            {"length_loss_weight": cfg["length_loss_weight"]} if kind in GRU_DECODER_KINDS else {}
        )
        model = build(kind, mcfg.get("args"), cfg["max_len"], **extra).to(device)
        fcfg = FitConfig(**cfg["fit"])
        bench(model, a, cfg, device, fcfg.amp == "bf16" and device == "cuda")
        return
    if a.scan or a.compile:
        ap.error("--scan and --compile only apply to --bench (set mamba.scan in the config)")
    phase = cfg.get("phase", "phase1")
    root = Path(a.out or cfg["results_root"]) / ("smoke" if a.smoke else phase)
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
    extra = {"length_loss_weight": cfg["length_loss_weight"]} if kind in GRU_DECODER_KINDS else {}
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
        keep = key == "test" and kind in ATTENTION_KINDS
        preds, attn = predict(model, data[split], mode, ebs, amp, keep_attention=keep)
        metrics[key] = sequence_report(data[split].labels, preds)
        rows = [
            {"name": r["name"], "dir": r["dir"], "target": r["label"], "pred": p}
            for r, p in zip(data[split].records, preds)
        ]
        (run_dir / f"pred_{key}.json").write_text(json.dumps(rows))
        if attn is not None:
            np.savez_compressed(run_dir / "attention_test.npz", attention=attn)

    diag = mamba_diagnostics(model, data["test"], ebs, amp)
    if diag is not None:
        np.savez_compressed(run_dir / "mamba_dt_test.npz", **diag)

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
    vocab = (
        {
            "attention": gru_vocab,
            "pooled": gru_vocab,
            "multihead": {"length_classes": f"1..{max_len}", "digit_classes": "0..9 per head"},
            "ctc": {"blank": 0, "digits": "class = 1 + digit", "size": 11},
        }[kind]
        if kind not in MIXER_KINDS
        else gru_vocab
    )
    grid = [512, 7, 14] if kind == "ctc" else [512, 7, 7]
    d_tok = model.encoder.dim if isinstance(model, MixerAttentionGRURecognizer) else 512
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
        }.get(
            kind,
            f"Linear({d_tok}, {max_len}) on the mean mixed token, lengths 1..{max_len}, "
            f"CE weight {w_len}, used by length-constrained decoding",
        ),
        "loss": {
            "attention": f"token CE (PAD ignored) + {cfg['length_loss_weight']} * length CE",
            "pooled": f"token CE (PAD ignored) + {cfg['length_loss_weight']} * length CE",
            "multihead": "length CE + sum of digit CE over the true positions",
            "ctc": "CTC (fp32)",
        }.get(kind, f"token CE (PAD ignored) + {cfg['length_loss_weight']} * length CE"),
        "teacher_forcing": "full" if kind in GRU_DECODER_KINDS else None,
        "decoding_modes": {k: MODE_TEXT[m] for k, (_, m) in EVALS.items() if m in modes}
        if kind in GRU_DECODER_KINDS
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
    if isinstance(model, MixerAttentionGRURecognizer):
        spec["architecture"] |= mixer_architecture(model, crop)
        spec["phase"] = phase
        spec["param_match"] = param_match(model)
        spec["mamba_diagnostics"] = (
            {
                "file": "mamba_dt_test.npz",
                "dt_mean": "test set, (N, n_directions, n_layers, 49) mean dt over channels per "
                "visual token, raster order, reverse direction mapped back to original order",
                "notes": str(diag["notes"]),
            }
            if diag is not None
            else None
        )
    write_json(history, run_dir / "history.json")
    write_json(spec, run_dir / "spec.json")
    write_json(metrics, run_dir / "metrics.json")
    logger.info(
        f"{run_dir} test exact {metrics['test']['exact_match']:.4f} "
        f"RMSE {metrics['test']['RMSE']:.4f}, whole exact {metrics['test_whole']['exact_match']:.4f}"
    )


if __name__ == "__main__":
    main()
