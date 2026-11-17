"""Train (or resume) one Task 3 recognizer on cached SVHN for one seed, then evaluate it.

Settings. configs/phase1_task3.yaml and phase2_task3.yaml use the oracle GT-box crop cache.
configs/phase1_task3_whole.yaml sets `cache_suffix: "_full"` (whole images, no ground-truth
box at train, validation or test) and `stress_split: null`. `stress_split` (default
"test_whole") names an extra cached split evaluated with the same checkpoint (the
whole-image letterbox stress test of the crop setting). It is read without the suffix.

Checkpoint selection (training time) is early stopping on validation exact match with
greedy decoding. It is kept because re-selecting would need retraining.

Reported decoding is never greedy (ocr_models.PRIMARY_DECODING): beam search for the GRU
decoders, CTC prefix beam search for CRNN-CTC, exact MAP for the multi-head CNN. The width W
is chosen on the validation split from BEAM_WIDTHS by exact match (ties to the smaller W)
and then used for test and the stress split. metrics.json keys: val, test (primary),
val_greedy, test_greedy (diagnostic), test_length_constrained (GRU decoders, diagnostic),
<stress>, <stress>_greedy and "decoding" (primary, width, val_by_width).

--eval-only reloads ckpt.pt's best_state of a finished run (no training, ckpt.pt and
fingerprint.json untouched), backs metrics.json up once as metrics_v1_greedy.json when it
predates this decoding, and rewrites metrics.json, the pred_*.json files, attention_test.npz
and spec.json's decoding entries.

Phase 2 (configs/phase2_task3.yaml for Setting A and phase2_task3_whole.yaml for Setting B,
`phase: phase2`) adds the token-mixer kinds of ocr_mamba. Their runs go to results/phase2/...
and Mamba runs also save the mean dt per visual token on the test set (mamba_dt_test.npz).

`compile: true` (config key, default false, Phase 2 token-mixer models only) compiles the
token mixer in place with nn.Module.compile for training and evaluation, on CUDA only. It is
a no-op on CPU (logged). The in-place compile keeps the state_dict keys of the eager model
(no `_orig_mod.` prefix), so ckpt.pt, resume and --eval-only work across eager and compiled
runs. The requested and active values are recorded in spec.json ("compile") and the
requested value in fingerprint.json. Dynamo does not trace nn.GRU, so in the BiGRU control
the GRU calls run eager (spec.json "compile.eager_in_mixer"). Train time is therefore
compile dependent and is not a method comparison.

`reverse_target: true` (config key, default false, GRU decoders only) trains a right-to-left
decoder on the reversed digit strings (configs/phase1_task3_r2l.yaml and
phase1_task3_whole_r2l.yaml). Its decoded strings are reversed back, so validation, metrics
and pred files read left-to-right. spec.json gets a "reverse_target" entry. It is the
backward model of the DRAM-style rescoring in scripts/rescore_task3.py.

--nbest (GRU decoders, opt-in) also writes nbest_val.json.gz and nbest_test.json.gz: the
finished hypotheses of the primary beam per image (width = the validation-selected width,
or --nbest-width) with their summed and length-normalised log-probabilities and the
length-head log-probabilities. Nothing else in the run changes.
"""

import argparse
import gzip
import json
import math
import shutil
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
from dl_final_prj.ocr_mamba import (
    P0_PARAMS,
    P0_PARTS,
    PARAM_TOL,
    MixerAttentionGRURecognizer,
    rel_param_diff,
)
from dl_final_prj.ocr_models import (
    DECODING_TEXT,
    PRIMARY_DECODING,
    GRUDecoderRecognizer,
    NBest,
    build,
    decoder_family,
    primary_mode,
    search_widths,
    select_width,
)
from dl_final_prj.svhn import DIGIT0, EOS, PAD, SOS, VOCAB, OcrBatch, SvhnSplit, encode
from dl_final_prj.ts_models import count_params

MIXER_KINDS = ("mamba_attention", "gru_mixer_attention")  # Phase 2, ocr_mamba
GRU_DECODER_KINDS = ("attention", "pooled", *MIXER_KINDS)
ATTENTION_KINDS = ("attention", *MIXER_KINDS)
SMOKE_SIZES = {"train": 2000, "val": 500, "test": 500}
SMOKE_STRESS = 500
TRAINING_SELECTION = (
    "early stopping on validation exact-match with greedy decoding (training-time checkpoint "
    "selection rule, kept for the non-greedy evaluation because re-selecting needs retraining)"
)
WIDTH_SELECTION = (
    "validation exact match, ties to the smaller width, chosen before any test evaluation "
    "and reused for test and the stress split"
)
COMPILE_TARGET = (
    "encoder.mixer (the token mixer), torch.compile in place via nn.Module.compile, the same "
    "for every token-mixer model. The CNN, the projection and the GRU decoder stay eager"
)
# Dynamo does not trace recurrent modules. In a mixer that holds one (the t3_m1 BiGRU) each
# RNN call is a graph break and runs as eager cuDNN, so only RMSNorm and the fusion Linear
# are compiled there. allow_rnn stays off. Accuracy comparisons are not affected, but train
# time is compile dependent and is not a method comparison.
EAGER_RNN = (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)
COMPILE_TIMING_NOTE = (
    "train time depends on what torch.compile can trace (see eager_in_mixer), so it is not a "
    "method comparison between token mixers"
)


def compile_requested(cfg: dict) -> bool:
    return bool(cfg.get("compile", False))


def reverse_target(cfg: dict) -> bool:
    """Config key `reverse_target` (default false): a right-to-left GRU decoder."""
    return bool(cfg.get("reverse_target", False))


REVERSE_TEXT = (
    "right-to-left decoder: trained on the reversed digit string (tokens of label[::-1], EOS), "
    "decoded strings reversed back, so predictions and metrics read left-to-right. The "
    "attention steps of attention_test.npz run right-to-left"
)


def compile_mixer(model: MixerAttentionGRURecognizer, backend: str = "inductor") -> None:
    """torch.compile the token mixer in place. Unlike `torch.compile(module)` this does not
    wrap the module, so the state_dict keys stay those of the eager model. `backend` is
    inductor in runs. The tests use aot_eager to trace and run the mixer on CPU."""
    model.encoder.mixer.compile(backend=backend)


def eager_in_mixer(model) -> list[str]:
    """Recurrent module types in the token mixer that Dynamo leaves eager (graph breaks)."""
    if not isinstance(model, MixerAttentionGRURecognizer):
        return []
    found = {type(m).__name__ for m in model.encoder.mixer.modules() if isinstance(m, EAGER_RNN)}
    return sorted(found)


def strip_compile_prefix(state: dict) -> dict:
    """state_dict without `_orig_mod.` segments (left by a torch.compile(module) wrapper)."""
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def apply_compile(model, cfg: dict, device: str) -> dict:
    """Applies the config's `compile` key. Active only on CUDA (no-op on CPU, logged).
    Returns the spec.json record."""
    requested = compile_requested(cfg)
    if requested and not isinstance(model, MixerAttentionGRURecognizer):
        raise SystemExit("compile: true applies to the Phase 2 token-mixer models only")
    active = requested and device == "cuda"
    if active:
        assert isinstance(model, MixerAttentionGRURecognizer)
        compile_mixer(model)
        logger.info("compile: torch.compile on the token mixer (in place)")
    elif requested:
        logger.info(f"compile: requested but not active on {device} (CUDA only), eager run")
    return {
        "requested": requested,
        "active": active,
        "target": COMPILE_TARGET,
        "eager_in_mixer": eager_in_mixer(model),
        "timing_note": COMPILE_TIMING_NOTE,
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


def report(split: SvhnSplit, preds: list[str]) -> dict:
    return sequence_report(split.labels, preds)


def evaluate(
    model, data: dict[str, SvhnSplit], stress: str | None, bs: int, amp: bool, keep_attention: bool
):
    """All Task 3 evaluations of a trained model. The width is selected on data["val"] only,
    before data["test"] is touched. Returns (metrics, predictions by metrics key, attention
    arrays for attention_test.npz or None)."""
    fam = decoder_family(model)
    widths = search_widths(model)
    t0 = time.time()
    val_by_width: dict[str, float] = {}
    if widths:
        val_preds = {}
        for w in widths:
            val_preds[w], _ = predict(model, data["val"], primary_mode(model, w), bs, amp)
            val_by_width[str(w)] = report(data["val"], val_preds[w])["exact_match"]
            logger.info(f"val {primary_mode(model, w)} exact {val_by_width[str(w)]:.4f}")
        width: int | None = select_width({w: val_by_width[str(w)] for w in widths})
        val_primary = val_preds[width]
    else:
        width = None
        val_primary, _ = predict(model, data["val"], primary_mode(model, None), bs, amp)
    mode = primary_mode(model, width)
    preds: dict[str, list[str]] = {"val": val_primary}
    preds["val_greedy"], _ = predict(model, data["val"], "greedy", bs, amp)
    # test is only read from here on, with the width fixed above
    preds["test"], att = predict(model, data["test"], mode, bs, amp, keep_attention)
    preds["test_greedy"], att_greedy = predict(
        model, data["test"], "greedy", bs, amp, keep_attention
    )
    if fam == "gru":
        preds["test_length_constrained"], _ = predict(model, data["test"], "length", bs, amp)
    if stress is not None:
        preds[stress], _ = predict(model, data[stress], mode, bs, amp)
        preds[f"{stress}_greedy"], _ = predict(model, data[stress], "greedy", bs, amp)
    split_of = {k: data["val"] if k.startswith("val") else data["test"] for k in preds}
    if stress is not None:
        split_of[stress] = split_of[f"{stress}_greedy"] = data[stress]
    metrics: dict = {k: report(split_of[k], p) for k, p in preds.items()}
    greedy_text = DECODING_TEXT[f"greedy_{fam}"]
    metrics["decoding"] = {
        "primary": PRIMARY_DECODING[fam],
        "mode": mode,
        "width": width,
        "candidate_widths": list(widths),
        "val_by_width": val_by_width,
        "width_selection": WIDTH_SELECTION if widths else None,
        "description": DECODING_TEXT[PRIMARY_DECODING[fam]],
        "keys": {
            "val": f"primary ({mode})",
            "test": f"primary ({mode})",
            "val_greedy": greedy_text,
            "test_greedy": greedy_text,
        }
        | ({"test_length_constrained": DECODING_TEXT["length"]} if fam == "gru" else {})
        | (
            {stress: f"primary ({mode}, val-selected width)", f"{stress}_greedy": greedy_text}
            if stress is not None
            else {}
        ),
        "stress": {"split": stress, "mode": mode, "width": width} if stress is not None else None,
        "checkpoint_selection": TRAINING_SELECTION,
        "seconds": time.time() - t0,
    }
    attention = None
    if att is not None:
        attention = {"attention": att, "mode": np.array(mode)}
        if att_greedy is not None:
            attention["attention_greedy"] = att_greedy
    return metrics, (preds, split_of), attention


def write_eval(run_dir: Path, metrics: dict, preds_split, attention: dict | None) -> None:
    preds, split_of = preds_split
    for key, p in preds.items():
        rows = [
            {"name": r["name"], "dir": r["dir"], "target": r["label"], "pred": x}
            for r, x in zip(split_of[key].records, p, strict=True)
        ]
        (run_dir / f"pred_{key}.json").write_text(json.dumps(rows))
    if attention is not None:
        np.savez_compressed(run_dir / "attention_test.npz", **attention)
    write_json(metrics, run_dir / "metrics.json")


def decoding_spec(metrics: dict) -> dict:
    """spec.json entries describing the evaluation decoding."""
    dec = metrics["decoding"]
    return {
        "decoding_modes": dec["keys"],
        "decoding": {k: v for k, v in dec.items() if k not in ("keys", "seconds")},
        "selection": TRAINING_SELECTION,
    }


def load_data(
    cfg: dict, cache: Path, device: str, smoke: bool, train: bool = True
) -> dict[str, SvhnSplit]:
    """train/val/test from the `cache_suffix` cache and the stress split (no suffix)."""
    suffix = cfg.get("cache_suffix") or ""
    stress = stress_split(cfg)
    kw = {"crop": cfg["crop"], "jitter": cfg["color_jitter"]}
    if reverse_target(cfg):  # training targets of a right-to-left decoder
        kw["reverse"] = True
    names = [s for s in ("train", "val", "test") if train or s != "train"]
    data = {
        s: SvhnSplit(
            cache,
            s,
            cfg["max_len"],
            device,
            limit=SMOKE_SIZES[s] if smoke else None,
            suffix=suffix,
            **kw,
        )
        for s in names
    }
    if stress is not None:
        lim = SMOKE_STRESS if smoke else None
        data[stress] = SvhnSplit(cache, stress, cfg["max_len"], device, limit=lim, **kw)
    logger.info(f"loaded {', '.join(f'{s} {len(d)}' for s, d in data.items())} on {device}")
    return data


def stress_split(cfg: dict) -> str | None:
    return cfg.get("stress_split", "test_whole")


@torch.no_grad()
def mamba_diagnostics(model, split: SvhnSplit, bs: int, amp: bool) -> dict | None:
    """Mean dt per visual token (N, n_dirs, n_layers, 49) for a Mamba token mixer, else None."""
    if not isinstance(model, MixerAttentionGRURecognizer):
        return None
    model.eval()
    dts, diag = [], {}
    for b in split.iterate(bs):
        # eager even when the mixer is compiled: the diagnostics path returns python objects
        with (
            torch.compiler.set_stance("force_eager"),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp),
        ):
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
    """spec.json record of the plan 2.3 budget (see configs/phase2_task3.yaml)."""
    n = count_params(model)
    return {
        "matched_control": "t3_m1 (and t3_m2..t3_m4), all within 5% of t3_m0",
        "tolerance": PARAM_TOL,
        "t3_p0_cnn_attention_gru_params": P0_PARAMS,
        "t3_p0_parts": P0_PARTS,
        "rel_diff_vs_t3_p0": rel_param_diff(n, P0_PARAMS),
        "note": "plan 2.3 fixes T3-M0 (d_model 128, two per-layer-fused bidirectional Mamba-1 "
        "layers, d_state 16, d_conv 4, expand 2, 2-D positions) within 5% of T3-P0, and "
        "T3-M1..M4 within 5% of T3-M0. M0 matches P0 in total only. P0 reads 512-d tokens, "
        "so about 0.6M parameters of its positions and decoder sit in the M0 projection and "
        "mixer (compare t3_p0_parts with n_params_breakdown)",
    }


def bench(model, a, cfg: dict, device: str, amp: bool) -> None:
    """Times `a.bench` training steps (AdamW, clip, bf16 autocast on CUDA) on random images
    and labels at the config batch size. Logs ms/step, peak CUDA memory and the projected
    train time per epoch and per run (validation and evaluation excluded). Writes nothing."""
    bs, max_len, fcfg = cfg["batch_size"], cfg["max_len"], FitConfig(**cfg["fit"])
    if a.compile:  # the same in-place compile as `compile: true` in a run
        if not isinstance(model, MixerAttentionGRURecognizer):
            raise SystemExit("--compile applies to the Phase 2 token-mixer models only")
        compile_mixer(model)
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


# --- n-best lists (secondary decodings, see ocr_models and scripts/rescore_task3.py) ------

NBEST_FORMAT = 1


@torch.no_grad()
def collect_nbest(
    model: GRUDecoderRecognizer, split: SvhnSplit, name: str, width: int, bs: int, amp: bool
) -> NBest:
    """n-best lists of the primary beam of `width` on one split (center crops)."""
    model.eval()
    hyps, len_lp = [], []
    for b in split.iterate(bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            h, lp = model.nbest(b.images, width)
        hyps += h
        len_lp.append(lp.float().cpu().numpy())
    return NBest.from_hyps(name, width, hyps, np.concatenate(len_lp))


@torch.no_grad()
def score_nbest(
    model: GRUDecoderRecognizer, split: SvhnSplit, nb: NBest, bs: int, amp: bool
) -> np.ndarray:
    """Teacher-forced log-probability of every n-best hypothesis under `model` (N, K), -inf
    padding. For a reverse_target model this is log P_R2L(reverse(hyp))."""
    model.eval()
    hyps, rows = nb.hyps(), []
    for b in split.iterate(bs):
        idx = b.index.cpu().tolist()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            rows += model.score_strings(b.images, [hyps[i] for i in idx])
    return nb.r2l_matrix(rows)


def nbest_path(run_dir: Path, split: str) -> Path:
    return run_dir / f"nbest_{split}.json.gz"


def write_nbest(path: Path, nb: NBest, split: SvhnSplit, meta: dict) -> None:
    """nbest_{split}.json.gz: meta plus one row per image (records order) with name, dir,
    target, len_logp (lengths 1..5) and hyps [{pred, logp, logp_norm, n_tokens}], best
    primary score first. pred reads left-to-right."""
    rows = []
    for r, preds, lp, ln, nt, ll in zip(
        split.records, nb.preds, nb.logp, nb.logp_norm, nb.n_tokens, nb.len_logp, strict=True
    ):
        hyps = [
            {"pred": p, "logp": float(a), "logp_norm": float(b), "n_tokens": int(c)}
            for p, a, b, c in zip(preds, lp, ln, nt, strict=True)
            if math.isfinite(a)
        ]
        row = {"name": r["name"], "dir": r["dir"], "target": r["label"]}
        rows.append(row | {"len_logp": [float(x) for x in ll], "hyps": hyps})
    out = {"format": NBEST_FORMAT, "split": nb.split, "width": nb.width} | meta | {"rows": rows}
    with gzip.open(path, "wt") as f:
        json.dump(out, f)


def read_nbest(path: Path) -> tuple[NBest, dict]:
    """(NBest, the file without rows) of a write_nbest file."""
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    rows = d.pop("rows")
    hyps = [
        [(h["pred"], h["logp"], h["logp_norm"], h["n_tokens"]) for h in r["hyps"]] for r in rows
    ]
    nb = NBest.from_hyps(d["split"], d["width"], hyps, np.array([r["len_logp"] for r in rows]))
    d["keys"] = [(r["dir"], r["name"], r["target"]) for r in rows]
    return nb, d


def nbest_meta(model, width: int, primary_width: int | None, cfg: dict, smoke: bool) -> dict:
    return {
        "primary_width": primary_width,
        "mode": f"beam{width}",
        "reverse_target": bool(getattr(model, "reverse_target", False)),
        "cache_suffix": cfg.get("cache_suffix") or "",
        "smoke": smoke,
        "score": "logp = summed log-probability of digits + EOS (PAD, SOS masked), logp_norm "
        "= logp / n_tokens (the primary beam score)",
    }


def write_nbest_files(model, data, metrics: dict, run_dir: Path, cfg: dict, a, amp: bool):
    """--nbest: nbest_val.json.gz and nbest_test.json.gz of a GRU-decoder run."""
    primary = metrics["decoding"]["width"]
    width = a.nbest_width or primary
    meta = nbest_meta(model, width, primary, cfg, a.smoke)
    for s in ("val", "test"):  # val first, as everywhere
        nb = collect_nbest(model, data[s], s, width, cfg["eval_batch_size"], amp)
        write_nbest(nbest_path(run_dir, s), nb, data[s], meta)
    logger.info(f"wrote nbest_val/test.json.gz (width {width}) to {run_dir}")


def build_spec(
    a, cfg: dict, mcfg: dict, model, data, history, metrics, diag, fcfg, device, t0, run_dir, cache
) -> dict:
    kind, max_len, crop = mcfg["kind"], cfg["max_len"], cfg["crop"]
    phase = cfg.get("phase", "phase1")
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
    suffix = cfg.get("cache_suffix") or ""
    stress = stress_split(cfg)
    size = cfg["size"]
    if suffix:
        cache_text = (
            f"whole original image resized to {size}x{size} ignoring aspect, no ground-truth "
            f"box anywhere (files {{split}}{suffix}_*)"
        )
    else:
        cache_text = (
            f"GT digit-box union squared, expanded 1.3x, clamped, resized to {size}x{size} "
            "ignoring aspect (oracle localization at test)"
        )
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
        "setting": "whole_image" if suffix else "oracle_crop",
        "cache_suffix": suffix,
        "crop_protocol": {
            "cache": cache_text,
            "train": f"random {crop}x{crop} crop, ColorJitter brightness/contrast/saturation "
            f"factors in [{1 - cfg['color_jitter']:.1f}, {1 + cfg['color_jitter']:.1f}]",
            "eval": f"center {crop}x{crop} crop",
            "stress_split": stress,
            "normalization": "mean 0.5, std 0.5 per channel",
        }
        | (
            {"test_whole": f"whole image letterboxed to {size}x{size}, center crop"}
            if stress == "test_whole"
            else {}
        ),
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
        **decoding_spec(metrics),
        "optimizer": "AdamW",
        "fit": vars(fcfg),
        "batch_size": cfg["batch_size"],
        "epochs_run": len(history),
        "best_epoch": best_epoch,
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
    return spec


def log_result(run_dir: Path, metrics: dict, stress: str | None) -> None:
    t, mode = metrics["test"], metrics["decoding"]["mode"]
    msg = f"{run_dir} test ({mode}) exact {t['exact_match']:.4f} RMSE {t['RMSE']:.4f}"
    msg += f", greedy exact {metrics['test_greedy']['exact_match']:.4f}"
    if stress is not None:
        msg += f", {stress} exact {metrics[stress]['exact_match']:.4f}"
    logger.info(msg)


# config keys that decide which images are evaluated and how (a missing key means the default)
EVAL_DATA_KEYS = {"cache_suffix": "", "crop": None, "size": None, "max_len": None}


def check_eval_fingerprint(fp: dict, a, cfg: dict, mcfg: dict, cache: Path, suffix: str) -> None:
    """Refuse --eval-only when the model, seed or evaluation data differ from training."""
    if fp.get("model") != mcfg or fp.get("seed") != a.seed:
        raise SystemExit("the run was trained with another model config or seed")
    old_cfg = fp.get("config", {})
    if reverse_target(old_cfg) != reverse_target(cfg):
        raise SystemExit("--eval-only: reverse_target differs from training")

    def norm(c: dict, k: str):  # a null cache_suffix is the empty suffix
        v = c.get(k, EVAL_DATA_KEYS[k])
        return (v or "") if k == "cache_suffix" else v

    data_diff = sorted(k for k in EVAL_DATA_KEYS if norm(old_cfg, k) != norm(cfg, k))
    if data_diff:
        raise SystemExit(f"--eval-only: data config keys differ from training: {data_diff}")
    now = {k: v for k, v in cfg.items() if k not in ("models", "seeds")}
    diff = sorted(k for k in set(now) | set(old_cfg) if old_cfg.get(k) != now.get(k))
    if diff:
        logger.warning(f"config keys differ from the training fingerprint: {diff}")
    # the cache file the run was trained on (path follows svhn_dir or --cache)
    path = cache / f"train{suffix}_images.npy"
    if not path.exists():
        raise SystemExit(f"--eval-only: {path} is missing (wrong --cache or cache_suffix?)")
    trained, current = fp.get("data"), file_identity(path)
    if trained is None:
        logger.warning("the fingerprint has no data identity, the cache cannot be checked")
    elif trained.get("size") != current["size"]:
        raise SystemExit(
            f"--eval-only: {current['path']} has {current['size']} bytes, training saw "
            f"{trained.get('size')} (the cache was rebuilt or another cache is used)"
        )
    elif trained != current:
        logger.warning(
            f"cache identity differs from training (same size): trained {trained}, now "
            f"{current}. The cache may have been rebuilt, check the records."
        )


def eval_only(a, cfg: dict, mcfg: dict, run_dir: Path, cache: Path) -> None:
    """Re-evaluate a finished run from ckpt.pt's best_state with the current decoding."""
    for f in ("ckpt.pt", "metrics.json", "spec.json"):
        if not (run_dir / f).exists():
            raise SystemExit(f"--eval-only needs a finished run, {run_dir / f} is missing")
    suffix = cfg.get("cache_suffix") or ""
    fp_path = run_dir / "fingerprint.json"
    if fp_path.exists():
        check_eval_fingerprint(json.loads(fp_path.read_text()), a, cfg, mcfg, cache, suffix)
    else:
        logger.warning(f"{fp_path} is missing, the data identity cannot be checked")
    ck = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    if not ck.get("stopped", False):
        raise SystemExit(f"{run_dir}/ckpt.pt is not a finished run")
    device = pick_device()
    kind = mcfg["kind"]
    extra = {"length_loss_weight": cfg["length_loss_weight"]} if kind in GRU_DECODER_KINDS else {}
    model = build(kind, mcfg.get("args"), cfg["max_len"], reverse_target(cfg), **extra)
    model.load_state_dict(strip_compile_prefix(ck["best_state"]))
    model = model.to(device).eval()
    del ck
    compile_info = apply_compile(model, cfg, device)
    amp = FitConfig(**cfg["fit"]).amp == "bf16" and device == "cuda"
    data = load_data(cfg, cache, device, a.smoke, train=False)
    stress = stress_split(cfg)
    old = json.loads((run_dir / "metrics.json").read_text())
    backup = run_dir / "metrics_v1_greedy.json"
    if "decoding" not in old and not backup.exists():
        shutil.copy2(run_dir / "metrics.json", backup)
        logger.info(f"backed up the greedy metrics to {backup}")
    t0 = time.time()
    metrics, preds, attention = evaluate(
        model, data, stress, cfg["eval_batch_size"], amp, keep_attention=kind in ATTENTION_KINDS
    )
    write_eval(run_dir, metrics, preds, attention)
    if a.nbest:
        write_nbest_files(model, data, metrics, run_dir, cfg, a, amp)
    spec = json.loads((run_dir / "spec.json").read_text())
    spec |= decoding_spec(metrics)
    # runs trained before the two settings existed are the oracle crop setting (suffix "")
    spec["setting"] = "whole_image" if suffix else "oracle_crop"
    spec["cache_suffix"] = suffix
    if isinstance(spec.get("crop_protocol"), dict):
        spec["crop_protocol"]["stress_split"] = stress
    spec["eval_only"] = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "seconds": time.time() - t0,
        "previous_metrics": backup.name if backup.exists() else None,
        "data": {s: file_identity(cache / f"{s}{suffix}_images.npy") for s in ("val", "test")},
        "note": "re-evaluated from ckpt.pt best_state, no training",
    }
    if isinstance(model, MixerAttentionGRURecognizer):
        spec["eval_only"]["compile"] = compile_info
    write_json(spec, run_dir / "spec.json")
    log_result(run_dir, metrics, stress)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/phase1_task3.yaml")
    ap.add_argument("--cache", default=None, help="override the SVHN cache directory")
    ap.add_argument("--out", default=None, help="override results_root")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 2,000/500/500 images")
    ap.add_argument("--force", action="store_true")
    eval_help = "re-evaluate a finished run from ckpt.pt (no training) with the current decoding"
    ap.add_argument("--eval-only", action="store_true", help=eval_help)
    bench_help = "time N training steps on random data and exit, no run directory (Phase 2)"
    ap.add_argument("--bench", type=int, default=0, metavar="N", help=bench_help)
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--scan", choices=["parallel", "loop"], help="--bench only: Mamba scan")
    compile_help = "--bench only: torch.compile the mixer (runs use the config key `compile`)"
    ap.add_argument("--compile", action="store_true", help=compile_help)
    nbest_help = "GRU decoders: also write nbest_{val,test}.json.gz (secondary decodings)"
    ap.add_argument("--nbest", action="store_true", help=nbest_help)
    width_help = "--nbest list width (default: the validation-selected primary width)"
    ap.add_argument("--nbest-width", type=int, default=None, help=width_help)
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
        model = build(kind, mcfg.get("args"), cfg["max_len"], reverse_target(cfg), **extra)
        model = model.to(device)
        fcfg = FitConfig(**cfg["fit"])
        bench(model, a, cfg, device, fcfg.amp == "bf16" and device == "cuda")
        return
    if a.scan or a.compile:
        ap.error("--scan and --compile only apply to --bench (set mamba.scan in the config)")
    if a.eval_only and a.force:
        ap.error("--eval-only never retrains, do not combine it with --force")
    if a.nbest and kind not in GRU_DECODER_KINDS:
        ap.error("--nbest applies to the GRU decoders only")
    if a.nbest_width is not None and (not a.nbest or a.nbest_width < 1):
        ap.error("--nbest-width needs --nbest and a width >= 1")
    if reverse_target(cfg) and kind not in GRU_DECODER_KINDS:
        raise SystemExit("reverse_target: true applies to the GRU decoders only")
    phase = cfg.get("phase", "phase1")
    root = Path(a.out or cfg["results_root"]) / ("smoke" if a.smoke else phase)
    run_dir = root / "runs" / "task3" / a.model / f"seed{a.seed}"
    cache = Path(a.cache or cfg["svhn_dir"])
    if a.eval_only:
        eval_only(a, cfg, mcfg, run_dir, cache)
        return
    if (run_dir / "metrics.json").exists() and not a.force:
        logger.info(f"{run_dir} already done")
        return
    suffix = cfg.get("cache_suffix") or ""
    fingerprint = {
        "model": mcfg,
        "config": {k: v for k, v in cfg.items() if k not in ("models", "seeds")},
        "seed": a.seed,
        "smoke": a.smoke,
        "data": file_identity(cache / f"train{suffix}_images.npy"),
    }
    if kind in MIXER_KINDS:  # Phase 1 fingerprints stay as they were
        fingerprint["compile"] = compile_requested(cfg)
    elif compile_requested(cfg):
        raise SystemExit("compile: true applies to the Phase 2 token-mixer models only")
    prepare_run_dir(run_dir, fingerprint, a.force)
    device = pick_device()
    data = load_data(cfg, cache, device, a.smoke)
    stress = stress_split(cfg)

    set_seed(a.seed)  # seeds the weight initialization too, not only training
    extra = {"length_loss_weight": cfg["length_loss_weight"]} if kind in GRU_DECODER_KINDS else {}
    model = build(kind, mcfg.get("args"), cfg["max_len"], reverse_target(cfg), **extra)
    model = model.to(device)
    compile_info = apply_compile(model, cfg, device)
    fcfg = FitConfig(**cfg["fit"])
    amp = fcfg.amp == "bf16" and device == "cuda"
    if not amp:
        fcfg.amp = "none"
    if a.smoke:
        fcfg.epochs = 2
    bs, ebs = cfg["batch_size"], cfg["eval_batch_size"]

    def train_batches(epoch: int, g: torch.Generator):
        return data["train"].iterate(bs, train=True, g=g)

    def step_loss(m, b, epoch: int) -> torch.Tensor:
        return m.loss(b)

    def validate(m) -> float:  # training-time selection, greedy (see TRAINING_SELECTION)
        p, _ = predict(m, data["val"], "greedy", ebs, amp)
        return 1.0 - float(np.mean([x == y for x, y in zip(p, data["val"].labels)]))

    t0 = time.time()
    groups = [{"params": list(model.parameters())}]
    model, history = fit(run_dir, model, groups, train_batches, step_loss, validate, fcfg, a.seed)

    metrics, preds, attention = evaluate(
        model, data, stress, ebs, amp, keep_attention=kind in ATTENTION_KINDS
    )
    diag = mamba_diagnostics(model, data["test"], ebs, amp)
    if diag is not None:
        np.savez_compressed(run_dir / "mamba_dt_test.npz", **diag)
    spec = build_spec(
        a, cfg, mcfg, model, data, history, metrics, diag, fcfg, device, t0, run_dir, cache
    )
    if isinstance(model, MixerAttentionGRURecognizer):
        spec["compile"] = compile_info
    if reverse_target(cfg):  # only right-to-left runs get the key
        spec["reverse_target"] = {"enabled": True, "description": REVERSE_TEXT}
    if a.nbest:
        write_nbest_files(model, data, metrics, run_dir, cfg, a, amp)
    write_json(history, run_dir / "history.json")
    write_json(spec, run_dir / "spec.json")
    write_eval(run_dir, metrics, preds, attention)  # metrics.json last: it marks the run done
    log_result(run_dir, metrics, stress)


if __name__ == "__main__":
    main()
