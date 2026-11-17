"""Secondary Task 3 decodings of finished GRU-decoder runs (plan 2.3), no training.

The primary result of a run (metrics.json, validation-selected beam) is never touched. This
script reloads ckpt.pt's best_state of a left-to-right (L2R) run, builds the n-best list of
its primary beam (nbest_{val,test}.json.gz, reused when present and matching) and reranks it:

- mbr_edit (plan name mbr_edit_secondary): edit-distance minimum Bayes risk over the n-best
  list (Kumar and Byrne 2004). No tuned parameter.
- length_rescore (length_rescore_secondary): primary score + lambda * log P_length.
- bidir_rescore (dram_fb_secondary, only with a paired right-to-left run): DRAM-style
  forward-backward rescoring (Ba et al. 2015), (log P_L2R + lambda * log P_R2L(reverse)) /
  tokens. The R2L run is the same model trained with `reverse_target: true`
  (configs/phase1_task3_r2l.yaml, phase1_task3_whole_r2l.yaml). L2R seed s pairs with R2L
  seed s.

Selection uses the validation split only. Every validation n-best list and every lambda is
fixed before any test image is decoded (the test split is not even loaded until then).
Lambdas come from ocr_models.LAMBDA_GRID, ties to the smaller lambda.

Reported values follow the plan's setting-level rule when the seeds of one model and setting
are passed together (--root ... --seeds 0 1 2). The primary beam width is the width with the
best mean validation exact match across the seeds (metrics.json decoding.val_by_width), and
the n-best list of every seed is built at that width. Each lambda is the one with the best mean
validation exact match across the seeds. Both are applied unchanged to every seed and to test.
Each seed's own lambda is kept as a labelled diagnostic ("per_run"). With a single run the
width and lambdas are that run's own (selection_scope "single_run"), with a warning that this
is not the plan's rule. rescoring.json records the scope and whether the seeds are 0, 1, 2
(plan_conformant).

Outputs in each L2R run directory: rescoring.json (per method: hyperparameter, validation and
test metrics.sequence_report), pred_val_{method}.json and pred_test_{method}.json with the
reported lambda (plus pred_{val,test}_{method}_per_run.json with several runs),
nbest_val.json.gz and nbest_test.json.gz.
The cache suffix, crop and smoke limits of each run come from its fingerprint.json, so Phase 1
and Phase 2 runs in both settings work.

Examples:
  rescore_task3.py --root results/phase1 --model cnn_attention_gru \\
      --r2l-model cnn_attention_gru_r2l --seeds 0 1 2
  rescore_task3.py --l2r results/phase2/runs/task3/t3_m0/seed0
"""

import argparse
import importlib.util
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from dl_final_prj.engine import FitConfig, pick_device, write_json
from dl_final_prj.metrics import sequence_report
from dl_final_prj.ocr_models import (
    LAMBDA_GRID,
    SECONDARY_PLAN_NAMES,
    SECONDARY_TEXT,
    GRUDecoderRecognizer,
    NBest,
    bidir_rescore,
    build,
    exact_by_lambda,
    length_rescore,
    mbr_edit,
    select_lambda,
    select_width,
)
from dl_final_prj.svhn import SvhnSplit


def _load_run_ocr():
    spec = importlib.util.spec_from_file_location("run_ocr", Path(__file__).with_name("run_ocr.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ro = _load_run_ocr()

LAMBDA_METHODS = {"length_rescore": "lambda_length", "bidir_rescore": "lambda_fb"}
SELECTION_TEXT = (
    "validation exact match only, ties to the smaller lambda. Every validation n-best list and "
    "lambda is fixed before the test split is loaded, then applied unchanged to test"
)
REFERENCE_TEXT = "rank 1 of the n-best list, i.e. the primary beam (reference, not secondary)"
OTHER_WIDTH_TEXT = (
    "rank 1 of an n-best list whose width differs from the validation-selected primary width "
    "(--nbest-width), a diagnostic, not the primary result"
)
PLAN_SEEDS = (0, 1, 2)  # the plan's Task 3 seeds, over which width and lambdas are pooled
SETTING_TEXT = (
    "setting-level selection (plan 2.3): the beam width (from metrics.json val_by_width) and "
    "each lambda maximise the mean validation exact match across the given runs (seeds), ties "
    "to the smaller value, and are applied unchanged to every run and to test"
)
SINGLE_RUN_TEXT = (
    "single-run selection: width and lambda selected on this run's validation split alone. "
    "Not the plan's setting-level rule, which needs all seeds in one call"
)
PER_RUN_TEXT = (
    "diagnostic only: lambda selected on this run's own validation split at the setting-level "
    "n-best width. Not the reported value"
)
# R2L config keys that must equal the L2R run (same data, recipe and decoding)
SAME_KEYS = (
    "cache_suffix",
    "crop",
    "size",
    "max_len",
    "svhn_dir",
    "fit",
    "batch_size",
    "length_loss_weight",
    "color_jitter",
)


# --- pure selection core (no files, no model) ---------------------------------------------


@dataclass
class SplitInputs:
    """One split of one run: its n-best list, labels and R2L scores (N, K) or None."""

    nb: NBest
    labels: list[str]
    r2l: np.ndarray | None = None


def decode_method(x: SplitInputs, method: str, lam: float | None = None) -> list[str]:
    if method == "beam_primary":
        return x.nb.top1()
    if method == "mbr_edit":
        return mbr_edit(x.nb)
    if lam is None:
        raise ValueError(f"{method} needs a lambda")
    if method == "length_rescore":
        return length_rescore(x.nb, lam)
    if method == "bidir_rescore":
        if x.r2l is None:
            raise ValueError("bidir_rescore needs R2L scores")
        r2l = x.r2l
        return bidir_rescore(x.nb, r2l, lam)
    raise ValueError(f"unknown method {method}")


def methods_of(x: SplitInputs) -> list[str]:
    out = ["beam_primary", "mbr_edit", "length_rescore"]
    return out + (["bidir_rescore"] if x.r2l is not None else [])


def validation_stage(val: SplitInputs, grid: Sequence[float]) -> dict:
    """Validation exact match of every lambda method over the grid (validation split only)."""
    if val.nb.split != "val":
        raise ValueError(f"the selection stage takes the validation split, got {val.nb.split!r}")
    return {
        m: exact_by_lambda(val.nb, val.labels, lambda lam, m=m: decode_method(val, m, lam), grid)
        for m in methods_of(val)
        if m in LAMBDA_METHODS
    }


def pooled_lambdas(val_tables: Sequence[dict]) -> dict[str, tuple[float, dict[float, float]]]:
    """Setting-level lambda per method from the mean validation exact match across runs."""
    out = {}
    for m in val_tables[0]:
        grid = list(val_tables[0][m])
        mean = {lam: float(np.mean([t[m][lam] for t in val_tables])) for lam in grid}
        out[m] = (select_lambda(mean, "val"), mean)
    return out


def setting_width(val_by_width: Sequence[Mapping[str, float]]) -> tuple[int, dict[int, float]]:
    """Setting-level primary beam width (plan 2.3): the width with the best mean validation
    exact match across the runs (seeds), ties to the smaller width. val_by_width are the
    metrics.json decoding.val_by_width tables of the runs, all over the same widths."""
    if not val_by_width or not val_by_width[0]:
        raise ValueError("no validation exact match by width to select from")
    widths = set(val_by_width[0])
    if any(set(t) != widths for t in val_by_width):
        raise ValueError("the runs were evaluated at different candidate widths")
    mean = {int(w): float(np.mean([t[w] for t in val_by_width])) for w in widths}
    return select_width(mean), dict(sorted(mean.items()))


def apply_stage(x: SplitInputs, lambdas: dict[str, float]) -> dict[str, list[str]]:
    """Predictions of every method with frozen lambdas (any split)."""
    return {m: decode_method(x, m, lambdas.get(m)) for m in methods_of(x)}


def rescore_core(
    vals: Sequence[SplitInputs],
    get_test: Callable[[int], SplitInputs],
    grid: Sequence[float] = LAMBDA_GRID,
) -> list[dict]:
    """Selection on validation, then test. get_test(i) is called only after every lambda is
    fixed.

    With several runs (the seeds of one model and setting) the reported lambda of each method
    is the plan's setting-level lambda (mean validation exact match across the runs), the same
    for every run. Each run's own selection is kept as a labelled diagnostic ("per_run"). With
    one run the reported lambda is that run's selection (scope "single_run", not the plan's
    rule). Returns per run {"scope", "val_tables", "lambdas" (reported), "setting" (method ->
    (lambda, mean table)) or None, "per_run_lambdas", "preds": {split: {method: preds}},
    "per_run_preds" (scope "setting" only), "test_labels"}."""
    tables = [validation_stage(v, grid) for v in vals]
    per_run = [{m: select_lambda(t[m], "val") for m in t} for t in tables]
    setting = None
    if len(vals) > 1:
        if len({frozenset(t) for t in tables}) != 1:
            raise ValueError("pooling needs the same methods in every run (R2L for all or none)")
        setting = pooled_lambdas(tables)
    out = []
    for i, v in enumerate(vals):
        reported = {m: p[0] for m, p in setting.items()} if setting is not None else per_run[i]
        r = {
            "scope": "setting" if setting is not None else "single_run",
            "val_tables": tables[i],
            "lambdas": reported,
            "setting": setting,
            "per_run_lambdas": per_run[i],
            "preds": {"val": apply_stage(v, reported)},
        }
        if setting is not None:
            r["per_run_preds"] = {"val": apply_stage(v, per_run[i])}
        out.append(r)
    # --- test: every selection above is frozen ---
    for i, r in enumerate(out):
        t = get_test(i)
        if t.nb.split != "test":
            raise ValueError(f"get_test returned the {t.nb.split!r} split")
        r["preds"]["test"] = apply_stage(t, r["lambdas"])
        r["test_labels"] = t.labels
        if "per_run_preds" in r:
            r["per_run_preds"]["test"] = apply_stage(t, r["per_run_lambdas"])
    return out


# --- runs on disk -------------------------------------------------------------------------


@dataclass
class Run:
    run_dir: Path
    fp: dict
    metrics: dict

    @property
    def cfg(self) -> dict:
        return self.fp["config"]

    @property
    def mcfg(self) -> dict:
        return self.fp["model"]

    @property
    def seed(self) -> int:
        return int(self.fp["seed"])

    @property
    def smoke(self) -> bool:
        return bool(self.fp.get("smoke", False))

    @property
    def name(self) -> str:
        return self.run_dir.parent.name


def open_run(run_dir: Path) -> Run:
    for f in ("ckpt.pt", "metrics.json", "fingerprint.json"):
        if not (run_dir / f).exists():
            raise SystemExit(f"{run_dir / f} is missing (not a finished Task 3 run)")
    fp = json.loads((run_dir / "fingerprint.json").read_text())
    if fp["model"]["kind"] not in ro.GRU_DECODER_KINDS:
        raise SystemExit(f"{run_dir}: secondary decodings apply to the GRU decoders only")
    return Run(run_dir, fp, json.loads((run_dir / "metrics.json").read_text()))


def check_pair(l2r: Run, r2l: Run) -> None:
    """The R2L run must be the same model, recipe, data and seed with reverse_target true."""
    if ro.reverse_target(l2r.cfg):
        raise SystemExit(f"{l2r.run_dir} is a right-to-left run, pass it with --r2l")
    if not ro.reverse_target(r2l.cfg):
        raise SystemExit(f"{r2l.run_dir} was not trained with reverse_target: true")
    if r2l.seed != l2r.seed or r2l.smoke != l2r.smoke:
        raise SystemExit(f"seed/smoke differ: {l2r.run_dir} vs {r2l.run_dir}")
    if r2l.mcfg != l2r.mcfg:
        raise SystemExit(f"model configs differ: {l2r.mcfg} vs {r2l.mcfg}")
    norm = lambda c, k: (c.get(k) or "") if k == "cache_suffix" else c.get(k)  # noqa: E731
    diff = [k for k in SAME_KEYS if norm(l2r.cfg, k) != norm(r2l.cfg, k)]
    if diff:
        raise SystemExit(f"R2L config differs from the L2R run in {diff}")
    other = sorted(
        k
        for k in set(l2r.cfg) | set(r2l.cfg)
        if k not in (*SAME_KEYS, "reverse_target") and l2r.cfg.get(k) != r2l.cfg.get(k)
    )
    if other:
        logger.warning(f"R2L config keys differ from the L2R run: {other}")


def load_model(run: Run, device: str) -> GRUDecoderRecognizer:
    cfg, kind = run.cfg, run.mcfg["kind"]
    ck = torch.load(run.run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    if not ck.get("stopped", False):
        raise SystemExit(f"{run.run_dir}/ckpt.pt is not a finished run")
    model = build(
        kind,
        run.mcfg.get("args"),
        cfg["max_len"],
        ro.reverse_target(cfg),
        length_loss_weight=cfg["length_loss_weight"],
    )
    model.load_state_dict(ro.strip_compile_prefix(ck["best_state"]))
    assert isinstance(model, GRUDecoderRecognizer)
    return model.to(device).eval()


def load_split(run: Run, cache: Path, device: str, split: str) -> SvhnSplit:
    cfg = run.cfg
    return SvhnSplit(
        cache,
        split,
        cfg["max_len"],
        device,
        crop=cfg["crop"],
        jitter=cfg["color_jitter"],
        limit=ro.SMOKE_SIZES[split] if run.smoke else None,
        suffix=cfg.get("cache_suffix") or "",
    )


def records_key(split: SvhnSplit) -> list[tuple]:
    return [(r["dir"], r["name"], r["label"]) for r in split.records]


def get_nbest(
    run: Run,
    model,
    split: SvhnSplit,
    name: str,
    width: int,
    primary_width: int,
    bs: int,
    amp: bool,
    regen: bool,
) -> tuple[NBest, bool]:
    """The n-best list of the primary beam (`width`), reused from nbest_{split}.json.gz when
    it matches (width, order of the records, left-to-right model). Returns (NBest,
    generated)."""
    path = ro.nbest_path(run.run_dir, name)
    if path.exists() and not regen:
        nb, meta = ro.read_nbest(path)
        same = (
            meta["width"] == width
            and not meta.get("reverse_target", False)
            and [tuple(k) for k in meta["keys"]] == records_key(split)
        )
        if same:
            return nb, False
        logger.info(f"{path} does not match this request, regenerating it")
    nb = ro.collect_nbest(model, split, name, width, bs, amp)
    meta = ro.nbest_meta(model, width, primary_width, run.cfg, run.smoke)
    ro.write_nbest(path, nb, split, meta)
    return nb, True


def primary_agreement(run: Run, split: str, nb: NBest) -> float | None:
    """Share of images where rank 1 of the n-best list equals pred_{split}.json (the stored
    primary result). 1.0 is expected when the n-best width is the primary width."""
    path = run.run_dir / f"pred_{split}.json"
    if not path.exists():
        return None
    stored = [r["pred"] for r in json.loads(path.read_text())]
    if len(stored) != len(nb):
        return None
    return float(np.mean([a == b for a, b in zip(stored, nb.top1(), strict=True)]))


def write_preds(path: Path, split: SvhnSplit, preds: list[str]) -> None:
    rows = [
        {"name": r["name"], "dir": r["dir"], "target": r["label"], "pred": p}
        for r, p in zip(split.records, preds, strict=True)
    ]
    path.write_text(json.dumps(rows))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--l2r", nargs="+", type=Path, help="left-to-right run directories")
    ap.add_argument("--r2l", nargs="+", type=Path, help="paired right-to-left run directories")
    ap.add_argument("--root", type=Path, help="results root of a phase, e.g. results/phase1")
    ap.add_argument("--model", help="with --root: L2R model name")
    ap.add_argument("--r2l-model", help="with --root: R2L model name")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="with --root")
    ap.add_argument("--cache", type=Path, default=None, help="override the SVHN cache dir")
    ap.add_argument("--nbest-width", type=int, default=None, help="default: primary width")
    ap.add_argument("--grid", nargs="+", type=float, default=list(LAMBDA_GRID))
    ap.add_argument("--regen-nbest", action="store_true", help="recompute the n-best lists")
    ap.add_argument("--batch-size", type=int, default=None, help="default: eval_batch_size")
    a = ap.parse_args()

    if a.root is not None:
        if a.l2r or a.r2l or not a.model:
            ap.error("--root takes --model (and --r2l-model, --seeds), not --l2r/--r2l")
        base = a.root / "runs" / "task3"
        l2r_dirs = [base / a.model / f"seed{s}" for s in a.seeds]
        r2l_dirs = [base / a.r2l_model / f"seed{s}" for s in a.seeds] if a.r2l_model else None
    else:
        if not a.l2r or a.model or a.r2l_model:
            ap.error("give --l2r DIR... (and --r2l DIR...) or --root with --model")
        l2r_dirs, r2l_dirs = a.l2r, a.r2l
    if r2l_dirs is not None and len(r2l_dirs) != len(l2r_dirs):
        ap.error("--r2l needs one directory per --l2r directory")
    if a.nbest_width is not None and a.nbest_width < 1:
        ap.error("--nbest-width must be >= 1")

    runs = [open_run(d) for d in l2r_dirs]
    r2l_runs = [open_run(d) for d in r2l_dirs] if r2l_dirs is not None else None
    if r2l_runs is not None:
        for fwd, bwd in zip(runs, r2l_runs, strict=True):
            check_pair(fwd, bwd)
    else:
        for r in runs:
            if ro.reverse_target(r.cfg):
                raise SystemExit(f"{r.run_dir} is a right-to-left run, pass it with --r2l")
    if len({r.seed for r in runs}) != len(runs):
        raise SystemExit("the L2R runs must be distinct seeds")
    if len(runs) > 1:
        keys = {json.dumps({k: r.cfg.get(k) for k in SAME_KEYS}, sort_keys=True) for r in runs}
        if len(keys) != 1 or len({json.dumps(r.mcfg, sort_keys=True) for r in runs}) != 1:
            raise SystemExit("pooling needs runs of one model and one setting")

    # primary width: the plan's setting-level width with several runs, else the run's own
    per_run_width = [int(r.metrics["decoding"]["width"]) for r in runs]
    width_table = None
    if len(runs) > 1:
        ref_width, width_table = setting_width(
            [r.metrics["decoding"]["val_by_width"] for r in runs]
        )
        ref_widths = [ref_width] * len(runs)
    else:
        ref_widths = per_run_width
    scope = "setting" if len(runs) > 1 else "single_run"
    seeds = sorted(r.seed for r in runs)
    plan_conformant = len(runs) > 1 and seeds == list(PLAN_SEEDS)
    if not plan_conformant:
        logger.warning(
            f"seeds {seeds}: the plan selects the beam width and the lambdas once per model and "
            f"setting over seeds {list(PLAN_SEEDS)}, so these numbers are "
            f"{'a single-run selection' if scope == 'single_run' else 'a partial-seed selection'}"
            ", not the plan's reported values"
        )

    device = pick_device()
    t0 = time.time()
    ctx = []
    for i, run in enumerate(runs):
        cfg = run.cfg
        cache = a.cache or Path(cfg["svhn_dir"])
        amp = FitConfig(**cfg["fit"]).amp == "bf16" and device == "cuda"
        bs = a.batch_size or cfg["eval_batch_size"]
        width = a.nbest_width or ref_widths[i]
        model = load_model(run, device)
        r2l_model = load_model(r2l_runs[i], device) if r2l_runs is not None else None
        ctx.append(
            {
                "cache": cache,
                "amp": amp,
                "bs": bs,
                "width": width,
                "model": model,
                "r2l": r2l_model,
                "splits": {},
                "generated": {},
            }
        )

    def inputs(i: int, split: str) -> SplitInputs:
        run, c = runs[i], ctx[i]
        data = load_split(run, c["cache"], device, split)
        nb, gen = get_nbest(
            run,
            c["model"],
            data,
            split,
            c["width"],
            ref_widths[i],
            c["bs"],
            c["amp"],
            a.regen_nbest,
        )
        r2l = None
        if c["r2l"] is not None:
            r2l = ro.score_nbest(c["r2l"], data, nb, c["bs"], c["amp"])
        c["splits"][split], c["generated"][split] = (data, nb), gen
        logger.info(f"{run.run_dir} {split}: n-best width {c['width']}, {len(nb)} images")
        return SplitInputs(nb, list(data.labels), r2l)

    vals = [inputs(i, "val") for i in range(len(runs))]  # validation only up to selection
    results = rescore_core(vals, lambda i: inputs(i, "test"), a.grid)

    for i, (run, res) in enumerate(zip(runs, results, strict=True)):
        c = ctx[i]
        splits = {s: c["splits"][s][0] for s in ("val", "test")}
        labels = {s: list(splits[s].labels) for s in ("val", "test")}
        is_primary = c["width"] == ref_widths[i]  # else rank 1 is another beam width
        setting = res["setting"]
        methods: dict = {}
        for m, val_preds in res["preds"]["val"].items():
            entry: dict = {
                "plan_name": SECONDARY_PLAN_NAMES.get(m, "beam_primary"),
                "description": SECONDARY_TEXT.get(m, REFERENCE_TEXT),
            }
            if m == "beam_primary" and not is_primary:
                entry = {"plan_name": None, "description": OTHER_WIDTH_TEXT}
            if m in LAMBDA_METHODS:
                hp = {
                    "name": LAMBDA_METHODS[m],
                    "value": res["lambdas"][m],
                    "selected_on": "val",
                    "rule": SETTING_TEXT if setting is not None else SINGLE_RUN_TEXT,
                    "seeds": seeds,
                    "plan_conformant": plan_conformant,
                    "val_exact_by_lambda": {str(k): v for k, v in res["val_tables"][m].items()},
                }
                if setting is not None:
                    hp["mean_val_exact_by_lambda"] = {str(k): v for k, v in setting[m][1].items()}
                entry["hyperparameter"] = hp
            else:
                entry["hyperparameter"] = None
            entry["val"] = sequence_report(labels["val"], val_preds)
            entry["test"] = sequence_report(labels["test"], res["preds"]["test"][m])
            if m != "beam_primary":
                for s in ("val", "test"):
                    write_preds(run.run_dir / f"pred_{s}_{m}.json", splits[s], res["preds"][s][m])
                entry["pred_files"] = {s: f"pred_{s}_{m}.json" for s in ("val", "test")}
            if setting is not None and m in LAMBDA_METHODS:
                pr = res["per_run_preds"]
                files = {s: f"pred_{s}_{m}_per_run.json" for s in ("val", "test")}
                for s in ("val", "test"):
                    write_preds(run.run_dir / files[s], splits[s], pr[s][m])
                entry["per_run"] = {
                    "description": PER_RUN_TEXT,
                    "value": res["per_run_lambdas"][m],
                    "val": sequence_report(labels["val"], pr["val"][m]),
                    "test": sequence_report(labels["test"], pr["test"][m]),
                    "pred_files": files,
                }
            key = m if m != "beam_primary" or is_primary else f"beam{c['width']}_top1"
            methods[key] = entry
        suffix = run.cfg.get("cache_suffix") or ""
        # rank 1 vs pred_{split}.json, only defined at the width that produced pred_*.json
        same_as_run = c["width"] == per_run_width[i]
        out = {
            "model": run.name,
            "seed": run.seed,
            "kind": run.mcfg["kind"],
            "run_dir": str(run.run_dir),
            "r2l_run_dir": str(r2l_runs[i].run_dir) if r2l_runs is not None else None,
            "r2l_model": r2l_runs[i].name if r2l_runs is not None else None,
            "setting": "whole_image" if suffix else "oracle_crop",
            "cache_suffix": suffix,
            "smoke": run.smoke,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "selection_scope": scope,
            "selection_seeds": seeds,
            "plan_conformant": plan_conformant,
            "reported_rule": SETTING_TEXT if scope == "setting" else SINGLE_RUN_TEXT,
            "nbest": {
                "width": c["width"],
                "primary_width": ref_widths[i],
                "run_width": per_run_width[i],
                "setting_width": None
                if width_table is None
                else {
                    "value": ref_widths[i],
                    "seeds": seeds,
                    "mean_val_exact_by_width": {str(k): v for k, v in width_table.items()},
                },
                "files": {s: ro.nbest_path(run.run_dir, s).name for s in ("val", "test")},
                "generated_now": c["generated"],
                "primary_agreement": {
                    s: primary_agreement(run, s, c["splits"][s][1]) if same_as_run else None
                    for s in ("val", "test")
                },
            },
            "lambda_grid": [float(x) for x in a.grid],
            "selection": SELECTION_TEXT,
            "methods": methods,
            "seconds": time.time() - t0,
        }
        write_json(out, run.run_dir / "rescoring.json")
        msg = ", ".join(f"{m} {e['test']['exact_match']:.4f}" for m, e in methods.items())
        logger.info(f"{run.run_dir} test exact: {msg}")


if __name__ == "__main__":
    main()
