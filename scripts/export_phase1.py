# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOperatorIssue=false
"""Collect Phase 1 runs into the CSV/JSON tables and figures listed in docs/experiment_plan.md.

Outputs under results/phase1/ (tables) and results/figures/phase1/ (PNG + SVG). Missing runs
are reported and skipped, so this can be rerun while training is still going.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from dl_final_prj.engine import write_json
from dl_final_prj.metrics import per_sample_losses
from dl_final_prj.stats import (
    bootstrap_image_diff,
    bootstrap_loss_diff,
    dm_test,
    mcnemar,
)

# Reference palette (dataviz skill, light mode). Color follows the model, never its rank.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {
    "bigru_attention": "#2a78d6",
    "seq2seq_attention": "#2a78d6",
    "cnn_attention_gru": "#2a78d6",
    "bigru": "#eb6834",
    "seq2seq": "#eb6834",
    "cnn_pool_gru": "#eb6834",
    "ridge": "#1baf7a",
    "crnn_ctc": "#1baf7a",
    "seasonal_weekly": "#eda100",
    "multihead_cnn": "#eda100",
    "seasonal_daily": "#e87ba4",
    "ridge_window": "#008300",
    "persistence": "#8a8984",
}
STYLES = {
    "persistence": ":",
    "seasonal_daily": "--",
    "seasonal_weekly": "--",
    "ridge": "-.",
    "ridge_window": "-.",
}
LABELS = {
    "bigru_attention": "Residual attentive BiGRU-2 (main)",
    "bigru": "Residual BiGRU-2, no attention",
    "seq2seq_attention": "Residual attentive LSTM enc-dec (main)",
    "seq2seq": "Residual LSTM enc-dec, no attention",
    "cnn_attention_gru": "CNN + attention GRU (main)",
    "cnn_pool_gru": "CNN + GRU, no attention",
    "multihead_cnn": "Multi-head CNN (Goodfellow)",
    "crnn_ctc": "CRNN-CTC",
    "ridge": "Ridge (sparse lags + calendar)",
    "ridge_window": "Ridge (168-h window + calendar)",
    "seasonal_weekly": "Seasonal naive, weekly",
    "seasonal_daily": "Seasonal naive, daily",
    "persistence": "Persistence",
}
OCR_FAMILIES = ["crnn_ctc", "multihead_cnn", "cnn_pool_gru", "cnn_attention_gru"]
WHOLE = "_whole"  # Setting B (whole image, no GT boxes) runs carry this suffix
for _m in OCR_FAMILIES:
    COLORS[_m + WHOLE] = COLORS[_m]
    LABELS[_m + WHOLE] = LABELS[_m]
TS_BASE = ["persistence", "seasonal_daily", "seasonal_weekly", "ridge", "ridge_window"]
ORDER = {
    1: [*TS_BASE, "bigru", "bigru_attention"],
    2: [*TS_BASE, "seq2seq", "seq2seq_attention"],
    3: [*OCR_FAMILIES, *(m + WHOLE for m in OCR_FAMILIES)],
}
MAIN = {1: "bigru_attention", 2: "seq2seq_attention", 3: "cnn_attention_gru"}
CONTRASTS = {
    1: [
        ("bigru_attention", "bigru"),
        ("bigru_attention", "ridge_window"),
        ("bigru_attention", "ridge"),
        ("bigru_attention", "seasonal_daily"),
    ],
    2: [
        ("seq2seq_attention", "seq2seq"),
        ("seq2seq_attention", "ridge_window"),
        ("seq2seq_attention", "ridge"),
        ("seq2seq_attention", "seasonal_daily"),
    ],
    3: [
        ("cnn_attention_gru", "cnn_pool_gru"),
        ("cnn_attention_gru", "multihead_cnn"),
        ("cnn_attention_gru", "crnn_ctc"),
        ("cnn_attention_gru_whole", "cnn_pool_gru_whole"),
        ("cnn_attention_gru_whole", "multihead_cnn_whole"),
        ("cnn_attention_gru_whole", "crnn_ctc_whole"),
    ],
}
# metrics.json keys written by run_ocr.py. "val"/"test" are the primary non-greedy decodings
# (validation-selected beam width, CTC prefix beam or multi-head MAP); *_greedy are diagnostics;
# test_whole is the crop-trained model on whole images (stress test, Setting A only).
OCR_MODES = (
    "val",
    "val_greedy",
    "test",
    "test_greedy",
    "test_length_constrained",
    "test_whole",
    "test_whole_greedy",
)


def setting_of(model: str) -> str:
    return "B_whole_image" if model.endswith(WHOLE) else "A_oracle_crop"


def family_of(model: str) -> str:
    return model.removesuffix(WHOLE)


METRICS = ["R2", "NSE", "MAE", "RMSE"]
OCR_METRICS = [
    *METRICS,
    "exact_match",
    "char_accuracy",
    "length_accuracy",
    "norm_edit_distance",
    "CER",
]


def style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_runs(root: Path, task: int) -> dict[str, dict[int, Path]]:
    runs: dict[str, dict[int, Path]] = {}
    for d in sorted((root / "runs" / f"task{task}").glob("*/seed*")):
        if (d / "metrics.json").exists():
            runs.setdefault(d.parent.name, {})[int(d.name[4:])] = d
    return runs


def summarize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("model", sort=False)[cols]
    out = pd.concat({"mean": g.mean(), "std": g.std(ddof=1), "median": g.median()}, axis=1)
    out.columns = [f"{c}_{s}" for s, c in out.columns]
    return out.reset_index()


# --- tables -------------------------------------------------------------------


def ts_tables(root: Path, task: int, runs: dict) -> None:
    rows, hrows = [], []
    for model in [m for m in ORDER[task] if m in runs] + [m for m in runs if m not in ORDER[task]]:
        for seed, d in sorted(runs[model].items()):
            met = json.loads((d / "metrics.json").read_text())
            spec = json.loads((d / "spec.json").read_text())
            for split in ("val", "test"):
                m = met[split] if task == 1 else met[split]["pooled"]
                rows.append(
                    {"model": model, "seed": seed, "split": split, **{k: m[k] for k in METRICS},
                     "n_params": spec.get("n_params"), "epochs_run": spec.get("epochs_run"),
                     "best_epoch": spec.get("best_epoch"), "train_seconds": spec.get("train_seconds")}
                )  # fmt: skip
                if task == 2:
                    for h, rep in met[split].items():
                        hrows.append(
                            {"model": model, "seed": seed, "split": split, "horizon": h, **rep}
                        )
    df = pd.DataFrame(rows)
    df.to_csv(root / f"task{task}_metrics_by_seed.csv", index=False)
    summarize(df[df.split == "test"], [*METRICS, "n_params", "train_seconds"]).to_csv(
        root / f"task{task}_metrics_summary.csv", index=False
    )
    if task == 2:
        hd = pd.DataFrame(hrows)
        hd.to_csv(root / "task2_metrics_by_horizon.csv", index=False)


def ocr_tables(root: Path, runs: dict) -> None:
    rows = []
    for model in [m for m in ORDER[3] if m in runs]:
        for seed, d in sorted(runs[model].items()):
            met = json.loads((d / "metrics.json").read_text())
            spec = json.loads((d / "spec.json").read_text())
            dec = met.get("decoding", {})
            for split in [x for x in OCR_MODES if x in met]:
                m = met[split]
                rows.append(
                    {"model": model, "family": family_of(model), "setting": setting_of(model),
                     "seed": seed, "split": split,
                     "decoding": "greedy" if split.endswith("_greedy") or split == "test_length_constrained"
                     else dec.get("mode", "greedy (v1)"),
                     "width": None if split.endswith("_greedy") else dec.get("width"),
                     **{k: m[k] for k in OCR_METRICS},
                     **{k: v for k, v in m.items() if k.startswith(("pos_macro", "sample_macro"))},
                     "n_params": spec["n_params"], "epochs_run": spec["epochs_run"],
                     "best_epoch": spec["best_epoch"], "train_seconds": spec["train_seconds"]}
                )  # fmt: skip
            rows += rescoring_rows(d, model, seed, spec)
    df = pd.DataFrame(rows)
    df.to_csv(root / "task3_metrics_by_seed.csv", index=False)
    parts = []
    test_splits = [x for x in dict.fromkeys(df.split) if x.startswith("test")]
    for split in test_splits:
        sm = summarize(df[df.split == split], [*OCR_METRICS, "n_params", "train_seconds"])
        sm["setting"] = sm.model.map(setting_of)
        parts.append(sm.assign(split=split))
    pd.concat(parts).to_csv(root / "task3_metrics_summary.csv", index=False)


def rescoring_rows(d: Path, model: str, seed: int, spec: dict) -> list[dict]:
    """Rows from scripts/rescore_task3.py (GRU decoders). beam_primary uses the setting-level
    width (mean validation exact match over seeds 0-2), the plan's reported primary decoding;
    the secondary methods rerank that n-best list with setting-level lambdas."""
    f = d / "rescoring.json"
    if not f.exists():
        return []
    rs = json.loads(f.read_text())
    rows = []
    for name, m in rs["methods"].items():
        for split in ("val", "test"):
            if not isinstance(m.get(split), dict):
                continue
            hp = m.get("hyperparameter") or {}
            rows.append(
                {"model": model, "family": family_of(model), "setting": setting_of(model),
                 "seed": seed, "split": f"{split}_{m.get('plan_name', name)}", "decoding": name,
                 "width": rs["nbest"].get("primary_width"),
                 "lambda": hp.get("value") if isinstance(hp, dict) else None,
                 "plan_conformant": rs.get("plan_conformant"),
                 **{k: m[split][k] for k in OCR_METRICS},
                 "n_params": spec["n_params"], "epochs_run": spec["epochs_run"],
                 "best_epoch": spec["best_epoch"], "train_seconds": spec["train_seconds"]}
            )  # fmt: skip
    return rows


def model_specs(root: Path, task: int, runs: dict) -> None:
    out = {}
    for model, seeds in runs.items():
        specs = {s: json.loads((d / "spec.json").read_text()) for s, d in seeds.items()}
        first = specs[min(specs)]
        per_seed = {
            s: {k: sp.get(k) for k in ("epochs_run", "best_epoch", "train_seconds", "checkpoint")}
            for s, sp in specs.items()
        }
        out[model] = {k: v for k, v in first.items() if k not in ("seed", "epochs_run", "best_epoch",
                      "train_seconds", "wall_seconds_this_session", "checkpoint")} | {
            "seeds": sorted(specs), "per_seed": per_seed}  # fmt: skip
    write_json(out, root / f"task{task}_model_specs.json")


def load_ts_preds(d: Path) -> dict[str, np.ndarray]:
    z = np.load(d / "pred_test.npz")
    return {k: z[k] for k in z.files}


def paired_tests(root: Path, runs: dict[int, dict], reps: int) -> None:
    """Seed-averaged paired comparisons for the Phase 1 contrasts (plan Section 4.3)."""
    rows = []
    for task in (1, 2):
        for a, b in CONTRASTS[task]:
            if a not in runs[task] or b not in runs[task]:
                continue
            seeds = sorted(set(runs[task][a]) & set(runs[task][b]))
            pa = [load_ts_preds(runs[task][a][s]) for s in seeds]
            pb = [load_ts_preds(runs[task][b][s]) for s in seeds]
            y = pa[0]["y_true"]
            # Loss per origin: the seed-average of the per-seed loss (not the loss of the average forecast)
            se_a = np.mean([((p["y_pred"] - y) ** 2).mean(1) for p in pa], 0)
            se_b = np.mean([((p["y_pred"] - y) ** 2).mean(1) for p in pb], 0)
            ae_a = np.mean([np.abs(p["y_pred"] - y).mean(1) for p in pa], 0)
            ae_b = np.mean([np.abs(p["y_pred"] - y).mean(1) for p in pb], 0)
            boot = bootstrap_loss_diff(y, se_a, se_b, ae_a, ae_b, block=168, reps=reps, seed=task)
            for loss, la, lb in (("squared", se_a, se_b), ("absolute", ae_a, ae_b)):
                dm = dm_test(la, lb, lag=24)
                rows.append({"task": task, "model_a": a, "model_b": b, "seeds": seeds, "test": "DM",
                             "loss": loss, **dm, "seed_rule": "per-origin loss averaged over seeds"})  # fmt: skip
            for k, v in boot.items():
                rows.append({"task": task, "model_a": a, "model_b": b, "seeds": seeds,
                             "test": "block_bootstrap_168h", "metric": k, **v,
                             "note": "A - B from per-origin losses averaged over seeds"})  # fmt: skip
            if task == 2:  # plan 4.3: additionally test each horizon
                for h in range(y.shape[1]):
                    la = np.mean([(p["y_pred"][:, h] - y[:, h]) ** 2 for p in pa], 0)
                    lb = np.mean([(p["y_pred"][:, h] - y[:, h]) ** 2 for p in pb], 0)
                    rows.append({"task": task, "model_a": a, "model_b": b, "seeds": seeds,
                                 "test": "DM", "loss": "squared", "horizon": h + 1,
                                 **dm_test(la, lb, lag=24), "seed_rule": "per-origin loss averaged over seeds"})  # fmt: skip
    for a, b in CONTRASTS[3]:
        if a not in runs[3] or b not in runs[3]:
            continue
        seeds = sorted(set(runs[3][a]) & set(runs[3][b]))
        per = {}
        for m in (a, b):
            ps = [json.loads((runs[3][m][s] / "pred_test.json").read_text()) for s in seeds]
            losses = [
                per_sample_losses([r["target"] for r in p], [r["pred"] for r in p]) for p in ps
            ]
            per[m] = {k: np.mean([lo[k] for lo in losses], 0) for k in losses[0]}
            per[m]["_by_seed"] = losses
        boot = bootstrap_image_diff(per[a], per[b], reps=reps, seed=3)
        for k, v in boot.items():
            rows.append({"task": 3, "model_a": a, "model_b": b, "seeds": seeds, "test": "image_bootstrap",
                         "metric": k, **v, "note": "per-image losses averaged over seeds"})  # fmt: skip
        for s, la, lb in zip(seeds, per[a]["_by_seed"], per[b]["_by_seed"]):
            rows.append({"task": 3, "model_a": a, "model_b": b, "seeds": [s], "test": "McNemar",
                         "metric": "exact_match", **mcnemar(la["exact"] > 0, lb["exact"] > 0)})  # fmt: skip
    pd.DataFrame(rows).to_csv(root / "paired_tests.csv", index=False)


# --- figures ------------------------------------------------------------------


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = x - x.mean()
    v = (x * x).sum()
    return np.array([(x[: len(x) - k] * x[k:]).sum() / v for k in range(lags + 1)])


def fig_dataset(npz: Path, figs: Path) -> None:
    z = np.load(npz)
    names = [str(n) for n in z["names"]]
    stable = [str(n) for n in z["stable"]]
    hours = pd.DatetimeIndex(z["hours"])
    y = z["values"][:, names.index("MT_235")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    ax = axes[0]
    w = slice(24 * 7 * 60, 24 * 7 * 62)  # two weeks in early 2012
    ax.plot(hours[w], y[w], color=COLORS["bigru_attention"], lw=1.4)
    ax.set_title("MT_235, two weeks of hourly load", fontsize=9, loc="left")
    ax.set_ylabel("kW")
    ax.tick_params(axis="x", rotation=30)
    ax = axes[1]
    prof = pd.DataFrame({"y": y, "h": hours.hour, "dow": hours.day_of_week})
    weekday = prof[prof.dow < 5].groupby("h").y.mean()
    weekend = prof[prof.dow >= 5].groupby("h").y.mean()
    ax.plot(weekday.index, weekday.values, color=COLORS["bigru_attention"], lw=1.6, label="Mon-Fri")
    ax.plot(weekend.index, weekend.values, color=COLORS["bigru"], lw=1.6, ls="--", label="Sat-Sun")
    ax.set_title("Mean load by hour of day", fontsize=9, loc="left")
    ax.set_xlabel("hour")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    r = acf(y, 200)
    ax.vlines(np.arange(len(r)), 0, r, color=COLORS["bigru_attention"], lw=0.8)
    for k in (1, 24, 168):
        ax.annotate(f"lag {k}: {r[k]:.2f}", (k, r[k]), textcoords="offset points", xytext=(4, 6),
                    fontsize=7, color=INK2)  # fmt: skip
    ax.set_title("Autocorrelation of MT_235", fontsize=9, loc="left")
    ax.set_xlabel("lag (hours)")
    for a in axes:
        style_axes(a)
    fig.suptitle(f"UCI ElectricityLoadDiagrams20112014, {len(y):,} hourly rows, {len(stable)} stable clients",
                 fontsize=10, color=INK, x=0.01, ha="left")  # fmt: skip
    save(fig, figs / "dataset_overview")


def fig_ts_forecasts(runs: dict, task: int, figs: Path) -> None:
    models = [m for m in ORDER[task] if m in runs and m != "seasonal_daily"]
    if MAIN[task] not in runs:
        return
    p = {m: load_ts_preds(runs[m][min(runs[m])]) for m in models}
    ref = p[MAIN[task]]
    hours = pd.DatetimeIndex(ref["hour"])
    n = len(hours)
    h_idx = 1 if task == 2 else 0  # Task 2 panel shows horizon t+2 for comparability
    w = slice(n // 2, n // 2 + 24 * 7)
    tgt_hours = hours[w] + pd.Timedelta(hours=2)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    for ax, sl in ((axes[0], w), (axes[1], slice(w.start + 24 * 3, w.start + 24 * 4))):
        th = hours[sl] + pd.Timedelta(hours=2)
        ax.plot(th, ref["y_true"][sl, h_idx], color=INK, lw=1.8, label="observed")
        for m in models:
            ax.plot(
                th,
                p[m]["y_pred"][sl, h_idx],
                color=COLORS[m],
                lw=1.3,
                ls=STYLES.get(m, "-"),
                label=LABELS[m],
            )
        ax.set_ylabel("kW")
        style_axes(ax)
    axes[0].set_title(f"Task {task} test forecasts of MT_235 at t+2 (seed 0), one week from {tgt_hours[0]:%Y-%m-%d}",
                      fontsize=9, loc="left")  # fmt: skip
    axes[1].set_title("Zoom on one day", fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    save(fig, figs / f"task{task}_baseline_forecasts")


def fig_ts_attention(runs: dict, task: int, figs: Path) -> None:
    if MAIN[task] not in runs:
        return
    seeds = runs[MAIN[task]]
    att = [load_ts_preds(d).get("attention") for d in seeds.values()]
    att = [a.astype(np.float32) for a in att if a is not None]
    if not att:
        return
    a = np.mean([x.mean(0) for x in att], 0)  # task1 (L,), task2 (H, L)
    L = a.shape[-1]
    lag = np.arange(L)[::-1]  # position L-1 is the origin, lag 0
    if task == 1:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(lag, a, color=COLORS[MAIN[1]], lw=1.4)
        ax.set_xlabel("hours before the forecast origin")
        ax.set_ylabel("mean attention weight")
        for k in (22, 46, 166):  # same hour as the target (t+2) one day / two days / one week back
            ax.axvline(k, color=INK2, lw=0.7, ls=":")
            ax.annotate(
                f"{k} h", (k, ax.get_ylim()[1]), fontsize=7, color=INK2, ha="center", va="bottom"
            )
        ax.invert_xaxis()
        ax.set_title(
            "Task 1 temporal attention, test set mean over 3 seeds", fontsize=9, loc="left"
        )
        style_axes(ax)
    else:
        fig, ax = plt.subplots(figsize=(10, 2.8))
        im = ax.imshow(
            a, aspect="auto", cmap="Blues", extent=(L - 0.5, -0.5, a.shape[0] + 0.5, 0.5)
        )
        ax.set_xlabel("hours before the forecast origin")
        ax.set_ylabel("decoder step h")
        ax.set_title("Task 2 decoder attention, test set mean over 3 seeds", fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, label="mean weight")
        style_axes(ax)
        ax.grid(False)
    save(fig, figs / f"task{task}_attention")


def fig_task2_horizon(root: Path, figs: Path) -> None:
    f = root / "task2_metrics_by_horizon.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    df = df[(df.split == "test") & df.horizon.str.match(r"h\d")]
    df["h"] = df.horizon.str[1:].astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, met in zip(axes, ("RMSE", "MAE", "R2")):
        for m in [m for m in ORDER[2] if m in set(df.model)]:
            g = df[df.model == m].groupby("h")[met]
            mu, sd = g.mean(), g.std(ddof=1).fillna(0)
            ax.errorbar(mu.index, mu.values, yerr=sd.values, color=COLORS[m], lw=1.5, ls=STYLES.get(m, "-"),
                        marker="o", ms=4, capsize=2, label=LABELS[m])  # fmt: skip
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("horizon (hours ahead)")
        ax.set_title(f"{met} by horizon (NSE equals R2)" if met == "R2" else f"{met} by horizon (kW)",
                     fontsize=9, loc="left")  # fmt: skip
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, figs / "task2_horizon_metrics")


SETTING_TEXT = {
    "A_oracle_crop": "Setting A, oracle GT-box crop",
    "B_whole_image": "Setting B, whole image",
}


def fig_task3(runs: dict, svhn_dir: Path, figs: Path, setting: str) -> None:
    """Attention examples and length/error breakdown for one Task 3 setting (primary decoding)."""
    suffix = WHOLE if setting == "B_whole_image" else ""
    images_file = "test_full_images.npy" if suffix else "test_images.npy"
    tag = "whole" if suffix else "crop"
    main = MAIN[3] + suffix
    if main in runs and (svhn_dir / images_file).exists():
        d0 = runs[main][min(runs[main])]
        preds = json.loads((d0 / "pred_test.json").read_text())
        att_f = d0 / "attention_test.npz"
        imgs = np.load(svhn_dir / images_file, mmap_mode="r")
        if att_f.exists():
            att = np.load(att_f)["attention"].astype(np.float32)
            ok = [
                i for i, r in enumerate(preds) if r["target"] == r["pred"] and len(r["target"]) >= 3
            ]
            bad = [i for i, r in enumerate(preds) if r["target"] != r["pred"]]
            rows = ok[:3] + bad[:3]
            steps = 1 + max(len(preds[i]["target"]) for i in rows)
            fig, axes = plt.subplots(
                len(rows), steps + 1, figsize=(1.4 * (steps + 1), 1.5 * len(rows) + 0.5)
            )
            g = math.isqrt(att.shape[-1])  # square token grid (7x7 for the VGG encoder)
            for r, i in enumerate(rows):
                img = np.asarray(imgs[i])[5:59, 5:59]  # the center 54x54 crop the model saw
                axes[r, 0].imshow(img)
                kind = "correct" if i in ok[:3] else "error"
                axes[r, 0].set_title(f"{kind}: true {preds[i]['target']} / pred {preds[i]['pred'] or '-'}",
                                     fontsize=7, loc="left", color=INK)  # fmt: skip
                for k in range(steps):
                    ax = axes[r, k + 1]
                    ax.imshow(img)
                    if k <= len(preds[i]["pred"]):
                        heat = att[i, k].reshape(g, g)
                        ax.imshow(heat, cmap="magma", alpha=0.55, extent=(0, 54, 54, 0),
                                  interpolation="bilinear")  # fmt: skip
                        ch = preds[i]["pred"][k] if k < len(preds[i]["pred"]) else "EOS"
                        ax.set_title(f"step {k + 1}: {ch}", fontsize=7, color=INK2)
                for ax in axes[r]:
                    ax.axis("off")
            fig.suptitle(f"Task 3, {SETTING_TEXT[setting]}: attention of each decoding step "
                         "(primary beam decode, seed 0)", fontsize=9, x=0.01, ha="left", color=INK)  # fmt: skip
            save(fig, figs / f"task3_examples_attention_{tag}")

    models = [m for m in ORDER[3] if m in runs and setting_of(m) == setting]
    if not models:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    width = 0.8 / len(models)
    kinds = ["correct", "too short", "too long", "wrong digit"]
    for j, m in enumerate(models):
        acc_by_len, types = [], []
        for d in runs[m].values():
            df = pd.DataFrame(json.loads((d / "pred_test.json").read_text()))
            df["L"] = df.target.str.len()
            df["ok"] = df.target == df.pred
            acc_by_len.append(df.groupby("L").ok.mean())
            lp = df.pred.str.len()
            t = np.where(
                df.ok,
                "correct",
                np.where(lp < df.L, "too short", np.where(lp > df.L, "too long", "wrong digit")),
            )
            types.append(pd.Series(t).value_counts(normalize=True).reindex(kinds, fill_value=0))
        acc = pd.concat(acc_by_len, axis=1)
        x = np.arange(len(acc)) + (j - (len(models) - 1) / 2) * width
        axes[0].bar(x, acc.mean(axis=1), width * 0.92, yerr=acc.std(axis=1, ddof=1).fillna(0),
                    color=COLORS[m], label=LABELS[m], capsize=2, edgecolor="white", linewidth=1)  # fmt: skip
        axes[0].set_xticks(np.arange(len(acc)), [f"{int(L)} digits" for L in acc.index])
        tp = pd.concat(types, axis=1).mean(axis=1)
        xx = np.arange(1, len(kinds)) + (j - (len(models) - 1) / 2) * width
        axes[1].bar(xx, tp.values[1:] * 100, width * 0.92, color=COLORS[m], label=LABELS[m],
                    edgecolor="white", linewidth=1)  # fmt: skip
    axes[1].set_xticks(np.arange(1, len(kinds)), kinds[1:])
    axes[0].set_title(f"{SETTING_TEXT[setting]}: exact-string accuracy by length (test, mean ± sd)",
                      fontsize=9, loc="left")  # fmt: skip
    axes[1].set_title("Error types, % of test images (primary decoding)", fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    for a in axes:
        style_axes(a)
    save(fig, figs / f"task3_length_errors_{tag}")


def fig_task3_protocols(root: Path, figs: Path) -> None:
    """Exact-string accuracy per setting and decoding, by model family, mean ± sd over seeds."""
    f = root / "task3_metrics_by_seed.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    df = df[df.split != "val"].copy()
    cats = [
        ("A_oracle_crop", "test", "A: GT crop\nprimary decode"),
        ("A_oracle_crop", "test_greedy", "A: GT crop\ngreedy (diagnostic)"),
        ("A_oracle_crop", "test_whole", "A model on whole\nimages (stress)"),
        ("B_whole_image", "test", "B: whole image\nprimary decode"),
        ("B_whole_image", "test_greedy", "B: whole image\ngreedy (diagnostic)"),
    ]
    cats = [c for c in cats if ((df.setting == c[0]) & (df.split == c[1])).any()]
    fams = [m for m in OCR_FAMILIES if m in set(df.family)]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    width = 0.8 / max(1, len(fams))
    for j, fam in enumerate(fams):
        mu, sd = [], []
        for setting, split, _ in cats:
            v = df[(df.family == fam) & (df.setting == setting) & (df.split == split)].exact_match
            mu.append(v.mean() * 100 if len(v) else np.nan)
            sd.append(v.std(ddof=1) * 100 if len(v) > 1 else 0)
        x = np.arange(len(cats)) + (j - (len(fams) - 1) / 2) * width
        ax.bar(x, mu, width * 0.92, yerr=sd, color=COLORS[fam], label=LABELS[fam], capsize=2,
               edgecolor="white", linewidth=1)  # fmt: skip
    ax.set_xticks(np.arange(len(cats)), [c[2] for c in cats], fontsize=7)
    ax.set_ylabel("exact-string accuracy (%)")
    ax.set_title("Task 3 test accuracy by setting and decoding (mean ± sd over 3 seeds)",
                 fontsize=9, loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=7, ncol=2)
    style_axes(ax)
    save(fig, figs / "task3_protocols")


def fig_tradeoff(root: Path, figs: Path) -> None:
    pts = []
    for task, key, lab in ((1, "RMSE", "test RMSE (kW)"), (2, "RMSE", "test pooled RMSE (kW)"),
                           (3, "exact_match", "test exact-string accuracy")):  # fmt: skip
        f = root / f"task{task}_metrics_summary.csv"
        if f.exists():
            df = pd.read_csv(f)
            if "split" in df:
                df = df[df.split == "test"]
            df = df[df.n_params_mean > 0]
            pts.append((task, key, lab, df))
    if not pts:
        return
    fig, axes = plt.subplots(1, len(pts), figsize=(4.4 * len(pts), 3.3))
    axes = np.atleast_1d(axes)
    for ax, (task, key, lab, df) in zip(axes, pts):
        for _, r in df.iterrows():
            ax.scatter(r.train_seconds_mean / 60, r[f"{key}_mean"], s=30 + r.n_params_mean / 2e5,
                       color=COLORS.get(r.model, INK2), edgecolor="white", linewidth=1.5, zorder=3)  # fmt: skip
            ax.annotate(f"{LABELS.get(r.model, r.model)}\n{r.n_params_mean / 1e6:.2f}M params",
                        (r.train_seconds_mean / 60, r[f"{key}_mean"]), fontsize=6.5, color=INK2,
                        textcoords="offset points", xytext=(6, 4))  # fmt: skip
        ax.set_xlabel("training time per run (min, A100)")
        ax.set_ylabel(lab)
        ax.set_title(f"Task {task}", fontsize=9, loc="left")
        style_axes(ax)
    save(fig, figs / "compute_accuracy_tradeoff")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--config3", default="configs/phase1_task3.yaml")
    ap.add_argument("--reps", type=int, default=10_000)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    root = Path(cfg["results_root"]) / "phase1"
    figs = Path(cfg["results_root"]) / "figures" / "phase1"
    runs = {t: load_runs(root, t) for t in (1, 2, 3)}
    for t in (1, 2, 3):
        have = {m: sorted(s) for m, s in runs[t].items()}
        missing = [m for m in ORDER[t] if m not in runs[t] or len(runs[t][m]) < len(cfg["seeds"])]
        logger.info(f"task{t} runs {have} incomplete {missing}")
    for t in (1, 2):
        if runs[t]:
            ts_tables(root, t, runs[t])
            model_specs(root, t, runs[t])
    if runs[3]:
        ocr_tables(root, runs[3])
        model_specs(root, 3, runs[3])
    paired_tests(root, runs, a.reps)
    fig_dataset(Path(cfg["electricity_npz"]), figs)
    for t in (1, 2):
        fig_ts_forecasts(runs[t], t, figs)
        fig_ts_attention(runs[t], t, figs)
    fig_task2_horizon(root, figs)
    cfg3 = yaml.safe_load(Path(a.config3).read_text()) if Path(a.config3).exists() else cfg
    for setting in ("A_oracle_crop", "B_whole_image"):
        fig_task3(runs[3], Path(cfg3.get("svhn_dir", "data/processed/svhn")), figs, setting)
    fig_task3_protocols(root, figs)
    fig_tradeoff(root, figs)
    index = [{"file": str(p), "task": p.stem.split("_")[0]} for p in sorted(figs.glob("*.png"))]
    write_json(index, figs / "figure_index.json")
    logger.info(f"exported tables to {root} and {len(index)} figures to {figs}")


if __name__ == "__main__":
    main()
