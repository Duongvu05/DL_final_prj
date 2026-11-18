# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOperatorIssue=false
"""Task 3 improvement study export (docs/experiment_plan.md Section 3).

Compares every improvement I1-I6 with the frozen Phase 1 reference I0 (cnn_attention_gru,
validation-selected setting-level beam) in each setting, with a 10,000-replicate paired image
bootstrap on per-image losses averaged over the same seeds, two-sided bootstrap p-values, Holm
correction over the setting's primary families and descriptive per-seed McNemar tests.

Writes results/improve/task3_improve_{by_seed,summary,tests}.csv and
results/figures/improve/task3_improve_effects.{png,svg}.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from dl_final_prj.metrics import per_sample_losses, sequence_report
from dl_final_prj.stats import bootstrap_image_diff, holm, mcnemar

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEEDS = (0, 1, 2)
# variant -> (results phase, run name, prediction file stem, description)
SETTINGS = {
    "A_oracle_crop": {
        "I0": (
            "phase1",
            "cnn_attention_gru",
            "pred_test",
            "Phase 1 reference, beam (val-selected)",
        ),
        "I1": (
            "phase1",
            "cnn_attention_gru",
            "pred_test_bidir_rescore",
            "DRAM forward-backward rescoring",
        ),
        "I2": ("improve", "i2_rrc", "pred_test", "RandomResizedCrop 0.70-1.00"),
        "I3": ("improve", "i3_ls", "pred_test", "label smoothing 0.05"),
    },
    "B_whole_image": {
        "I0": (
            "phase1",
            "cnn_attention_gru_whole",
            "pred_test",
            "Phase 1 reference, beam (val-selected)",
        ),
        "I1": (
            "phase1",
            "cnn_attention_gru_whole",
            "pred_test_bidir_rescore",
            "DRAM forward-backward rescoring",
        ),
        "I2": ("improve", "i2_rrc_whole", "pred_test", "RandomResizedCrop 0.70-1.00"),
        "I3": ("improve", "i3_ls_whole", "pred_test", "label smoothing 0.05"),
        "I4": ("improve", "i4_stn_whole", "pred_test", "affine spatial transformer"),
        "I5": ("improve", "i5_res96_whole", "pred_test", "96x96 input, 80x80 crops"),
        "I6": ("improve", "i6_combo_whole", "pred_test", "I2 + I3 + I4 + I5"),
    },
}
PRIMARY = {"A_oracle_crop": ["I1", "I2"], "B_whole_image": ["I1", "I2", "I4", "I5"]}
METRICS = ["exact_match", "char_accuracy", "length_accuracy", "norm_edit_distance", "CER",
           "R2", "NSE", "MAE", "RMSE"]  # fmt: skip
COLOR = {"A_oracle_crop": "#2a78d6", "B_whole_image": "#eb6834"}


def load_preds(path: Path) -> tuple[list[str], list[str]]:
    rows = json.loads(path.read_text())
    return [r["target"] for r in rows], [r["pred"] for r in rows]


def run_dir(root: Path, phase: str, name: str, seed: int) -> Path:
    return root / phase / "runs" / "task3" / name / f"seed{seed}"


def variant_info(root: Path, phase: str, name: str, stem: str, seed: int) -> dict:
    d = run_dir(root, phase, name, seed)
    spec = json.loads((d / "spec.json").read_text())
    info = {"n_params": spec["n_params"], "train_seconds": spec.get("train_seconds")}
    if stem == "pred_test_bidir_rescore":
        rs = json.loads((d / "rescoring.json").read_text())
        m = rs["methods"]["bidir_rescore"]
        hp = m.get("hyperparameter") or {}
        r2l = run_dir(root, phase, rs.get("r2l_model") or f"{name}_r2l", seed)
        info["n_params"] += json.loads((r2l / "spec.json").read_text())["n_params"]
        info["decoding"] = f"beam{rs['nbest']['primary_width']} + R2L rescoring"
        info["lambda"] = hp.get("value") if isinstance(hp, dict) else None
    else:
        dec = json.loads((d / "metrics.json").read_text())["decoding"]
        info["decoding"] = dec["mode"]
    return info


def collect(root: Path) -> tuple[pd.DataFrame, dict]:
    rows, preds = [], {}
    for setting, variants in SETTINGS.items():
        for vid, (phase, name, stem, desc) in variants.items():
            for s in SEEDS:
                f = run_dir(root, phase, name, s) / f"{stem}.json"
                if not f.exists():
                    logger.warning(f"missing {f}")
                    continue
                t, p = load_preds(f)
                preds[(setting, vid, s)] = (t, p)
                rep = sequence_report(t, p)
                rows.append({"setting": setting, "variant": vid, "description": desc, "run": name,
                             "seed": s, **variant_info(root, phase, name, stem, s),
                             **{k: rep[k] for k in METRICS}})  # fmt: skip
    return pd.DataFrame(rows), preds


def paired_tests(preds: dict, reps: int) -> pd.DataFrame:
    rows = []
    for setting, variants in SETTINGS.items():
        if any((setting, "I0", s) not in preds for s in SEEDS):
            continue
        ref = [preds[(setting, "I0", s)] for s in SEEDS]
        ref_l = [per_sample_losses(*r) for r in ref]
        for vid in variants:
            if vid == "I0":
                continue
            if any((setting, vid, s) not in preds for s in SEEDS):
                continue
            cur = [preds[(setting, vid, s)] for s in SEEDS]
            assert all(c[0] == r[0] for c, r in zip(cur, ref, strict=True)), "test order differs"
            cur_l = [per_sample_losses(*c) for c in cur]
            avg = lambda ls: {k: np.mean([x[k] for x in ls], 0) for k in ls[0]}  # noqa: E731
            boot = bootstrap_image_diff(avg(cur_l), avg(ref_l), reps=reps, seed=7)
            for metric, v in boot.items():
                rows.append({"setting": setting, "variant": vid, "vs": "I0", "test": "paired_image_bootstrap",
                             "metric": metric, **v, "primary": vid in PRIMARY[setting],
                             "seed_rule": "per-image losses averaged over seeds 0-2"})  # fmt: skip
            for s, c, r in zip(SEEDS, cur_l, ref_l, strict=True):
                rows.append({"setting": setting, "variant": vid, "vs": "I0", "test": "McNemar_descriptive",
                             "metric": "exact_match", "seed": s, **mcnemar(c["exact"] > 0, r["exact"] > 0)})  # fmt: skip
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for setting in SETTINGS:
        m = (
            (df.setting == setting)
            & (df.test == "paired_image_bootstrap")
            & df.primary.astype(bool)
        )
        m &= df.metric == "exact_match"
        if m.any():
            df.loc[m, "p_holm"] = holm(df.loc[m, "p_value"].tolist())
    return df


def figure(tests: pd.DataFrame, out: Path) -> None:
    t = tests[(tests.test == "paired_image_bootstrap") & (tests.metric == "exact_match")]
    if t.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=False)
    for ax, setting in zip(axes, SETTINGS, strict=True):
        d = t[t.setting == setting].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        ax.errorbar(d["diff"] * 100, y, xerr=[(d["diff"] - d.ci_low) * 100, (d.ci_high - d["diff"]) * 100],
                    fmt="o", color=COLOR[setting], ms=6, capsize=3, lw=1.5)  # fmt: skip
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        labels = [
            f"{v}: {SETTINGS[setting][v][3]}" + (" *" if p else "")
            for v, p in zip(d.variant, d.primary)
        ]
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_xlabel("exact-string accuracy difference vs I0 (points, 95% CI)")
        ax.set_title(setting.replace("_", " "), fontsize=9, loc="left", color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=7)
    fig.suptitle("Task 3 improvements vs the Phase 1 CNN + attention GRU (paired image bootstrap, "
                 "seeds 0-2; * primary family, Holm-corrected)", fontsize=9, x=0.01, ha="left")  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--reps", type=int, default=10_000)
    a = ap.parse_args()
    root = Path(a.results)
    by_seed, preds = collect(root)
    out = root / "improve"
    out.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(out / "task3_improve_by_seed.csv", index=False)
    g = by_seed.groupby(["setting", "variant", "description"], sort=False)
    summ = g[[*METRICS, "n_params"]].agg(["mean", "std"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ["n_seeds"] = g.size()
    summ.reset_index().to_csv(out / "task3_improve_summary.csv", index=False)
    tests = paired_tests(preds, a.reps)
    tests.to_csv(out / "task3_improve_tests.csv", index=False)
    figure(tests, root / "figures" / "improve" / "task3_improve_effects")
    show = summ.reset_index()[
        ["setting", "variant", "exact_match_mean", "exact_match_std", "n_seeds"]
    ]
    logger.info("\n" + show.to_string(index=False))


if __name__ == "__main__":
    main()
