# DL final project, group 12

Electricity load forecasting on UCI ElectricityLoadDiagrams20112014 (Tasks 1 and 2, target
`MT_235`) and full house-number recognition on SVHN Format 1 (Task 3). The experiment design
is in [docs/experiment_plan.md](docs/experiment_plan.md).

## Layout

- `rust/dlfast` - Rust extension (PyO3 + rayon) for the CPU hot paths. It parses the UCI
  table, letterboxes SVHN images into a uint8 cache, runs the moving-block bootstrap and
  computes batched edit distances. Every function has a NumPy reference in
  `src/dl_final_prj/fast.py`, and the tests compare the two.
- `src/dl_final_prj` - data (`electricity.py`, `svhn.py`), models (`ts_models.py`,
  `ocr_models.py`), baselines, metrics, paired tests and the resumable training loop.
- `scripts` - data preparation, one script per training run, the Phase 1 queue and the export.
- `configs/phase1.yaml` - all Phase 1 hyperparameters and seeds.

## Local development

```bash
uv sync          # builds the Rust extension, needs cargo
uv run pytest
```

## On the A100 pod

```bash
bash scripts/setup_pod_env.sh        # venv on top of the image's PyTorch 2.8
.venv/bin/python scripts/prepare_data.py
bash scripts/phase1.sh               # rerun after a preemption, it resumes
.venv/bin/python scripts/export_phase1.py
```

Raw data goes in `data/raw/uci_electricity/LD2011_2014.txt` and `data/raw/svhn/{train,test}/`.
Results land in `results/` (runs, tables, figures and manifests).
