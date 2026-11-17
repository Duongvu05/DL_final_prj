"""Training loop with checkpoint/resume for the preemptible pod (plan Section 4.2).

A checkpoint is written atomically at the end of every epoch. It holds the model, optimizer,
scheduler, epoch counter, early-stopping state, best weights, history and all RNG states.
The shuffling order of epoch e is drawn from a generator seeded by (seed, e), so a resumed
run sees exactly the same batches as an uninterrupted one.
"""

import copy
import json
import os
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epoch_generator(seed: int, epoch: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed * 100_003 + epoch)
    return g


def atomic_save(obj: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


@dataclass
class FitConfig:
    epochs: int
    patience: int
    lr: float
    weight_decay: float = 1e-4
    clip: float = 1.0
    plateau_patience: int = 3
    plateau_factor: float = 0.5
    amp: str = "none"  # "bf16" or "none"
    schedule: str = "plateau"  # "plateau" (ReduceLROnPlateau on val) or "step"
    step_epoch: int = 10  # "step": multiply the lr by step_gamma after this many epochs
    step_gamma: float = 0.1


def fit(
    run_dir: Path,
    model: nn.Module,
    param_groups: list[dict],
    train_batches: Callable[[int, torch.Generator], Iterable],
    step_loss: Callable[[nn.Module, object, int], torch.Tensor],
    validate: Callable[[nn.Module], float],
    cfg: FitConfig,
    seed: int,
    on_epoch_start: Callable[[nn.Module, int], None] | None = None,
) -> tuple[nn.Module, list[dict]]:
    """Train with early stopping on `validate` (lower is better). Resumes from run_dir/ckpt.pt."""
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.pt"
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.schedule == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_epoch, gamma=cfg.step_gamma)
    elif cfg.schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience
        )
    else:
        raise ValueError(f"unknown schedule {cfg.schedule}")
    start, best, bad, history = 1, float("inf"), 0, []
    best_state = copy.deepcopy(model.state_dict())
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        best_state, history = ck["best_state"], ck["history"]
        _set_rng_state(ck["rng"])
        logger.info(f"resumed {run_dir} at epoch {start} (best {best:.5f}, bad {bad})")
        if ck.get("stopped"):
            model.load_state_dict(best_state)
            return model, history
    else:
        set_seed(seed)
    dtype = torch.bfloat16 if cfg.amp == "bf16" else None
    for epoch in range(start, cfg.epochs + 1):
        model.train()
        if on_epoch_start is not None:
            on_epoch_start(model, epoch)  # after train(), so it can put frozen parts in eval mode
        t0, total, steps = time.time(), 0.0, 0
        for batch in train_batches(epoch, epoch_generator(seed, epoch)):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                loss = step_loss(model, batch, epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            total += float(loss.detach())
            steps += 1
        model.eval()
        with torch.no_grad():
            val = float(validate(model))
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(val)
        else:
            sched.step()
        improved = val < best
        if improved:
            best, bad = val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        stopped = bad >= cfg.patience or epoch == cfg.epochs
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(1, steps),
                "val": val,
                "lr": opt.param_groups[0]["lr"],
                "seconds": time.time() - t0,
            }
        )
        logger.info(
            f"{run_dir.name} epoch {epoch} loss {history[-1]['train_loss']:.5f} val {val:.5f}"
            f"{' *' if improved else ''} ({history[-1]['seconds']:.1f}s)"
        )
        atomic_save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "best": best,
                "bad": bad,
                "best_state": best_state,
                "history": history,
                "rng": _rng_state(),
                "stopped": stopped,
            },
            ckpt_path,
        )
        if stopped:
            break
    model.load_state_dict(best_state)
    return model, history


def prepare_run_dir(run_dir: Path, fingerprint: dict, force: bool) -> None:
    """Guard against resuming a checkpoint made with other data or settings.

    With force the run directory is wiped, so the run really restarts. Otherwise an
    existing run_dir/fingerprint.json must equal `fingerprint` (config, model and data
    identity), else a ValueError asks for --force instead of silently reusing ckpt.pt.
    """
    import shutil

    fp_path = run_dir / "fingerprint.json"
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    current = json.loads(json.dumps(fingerprint, default=str, sort_keys=True))
    if fp_path.exists():
        saved = json.loads(fp_path.read_text())
        if saved != current:
            raise ValueError(
                f"{run_dir} was started with different settings or data, rerun with --force"
            )
    elif (run_dir / "ckpt.pt").exists():
        raise ValueError(f"{run_dir} has a checkpoint without a fingerprint, rerun with --force")
    write_json(current, fp_path)


def file_identity(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime": int(st.st_mtime)}


def pick_device() -> str:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return "cuda"
    return "cpu"
