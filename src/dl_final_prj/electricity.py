"""UCI ElectricityLoadDiagrams20112014 for Tasks 1 and 2 (plan Sections 1.1 and 4.1).

prepare() turns the raw 15-minute file into an hourly matrix once. ElectricityData then
builds leakage-safe windows. Split rule: the hourly rows are split 70/15/15 in time order,
an origin t (last observed hour) belongs to the split that contains ALL of its target hours,
and the input window may reach back across a split boundary (that history is observed).
The scaler is fitted on train rows only.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dl_final_prj.fast import parse_semicolon_table

TARGET = "MT_235"
CALENDAR = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")
RIDGE_LAGS = (1, 2, 3, 24, 48, 168)
URL = "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"


def split_bounds(
    n: int, train_frac: float = 0.7, val_frac: float = 0.15
) -> dict[str, tuple[int, int]]:
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return {"train": (0, i_train), "val": (i_train, i_val), "test": (i_val, n)}


def calendar_features(stamps: pd.DatetimeIndex) -> np.ndarray:
    hour = stamps.hour.to_numpy()
    dow = stamps.day_of_week.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]
    ).astype(np.float32)


def stable_clients(
    values: np.ndarray, names: list[str], train_end: int, max_zero_frac: float = 0.01
) -> list[str]:
    """Clients nonzero within the first 24 hours and with < 1% zero hours in the train rows.

    Only train rows are used, so the choice cannot leak validation or test information.
    """
    head_live = (values[:24] > 0).any(axis=0)
    zero_frac = (values[:train_end] == 0).mean(axis=0)
    keep = head_live & (zero_frac < max_zero_frac)
    return [n for n, k in zip(names, keep, strict=True) if k]


def prepare(raw_txt: Path, out_dir: Path) -> dict:
    """Parse LD2011_2014.txt, aggregate to hourly means, freeze the client list, write npz + audit."""
    names, stamps, raw = parse_semicolon_table(str(raw_txt))
    ts = pd.DatetimeIndex(pd.to_datetime(stamps))
    steps = np.diff(ts.asi8)
    if not np.all(steps == 15 * 60 * 10**9):
        raise ValueError("raw timestamps are not a regular 15-minute grid")
    if len(raw) % 4 or int(ts.minute[3]) != 0:
        raise ValueError("expected rows :15 :30 :45 :00 per hour, starting at 00:15")
    # Right-closed, right-labelled hours: rows 00:15..01:00 become the hour labelled 01:00.
    hourly = raw.reshape(-1, 4, raw.shape[1]).mean(axis=1)
    hours = ts[3::4]
    bounds = split_bounds(len(hourly))
    stable = stable_clients(hourly, names, bounds["train"][1])
    if TARGET not in stable:
        raise ValueError(f"{TARGET} is not in the stable-history client set")
    first_nz = np.argmax(hourly > 0, axis=0)
    t = hourly[:, names.index(TARGET)]
    audit = {
        "dataset": "ElectricityLoadDiagrams20112014",
        "url": URL,
        "raw_file": raw_txt.name,
        "raw_rows": len(raw),
        "raw_frequency": "15 minutes",
        "raw_first": str(ts[0]),
        "raw_last": str(ts[-1]),
        "hourly_rows": len(hourly),
        "hourly_first": str(hours[0]),
        "hourly_last": str(hours[-1]),
        "aggregation": "mean of the four 15-minute readings in (h-1, h], labelled h",
        "parsing": "semicolon separated, decimal comma, first column quoted timestamp",
        "n_attributes": len(names),
        "attributes": names,
        "n_stable_clients": len(stable),
        "stable_clients": stable,
        "stable_rule": "nonzero within the first 24 hours and < 1% zero hours in the train rows",
        "missing_values": int(np.isnan(raw).sum()),
        "negative_values": int((raw < 0).sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "clients_with_structural_leading_zeros": int((first_nz > 24).sum()),
        "pred_column": TARGET,
        "target_stats": {
            "mean": float(t.mean()),
            "std": float(t.std()),
            "min": float(t.min()),
            "max": float(t.max()),
            "zero_hours": int((t == 0).sum()),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "electricity_hourly.npz",
        values=hourly.astype(np.float64),
        hours=hours.asi8,
        names=np.array(names),
        stable=np.array(stable),
    )
    (out_dir / "electricity_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


@dataclass
class Batch:
    x: torch.Tensor  # (B, L, F) scaled inputs
    y: torch.Tensor  # (B, H) scaled targets
    cal_future: torch.Tensor  # (B, H, 4) calendar of the target hours
    origin: torch.Tensor  # (B,) row index of the last observed hour


class ElectricityData:
    """Scaled feature matrix on the device plus origin lists per split.

    Features are the stable clients (loads, standardized with train statistics) followed by
    the four calendar columns. `offsets` are the target steps after the origin, (2,) for
    Task 1 and (1, 2, 3, 4, 5) for Task 2.
    """

    def __init__(
        self,
        npz: Path,
        lookback: int,
        offsets: tuple[int, ...],
        device: str = "cpu",
        target_only: bool = False,
        use_calendar: bool = True,
    ) -> None:
        z = np.load(npz, allow_pickle=False)
        names = [str(n) for n in z["names"]]
        stable = [str(n) for n in z["stable"]]
        cols = [TARGET] if target_only else stable
        values = z["values"][:, [names.index(c) for c in cols]]
        self.hours = pd.DatetimeIndex(z["hours"])
        self.lookback, self.offsets = lookback, tuple(offsets)
        self.n = len(values)
        self.bounds = split_bounds(self.n)
        tr0, tr1 = self.bounds["train"]
        self.mean = values[tr0:tr1].mean(0)
        self.std = values[tr0:tr1].std(0) + 1e-8
        self.feature_names = list(cols)
        self.target_col = cols.index(TARGET)
        scaled = (values - self.mean) / self.std
        cal = calendar_features(self.hours)
        feats = np.hstack([scaled, cal]) if use_calendar else scaled
        if use_calendar:
            self.feature_names += list(CALENDAR)
        self.y_raw = values[:, self.target_col].astype(np.float64)
        self.device = device
        self.X = torch.tensor(feats, dtype=torch.float32, device=device)
        self.Y = torch.tensor(scaled[:, self.target_col], dtype=torch.float32, device=device)
        self.C = torch.tensor(cal, dtype=torch.float32, device=device)
        self._ar = torch.arange(lookback, device=device)
        self._off = torch.tensor(self.offsets, device=device)
        self.origins = {name: self._origins(name) for name in self.bounds}

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def horizon(self) -> int:
        return len(self.offsets)

    def _origins(self, split: str) -> np.ndarray:
        lo, hi = self.bounds[split]
        first, last = min(self.offsets), max(self.offsets)
        t = np.arange(self.lookback - 1, self.n - last)
        return t[(t + first >= lo) & (t + last < hi)]

    def target_scale(self) -> tuple[float, float]:
        return float(self.mean[self.target_col]), float(self.std[self.target_col])

    def inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        m, s = self.target_scale()
        return np.asarray(y_scaled, np.float64) * s + m

    def true_targets(self, split: str) -> np.ndarray:
        """Original-unit targets (n_origins, H)."""
        o = self.origins[split]
        return self.y_raw[o[:, None] + np.array(self.offsets)]

    def batch(self, origins: torch.Tensor) -> Batch:
        o = origins.to(self.device)
        rows = o[:, None] - self.lookback + 1 + self._ar
        tgt = o[:, None] + self._off
        return Batch(self.X[rows], self.Y[tgt], self.C[tgt], o)

    def iterate(self, split: str, batch_size: int, generator: torch.Generator | None = None):
        """Yield batches. Shuffled with `generator` when given, else in time order."""
        o = torch.as_tensor(self.origins[split])
        if generator is not None:
            o = o[torch.randperm(len(o), generator=generator)]
        for i in range(0, len(o), batch_size):
            yield self.batch(o[i : i + batch_size])

    def split_summary(self) -> dict:
        out = {}
        for name, (lo, hi) in self.bounds.items():
            o = self.origins[name]
            out[name] = {
                "rows": [int(lo), int(hi)],
                "first_hour": str(self.hours[lo]),
                "last_hour": str(self.hours[hi - 1]),
                "n_windows": len(o),
                "first_target_hour": str(self.hours[o[0] + min(self.offsets)]),
                "last_target_hour": str(self.hours[o[-1] + max(self.offsets)]),
            }
        return out


def ridge_design(
    y_raw: np.ndarray, cal: np.ndarray, origins: np.ndarray, offset: int
) -> np.ndarray:
    """Ridge features for target t+offset. Lag k means y[t+1-k], so lag 1 is the last observation."""
    lags = np.column_stack([y_raw[origins + 1 - k] for k in RIDGE_LAGS])
    return np.hstack([lags, cal[origins + offset]])
