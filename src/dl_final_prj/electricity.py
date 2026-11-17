"""UCI ElectricityLoadDiagrams20112014 for Tasks 1 and 2 (plan Sections 1.1, 2.1, 2.2 and 4.1).

prepare() turns the raw 15-minute file into an hourly matrix once. ElectricityData then builds
leakage-safe, target-only, seasonal-residual windows for the fixed target MT_235.

Split rule. The hourly rows are split 70/15/15 in time order. An origin t (last observed hour)
belongs to the split that contains ALL of its target hours, and the input window may reach back
across a split boundary (that history is observed at forecast time). Every scaler is fitted on
train rows only.

Timestamp-aligned lags. The hourly grid is checked to be regular (one row per hour, no gaps, no
duplicates), so row s-k is exactly the timestamp s-k hours. A lag lookup by row index is
therefore the timestamp-aligned lookup y(s-k h) the plan asks for. The UCI file keeps a 24-row
day also on DST change days. Its 15-minute readings stamped 01:00 to 01:45 are anomalous on those
days (near zero in March, two hours of load in October). With right-closed hourly means this
anomaly lands in the hourly rows labelled 01:00 (one reading) and 02:00 (three readings). Both
rows are kept as observed and marked by the dst_flag calendar feature instead of being moved.

Seasonal residual. For a target hour tau the model predicts r(tau) = y(tau) - y(tau-24), scaled
with the train mean/std of r. The kW forecast is y(tau-24) + r_hat * std_r + mean_r.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dl_final_prj.fast import parse_semicolon_table

TARGET = "MT_235"
CALENDAR = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "dst_flag")
INPUT_LAGS = (24, 48, 168)  # extra input channels y(s-k) for every window step s
SEASONAL_LAGS = (24, 48, 168)  # counted back from the target hour tau = t+h
RECENT_LAGS = (1, 2, 3)  # counted back from the origin t: y[t], y[t-1], y[t-2]
INPUT_CHANNELS = ("y", "y_lag24", "y_lag48", "y_lag168", *CALENDAR)
TARGET_FEATURES = (*(f"target_{c}" for c in CALENDAR), "b24", "b48", "b168")
DST_RULE = (
    "hourly rows labelled 01:00 and 02:00 on the last Sunday of March and of October, which "
    "hold the anomalous UCI readings stamped 01:00 to 01:45 (1 and 3 of 4 readings)"
)
RESIDUAL_DEFINITION = "r(tau) = y(tau) - y(tau - 24 h), standardized with train mean/std of r"
URL = "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"


def split_bounds(
    n: int, train_frac: float = 0.7, val_frac: float = 0.15
) -> dict[str, tuple[int, int]]:
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return {"train": (0, i_train), "val": (i_train, i_val), "test": (i_val, n)}


def dst_flags(stamps: pd.DatetimeIndex) -> np.ndarray:
    """True for the hourly rows labelled 01:00 and 02:00 on the last Sunday of March and October.

    On those change days the UCI readings stamped 01:00, 01:15, 01:30 and 01:45 are anomalous
    (near zero in March, the sum of two hours in October). The right-closed hourly mean puts
    one of them in the row labelled 01:00 and three in the row labelled 02:00, so both rows are
    flagged. The flag is a pure calendar rule, known at forecast time, and is set even when a
    client shows no anomaly on that day. March and October have 31 days, so their last Sunday
    is the Sunday that falls on day 25 or later.
    """
    month = stamps.month.to_numpy()
    return (
        np.isin(month, (3, 10))
        & (stamps.day.to_numpy() >= 25)
        & (stamps.day_of_week.to_numpy() == 6)
        & np.isin(stamps.hour.to_numpy(), (1, 2))
        & (stamps.minute.to_numpy() == 0)
    )


def calendar_features(stamps: pd.DatetimeIndex) -> np.ndarray:
    """The 7 CALENDAR columns for each timestamp, float32 (n, 7). Known at forecast time."""
    hour = stamps.hour.to_numpy()
    dow = stamps.day_of_week.to_numpy()
    doy = stamps.day_of_year.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            dst_flags(stamps).astype(np.float64),
        ]
    ).astype(np.float32)


def check_regular_hourly(hours: np.ndarray) -> None:
    """Raise unless the timestamps are strictly increasing in steps of exactly one hour."""
    steps = np.diff(np.asarray(hours).astype("datetime64[s]").astype(np.int64))
    if not np.all(steps == 3600):
        raise ValueError("hourly timestamps are not a regular one-hour grid")


def lag(a: np.ndarray, k: int) -> np.ndarray:
    """Row-shifted copy, out[s] = a[s-k]. Rows s < k have no history and are set to 0."""
    out = np.zeros_like(a)
    out[k:] = a[:-k]
    return out


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
    steps = np.diff(ts.to_numpy("datetime64[s]").astype(np.int64))
    if not np.all(steps == 15 * 60):
        raise ValueError("raw timestamps are not a regular 15-minute grid")
    if len(raw) % 4 or int(ts.minute[3]) != 0:
        raise ValueError("expected rows :15 :30 :45 :00 per hour, starting at 00:15")
    # Right-closed, right-labelled hours: rows 00:15..01:00 become the hour labelled 01:00.
    hourly = raw.reshape(-1, 4, raw.shape[1]).mean(axis=1)
    hours = ts[3::4]
    check_regular_hourly(hours.to_numpy("datetime64[s]"))
    bounds = split_bounds(len(hourly))
    stable = stable_clients(hourly, names, bounds["train"][1])
    if TARGET not in stable:
        raise ValueError(f"{TARGET} is not in the stable-history client set")
    first_nz = np.argmax(hourly > 0, axis=0)
    t = hourly[:, names.index(TARGET)]
    dst = dst_flags(hours)
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
        "hourly_regular": True,
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
        "dst_flag_rule": DST_RULE,
        "dst_flag_rows": int(dst.sum()),
        "dst_flag_hours": [str(h) for h in hours[dst]],
        "pred_column": TARGET,
        "target_stats": {
            "mean": float(t.mean()),
            "std": float(t.std()),
            "min": float(t.min()),
            "max": float(t.max()),
            "zero_hours": int((t == 0).sum()),
            "dst_flag_values": [float(v) for v in t[dst]],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "electricity_hourly.npz",
        values=hourly.astype(np.float64),
        hours=hours.to_numpy("datetime64[s]"),
        names=np.array(names),
        stable=np.array(stable),
    )
    (out_dir / "electricity_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


@dataclass
class Batch:
    x: torch.Tensor  # (B, L, F) encoder inputs, INPUT_CHANNELS (+ clients if not target-only)
    y: torch.Tensor  # (B, H) standardized residual r(t+h) targets
    target_feats: torch.Tensor  # (B, H, 10) TARGET_FEATURES of each target hour
    cal_future: torch.Tensor  # (B, H, 7) calendar of the target hours
    r0: torch.Tensor  # (B,) standardized residual r(t) at the origin
    origin: torch.Tensor  # (B,) row index of the last observed hour


class ElectricityData:
    """Target-only seasonal-residual windows for MT_235, held on the device.

    For an origin t and every input step s in [t-L+1, t], the encoder sees the 11 channels
    INPUT_CHANNELS: standardized y(s), y(s-24), y(s-48), y(s-168) (target scaler) and the
    calendar of s. For each offset h (target hour tau = t+h) the TARGET_FEATURES are the calendar
    of tau and standardized b24 = y(tau-24), b48 = y(tau-48), b168 = y(tau-168). All of these are
    observed at time t because h < 24. `offsets` is (2,) for Task 1 and (1, ..., 5) for Task 2.

    target_only=False appends the other stable clients as extra standardized input channels.
    That variant is a negative control only (plan Section 1.1), not a Phase 1/2 input.
    """

    def __init__(
        self,
        npz: Path,
        lookback: int,
        offsets: tuple[int, ...],
        device: str = "cpu",
        target_only: bool = True,
    ) -> None:
        self.lookback, self.offsets = lookback, tuple(offsets)
        if max(self.offsets) >= min(SEASONAL_LAGS) or min(self.offsets) < 1:
            raise ValueError("offsets must lie in [1, 23] so target-side lags are observed")
        z = np.load(npz, allow_pickle=False)
        names = [str(n) for n in z["names"]]
        hours = z["hours"].astype("datetime64[s]")
        check_regular_hourly(hours)
        self.hours = pd.DatetimeIndex(hours)
        self.target_only = target_only
        y = z["values"][:, names.index(TARGET)].astype(np.float64)
        self.y_raw = y
        self.n = len(y)
        self.bounds = split_bounds(self.n)
        tr0, tr1 = self.bounds["train"]

        # Target scaler (train rows only), shared by y and its lags.
        self.y_mean = float(y[tr0:tr1].mean())
        self.y_std = float(y[tr0:tr1].std()) + 1e-8
        ys = (y - self.y_mean) / self.y_std
        self.cal = calendar_features(self.hours)
        lags = [lag(ys, k) for k in INPUT_LAGS]
        feats = np.column_stack([ys, *lags, self.cal])
        self.feature_names = list(INPUT_CHANNELS)
        self.clients: list[str] = []
        if not target_only:
            self.clients = [str(c) for c in z["stable"] if str(c) != TARGET]
            cv = z["values"][:, [names.index(c) for c in self.clients]]
            mu, sd = cv[tr0:tr1].mean(0), cv[tr0:tr1].std(0) + 1e-8
            feats = np.hstack([feats, (cv - mu) / sd])
            self.feature_names += self.clients
        tfeats = np.column_stack([self.cal, *lags])  # row tau holds the features of hour tau

        # Seasonal residual r(tau) = y(tau) - y(tau-24), scaled with train rows tau >= 24.
        r = y - lag(y, 24)
        r_train = r[max(24, tr0) : tr1]
        self.r_mean = float(r_train.mean())
        self.r_std = float(r_train.std()) + 1e-8
        rs = (r - self.r_mean) / self.r_std
        rs[:24] = 0.0  # no y(tau-24) there, never used by a valid origin

        self.device = device
        self.X = torch.tensor(feats, dtype=torch.float32, device=device)
        self.TF = torch.tensor(tfeats, dtype=torch.float32, device=device)
        self.C = torch.tensor(self.cal, dtype=torch.float32, device=device)
        self.R = torch.tensor(rs, dtype=torch.float32, device=device)
        self._ar = torch.arange(lookback, device=device)
        self._off = torch.tensor(self.offsets, device=device)
        # First origin whose oldest input step s = t-L+1 still has y(s-168).
        self.first_origin = lookback - 1 + max(INPUT_LAGS)
        self.origins = {name: self._origins(name) for name in self.bounds}

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_target_feats(self) -> int:
        return self.TF.shape[1]

    @property
    def horizon(self) -> int:
        return len(self.offsets)

    def _origins(self, split: str) -> np.ndarray:
        lo, hi = self.bounds[split]
        first, last = min(self.offsets), max(self.offsets)
        t = np.arange(self.first_origin, self.n - last)
        return t[(t + first >= lo) & (t + last < hi)]

    def target_scale(self) -> tuple[float, float]:
        """Train mean and std of the target load in kW."""
        return self.y_mean, self.y_std

    def residual_scale(self) -> tuple[float, float]:
        """Train mean and std of the seasonal residual r in kW."""
        return self.r_mean, self.r_std

    def _target_rows(self, origins: np.ndarray) -> np.ndarray:
        return np.asarray(origins, np.int64)[:, None] + np.array(self.offsets)

    def base_b24(self, origins: np.ndarray) -> np.ndarray:
        """Daily seasonal base y(tau-24) in kW for each origin and offset, (n, H)."""
        return self.y_raw[self._target_rows(origins) - 24]

    def inverse(self, r_scaled: np.ndarray, origins: np.ndarray) -> np.ndarray:
        """kW forecast y(tau-24) + r_scaled * std_r + mean_r, shape (n, H)."""
        r = np.asarray(r_scaled, np.float64).reshape(len(origins), self.horizon)
        return self.base_b24(origins) + r * self.r_std + self.r_mean

    def true_targets(self, split: str) -> np.ndarray:
        """Original-unit targets (n_origins, H)."""
        return self.y_raw[self._target_rows(self.origins[split])]

    def batch(self, origins: torch.Tensor) -> Batch:
        o = origins.to(self.device)
        rows = o[:, None] - self.lookback + 1 + self._ar
        tgt = o[:, None] + self._off
        return Batch(
            x=self.X[rows],
            y=self.R[tgt],
            target_feats=self.TF[tgt],
            cal_future=self.C[tgt],
            r0=self.R[o],
            origin=o,
        )

    def iterate(self, split: str, batch_size: int, generator: torch.Generator | None = None):
        """Yield batches. Shuffled with `generator` when given, else in time order."""
        o = torch.as_tensor(self.origins[split])
        if generator is not None:
            o = o[torch.randperm(len(o), generator=generator)]
        for i in range(0, len(o), batch_size):
            yield self.batch(o[i : i + batch_size])

    def window_rule(self) -> dict:
        """Candidate-origin rule and count over the whole series, before the split assignment.

        Origins are t in [L - 1 + 168, N - max_offset), so there are N - (L - 1 + 168) -
        max_offset of them. Origins whose target hours straddle a split boundary are dropped,
        so the three split counts sum to less than this.
        """
        last = max(self.offsets)
        n_all = self.n - self.first_origin - last
        return {
            "first_origin_row": self.first_origin,
            "first_origin_rule": f"L - 1 + {max(INPUT_LAGS)} = {self.lookback} - 1 + "
            f"{max(INPUT_LAGS)} = {self.first_origin}",
            "formula_all_windows": f"N - (L - 1 + {max(INPUT_LAGS)}) - max_offset = {self.n} - "
            f"{self.first_origin} - {last} = {n_all}",
            "n_all_windows": n_all,
            "n_split_windows": int(sum(len(o) for o in self.origins.values())),
        }

    def split_summary(self) -> dict:
        """Per split rows, hours, window counts and DST rows, plus the overall window rule."""
        dst = self.cal[:, CALENDAR.index("dst_flag")] > 0
        out: dict = {}
        for name, (lo, hi) in self.bounds.items():
            o = self.origins[name]
            out[name] = {
                "rows": [int(lo), int(hi)],
                "first_hour": str(self.hours[lo]),
                "last_hour": str(self.hours[hi - 1]),
                "n_windows": len(o),
                "first_target_hour": str(self.hours[o[0] + min(self.offsets)]),
                "last_target_hour": str(self.hours[o[-1] + max(self.offsets)]),
                "dst_flag_rows": int(dst[lo:hi].sum()),
            }
        out["all_windows"] = self.window_rule()
        return out

    def feature_summary(self) -> dict:
        """Input, target-side and residual definitions plus the scalers, for spec.json."""
        dst = self.cal[:, CALENDAR.index("dst_flag")] > 0
        return {
            "target": TARGET,
            "target_only": self.target_only,
            "input_channels": self.feature_names,
            "input_channel_rule": "per input step s: standardized y(s), y(s-24), y(s-48), "
            "y(s-168) and the calendar of s",
            "target_features": list(TARGET_FEATURES),
            "target_feature_rule": "per target hour tau = t+h: calendar of tau and standardized "
            "y(tau-24), y(tau-48), y(tau-168)",
            "calendar": list(CALENDAR),
            "residual": RESIDUAL_DEFINITION,
            "prediction": "y_hat(tau) = y(tau-24) + r_hat * residual_std + residual_mean",
            "target_scaler": {"mean": self.y_mean, "std": self.y_std},
            "residual_scaler": {"mean": self.r_mean, "std": self.r_std},
            "scaler_fit_rows": "train rows only (residual: train rows with tau >= 24)",
            "lag_lookup": "row index on a verified regular hourly grid, equal to timestamp lag",
            "first_origin_row": self.first_origin,
            "rows_without_full_lag_history": self.first_origin,
            "windows": self.window_rule(),
            "dst_flag_rule": DST_RULE,
            "dst_flag_rows": int(dst.sum()),
            "dst_flag_hours": [str(h) for h in self.hours[dst]],
            "dst_seasonal_base_note": "on the day after a change day, b24 at 01:00 and 02:00 "
            "reads a flagged row, so that base carries the DST anomaly",
        }


def ridge_design(
    y_raw: np.ndarray, cal: np.ndarray, origins: np.ndarray, offset: int
) -> np.ndarray:
    """Corrected sparse ridge features for the target hour tau = t+offset.

    Recent lags k in {1, 2, 3} are y[t+1-k] (the last three observations). Seasonal lags
    k in {24, 48, 168} are aligned to the target hour, y[tau-k], i.e. the same hour one day,
    two days and one week earlier. These are observed at time t because offset < 24.
    Plus the 7 calendar features of tau. Shape (n, 13).
    """
    if offset >= min(SEASONAL_LAGS):
        raise ValueError("seasonal lags would need unobserved values")
    recent = [y_raw[origins + 1 - k] for k in RECENT_LAGS]
    seasonal = [y_raw[origins + offset - k] for k in SEASONAL_LAGS]
    return np.hstack([np.column_stack(recent + seasonal), cal[origins + offset]])


def ridge_window_design(
    y_raw: np.ndarray, cal: np.ndarray, origins: np.ndarray, offset: int, lookback: int = 168
) -> np.ndarray:
    """Full-window target-only ridge features for tau = t+offset.

    The `lookback` raw target values y[t-lookback+1 .. t] (oldest first), then the 7 calendar
    features of tau. Shape (n, lookback + 7).
    """
    rows = np.asarray(origins)[:, None] - lookback + 1 + np.arange(lookback)
    return np.hstack([y_raw[rows], cal[origins + offset]])
