from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from dl_final_prj.data.splits import StandardScaler, chronological_split, make_windows


@dataclass
class TimeSeriesConfig:
    """Everything needed to build forecasting windows from one CSV file."""

    path: str
    pred_column: str
    lookback: int = 48
    offset: int = 1
    horizon: int = 1
    datetime_col: str | None = None
    datetime_format: str | None = None
    feature_cols: list[str] | None = None
    resample: str | None = None
    train_frac: float = 0.7
    val_frac: float = 0.15
    max_rows: int | None = None


class WindowDataset(Dataset):
    """Pairs of (x, y). x holds only rows up to the current step t, y holds the future targets."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]


@dataclass
class TimeSeriesData:
    """Scaled windows per split. The scaler was fitted on the train rows only."""

    feature_names: list[str]
    target_idx: int
    scaler: StandardScaler
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def loader(self, name: str, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
        return DataLoader(WindowDataset(*self.splits[name]), batch_size=batch_size, shuffle=shuffle)

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        """Map scaled target values (any shape) back to original units."""
        return y * self.scaler.std[self.target_idx] + self.scaler.mean[self.target_idx]


def load_frame(cfg: TimeSeriesConfig) -> pd.DataFrame:
    """Read the CSV, index it by time, optionally resample, and keep the model columns."""
    df = pd.read_csv(cfg.path)
    if cfg.datetime_col:
        stamp = pd.to_datetime(df.pop(cfg.datetime_col), format=cfg.datetime_format)
        df.index = pd.DatetimeIndex(stamp)
        df = df[~df.index.duplicated()].sort_index()
    if cfg.resample:
        if not cfg.datetime_col:
            raise ValueError("resample needs datetime_col")
        df = df.select_dtypes("number").resample(cfg.resample).mean()
    cols = list(cfg.feature_cols) if cfg.feature_cols else list(df.select_dtypes("number").columns)
    if cfg.pred_column not in cols:
        cols.append(cfg.pred_column)
    df = (
        pd.DataFrame(df[cols]).ffill().dropna()
    )  # forward fill only, so no future value leaks backwards
    return df.iloc[: cfg.max_rows] if cfg.max_rows else df


def build_time_series(cfg: TimeSeriesConfig, df: pd.DataFrame | None = None) -> TimeSeriesData:
    """Split in time order, fit the scaler on train rows, then window each split on its own."""
    df = load_frame(cfg) if df is None else df
    names = list(df.columns)
    values = df.to_numpy(dtype=np.float64)
    target_idx = names.index(cfg.pred_column)
    slices = chronological_split(len(values), cfg.train_frac, cfg.val_frac)
    scaler = StandardScaler.fit(values[slices[0]])
    scaled = scaler.transform(values)
    data = TimeSeriesData(feature_names=names, target_idx=target_idx, scaler=scaler)
    for name, sl in zip(("train", "val", "test"), slices):
        rows = scaled[sl]
        if len(rows) < cfg.lookback + cfg.offset + cfg.horizon:
            raise ValueError(f"{name} split has only {len(rows)} rows, too few for one window")
        data.splits[name] = make_windows(
            rows, rows[:, target_idx], cfg.lookback, cfg.offset, cfg.horizon
        )
    return data


def make_synthetic_series(
    n: int = 1000, n_features: int = 3, seed: int = 0, freq: str = "h"
) -> pd.DataFrame:
    """Seeded multivariate series with trend, daily cycle and a lagged dependence on f0."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cols = {
        f"f{k}": rng.normal(size=n).cumsum() * 0.1 + np.sin(2 * np.pi * t / (24 + k))
        for k in range(n_features)
    }
    lagged = np.roll(cols["f0"], 3)
    target = 0.005 * t + 2 * np.sin(2 * np.pi * t / 24) + 0.5 * lagged + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=n, freq=freq), **cols})
    df["target"] = target
    return df
