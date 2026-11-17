from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def write_synthetic_npz(path: Path, n_hours: int = 2400, n_clients: int = 6, seed: int = 0) -> Path:
    """Hourly matrix with daily and weekly cycles. MT_235 is a stable client, MT_999 is late-start."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    names = [f"MT_{i:03d}" for i in range(n_clients - 2)] + ["MT_235", "MT_999"]
    base = 50 + 10 * np.sin(2 * np.pi * t / 24)[:, None] + 5 * np.sin(2 * np.pi * t / 168)[:, None]
    values = base + rng.normal(0, 1, (n_hours, n_clients))
    values[: n_hours // 3, -1] = 0.0  # structural zeros
    hours = pd.date_range("2012-01-01 01:00", periods=n_hours, freq="h")
    np.savez_compressed(
        path,
        values=values,
        hours=hours.to_numpy("datetime64[s]"),
        names=np.array(names),
        stable=np.array(names[:-1]),
    )
    return path


@pytest.fixture
def synthetic_npz(tmp_path: Path) -> Path:
    return write_synthetic_npz(tmp_path / "electricity_hourly.npz")
