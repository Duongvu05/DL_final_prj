from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        return cls(mean=x.mean(axis=0), std=x.std(axis=0) + 1e-8)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def chronological_split(
    n: int, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[slice, slice, slice]:
    """Split raw row indices in time order. Windows are built inside each slice."""
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return slice(0, i_train), slice(i_train, i_val), slice(i_val, n)


def make_windows(
    features: np.ndarray, target: np.ndarray, lookback: int, offset: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build (x, y) windows with no look-ahead.

    x[i] is features[i : i + lookback], so the last observed step is t = i + lookback - 1.
    y[i] is target[t + offset : t + offset + horizon].
    Task 1 uses offset=2, horizon=1 to predict y[t+2].
    Task 2 uses offset=1, horizon=5 to predict y[t+1 .. t+5].
    """
    xs, ys = [], []
    last = len(features) - lookback - offset - horizon + 2
    for i in range(max(0, last)):
        t = i + lookback - 1
        xs.append(features[i : i + lookback])
        ys.append(target[t + offset : t + offset + horizon])
    return np.stack(xs), np.stack(ys)
