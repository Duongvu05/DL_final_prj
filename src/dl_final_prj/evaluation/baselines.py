import numpy as np


def persistence(x: np.ndarray, target_idx: int, horizon: int) -> np.ndarray:
    """Repeat the last observed target value. x (n, lookback, F) -> (n, horizon)."""
    return np.repeat(x[:, -1, target_idx][:, None], horizon, axis=1)


def seasonal_naive(
    x: np.ndarray, target_idx: int, offset: int, horizon: int, period: int
) -> np.ndarray:
    """Predict y[t+k] with the target one period earlier, y[t+k-period]."""
    lookback = x.shape[1]
    steps = np.arange(offset, offset + horizon)
    if period < steps[-1] or period > lookback - 1 + steps[0]:
        raise ValueError(f"period {period} is not usable with lookback {lookback} and these steps")
    return x[:, lookback - 1 + steps - period, target_idx]


class LinearRegressionBaseline:
    """Closed form ridge on the flattened window. A tiny penalty keeps the solve stable."""

    def __init__(self, ridge: float = 1e-6) -> None:
        self.ridge = ridge
        self.weights: np.ndarray | None = None

    @staticmethod
    def _design(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(len(x), -1)
        return np.hstack([flat, np.ones((len(flat), 1))])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegressionBaseline":
        a = self._design(x)
        gram = a.T @ a + self.ridge * np.eye(a.shape[1])
        self.weights = np.linalg.solve(gram, a.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("fit must be called first")
        return self._design(x) @ self.weights


def baseline_predictions(
    train: tuple[np.ndarray, np.ndarray],
    x_eval: np.ndarray,
    target_idx: int,
    offset: int,
    horizon: int,
    period: int | None = None,
) -> dict[str, np.ndarray]:
    """All baselines on the same windows. Inputs and outputs are in scaled space.

    Persistence and seasonal naive are affine equivariant, so scaling does not change them.
    """
    preds = {
        "persistence": persistence(x_eval, target_idx, horizon),
        "linear_regression": LinearRegressionBaseline().fit(*train).predict(x_eval),
    }
    if period:
        preds["seasonal_naive"] = seasonal_naive(x_eval, target_idx, offset, horizon, period)
    return preds
