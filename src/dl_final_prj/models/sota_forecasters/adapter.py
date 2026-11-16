"""Thin adapter that gives the SOTA models the same interface as our RNN forecasters."""

from typing import Any

import torch
from torch import nn

from dl_final_prj.models.sota_forecasters.itransformer import ITransformer
from dl_final_prj.models.sota_forecasters.layers import ForecastConfig
from dl_final_prj.models.sota_forecasters.patchtst import PatchTST

MODELS: dict[str, type[nn.Module]] = {"patchtst": PatchTST, "itransformer": ITransformer}


class SOTAForecaster(nn.Module):
    """Wraps PatchTST or iTransformer. Input (B, lookback, n_features), output (B, horizon).

    The backbone forecasts every variate. The adapter keeps only the target column, so the
    loss and the metrics only ever see the target.

    Task 1 uses horizon=2 with windows built at offset=1. Column 0 is y[t+1] and the LAST
    column (index -1) is y[t+2], which is the Task 1 target. Score `pred[:, -1]` for Task 1.
    Task 2 uses horizon=5 and scores all five steps.
    """

    def __init__(
        self,
        name: str,
        n_features: int,
        target_idx: int,
        lookback: int,
        horizon: int,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if name not in MODELS:
            raise ValueError(f"unknown model {name}, choose from {sorted(MODELS)}")
        self.target_idx, self.horizon = target_idx, horizon
        cfg = ForecastConfig(seq_len=lookback, pred_len=horizon, enc_in=n_features, **kwargs)
        self.backbone = MODELS[name](cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, lookback, n_features) -> (B, horizon) in the scaled target space."""
        return self.backbone(x)[:, :, self.target_idx]
