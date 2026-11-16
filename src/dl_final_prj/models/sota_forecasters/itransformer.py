# Adapted from https://github.com/thuml/Time-Series-Library (models/iTransformer.py)
# Commit: 4e938a1767106324dd753b2a44832bf870a0252e
# License: MIT (third_party/Time-Series-Library/LICENSE)
# Modifications: forecasting branch only, config dataclass instead of argparse namespace,
#   no time-mark inputs, shared layers moved to layers.py, type hints added.
"""iTransformer (Liu et al., ICLR 2024, https://arxiv.org/abs/2310.06625)."""

import torch
from torch import nn

from dl_final_prj.models.sota_forecasters.layers import (
    ForecastConfig,
    InvertedEmbedding,
    make_encoder,
    window_normalize,
)


class ITransformer(nn.Module):
    """Attention across variates. Each variate series is one token."""

    def __init__(self, cfg: ForecastConfig) -> None:
        super().__init__()
        self.pred_len = cfg.pred_len
        self.enc_embedding = InvertedEmbedding(cfg.seq_len, cfg.d_model, cfg.dropout)
        self.encoder = make_encoder(cfg, nn.LayerNorm(cfg.d_model))
        self.projection = nn.Linear(cfg.d_model, cfg.pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, L, n_vars) -> (B, pred_len, n_vars)."""
        n_vars = x.shape[2]
        x, means, stdev = window_normalize(x)
        enc_out = self.encoder(self.enc_embedding(x))
        out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        return out * stdev[:, 0:1, :] + means[:, 0:1, :]
