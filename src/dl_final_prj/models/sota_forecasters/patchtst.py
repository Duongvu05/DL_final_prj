# Adapted from https://github.com/thuml/Time-Series-Library (models/PatchTST.py)
# Commit: 4e938a1767106324dd753b2a44832bf870a0252e
# License: MIT (third_party/Time-Series-Library/LICENSE)
# Modifications: forecasting branch only, config dataclass instead of argparse namespace,
#   no dummy time-mark inputs, shared layers moved to layers.py, type hints added.
"""PatchTST (Nie et al., ICLR 2023, https://arxiv.org/abs/2211.14730)."""

import torch
from torch import nn

from dl_final_prj.models.sota_forecasters.layers import (
    FlattenHead,
    ForecastConfig,
    PatchEmbedding,
    Transpose,
    make_encoder,
    window_normalize,
)


class PatchTST(nn.Module):
    """Channel independent patch transformer. Every variate is forecast on its own."""

    def __init__(self, cfg: ForecastConfig) -> None:
        super().__init__()
        if cfg.seq_len < cfg.patch_len:
            raise ValueError(f"lookback {cfg.seq_len} is shorter than patch_len {cfg.patch_len}")
        self.pred_len = cfg.pred_len
        self.patch_embedding = PatchEmbedding(
            cfg.d_model, cfg.patch_len, cfg.stride, cfg.stride, cfg.dropout
        )
        norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(cfg.d_model), Transpose(1, 2))
        self.encoder = make_encoder(cfg, norm)
        n_patches = int((cfg.seq_len - cfg.patch_len) / cfg.stride + 2)
        self.head = FlattenHead(cfg.d_model * n_patches, cfg.pred_len, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, L, n_vars) -> (B, pred_len, n_vars)."""
        x, means, stdev = window_normalize(x)
        enc_out, n_vars = self.patch_embedding(x.permute(0, 2, 1))
        enc_out = self.encoder(enc_out)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        out = self.head(enc_out.permute(0, 1, 3, 2)).permute(0, 2, 1)
        return out * stdev[:, 0:1, :] + means[:, 0:1, :]
