# Adapted from https://github.com/thuml/Time-Series-Library
# Commit: 4e938a1767106324dd753b2a44832bf870a0252e
# License: MIT (third_party/Time-Series-Library/LICENSE)
# Source files: layers/Embed.py, layers/SelfAttention_Family.py, layers/Transformer_EncDec.py,
#   models/PatchTST.py (Transpose, FlattenHead).
# Modifications: merged into one module, kept only the layers PatchTST and iTransformer use,
#   dropped causal mask, de-stationary tau/delta, conv distilling and time-mark inputs, added
#   type hints and a ForecastConfig dataclass in place of the argparse namespace.
"""Small building blocks shared by PatchTST and iTransformer."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.functional import gelu, relu


@dataclass
class ForecastConfig:
    """Replaces the `configs` namespace of Time-Series-Library for the two models we use."""

    seq_len: int
    pred_len: int
    enc_in: int
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    activation: str = "gelu"
    patch_len: int = 16
    stride: int = 8


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positions."""

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    """Cut each variate into overlapping patches and embed every patch."""

    def __init__(
        self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float
    ) -> None:
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """x (B, n_vars, L) -> (B * n_vars, n_patches, d_model) and n_vars."""
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class InvertedEmbedding(nn.Module):
    """iTransformer embedding. The whole series of one variate becomes one token."""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, L, n_vars) -> (B, n_vars, d_model)."""
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))


class FullAttention(nn.Module):
    """Scaled dot-product attention without a mask."""

    def __init__(self, attention_dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """All inputs (B, L, H, E) -> (B, L, H, E)."""
        scale = 1.0 / math.sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attn, values).contiguous()


class AttentionLayer(nn.Module):
    """Multi-head wrapper with query, key, value and output projections."""

    def __init__(self, attention: FullAttention, d_model: int, n_heads: int) -> None:
        super().__init__()
        d_head = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_head * n_heads)
        self.key_projection = nn.Linear(d_model, d_head * n_heads)
        self.value_projection = nn.Linear(d_model, d_head * n_heads)
        self.out_projection = nn.Linear(d_head * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention over x (B, L, d_model)."""
        b, length, _ = x.shape
        h = self.n_heads
        q = self.query_projection(x).view(b, length, h, -1)
        k = self.key_projection(x).view(b, length, h, -1)
        v = self.value_projection(x).view(b, length, h, -1)
        out = self.inner_attention(q, k, v).view(b, length, -1)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    """Post-norm transformer encoder block with 1x1 convolutions as the feed forward net."""

    def __init__(
        self,
        attention: AttentionLayer,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = relu if activation == "relu" else gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(x))
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    """Stack of encoder layers with an optional final norm."""

    def __init__(self, layers: list[EncoderLayer], norm_layer: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x if self.norm is None else self.norm(x)


class Transpose(nn.Module):
    """Swap two dimensions. Used to run BatchNorm1d over the feature axis."""

    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class FlattenHead(nn.Module):
    """PatchTST head. Flatten (d_model, n_patches) and map linearly to the forecast window."""

    def __init__(self, nf: int, target_window: int, head_dropout: float = 0.0) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, n_vars, d_model, n_patches) -> (B, n_vars, target_window)."""
        return self.dropout(self.linear(self.flatten(x)))


def make_encoder(cfg: ForecastConfig, norm_layer: nn.Module) -> Encoder:
    """Build the encoder stack both models share."""
    layers = [
        EncoderLayer(
            AttentionLayer(FullAttention(cfg.dropout), cfg.d_model, cfg.n_heads),
            cfg.d_model,
            cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )
        for _ in range(cfg.e_layers)
    ]
    return Encoder(layers, norm_layer=norm_layer)


def window_normalize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-window, per-variate standardisation from the Non-stationary Transformer.

    Uses only the lookback window, so no future information is involved.
    """
    means = x.mean(1, keepdim=True).detach()
    x = x - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return x / stdev, means, stdev
