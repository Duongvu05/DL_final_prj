import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Bahdanau attention. Scores each key against a query and returns the weighted sum."""

    def __init__(self, query_dim: int, key_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.w_query = nn.Linear(query_dim, attn_dim, bias=False)
        self.w_key = nn.Linear(key_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """query (B, Q), keys (B, L, K) -> context (B, K), weights (B, L) that sum to 1."""
        scores = self.v(torch.tanh(self.w_query(query).unsqueeze(1) + self.w_key(keys)))
        weights = torch.softmax(scores.squeeze(-1), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class RNNForecaster(nn.Module):
    """Task 1. LSTM or GRU encoder, optional temporal attention, linear head with one output."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell: str = "lstm",
        attention: bool = False,
    ) -> None:
        super().__init__()
        if cell not in ("lstm", "gru"):
            raise ValueError(f"cell must be 'lstm' or 'gru', got {cell}")
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            n_features,
            hidden,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = AdditiveAttention(hidden, hidden) if attention else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if attention else 1), 1)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """x (B, L, F) -> prediction (B, 1). Attention weights (B, L) on request."""
        out, _ = self.rnn(x)
        feat, weights = out[:, -1], None
        if self.attn is not None:
            context, weights = self.attn(feat, out)
            feat = torch.cat([feat, context], dim=-1)
        pred = self.head(self.drop(feat))
        return (pred, weights) if return_attention else pred
