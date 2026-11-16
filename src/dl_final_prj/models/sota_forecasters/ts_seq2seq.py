# Adapted from https://github.com/leehyeonbeen/TimeSeriesSeq2Seq
# Source files architecture/{seq2seq,attention,rnn,mlp,init}.py, commit 9b981548b3c5e82c908fd5a1e68e456154c78682
# License MIT, Copyright (c) 2023 Hyeonbeen Lee, text in third_party/TimeSeriesSeq2Seq/LICENSE
# Modifications: kept LSTMCell, StackedLSTMCell, BahdanauAttention, MLP head, initialize and the
# AttentionLSTMSeq2Seq decode loop. Dropped bidirectional, layernorm, Skeleton and the unused Ws layer.
# Merged forward_auto and forward_labeled into one loop with per-sample scheduled sampling. This fixes
# a leak in the repo teacher-forced path, which fed y[t] as the input that predicts y[t]. The first
# decoder input is the last observed target, not zeros. Added the interface of
# models/seq2seq_forecaster.py and return_attention.
"""Task 2 reference model. LSTM encoder-decoder with Bahdanau attention from TimeSeriesSeq2Seq."""

import torch
from torch import Tensor, nn


class LSTMCell(nn.Module):
    """Plain LSTM cell (repo architecture/rnn.py). Gate order is input, forget, cell, output."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h0, c0 = state
        gates = x @ self.weight_ih.t() + self.bias_ih + h0 @ self.weight_hh.t() + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        c1 = torch.sigmoid(forgetgate) * c0 + torch.sigmoid(ingate) * torch.tanh(cellgate)
        h1 = torch.sigmoid(outgate) * torch.tanh(c1)
        return h1, c1


class StackedLSTMCell(nn.Module):
    """One time step through `num_layers` cells. States are (num_layers, B, H)."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        sizes = [input_size] + [hidden_size] * (num_layers - 1)
        self.cells = nn.ModuleList(LSTMCell(s, hidden_size) for s in sizes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h0_all, c0_all = state
        h_all, c_all = [], []
        out = x
        for i, cell in enumerate(self.cells):
            out, c1 = cell(out, (h0_all[i], c0_all[i]))
            h_all.append(self.dropout(out))
            c_all.append(self.dropout(c1))
        return torch.stack(h_all), torch.stack(c_all)


class LSTMEncoder(nn.Module):
    """Runs the stacked cell over time from learnable initial states (repo LSTMEncoder)."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.cell = StackedLSTMCell(input_size, hidden_size, num_layers, dropout)
        self.h0 = nn.Parameter(torch.empty(num_layers, 1, hidden_size).normal_(0, 1e-2))
        self.c0 = nn.Parameter(torch.empty(num_layers, 1, hidden_size).normal_(0, 1e-2))

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """x (B, L, F) -> top layer states (B, L, H) and the final (h, c) of every layer."""
        h, c = self.h0.expand(-1, len(x), -1), self.c0.expand(-1, len(x), -1)
        top = []
        for t in range(x.shape[1]):
            h, c = self.cell(x[:, t], (h, c))
            top.append(h[-1])
        return torch.stack(top, dim=1), (h, c)


class BahdanauAttention(nn.Module):
    """Additive attention (repo architecture/attention.py). Softmax runs over encoder time steps."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.Va = nn.Linear(hidden_size, 1)
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """q (B, 1, H), k and v (B, L, H) -> context (B, 1, H) and weights (B, L, 1)."""
        scores = self.Va(torch.tanh(self.Wa(q) + self.Ua(k))).squeeze(-1)  # (B, L)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)  # (B, L, 1), sums to 1 over L
        return (weights * v).sum(dim=1, keepdim=True), weights


class TSSeq2SeqForecaster(nn.Module):
    """Task 2 reference. Same interface as `Seq2SeqForecaster`.

    Decoder input is [previous decoder state, previous target value]. The new decoder state queries
    the encoder outputs and the attention context alone goes through the MLP head (repo design).
    No future covariates are used.
    """

    def __init__(
        self,
        n_features: int,
        target_idx: int,
        horizon: int = 5,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        attention: bool = True,
    ) -> None:
        super().__init__()
        if not attention:
            raise ValueError("the reference decoder always uses attention")
        self.target_idx, self.horizon, self.hidden = target_idx, horizon, hidden
        self.encoder = LSTMEncoder(n_features, hidden, num_layers, dropout)
        self.decoder = StackedLSTMCell(hidden + 1, hidden, num_layers, dropout)
        self.attention = BahdanauAttention(hidden)
        self.head = nn.Sequential(  # repo MLP with no extra hidden layers
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        for name, p in self.named_parameters():  # repo architecture/init.py
            if "bias" in name:
                nn.init.zeros_(p)
            else:
                nn.init.normal_(p, mean=0.0, std=0.01)

    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        teacher_forcing: float = 0.0,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """x (B, L, F), optional y (B, horizon) -> predictions (B, horizon, 1).

        With probability `teacher_forcing` per sample and step, the next input is the true target.
        Otherwise it is the model's own prediction. Use the default 0 for evaluation.
        """
        enc_all, state = self.encoder(x)
        prev_y = x[:, -1, self.target_idx].unsqueeze(-1)  # last observed target
        prev_h = torch.zeros(len(x), self.hidden, device=x.device)
        preds, weights = [], []
        for k in range(self.horizon):
            h, c = self.decoder(torch.cat([prev_h, prev_y], dim=-1), state)
            state, prev_h = (h, c), h[-1]
            context, w = self.attention(prev_h.unsqueeze(1), enc_all, enc_all)
            pred = self.head(context.squeeze(1))
            preds.append(pred)
            weights.append(w.squeeze(-1))
            prev_y = pred
            if y is not None and teacher_forcing > 0:
                use_truth = torch.rand(len(x), 1, device=x.device) < teacher_forcing
                prev_y = torch.where(use_truth, y[:, k].unsqueeze(-1), pred)
        out = torch.stack(preds, dim=1)
        return (out, torch.stack(weights, dim=1)) if return_attention else out
