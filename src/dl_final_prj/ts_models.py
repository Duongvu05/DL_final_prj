"""Seasonal-residual recurrent forecasters for Task 1 (BiGRU-2) and Task 2 (LSTM seq2seq).

Every model maps an electricity Batch to (pred (B, H) standardized residuals, aux dict). The
kW forecast is ElectricityData.inverse(pred, origins), i.e. y(tau-24) plus the unscaled
residual (plan Sections 2.1 and 2.2).
"""

import torch
from torch import nn

from dl_final_prj.electricity import Batch


class AdditiveAttention(nn.Module):
    """Bahdanau attention. score_l = v^T tanh(W_q q + W_k k_l), weights = softmax(score)."""

    def __init__(self, query_dim: int, key_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.w_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(key_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor, keys_proj: torch.Tensor | None = None
    ):
        """query (B, Q), keys (B, L, K) -> context (B, K), weights (B, L).

        keys_proj = w_k(keys) can be passed in to avoid recomputing it at each decoder step.
        """
        kp = self.w_k(keys) if keys_proj is None else keys_proj
        score = self.v(torch.tanh(self.w_q(query).unsqueeze(1) + kp)).squeeze(-1)
        weights = torch.softmax(score, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class BiRNNForecaster(nn.Module):
    """Task 1. Seasonal-residual (attentive) BiGRU-2 over the 168-hour target-only window.

    The final state is the concat of the last-layer forward and backward final states. With
    attention, a Bahdanau MLP scores all encoder outputs with the final state as query. The
    head sees [final, context (if attention), target features of every horizon flattened] and
    is LayerNorm -> Linear(head_dim) -> GELU -> Dropout -> Linear(H) standardized residuals.
    """

    def __init__(
        self,
        n_features: int,
        horizon: int = 1,
        n_target_feats: int = 10,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        attention: bool = True,
        attn_dim: int = 128,
        head_dim: int = 128,
        cell: str = "gru",
    ) -> None:
        super().__init__()
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(
            n_features, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.cell = cell
        state_dim = 2 * hidden
        self.attn = AdditiveAttention(state_dim, state_dim, attn_dim) if attention else None
        feat = state_dim * (2 if attention else 1) + horizon * n_target_feats
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, horizon),
        )

    def forward(self, b: Batch, teacher_forcing: float = 0.0):
        out, h = self.rnn(b.x)
        h_n = h[0] if self.cell == "lstm" else h
        final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        aux = {}
        parts = [final]
        if self.attn is not None:
            ctx, w = self.attn(final, out)
            parts.append(ctx)
            aux["attention"] = w
        parts.append(b.target_feats.flatten(1))
        return self.head(torch.cat(parts, dim=-1)), aux


class Seq2SeqForecaster(nn.Module):
    """Task 2. Two-layer LSTM encoder and autoregressive LSTMCell residual decoder.

    The decoder starts from the encoder's top-layer final (h, c). Its input at step k is
    [previous standardized residual, target features of hour t+k (10), learned horizon
    embedding k, attention context (if attention)]. The attention query is the decoder state
    before the update (Bahdanau). The first previous residual is r(t) at the origin. The output
    Linear([h, context]) gives one standardized residual per step.

    In training mode the previous residual is the true one from Batch.y with probability
    `teacher_forcing` (1.0 in Phase 1). In eval mode decoding is fully autoregressive whatever
    the ratio argument.
    """

    def __init__(
        self,
        n_features: int,
        horizon: int = 5,
        n_target_feats: int = 10,
        hidden: int = 64,
        enc_layers: int = 2,
        dropout: float = 0.2,
        attention: bool = True,
        attn_dim: int = 128,
        horizon_dim: int = 16,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(n_features, hidden, enc_layers, batch_first=True, dropout=dropout)
        self.attn = AdditiveAttention(hidden, hidden, attn_dim) if attention else None
        self.h_emb = nn.Embedding(horizon, horizon_dim)
        dec_in = 1 + n_target_feats + horizon_dim + (hidden if attention else 0)
        self.decoder = nn.LSTMCell(dec_in, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * (2 if attention else 1), 1)

    def forward(self, b: Batch, teacher_forcing: float = 0.0):
        enc, (h_n, c_n) = self.encoder(b.x)
        h, c = h_n[-1], c_n[-1]
        kp = self.attn.w_k(enc) if self.attn is not None else None
        prev = b.r0.unsqueeze(-1)
        preds, weights = [], []
        bsz = b.x.shape[0]
        tf = teacher_forcing if self.training else 0.0
        for k in range(self.horizon):
            emb = self.h_emb.weight[k].expand(bsz, -1)
            parts = [prev, b.target_feats[:, k], emb]
            ctx = None
            if self.attn is not None:
                ctx, w = self.attn(h, enc, kp)
                parts.append(ctx)
                weights.append(w)
            h, c = self.decoder(torch.cat(parts, dim=-1), (h, c))
            feat = self.drop(h if ctx is None else torch.cat([h, ctx], dim=-1))
            pred = self.out(feat)
            preds.append(pred)
            true = b.y[:, k : k + 1]
            if tf >= 1.0:
                prev = true
            elif tf > 0.0:
                use = torch.rand(bsz, 1, device=pred.device) < tf
                prev = torch.where(use, true, pred)
            else:
                prev = pred
        aux = {"attention": torch.stack(weights, dim=1)} if weights else {}
        return torch.cat(preds, dim=-1), aux


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
