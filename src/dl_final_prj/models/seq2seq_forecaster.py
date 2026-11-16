import torch
from torch import nn

from dl_final_prj.models.rnn_forecaster import AdditiveAttention


def teacher_forcing_ratio(
    epoch: int, start: float = 1.0, decay: float = 0.8, floor: float = 0.0
) -> float:
    """Scheduled sampling. Starts at `start` on epoch 1 and decays geometrically."""
    return max(floor, start * decay ** (epoch - 1))


class Seq2SeqForecaster(nn.Module):
    """Task 2. LSTM encoder, LSTM decoder with Bahdanau attention, autoregressive decoding.

    The decoder input at each step is only the previous target value (and the attention
    context). No future covariates are used. The first input is the last observed target.
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
        self.target_idx, self.horizon = target_idx, horizon
        drop = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(n_features, hidden, num_layers, batch_first=True, dropout=drop)
        self.attn = AdditiveAttention(hidden, hidden) if attention else None
        dec_in = 1 + (hidden if attention else 0)
        self.decoder = nn.LSTM(dec_in, hidden, num_layers, batch_first=True, dropout=drop)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden * (2 if attention else 1), 1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """x (B, L, F), optional y (B, horizon) -> predictions (B, horizon, 1).

        With probability `teacher_forcing` per sample and step, the next input is the true
        target. Otherwise it is the model's own prediction. Use the default 0 for evaluation.
        """
        enc_out, state = self.encoder(x)
        prev = x[:, -1, self.target_idx].unsqueeze(-1)
        preds, all_weights = [], []
        for k in range(self.horizon):
            inp, feat_ctx = prev, None
            if self.attn is not None:
                feat_ctx, weights = self.attn(state[0][-1], enc_out)
                inp = torch.cat([prev, feat_ctx], dim=-1)
                all_weights.append(weights)
            out, state = self.decoder(inp.unsqueeze(1), state)
            feat = out.squeeze(1) if feat_ctx is None else torch.cat([out.squeeze(1), feat_ctx], -1)
            pred = self.proj(self.drop(feat))
            preds.append(pred)
            prev = pred
            if y is not None and teacher_forcing > 0:
                use_truth = torch.rand(len(x), 1, device=x.device) < teacher_forcing
                prev = torch.where(use_truth, y[:, k].unsqueeze(-1), pred)
        out_seq = torch.stack(preds, dim=1)
        if return_attention:
            return out_seq, (torch.stack(all_weights, dim=1) if all_weights else None)
        return out_seq
