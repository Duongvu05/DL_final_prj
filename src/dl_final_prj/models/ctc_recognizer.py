"""CNN + BiLSTM with a CTC head, the Task 3 baseline. The CTC blank reuses the PAD index 0."""

import torch
import torch.nn.functional as fn
from torch import nn

from dl_final_prj.data.image_sequences import DIGIT_OFFSET, PAD, VOCAB_SIZE
from dl_final_prj.models.cnn_attention_decoder import ConvBiLSTMEncoder

BLANK = PAD


def ctc_collapse(ids: list[int]) -> list[int]:
    """Greedy CTC rule. Merge repeated ids first, then drop blanks."""
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out


class CTCRecognizer(nn.Module):
    def __init__(
        self,
        height: int = 32,
        hidden: int = 256,
        channels: tuple[int, int, int] = (64, 128, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ConvBiLSTMEncoder(height, channels, hidden, dropout)
        self.head = nn.Linear(self.encoder.out_dim, VOCAB_SIZE)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Log probabilities (T, B, V), the layout fn.ctc_loss expects."""
        return self.head(self.encoder(images)).log_softmax(dim=-1).transpose(0, 1)

    def loss(
        self, images: torch.Tensor, tokens: torch.Tensor, n_digits: torch.Tensor
    ) -> torch.Tensor:
        """tokens is BOS d1 ... dn EOS PAD ..., so tokens[:, 1:] holds the targets first."""
        log_probs = self(images)
        in_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
        return fn.ctc_loss(
            log_probs, tokens[:, 1:], in_len, n_digits.cpu(), blank=BLANK, zero_infinity=True
        )

    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor) -> list[str]:
        best = self(images).argmax(dim=-1).transpose(0, 1).tolist()  # (B, T)
        return [
            "".join(str(i - DIGIT_OFFSET) for i in ctc_collapse(row) if i >= DIGIT_OFFSET)
            for row in best
        ]
