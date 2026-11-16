"""Configurable CNN, sequence model and prediction recognizer on the clovaai modules.

The layout follows the four stage framework of the benchmark (feature extraction, sequence
modeling, prediction). TPS is skipped. The interface and vocabulary match our own models
in data/image_sequences.py. Tokens are PAD=0, BOS=1, EOS=2 and the digits. CTC reuses PAD
as the blank, like models/ctc_recognizer.py.
"""

from typing import Literal

import torch
import torch.nn.functional as fn
from torch import nn

from dl_final_prj.data.image_sequences import BOS, PAD, VOCAB_SIZE, decode
from dl_final_prj.models.ctc_recognizer import ctc_collapse
from dl_final_prj.models.sota_recognizers.clova_modules import (
    Attention,
    BidirectionalLSTM,
    ResNetFeatureExtractor,
    VGGFeatureExtractor,
)

Feature = Literal["vgg", "resnet"]
Sequence = Literal["none", "bilstm"]
Prediction = Literal["ctc", "attn"]


class ClovaRecognizer(nn.Module):
    """Image (B, 1, 32, W) in [0, 1] to a digit string.

    feature     vgg (CRNN) or resnet (FAN) feature extractor.
    sequence    none or two stacked BiLSTM layers.
    prediction  ctc or attn (LSTM decoder with additive attention).
    max_len     length of the padded token rows, BOS + digits + EOS. The attention decoder
                runs max_len - 1 steps, so it can emit up to max_len - 2 digits and EOS.
    """

    def __init__(
        self,
        feature: Feature = "vgg",
        sequence: Sequence = "bilstm",
        prediction: Prediction = "ctc",
        output_channel: int = 256,
        hidden: int = 256,
        max_len: int = 8,
    ) -> None:
        super().__init__()
        if feature not in ("vgg", "resnet"):
            raise ValueError(f"unknown feature extractor {feature}")
        if sequence not in ("none", "bilstm"):
            raise ValueError(f"unknown sequence model {sequence}")
        if prediction not in ("ctc", "attn"):
            raise ValueError(f"unknown prediction {prediction}")
        self.prediction, self.steps = prediction, max_len - 1
        extractor = VGGFeatureExtractor if feature == "vgg" else ResNetFeatureExtractor
        self.feature = extractor(1, output_channel)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))  # collapse the height axis
        if sequence == "bilstm":
            self.sequence = nn.Sequential(
                BidirectionalLSTM(output_channel, hidden, hidden),
                BidirectionalLSTM(hidden, hidden, hidden),
            )
            out_dim = hidden
        else:
            self.sequence = nn.Identity()
            out_dim = output_channel
        if prediction == "ctc":
            self.head = nn.Linear(out_dim, VOCAB_SIZE)
        else:
            self.head = Attention(out_dim, hidden, VOCAB_SIZE)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Contextual features (B, T, C). The benchmark normalises pixels to [-1, 1]."""
        f = self.feature(images * 2.0 - 1.0)  # (B, C, H', W')
        f = self.pool(f.permute(0, 3, 1, 2)).squeeze(3)  # (B, W', C)
        return self.sequence(f)

    def forward(self, images: torch.Tensor, tokens_in: torch.Tensor | None = None) -> torch.Tensor:
        """CTC log probabilities (T, B, V), or attention logits (B, steps, V).

        The attention decoder is teacher forced when tokens_in (B, steps, starts with BOS)
        is given and free running otherwise.
        """
        feats = self.encode(images)
        if self.prediction == "ctc":
            return self.head(feats).log_softmax(dim=-1).transpose(0, 1)
        logits, _ = self.head(feats, tokens_in, self.steps, BOS)
        return logits

    def loss(
        self, images: torch.Tensor, tokens: torch.Tensor, n_digits: torch.Tensor
    ) -> torch.Tensor:
        """tokens is BOS d1 ... dn EOS PAD .... CTC targets are tokens[:, 1:], n_digits long."""
        if self.prediction == "ctc":
            log_probs = self(images)
            in_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
            return fn.ctc_loss(
                log_probs, tokens[:, 1:], in_len, n_digits.cpu(), blank=PAD, zero_infinity=True
            )
        logits = self(images, tokens[:, : self.steps])
        return fn.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens[:, 1 : self.steps + 1].reshape(-1),
            ignore_index=PAD,
        )

    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Argmax ids and, for attn, the attention weights (B, steps, T).

        For CTC the ids are the raw per-frame argmax (B, T) and the weights are None.
        """
        feats = self.encode(images)
        if self.prediction == "ctc":
            return self.head(feats).argmax(dim=-1), None
        logits, alphas = self.head(feats, None, self.steps, BOS)
        return logits.argmax(dim=-1), alphas

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> list[str]:
        """Greedy decode to digit strings. PAD, BOS and everything after EOS are removed."""
        ids, _ = self.greedy_decode(images)
        rows = ids.tolist()
        if self.prediction == "ctc":
            rows = [ctc_collapse(row) for row in rows]
        return [decode(row) for row in rows]
