"""Task 3 recognizers (plan Section 2.3).

All three share one backbone: torchvision ResNet-18 (ImageNet weights) without the pooling
and classification head. The first block of layer3 and of layer4 uses stride (2, 1) instead
of 2, so a 64x256 input gives a 2x32 feature map (64 visual tokens) instead of 2x8. This is
the usual text-recognition change (as in CRNN/ASTER backbones). Pretrained weights still load
because strides have no parameters.
"""

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import BasicBlock

from dl_final_prj.svhn import DIGIT0, EOS, PAD, SOS, VOCAB, OcrBatch
from dl_final_prj.ts_models import AdditiveAttention


class ResNetBackbone(nn.Module):
    out_channels = 512

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        for layer in (net.layer3, net.layer4):
            block = cast(BasicBlock, layer[0])
            block.conv1.stride = (2, 1)
            cast(nn.Conv2d, cast(nn.Sequential, block.downsample)[0]).stride = (2, 1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.body = nn.Sequential(net.layer2, net.layer3, net.layer4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.stem(x))

    def set_stem_frozen(self, frozen: bool) -> None:
        for p in self.stem.parameters():
            p.requires_grad = not frozen
        if frozen:
            self.stem.eval()


class AttentionRecognizer(nn.Module):
    """CNN + 2-D positional embeddings + Bahdanau visual attention + stacked LSTM decoder.

    At step k the attention query is the top decoder state, the context is the weighted sum of
    the 64 visual tokens, and the LSTM input is [embedding(previous token), context].
    The initial states come from a linear map of the mean visual token (Show, Attend and Tell).
    """

    def __init__(
        self,
        max_len: int,
        hidden: int = 256,
        layers: int = 2,
        attn_dim: int = 256,
        emb_dim: int = 64,
        dropout: float = 0.2,
        pretrained: bool = True,
        grid: tuple[int, int] = (2, 32),
    ) -> None:
        super().__init__()
        self.max_len, self.layers, self.hidden = max_len, layers, hidden
        self.cnn = ResNetBackbone(pretrained)
        c = self.cnn.out_channels
        self.row = nn.Parameter(torch.zeros(grid[0], c))
        self.col = nn.Parameter(torch.zeros(grid[1], c))
        nn.init.normal_(self.row, std=0.02)
        nn.init.normal_(self.col, std=0.02)
        self.attn = AdditiveAttention(hidden, c, attn_dim)
        self.emb = nn.Embedding(VOCAB, emb_dim)
        self.cells = nn.ModuleList(
            [nn.LSTMCell(emb_dim + c if i == 0 else hidden, hidden) for i in range(layers)]
        )
        self.init_h = nn.Linear(c, hidden * layers)
        self.init_c = nn.Linear(c, hidden * layers)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden + c, VOCAB)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        f = self.cnn(images)  # (B, C, h, w)
        _, c, h, w = f.shape
        f = f + self.row[:h].T.reshape(1, c, h, 1) + self.col[:w].T.reshape(1, c, 1, w)
        return f.flatten(2).transpose(1, 2)  # (B, h*w, C) in raster order

    def forward(self, b: OcrBatch, teacher_forcing: float = 1.0):
        """Returns logits (B, max_len+1, VOCAB) and attention weights (B, max_len+1, tokens)."""
        feats = self.encode(b.images)
        kp = self.attn.w_k(feats)
        bsz = feats.shape[0]
        mean = feats.mean(1)
        hs = list(self.init_h(mean).tanh().chunk(self.layers, -1))
        cs = list(self.init_c(mean).tanh().chunk(self.layers, -1))
        prev = torch.full((bsz,), SOS, dtype=torch.long, device=feats.device)
        logits, weights = [], []
        for k in range(self.max_len + 1):
            ctx, w = self.attn(hs[-1], feats, kp)
            inp = torch.cat([self.emb(prev), ctx], -1)
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(inp, (hs[i], cs[i]))
                inp = self.drop(hs[i])
            step = self.out(torch.cat([self.drop(hs[-1]), ctx], -1))
            logits.append(step)
            weights.append(w)
            prev = step.argmax(-1)
            if self.training and teacher_forcing > 0:
                use = torch.rand(bsz, device=prev.device) < teacher_forcing
                prev = torch.where(use, b.tokens[:, k], prev)
        return torch.stack(logits, 1), {"attention": torch.stack(weights, 1)}


class PooledRecognizer(nn.Module):
    """No-attention counterpart. Global-average-pooled CNN vector initializes a 1-layer LSTM and
    is fed at every step together with the previous token embedding."""

    def __init__(
        self, max_len: int, hidden: int = 256, layers: int = 1, emb_dim: int = 64,
        dropout: float = 0.2, pretrained: bool = True,
    ) -> None:  # fmt: skip
        super().__init__()
        self.max_len, self.layers = max_len, layers
        self.cnn = ResNetBackbone(pretrained)
        c = self.cnn.out_channels
        self.emb = nn.Embedding(VOCAB, emb_dim)
        self.cells = nn.ModuleList(
            [nn.LSTMCell(emb_dim + c if i == 0 else hidden, hidden) for i in range(layers)]
        )
        self.init_h = nn.Linear(c, hidden * layers)
        self.init_c = nn.Linear(c, hidden * layers)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, VOCAB)

    def forward(self, b: OcrBatch, teacher_forcing: float = 1.0):
        v = self.cnn(b.images).mean(dim=(2, 3))
        bsz = v.shape[0]
        hs = list(self.init_h(v).tanh().chunk(self.layers, -1))
        cs = list(self.init_c(v).tanh().chunk(self.layers, -1))
        prev = torch.full((bsz,), SOS, dtype=torch.long, device=v.device)
        logits = []
        for k in range(self.max_len + 1):
            inp = torch.cat([self.emb(prev), v], -1)
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(inp, (hs[i], cs[i]))
                inp = self.drop(hs[i])
            step = self.out(inp)
            logits.append(step)
            prev = step.argmax(-1)
            if self.training and teacher_forcing > 0:
                use = torch.rand(bsz, device=prev.device) < teacher_forcing
                prev = torch.where(use, b.tokens[:, k], prev)
        return torch.stack(logits, 1), {}


class CTCRecognizer(nn.Module):
    """CRNN. Feature map averaged over height -> 32 time steps -> BiLSTM -> per-step classes.
    Class 0 is the CTC blank, class 1 + d is digit d."""

    def __init__(
        self, hidden: int = 256, layers: int = 2, dropout: float = 0.2, pretrained: bool = True
    ) -> None:
        super().__init__()
        self.cnn = ResNetBackbone(pretrained)
        self.rnn = nn.LSTM(
            self.cnn.out_channels,
            hidden,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 11)

    def forward(self, b: OcrBatch, teacher_forcing: float = 0.0):
        f = self.cnn(b.images).mean(2).transpose(1, 2)  # (B, T, C)
        h, _ = self.rnn(f)
        return self.out(self.drop(h)), {}


def tokens_to_ctc_targets(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder tokens (d+3 ... EOS PAD) -> concatenated CTC targets (d+1) and lengths."""
    digit = tokens >= DIGIT0
    lengths = digit.sum(1)
    return (tokens[digit] - DIGIT0 + 1), lengths


def ctc_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    logp = logits.float().log_softmax(-1).transpose(0, 1)  # (T, B, C)
    targets, lengths = tokens_to_ctc_targets(tokens)
    t = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device)
    return F.ctc_loss(logp, targets, t, lengths, blank=0, zero_infinity=True)


def ctc_greedy(logits: torch.Tensor) -> list[str]:
    best = logits.argmax(-1).cpu().tolist()
    out = []
    for seq in best:
        s, prev = [], 0
        for c in seq:
            if c != prev and c != 0:
                s.append(str(c - 1))
            prev = c
        out.append("".join(s))
    return out


def seq_loss(logits: torch.Tensor, tokens: torch.Tensor, smoothing: float) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, VOCAB),
        tokens.reshape(-1),
        ignore_index=PAD,
        label_smoothing=smoothing,
    )


def greedy_strings(logits: torch.Tensor) -> list[str]:
    """Logits from a decoder run with teacher_forcing=0 -> digit strings (stop at EOS)."""
    out = []
    for seq in logits.argmax(-1).cpu().tolist():
        s = []
        for t in seq:
            if t == EOS or t < DIGIT0:
                break
            s.append(str(t - DIGIT0))
        out.append("".join(s))
    return out
