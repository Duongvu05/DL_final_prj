"""Task 3 recognizers on the cropped 54x54 SVHN protocol (plan Section 2.3).

All four models share a VGG-BN encoder trained from scratch: four stages of two
(conv3x3, BatchNorm, ReLU) with 64, 128, 256 and 512 channels, 2x2 max-pooling (ceil mode)
after stages 1-3 and Dropout2d(0.1) after each stage. A 54x54 crop gives a 512x7x7 map.

- AttentionGRURecognizer (course model): 49 visual tokens with learned row/column position
  embeddings, a 2-layer GRU decoder with Bahdanau attention, and an auxiliary length head.
- PooledGRURecognizer: the same model without attention. The mean token replaces the context.
- MultiHeadCNN: Goodfellow-style length head plus five digit heads on a flattened feature map.
- CTCRecognizer: CRNN with (2,1) last pooling, 14 time steps, BiLSTM and CTC.

Every model has `loss(batch)` (fp32 training loss) and `predict(images, mode)` returning digit
strings and an aux dict. The CNN runs under the caller's autocast (bf16 on CUDA), while the
recurrent parts and the losses run in fp32.

Decoding. The reported (primary) decoding is never greedy:
- GRU decoders (Phase 1 and the Phase 2 token-mixer models): batched beam search "beam{W}",
  score = summed log-probability / number of tokens (digits + EOS), EOS forced at the
  max_len step.
- CRNN-CTC: CTC prefix beam search "prefix{W}" (blank 0), at most max_len digits.
- Multi-head CNN: exact MAP decoding "map" over (length, digits).
The width W of the first two is chosen on the validation split from BEAM_WIDTHS by exact
match, ties to the smaller width (select_width). "greedy" (and "length" for the GRU
decoders) remain as labelled diagnostics.

Secondary GRU decodings (opt-in, scripts/rescore_task3.py, never the primary result). They
rerank the n-best list of the primary beam (GRUDecoderRecognizer.nbest, the validation-selected
width): "mbr_edit" (edit-distance minimum Bayes risk), "length_rescore" (length head) and
"bidir_rescore" (DRAM-style forward-backward with a right-to-left model, score_strings). Their
lambdas are selected on validation only (select_lambda). A right-to-left model is the same
architecture built with reverse_target=True and trained on reversed targets
(SvhnSplit(reverse=True)). Its predictions are reversed back, so they read left-to-right.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from dl_final_prj.fast import levenshtein_batch
from dl_final_prj.svhn import (
    DIGIT0,
    EOS,
    MAX_LEN,
    PAD,
    SOS,
    VOCAB,
    OcrBatch,
    decode,
    encode,
    target_order,
)
from dl_final_prj.ts_models import AdditiveAttention

N_LENGTHS = MAX_LEN  # length classes 1..5
BEAM_WIDTHS = (3, 5, 10)  # candidate widths, one is selected on validation


def _fp32(x: torch.Tensor):
    """Context that switches autocast off, so the code inside runs in fp32."""
    return torch.autocast(device_type=x.device.type, enabled=False)


class VggEncoder(nn.Module):
    """VGG-BN CNN. `pools` gives the max-pool kernel (= stride) after stages 1-3."""

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 256, 512),
        pools: Sequence[tuple[int, int]] = ((2, 2), (2, 2), (2, 2)),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 3
        for i, c in enumerate(channels):
            for _ in range(2):
                layers += [
                    nn.Conv2d(c_in, c, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                ]
                c_in = c
            if i < len(pools):
                layers.append(nn.MaxPool2d(pools[i], pools[i], ceil_mode=True))
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenEncoder(nn.Module):
    """VGG encoder -> (B, h*w, C) tokens in raster order plus learned row/column embeddings."""

    def __init__(self, grid: tuple[int, int] = (7, 7), dropout: float = 0.1) -> None:
        super().__init__()
        self.cnn = VggEncoder(dropout=dropout)
        c = self.cnn.out_channels
        self.grid, self.dim = grid, c
        self.row = nn.Parameter(torch.randn(grid[0], c) * 0.02)
        self.col = nn.Parameter(torch.randn(grid[1], c) * 0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        f = self.cnn(images).float()
        _, c, h, w = f.shape
        if (h, w) != self.grid:
            raise ValueError(f"feature map {h}x{w} does not match the {self.grid} grid")
        f = f + self.row.T.reshape(1, c, h, 1) + self.col.T.reshape(1, c, 1, w)
        return f.flatten(2).transpose(1, 2)


class GRUDecoderRecognizer(nn.Module):
    """Shared CNN + stacked-GRUCell decoder. Subclasses choose attention or the mean token.

    Step k: context = attention(top hidden, tokens) or the mean token, input =
    [embedding(previous token), context], logits = Linear([new top hidden, context]).
    The initial hidden states are tanh(Linear(mean token)). The length head classifies
    the number of digits (1..5) from the mean token.
    """

    modes = ("greedy", "length", *(f"beam{w}" for w in BEAM_WIDTHS))
    attention = True
    # Right-to-left decoder (opt-in, build(reverse_target=True)): trained on the reversed digit
    # string. Every decoded string is reversed back, so predictions always read left-to-right.
    reverse_target = False

    def __init__(
        self,
        max_len: int = MAX_LEN,
        hidden: int = 256,
        layers: int = 2,
        attn_dim: int = 256,
        emb_dim: int = 64,
        length_loss_weight: float = 0.2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len, self.hidden, self.layers = max_len, hidden, layers
        self.length_loss_weight = length_loss_weight
        self.encoder = TokenEncoder(dropout=dropout)
        c = self.encoder.dim
        self.attn = AdditiveAttention(hidden, c, attn_dim) if self.attention else None
        self.emb = nn.Embedding(VOCAB, emb_dim)
        self.cells = nn.ModuleList(
            [nn.GRUCell(emb_dim + c if i == 0 else hidden, hidden) for i in range(layers)]
        )
        self.init_h = nn.Linear(c, hidden * layers)
        self.out = nn.Linear(hidden + c, VOCAB)
        self.length_head = nn.Linear(c, N_LENGTHS)

    # --- building blocks ---------------------------------------------------------

    def encode(self, images: torch.Tensor):
        """tokens (B, L, C), mean token (B, C), projected keys or None. All fp32."""
        feats = self.encoder(images)
        with _fp32(feats):
            mean = feats.mean(1)
            kp = self.attn.w_k(feats) if self.attn is not None else None
        return feats, mean, kp

    def init_state(self, mean: torch.Tensor) -> torch.Tensor:
        return self.init_h(mean).tanh().view(-1, self.layers, self.hidden)

    def step(self, prev, h, feats, kp, mean):
        """One decoder step. h (B, layers, hidden) -> logits (B, VOCAB), new h, weights or None."""
        if self.attn is not None:
            ctx, w = self.attn(h[:, -1], feats, kp)
        else:
            ctx, w = mean, None
        inp = torch.cat([self.emb(prev), ctx], -1)
        new = []
        for i, cell in enumerate(self.cells):
            inp = cell(inp, h[:, i])
            new.append(inp)
        return self.out(torch.cat([inp, ctx], -1)), torch.stack(new, 1), w

    # --- training ----------------------------------------------------------------

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        """Teacher-forced pass. Returns logits (B, T, VOCAB), length logits (B, 5) and the
        attention weights (B, T, L) or None, with T = max_len + 1."""
        feats, mean, kp = self.encode(images)
        with _fp32(feats):
            h = self.init_state(mean)
            prev = torch.full_like(tokens[:, 0], SOS)
            logits, weights = [], []
            for k in range(tokens.shape[1]):
                step, h, w = self.step(prev, h, feats, kp, mean)
                logits.append(step)
                weights.append(w)
                prev = tokens[:, k]
            att = torch.stack(weights, 1) if self.attn is not None else None
            return torch.stack(logits, 1), self.length_head(mean), att

    def loss(self, b: OcrBatch) -> torch.Tensor:
        logits, length_logits, _ = self(b.images, b.tokens)
        tok = F.cross_entropy(
            logits.float().reshape(-1, VOCAB), b.tokens.reshape(-1), ignore_index=PAD
        )
        return tok + self.length_loss_weight * F.cross_entropy(length_logits.float(), b.lengths)

    # --- decoding ----------------------------------------------------------------

    @staticmethod
    def _mask_special(logits: torch.Tensor) -> torch.Tensor:
        """SOS and PAD are never valid outputs."""
        logits = logits.float().clone()
        logits[:, [PAD, SOS]] = float("-inf")
        return logits

    def greedy(
        self, images: torch.Tensor, constrain: bool = False, lengths: torch.Tensor | None = None
    ):
        """Autoregressive argmax decoding. With constrain=True the output has exactly `lengths`
        digits (default: the length head's prediction). EOS is masked before that step and
        forced at it. EOS is always forced at the last step (safety stop), so no output has
        more than max_len digits. Returns token ids (B, T) and attention (B, T, L) or None."""
        feats, mean, kp = self.encode(images)
        with _fp32(feats):
            if constrain and lengths is None:
                lengths = self.length_head(mean).argmax(-1) + 1
            h = self.init_state(mean)
            prev = torch.full((feats.shape[0],), SOS, dtype=torch.long, device=feats.device)
            is_eos = torch.arange(VOCAB, device=feats.device) == EOS
            ids, weights = [], []
            for k in range(self.max_len + 1):
                logits, h, w = self.step(prev, h, feats, kp, mean)
                logits = self._mask_special(logits)
                if lengths is not None:
                    before = lengths.gt(k)[:, None] & is_eos
                    at = lengths.eq(k)[:, None] & ~is_eos
                    logits = logits.masked_fill(before | at, float("-inf"))
                if k == self.max_len:
                    logits = logits.masked_fill(~is_eos, float("-inf"))
                prev = logits.argmax(-1)
                ids.append(prev)
                weights.append(w)
            att = torch.stack(weights, 1) if self.attn is not None else None
            return torch.stack(ids, 1), att

    def beam(self, images: torch.Tensor, width: int = 3):
        """Batched beam search of any width without length constraint. Beams are pruned by the
        summed log-probability. The final hypothesis maximizes the summed log-probability
        divided by its number of tokens (digits plus EOS). EOS is forced at the last step, so
        no hypothesis has more than max_len digits. width=1 is greedy decoding.
        Returns token ids (B, T) and the attention (B, T, L) of the chosen beam, or None."""
        hist, scores, ntok, att, _ = self._beam_search(images, width)
        best = (scores / ntok.clamp(min=1)).argmax(-1)
        rows = torch.arange(hist.shape[0], device=hist.device)
        return hist[rows, best], (att[rows, best] if att is not None else None)

    def _beam_search(self, images: torch.Tensor, width: int):
        """The search behind beam() and nbest(). Returns the final beams: token ids
        (B, W, T), summed log-probabilities (B, W) (-inf for an unused beam), token counts
        (B, W) (digits + EOS), attention (B, W, T, L) or None, and the mean token (B, C)."""
        if width < 1:
            raise ValueError(f"beam width must be >= 1, got {width}")
        feats, mean, kp = self.encode(images)
        with _fp32(feats):
            bsz, k_, dev = feats.shape[0], width, feats.device
            rep = lambda t: t.repeat_interleave(k_, 0)  # noqa: E731
            feats, mean_k = rep(feats), rep(mean)
            kp = rep(kp) if kp is not None else None
            h = rep(self.init_state(mean))
            scores = torch.full((bsz, k_), float("-inf"), device=dev)
            scores[:, 0] = 0.0  # all beams start identical, keep only one
            finished = torch.zeros(bsz, k_, dtype=torch.bool, device=dev)
            ntok = torch.zeros(bsz, k_, device=dev)
            hist = torch.zeros(bsz, k_, 0, dtype=torch.long, device=dev)
            att: torch.Tensor | None = None
            prev = torch.full((bsz * k_,), SOS, dtype=torch.long, device=dev)
            pad_only = torch.full((VOCAB,), float("-inf"), device=dev)
            pad_only[PAD] = 0.0  # a finished beam continues with PAD at no cost
            base = torch.arange(bsz, device=dev)[:, None] * k_
            eos_only = torch.full((VOCAB,), float("-inf"), device=dev)
            eos_only[EOS] = 0.0
            for k in range(self.max_len + 1):
                logits, h, w = self.step(prev, h, feats, kp, mean_k)
                logp = self._mask_special(logits).log_softmax(-1).view(bsz, k_, VOCAB)
                if k == self.max_len:  # safety stop, the EOS probability still counts
                    logp = logp + eos_only
                logp = torch.where(finished[..., None], pad_only, logp)
                top, ix = (scores[..., None] + logp).view(bsz, -1).topk(k_, -1)
                src, tok = ix // VOCAB, ix % VOCAB
                was_done = finished.gather(1, src)
                ntok = ntok.gather(1, src) + (~was_done).float()
                finished = was_done | (tok == EOS)
                hist = torch.cat(
                    [hist.gather(1, src[..., None].expand(-1, -1, hist.shape[2])), tok[..., None]],
                    -1,
                )
                if w is not None:  # the weights of step k belong to the source beam
                    w_src = (
                        w.float()
                        .view(bsz, k_, 1, -1)
                        .gather(1, src[..., None, None].expand(-1, -1, 1, w.shape[-1]))
                    )
                    if att is not None:
                        idx = src[..., None, None].expand(-1, -1, *att.shape[2:])
                        w_src = torch.cat([att.gather(1, idx), w_src], 2)
                    att = w_src
                h = h[(src + base).view(-1)]
                scores, prev = top, tok.view(-1)
            return hist, scores, ntok, att, mean

    def nbest(self, images: torch.Tensor, width: int):
        """The `width` finished hypotheses of beam(images, width) per image, best first by the
        primary score (summed log-probability / tokens, stable order, so the first one is
        exactly beam()'s choice). Unused beams (-inf) are dropped. Predictions read
        left-to-right (a reverse_target model's strings are reversed back).
        Returns (hyps, length log-probabilities (B, N_LENGTHS) of the length head), where
        hyps[b] is a list of (pred, logp, logp_norm, n_tokens)."""
        hist, scores, ntok, _, mean = self._beam_search(images, width)
        with _fp32(mean):
            len_lp = self.length_head(mean.float()).float().log_softmax(-1)
        norm = scores / ntok.clamp(min=1)
        order = torch.sort(norm, dim=-1, descending=True, stable=True).indices
        ids, raw, nrm, cnt = (t.cpu().tolist() for t in (hist, scores, norm, ntok))
        out = []
        for b, row in enumerate(order.cpu().tolist()):
            out.append(
                [
                    (self._reported(decode(ids[b][j])), raw[b][j], nrm[b][j], int(cnt[b][j]))
                    for j in row
                    if math.isfinite(raw[b][j])
                ]
            )
        return out, len_lp

    def score_strings(self, images: torch.Tensor, strings: Sequence[Sequence[str]]):
        """Teacher-forced summed log-probability of given left-to-right digit strings,
        strings[b] for image b, with the softmax of decoding (PAD and SOS masked) over the
        digits and EOS. A reverse_target model scores the reversed string, so for a
        right-to-left model this is log P_R2L(reverse(s)). Returns a list per image."""
        feats, mean, kp = self.encode(images)
        flat = [s for ss in strings for s in ss]
        if not flat:
            return [[] for _ in strings]
        with _fp32(feats):
            dev = feats.device
            owner = torch.tensor([i for i, ss in enumerate(strings) for _ in ss], device=dev)
            tokens = torch.tensor(
                [encode(target_order(s, self.reverse_target), self.max_len) for s in flat],
                dtype=torch.long,
                device=dev,
            )
            f, m = feats[owner], mean[owner]
            k_rep = kp[owner] if kp is not None else None
            h = self.init_state(m)
            prev = torch.full((len(flat),), SOS, dtype=torch.long, device=dev)
            total = torch.zeros(len(flat), device=dev)
            for k in range(tokens.shape[1]):
                logits, h, _ = self.step(prev, h, f, k_rep, m)
                lp = self._mask_special(logits).log_softmax(-1)
                tok = tokens[:, k]
                total = total + torch.where(tok != PAD, lp.gather(1, tok[:, None])[:, 0], 0.0)
                prev = tok
        vals = total.cpu().tolist()
        out, i = [], 0
        for ss in strings:
            out.append(vals[i : i + len(ss)])
            i += len(ss)
        return out

    def _reported(self, s: str) -> str:
        """Left-to-right reading of a decoded string."""
        return target_order(s, self.reverse_target)

    def predict(self, images: torch.Tensor, mode: str = "greedy"):
        """Digit strings for mode beamW (primary, any width W >= 1), greedy (diagnostic) or
        length (greedy constrained to the length head, diagnostic). aux["attention"]
        (B, T, L) holds the attention of the returned sequence (attention models only)."""
        aux: dict[str, torch.Tensor] = {}
        att = None
        if mode == "greedy":
            ids, att = self.greedy(images)
        elif mode == "length":
            ids, _ = self.greedy(images, constrain=True)
        elif mode.startswith("beam") and mode[4:].isdigit():
            ids, att = self.beam(images, int(mode[4:]))
        else:
            raise ValueError(f"unknown decoding mode {mode}")
        if att is not None:
            aux["attention"] = att
        return [self._reported(decode(s)) for s in ids.cpu().tolist()], aux


class AttentionGRURecognizer(GRUDecoderRecognizer):
    """Course model: CNN + 2-D position embeddings + Bahdanau attention + 2-layer GRU."""

    attention = True


class PooledGRURecognizer(GRUDecoderRecognizer):
    """No-attention counterpart. The mean visual token is fed at every step."""

    attention = False


class MultiHeadCNN(nn.Module):
    """Goodfellow et al. (2014) style. Flattened CNN map -> 2 FC layers -> length head (5)
    and five digit heads (10). Loss = CE(length) + sum of CE over the true digit positions.
    Decoding: "map" (primary, multihead_map) or "greedy" (argmax length, then the argmax
    digit of the first L heads, diagnostic)."""

    modes = ("greedy", "map")

    def __init__(
        self,
        max_len: int = MAX_LEN,
        fc_dim: int = 1024,
        dropout: float = 0.3,
        cnn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.cnn = VggEncoder(dropout=cnn_dropout)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cnn.out_channels * 7 * 7, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
        )
        self.length_head = nn.Linear(fc_dim, max_len)
        self.digit_heads = nn.ModuleList([nn.Linear(fc_dim, 10) for _ in range(max_len)])

    def forward(self, images: torch.Tensor):
        """Length logits (B, max_len) and digit logits (B, max_len, 10), fp32."""
        z = self.fc(self.cnn(images))
        digits = torch.stack([head(z) for head in self.digit_heads], 1)
        return self.length_head(z).float(), digits.float()

    def loss(self, b: OcrBatch) -> torch.Tensor:
        length_logits, digit_logits = self(b.images)
        tok = b.tokens[:, : self.max_len]
        valid = tok >= DIGIT0
        target = (tok - DIGIT0).clamp(min=0)
        ce = F.cross_entropy(digit_logits.reshape(-1, 10), target.reshape(-1), reduction="none")
        per_image = F.cross_entropy(length_logits, b.lengths, reduction="none")
        per_image = per_image + (ce.view_as(target) * valid).sum(1)
        return per_image.mean()

    def predict(self, images: torch.Tensor, mode: str = "greedy"):
        length_logits, digit_logits = self(images)
        if mode == "greedy":
            lengths, digits = length_logits.argmax(-1) + 1, digit_logits.argmax(-1)
        elif mode == "map":
            lengths, digits = multihead_map(length_logits, digit_logits)
        else:
            raise ValueError(f"unknown decoding mode {mode}")
        return [
            "".join(str(d) for d in ds[:n])
            for ds, n in zip(digits.cpu().tolist(), lengths.cpu().tolist())
        ], {}


def multihead_map(
    length_logits: torch.Tensor, digit_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact MAP over (L, digits) of the multi-head model, which factorizes as
    p(L) * prod_{i<L} p(d_i | head i). For each L the best digits are the per-head argmax, so
    L* = argmax_L [log p(L) + sum_{i<L} max_d log p(d | head i)] (ties to the shorter L).
    length_logits (B, max_len), digit_logits (B, max_len, 10) -> lengths (B,) in 1..max_len
    and digits (B, max_len) (only the first L are read)."""
    len_lp = length_logits.float().log_softmax(-1)
    best_lp, digits = digit_logits.float().log_softmax(-1).max(-1)
    score = len_lp + best_lp.cumsum(-1)  # score[:, L - 1] for length L
    return score.argmax(-1) + 1, digits


class CTCRecognizer(nn.Module):
    """CRNN. Pooling (2,2), (2,2), (2,1) gives 512x7x14, averaged over height -> 14 steps ->
    2-layer BiLSTM -> 11 classes. Class 0 is the CTC blank, class 1 + d is digit d.
    Decoding: "prefixW" (primary, CTC prefix beam search of width W, at most max_len digits)
    or "greedy" (best path, diagnostic, may exceed max_len)."""

    modes = ("greedy", *(f"prefix{w}" for w in BEAM_WIDTHS))

    def __init__(
        self, hidden: int = 256, layers: int = 2, cnn_dropout: float = 0.1, max_len: int = MAX_LEN
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.cnn = VggEncoder(pools=((2, 2), (2, 2), (2, 1)), dropout=cnn_dropout)
        self.rnn = nn.LSTM(
            self.cnn.out_channels, hidden, layers, batch_first=True, bidirectional=True
        )
        self.out = nn.Linear(2 * hidden, 11)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Per-step logits (B, T, 11), fp32."""
        f = self.cnn(images).float().mean(2).transpose(1, 2)  # (B, T, C)
        with _fp32(f):
            h, _ = self.rnn(f)
            return self.out(h)

    def loss(self, b: OcrBatch) -> torch.Tensor:
        return ctc_loss(self(b.images), b.tokens)

    def predict(self, images: torch.Tensor, mode: str = "greedy"):
        if mode == "greedy":
            return ctc_greedy(self(images)), {}
        if mode.startswith("prefix") and mode[6:].isdigit():
            return ctc_prefix_beam(self(images), int(mode[6:]), self.max_len), {}
        raise ValueError(f"unknown decoding mode {mode}")


def tokens_to_ctc_targets(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder tokens (d+3 ... EOS PAD) -> concatenated CTC targets (d+1) and lengths."""
    digit = tokens >= DIGIT0
    return tokens[digit] - DIGIT0 + 1, digit.sum(1)


def ctc_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Mean CTC loss in fp32 (per-sequence loss divided by target length, then averaged)."""
    with _fp32(logits):
        logp = logits.float().log_softmax(-1).transpose(0, 1)  # (T, B, C)
        targets, lengths = tokens_to_ctc_targets(tokens)
        t = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device)
        return F.ctc_loss(logp, targets, t, lengths, blank=0, zero_infinity=True)


def ctc_greedy(logits: torch.Tensor) -> list[str]:
    """Best path: argmax per step, merge repeats, drop blanks."""
    out = []
    for seq in logits.argmax(-1).cpu().tolist():
        s, prev = [], 0
        for c in seq:
            if c != prev and c != 0:
                s.append(str(c - 1))
            prev = c
        out.append("".join(s))
    return out


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_prefix_beam_one(
    logp: Sequence[Sequence[float]], width: int, max_len: int | None = None
) -> list[tuple[tuple[int, ...], float]]:
    """CTC prefix beam search on one sequence of per-step log-probabilities (T, C), blank 0.

    Every prefix keeps two log-probabilities, of the paths that end in a blank (pb) and of
    those that end in its last label (pnb). Per step a prefix is extended by blank (stays),
    by its last label c (stays, only from pnb: a repeat merges) and by any label (grows).
    The same label after a blank grows the prefix ("1", blank, "1" reads "11"). The `width`
    most probable prefixes survive each step (ties by prefix, for determinism).
    With `max_len`, a prefix of max_len labels never grows, so no longer prefix takes a beam
    slot. This is exact: the probability of a prefix never depends on longer prefixes.
    Returns the final prefixes as (labels tuple, log-probability), best first."""
    if width < 1:
        raise ValueError(f"beam width must be >= 1, got {width}")
    ninf = -math.inf
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, ninf)}
    for row in logp:
        nxt: dict[tuple[int, ...], list[float]] = {}
        for prefix, (pb, pnb) in beams.items():
            total = _logaddexp(pb, pnb)
            stay = nxt.setdefault(prefix, [ninf, ninf])
            stay[0] = _logaddexp(stay[0], total + row[0])
            last = prefix[-1] if prefix else None
            full = max_len is not None and len(prefix) >= max_len
            for c in range(1, len(row)):
                p = row[c]
                if p == ninf:
                    continue
                if c == last:
                    stay[1] = _logaddexp(stay[1], pnb + p)  # repeat without blank merges
                    src = pb  # a repeat grows the prefix only after a blank
                else:
                    src = total
                if src != ninf and not full:
                    grown = nxt.setdefault((*prefix, c), [ninf, ninf])
                    grown[1] = _logaddexp(grown[1], src + p)
        ranked = sorted(nxt.items(), key=lambda kv: (-_logaddexp(*kv[1]), kv[0]))
        beams = {k: (v[0], v[1]) for k, v in ranked[:width]}
    return sorted(((k, _logaddexp(*v)) for k, v in beams.items()), key=lambda kv: (-kv[1], kv[0]))


def ctc_prefix_beam(logits: torch.Tensor, width: int, max_len: int = MAX_LEN) -> list[str]:
    """Prefix beam search for a batch of CTC logits (B, T, 11), class 1 + d is digit d.
    Prefixes never grow past max_len digits during the search, so the result is always a
    scored prefix: the most probable surviving prefix with at most max_len digits."""
    logp = logits.float().log_softmax(-1).cpu().tolist()
    out = []
    for seq in logp:
        best = ctc_prefix_beam_one(seq, width, max_len)[0][0]
        out.append("".join(str(c - 1) for c in best))
    return out


# --- decoding policy ----------------------------------------------------------------------

PRIMARY_DECODING = {"gru": "beam", "ctc": "ctc_prefix_beam", "multihead": "map"}
DECODING_TEXT = {
    "beam": "beam search of width W (selected on validation), unconstrained, score = summed "
    "log-probability / number of tokens (digits + EOS), EOS forced at the max_len step",
    "ctc_prefix_beam": "CTC prefix beam search of width W (selected on validation), blank 0, "
    "repeats merge unless separated by a blank, best prefix with at most max_len digits",
    "map": "exact MAP over (L, digits): L = argmax log p(L) + sum_{i<L} max_d log p(d | head i), "
    "then the argmax digits of the first L heads",
    "greedy_gru": "greedy (argmax per step), unconstrained, EOS forced at the max_len step "
    "(diagnostic only)",
    "greedy_ctc": "CTC best path (argmax per step, merge repeats, drop blanks), diagnostic only",
    "greedy_multihead": "argmax length head, then the argmax digits of the first L heads "
    "(diagnostic only)",
    "length": "greedy constrained to the length-head prediction (diagnostic only)",
}


def decoder_family(model: nn.Module) -> str:
    """gru (every GRUDecoderRecognizer, Phase 1 and 2), ctc or multihead."""
    if isinstance(model, GRUDecoderRecognizer):
        return "gru"
    if isinstance(model, CTCRecognizer):
        return "ctc"
    if isinstance(model, MultiHeadCNN):
        return "multihead"
    raise TypeError(f"no decoding policy for {type(model).__name__}")


def search_widths(model: nn.Module) -> tuple[int, ...]:
    """Candidate widths of the primary decoding, () when it has no width (multi-head MAP)."""
    return () if decoder_family(model) == "multihead" else BEAM_WIDTHS


def primary_mode(model: nn.Module, width: int | None) -> str:
    """predict() mode string of the primary (non-greedy) decoding at `width`."""
    fam = decoder_family(model)
    if fam == "multihead":
        return "map"
    if width is None:
        raise ValueError(f"{fam} decoding needs a width")
    return f"beam{width}" if fam == "gru" else f"prefix{width}"


def select_width(val_exact_by_width: Mapping[int, float]) -> int:
    """The width with the best validation exact match, ties to the smaller width."""
    if not val_exact_by_width:
        raise ValueError("no width to select from")
    return min(val_exact_by_width, key=lambda w: (-val_exact_by_width[w], w))


def build(
    kind: str,
    args: dict | None = None,
    max_len: int = MAX_LEN,
    reverse_target: bool = False,
    **extra,
) -> nn.Module:
    """Model from its config kind: attention, pooled, multihead or ctc (Phase 1), or
    mamba_attention or gru_mixer_attention (Phase 2, see ocr_mamba). reverse_target=True
    (GRU decoders only) makes a right-to-left decoder. It adds no parameter."""
    model = _build(kind, args, max_len, **extra)
    if reverse_target:
        if not isinstance(model, GRUDecoderRecognizer):
            raise ValueError(f"reverse_target applies to the GRU decoders only, not {kind}")
        model.reverse_target = True
    return model


def _build(kind: str, args: dict | None, max_len: int, **extra) -> nn.Module:
    args = dict(args or {})
    if kind in ("mamba_attention", "gru_mixer_attention"):
        from dl_final_prj.ocr_mamba import build_mixer_recognizer  # lazy: ocr_mamba imports us

        return build_mixer_recognizer(kind, max_len=max_len, **args, **extra)
    if kind in ("attention", "pooled"):
        cls = AttentionGRURecognizer if kind == "attention" else PooledGRURecognizer
        return cls(max_len=max_len, **args, **extra)
    if kind == "multihead":
        return MultiHeadCNN(max_len=max_len, **args)
    if kind == "ctc":
        return CTCRecognizer(**args, max_len=max_len)
    raise ValueError(f"unknown model kind {kind}")


# --- secondary GRU decodings: n-best rescoring (plan 2.3) ---------------------------------
#
# All of them rerank the frozen n-best list of the primary beam (validation-selected width)
# and never change the primary result. Lambdas come from LAMBDA_GRID and are selected on the
# validation split only (select_lambda, ties to the smaller lambda), then applied unchanged to
# test.

LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)  # plan 2.3 pre-registered grid
SECONDARY_METHODS = ("mbr_edit", "length_rescore", "bidir_rescore")
SECONDARY_PLAN_NAMES = {
    "mbr_edit": "mbr_edit_secondary",
    "length_rescore": "length_rescore_secondary",
    "bidir_rescore": "dram_fb_secondary",
}
SECONDARY_TEXT = {
    "mbr_edit": "minimum Bayes risk over the primary-beam n-best list (Kumar and Byrne 2004): "
    "posterior = softmax of the summed sequence log-probability over the list, loss = "
    "Levenshtein distance, the hypothesis with the least expected edit distance wins (ties to "
    "the better primary rank). No tuned parameter",
    "length_rescore": "n-best rescoring with the auxiliary length head: score = primary score "
    "(summed log-probability / tokens) + lambda * log P_length(number of digits), lambda "
    "selected on validation exact match. lambda 0 is the primary beam",
    "bidir_rescore": "DRAM-style forward-backward rescoring (Ba et al. 2015): score = (log "
    "P_L2R(hyp) + lambda * log P_R2L(reverse(hyp))) / tokens, with log P_R2L the "
    "teacher-forced log-probability (digits + EOS) under the same-seed right-to-left model. "
    "lambda selected on validation exact match. lambda 0 is the primary beam",
}


@dataclass
class NBest:
    """n-best lists of one split, padded to `width` columns (a padded entry has pred "" and
    logp -inf). Column 0 is the primary beam's choice. split names the data split and is
    checked by select_lambda, which accepts validation lists only."""

    split: str
    width: int
    preds: list[list[str]]  # (N, K) left-to-right digit strings, best primary score first
    logp: np.ndarray  # (N, K) summed log-probability of digits + EOS, -inf pad
    logp_norm: np.ndarray  # (N, K) primary score logp / n_tokens, -inf pad
    n_tokens: np.ndarray  # (N, K) digits + EOS, 0 pad
    len_logp: np.ndarray  # (N, N_LENGTHS) length-head log-probabilities, lengths 1..5

    @classmethod
    def from_hyps(
        cls, split: str, width: int, hyps: Sequence[Sequence[tuple]], len_logp: np.ndarray
    ) -> "NBest":
        n = len(hyps)
        if any(not h for h in hyps):
            raise ValueError("every image needs at least one hypothesis")
        k = max(width, max(len(h) for h in hyps))
        preds = [[""] * k for _ in range(n)]
        logp = np.full((n, k), -np.inf)
        norm = np.full((n, k), -np.inf)
        ntok = np.zeros((n, k), np.int64)
        for i, row in enumerate(hyps):
            for j, (p, lp, ln, nt) in enumerate(row):
                preds[i][j], logp[i, j], norm[i, j], ntok[i, j] = p, lp, ln, nt
        return cls(split, width, preds, logp, norm, ntok, np.asarray(len_logp, np.float64))

    def __len__(self) -> int:
        return len(self.preds)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.logp)

    def hyps(self) -> list[list[str]]:
        """The real (unpadded) hypotheses per image."""
        return [
            [p for p, ok in zip(row, v, strict=True) if ok]
            for row, v in zip(self.preds, self.valid, strict=True)
        ]

    def top1(self) -> list[str]:
        return [row[0] for row in self.preds]

    def pick(self, scores: np.ndarray) -> list[str]:
        """The argmax of `scores` (N, K) per image over the real hypotheses. Ties and rows
        with no finite score go to the earlier (better primary) rank."""
        s = np.where(self.valid, np.asarray(scores, np.float64), -np.inf)
        s = np.where(np.isnan(s), -np.inf, s)
        best = s.argmax(1)
        return [self.preds[i][j] for i, j in enumerate(best.tolist())]

    def r2l_matrix(self, r2l: Sequence[Sequence[float]]) -> np.ndarray:
        """Per-image lists of R2L scores (same order as hyps()) -> padded (N, K)."""
        out = np.full(self.logp.shape, -np.inf)
        for i, (row, v) in enumerate(zip(r2l, self.valid, strict=True)):
            if len(row) != int(v.sum()):
                raise ValueError(f"image {i}: {len(row)} R2L scores for {int(v.sum())} hyps")
            out[i, : len(row)] = row
        return out


def length_logp_of(nb: NBest) -> np.ndarray:
    """log P_length(number of digits) of every hypothesis (N, K). -inf for 0 digits or more
    than the head's classes (never chosen once lambda > 0) and for padding."""
    n_len = nb.len_logp.shape[1]
    lengths = np.array([[len(p) for p in row] for row in nb.preds], np.int64)
    ok = (lengths >= 1) & (lengths <= n_len) & nb.valid
    idx = np.clip(lengths - 1, 0, n_len - 1)
    lp = np.take_along_axis(nb.len_logp, idx, 1)
    return np.where(ok, lp, -np.inf)


def length_rescore(nb: NBest, lam: float) -> list[str]:
    """argmax primary score + lam * log P_length. lam = 0 is exactly the primary beam."""
    if lam == 0:
        return nb.pick(nb.logp_norm)
    return nb.pick(nb.logp_norm + lam * length_logp_of(nb))


def bidir_rescore(nb: NBest, r2l_logp: np.ndarray, lam: float) -> list[str]:
    """argmax (log P_L2R + lam * log P_R2L(reverse)) / n_tokens, written as the primary score
    plus lam * log P_R2L / n_tokens so that lam = 0 is exactly the primary beam.
    r2l_logp is (N, K) aligned with nb (NBest.r2l_matrix)."""
    if lam == 0:
        return nb.pick(nb.logp_norm)
    return nb.pick(nb.logp_norm + lam * r2l_logp / np.maximum(nb.n_tokens, 1))


def mbr_edit(nb: NBest) -> list[str]:
    """Edit-distance MBR over each n-best list. Posterior p_j = softmax(logp_j) over the
    list, risk_i = sum_j p_j * lev(h_i, h_j), the least risk wins (ties to the earlier rank).
    With one hypothesis carrying all the mass this is the MAP hypothesis."""
    a, b = [], []
    k = nb.logp.shape[1]
    for row in nb.preds:
        for x in row:
            a += [x] * k
            b += row
    dist = levenshtein_batch(a, b).astype(np.float64).reshape(len(nb), k, k)
    valid = nb.valid
    lp = np.where(valid, nb.logp, -np.inf)
    lp = lp - lp.max(1, keepdims=True)
    post = np.exp(lp)
    post /= post.sum(1, keepdims=True)
    risk = np.einsum("nij,nj->ni", dist, post)
    risk = np.where(valid, risk, np.inf)
    best = risk.argmin(1)
    return [nb.preds[i][j] for i, j in enumerate(best.tolist())]


def select_lambda(val_exact_by_lambda: Mapping[float, float], split: str = "val") -> float:
    """The lambda with the best validation exact match, ties to the smaller lambda.
    `split` names where the exact matches come from. Anything but "val" is refused."""
    if split != "val":
        raise ValueError(f"lambda selection uses the validation split only, got {split!r}")
    if not val_exact_by_lambda:
        raise ValueError("no lambda to select from")
    return min(val_exact_by_lambda, key=lambda x: (-val_exact_by_lambda[x], x))


def exact_by_lambda(
    nb: NBest,
    labels: Sequence[str],
    rescore: Callable[[float], list[str]],
    grid: Sequence[float] = LAMBDA_GRID,
) -> dict[float, float]:
    """Exact match of rescore(lambda) on nb for every lambda of the grid. nb must be the
    validation n-best list (this feeds select_lambda)."""
    if nb.split != "val":
        raise ValueError(f"lambda search runs on the validation split only, got {nb.split!r}")
    if len(labels) != len(nb):
        raise ValueError("labels and n-best lists differ in length")
    out = {}
    for lam in grid:
        preds = rescore(float(lam))
        out[float(lam)] = float(np.mean([p == t for p, t in zip(preds, labels, strict=True)]))
    return out
