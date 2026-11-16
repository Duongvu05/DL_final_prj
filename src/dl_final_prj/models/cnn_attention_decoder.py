"""CNN + BiLSTM encoder with an LSTM decoder and Bahdanau attention (Task 3)."""

import torch
import torch.nn.functional as fn
from torch import nn

from dl_final_prj.data.image_sequences import BOS, EOS, PAD, VOCAB_SIZE


class ConvBiLSTMEncoder(nn.Module):
    """Image (B, 1, H, W) to a feature sequence (B, T, 2 * hidden).

    Pooling halves the height three times but the width only once, so the width axis stays
    long enough to hold several frames per digit. Height is then folded into the channels.
    Images have a fixed width, so no padding mask is needed.
    """

    def __init__(
        self,
        height: int = 32,
        channels: tuple[int, int, int] = (64, 128, 256),
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pools = [(2, 2), (2, 1), (2, 1)]
        layers: list[nn.Module] = []
        c_in = 1
        for c_out, pool in zip(channels, pools):
            layers += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            ]
            c_in = c_out
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1] * (height // 8), hidden)
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)  # (B, C, H', W')
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        out, _ = self.rnn(self.drop(fn.relu(self.proj(f))))
        return self.drop(out)


class BahdanauAttention(nn.Module):
    """Additive attention. Returns the context vector and weights that sum to 1 over time."""

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 256) -> None:
        super().__init__()
        self.w_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.w_dec = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def project(self, enc: torch.Tensor) -> torch.Tensor:
        return self.w_enc(enc)

    def forward(
        self, h: torch.Tensor, enc: torch.Tensor, enc_proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.v(torch.tanh(enc_proj + self.w_dec(h)[:, None])).squeeze(-1)  # (B, T)
        alpha = scores.softmax(dim=-1)
        context = torch.bmm(alpha[:, None], enc).squeeze(1)
        return context, alpha


class AttentionRecognizer(nn.Module):
    """Seq2seq digit recognizer. With use_attention=False the context is the mean-pooled
    encoder state at every step (ablation)."""

    def __init__(
        self,
        height: int = 32,
        hidden: int = 256,
        channels: tuple[int, int, int] = (64, 128, 256),
        emb_dim: int = 64,
        attn_dim: int = 256,
        dropout: float = 0.1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.encoder = ConvBiLSTMEncoder(height, channels, hidden, dropout)
        enc_dim = self.encoder.out_dim
        self.embed = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=PAD)
        self.attn = BahdanauAttention(enc_dim, hidden, attn_dim)
        self.cell = nn.LSTMCell(emb_dim + enc_dim, hidden)
        self.init_h = nn.Linear(enc_dim, hidden)
        self.init_c = nn.Linear(enc_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden + enc_dim, VOCAB_SIZE)

    def _init_state(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = enc.mean(dim=1)
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def _step(
        self,
        prev: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        enc: torch.Tensor,
        enc_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        h, c = state
        if self.use_attention:
            context, alpha = self.attn(h, enc, enc_proj)
        else:
            context = enc.mean(dim=1)
            alpha = enc.new_full(enc.shape[:2], 1.0 / enc.shape[1])
        h, c = self.cell(torch.cat([self.embed(prev), context], dim=-1), (h, c))
        logits = self.out(self.drop(torch.cat([h, context], dim=-1)))
        return logits, (h, c), alpha

    def forward(self, images: torch.Tensor, tokens_in: torch.Tensor) -> torch.Tensor:
        """Teacher forcing. tokens_in (B, L) starts with BOS. Returns logits (B, L, V)."""
        enc = self.encoder(images)
        enc_proj = self.attn.project(enc)
        state = self._init_state(enc)
        steps = []
        for t in range(tokens_in.shape[1]):
            logits, state, _ = self._step(tokens_in[:, t], state, enc, enc_proj)
            steps.append(logits)
        return torch.stack(steps, dim=1)

    def loss(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Cross entropy on the next token, ignoring PAD."""
        logits = self(images, tokens[:, :-1])
        return fn.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), ignore_index=PAD
        )

    @torch.no_grad()
    def greedy_decode(
        self, images: torch.Tensor, max_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns token ids (B, max_len) with PAD after EOS, and attention (B, max_len, T)."""
        enc = self.encoder(images)
        enc_proj = self.attn.project(enc)
        state = self._init_state(enc)
        prev = torch.full((images.shape[0],), BOS, dtype=torch.long, device=images.device)
        done = torch.zeros_like(prev, dtype=torch.bool)
        ids, alphas = [], []
        for _ in range(max_len):
            logits, state, alpha = self._step(prev, state, enc, enc_proj)
            prev = logits.argmax(dim=-1)
            prev = prev.masked_fill(done, PAD)
            ids.append(prev)
            alphas.append(alpha)
            done = done | (prev == EOS)
        return torch.stack(ids, dim=1), torch.stack(alphas, dim=1)
