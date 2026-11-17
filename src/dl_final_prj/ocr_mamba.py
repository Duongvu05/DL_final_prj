"""Phase 2 Task 3 recognizers (plan Section 2.3 Phase 2, ablations T3-M0..M4).

The Phase 1 attentive GRU decoder (GRUDecoderRecognizer in ocr_models) is reused unchanged.
Only the token encoder differs from Phase 1:

    VGG-BN CNN (512x7x7) -> 49 raster-order tokens -> Linear(512, d_model)
    -> + learned row/column position embeddings (optional, off for T3-M4)
    -> token mixer -> (B, 49, d_model) tokens for attention, the mean token and the length head

Token mixers. Both default mixers share one skeleton, so the M0-M1 contrast changes only the
sequence operator (plan 2.3 "two residual bidirectional Mamba blocks", Vision Mamba style):

    n_layers x [ x + Dropout(Linear(2*d, d)(cat[fwd(RMSNorm(x)), flip(bwd(flip(RMSNorm(x))))])) ]
    -> final RMSNorm

- "mamba" (T3-M0, M2, M3, M4): VisionBiMamba. fwd and bwd are Mamba mixers with separate
  weights. The two directions are fused inside every layer, so layer 2 sees context from both
  directions. direction="forward" drops bwd and the fusion Linear (exactly mamba.MambaBlock
  per layer). `mamba: {fusion: late}` selects the older mamba.BiMamba instead (independent
  forward and reverse stacks joined only by one final Linear).
- "gru" (T3-M1): BiGRUMixer, the same skeleton with a 1-layer bidirectional nn.GRU per layer.
  Its hidden size is chosen so that the whole model is within 5% of T3-M0.

The projection, position embeddings and mixer run in fp32 with autocast off (the CNN still
runs under the caller's autocast), as the Phase 1 recurrent parts do. Mamba and GRU mixers
therefore get the same precision.

ONNX: `GreedyDecodeExport` wraps a recognizer so that `torch.onnx.export` traces the CNN,
the token mixer and the whole greedy decode (max_len + 1 unrolled steps, EOS forced at the
last step) into one graph that maps images to token ids. It is a deployment check only. The
reported Task 3 results use the inherited beam search (scripts/run_ocr.py).
"""

from typing import Literal, cast

import torch
from torch import nn

from dl_final_prj.mamba import (
    BiMamba,
    Mamba1Mixer,
    Mamba2Mixer,
    MambaBlock,
    MixerKind,
    RMSNorm,
    build_mixer,
)
from dl_final_prj.ocr_models import N_LENGTHS, GRUDecoderRecognizer, VggEncoder, _fp32
from dl_final_prj.svhn import MAX_LEN, VOCAB
from dl_final_prj.ts_models import AdditiveAttention, count_params

MixerName = Literal["mamba", "gru"]
Direction = Literal["bi", "forward"]
MAMBA_DEFAULTS = {
    "kind": "mamba1",
    "fusion": "layer",
    "n_layers": 2,
    "direction": "bi",
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "static": False,
    "dropout": 0.0,
}
# Trainable parameters of the Phase 1 cnn_attention_gru (T3-P0, configs/phase1_task3.yaml).
# Plan 2.3 asks for T3-M0 within 5% of it, which the plan's own fixed Mamba settings cannot
# meet (see configs/phase2_task3.yaml). It is recorded in spec.json for transparency.
P0_PARAMS = 6_202_898


def _flip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[1])


def _check_direction(direction: str) -> bool:
    if direction not in ("bi", "forward"):
        raise ValueError(f"unknown direction {direction!r}")
    return direction == "bi"


class BiMambaLayer(nn.Module):
    """Pre-norm residual bidirectional Mamba block (B, L, d_model) -> (B, L, d_model).

    u = RMSNorm(x). bi: x + Dropout(Linear(2d, d)(cat[M_f(u), flip(M_r(flip(u)))])).
    forward: x + Dropout(M_f(u)), the same computation as mamba.MambaBlock.
    """

    def __init__(
        self,
        d_model: int,
        direction: Direction = "bi",
        kind: MixerKind = "mamba1",
        dropout: float = 0.0,
        **mixer_kw,
    ) -> None:
        super().__init__()
        self.bi = _check_direction(direction)
        self.norm = RMSNorm(d_model)
        self.fwd = build_mixer(kind, d_model, **mixer_kw)
        if self.bi:
            self.bwd = build_mixer(kind, d_model, **mixer_kw)
            self.proj = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward_diag(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns (output, [mean dt per step (B, L) per direction, in original order])."""
        u = self.norm(x)
        f, dt_f = self.fwd.forward_diag(u)
        dts = [dt_f.detach().float().mean(-1)]
        if self.bi:
            r, dt_r = self.bwd.forward_diag(_flip(u))
            f = self.proj(torch.cat([f, _flip(r)], dim=-1))
            dts.append(_flip(dt_r.detach().float().mean(-1)))
        return x + self.drop(f), dts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.norm(x)
        f = self.fwd(u)
        if self.bi:
            f = self.proj(torch.cat([f, _flip(self.bwd(_flip(u)))], dim=-1))
        return x + self.drop(f)


class VisionBiMamba(nn.Module):
    """n_layers BiMambaLayers and a final RMSNorm, (B, L, d_model) -> (B, L, d_model).

    Same interface as mamba.BiMamba. forward(x, diagnostics=False) returns (y, diag). diag is
    {} unless diagnostics=True, then it holds "dt_mean" (B, n_directions, n_layers, L) in
    original token order, "directions", "static" and "notes". `features(x)` returns y alone.
    direction="forward" is causal in token order (output t depends on tokens <= t).
    """

    fusion = "layer"

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        direction: Direction = "bi",
        kind: MixerKind = "mamba1",
        static: bool = False,
        dropout: float = 0.0,
        final_norm: bool = True,
        **mixer_kw,
    ) -> None:
        super().__init__()
        _check_direction(direction)
        self.direction, self.static, self.n_layers = direction, static, n_layers
        self.layers = nn.ModuleList(
            BiMambaLayer(d_model, direction, kind, dropout, static=static, **mixer_kw)
            for _ in range(n_layers)
        )
        self.norm = RMSNorm(d_model) if final_norm else nn.Identity()

    def ssm_mixers(self) -> list[Mamba1Mixer | Mamba2Mixer]:
        return [m for m in self.modules() if isinstance(m, Mamba1Mixer | Mamba2Mixer)]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor, diagnostics: bool = False) -> tuple[torch.Tensor, dict]:
        if not diagnostics:
            return self.features(x), {}
        per_layer = []
        for layer in self.layers:
            x, dts = cast(BiMambaLayer, layer).forward_diag(x)
            per_layer.append(torch.stack(dts, dim=1))  # (B, n_dirs, L)
        diag: dict = {
            "dt_mean": torch.stack(per_layer, dim=2),  # (B, n_dirs, n_layers, L)
            "directions": ["forward", "reverse"][: 2 if self.direction == "bi" else 1],
            "static": self.static,
            "notes": (
                "static-state control: dt, B and C are learned time-invariant parameters, "
                "so dt_mean is constant over tokens"
                if self.static
                else "selective: dt, B and C are computed from the input at every token"
            ),
        }
        return self.norm(x), diag


class BiGRULayer(nn.Module):
    """Pre-norm residual GRU layer, x + Dropout(Linear(GRU(RMSNorm(x)))). bi uses a
    1-layer bidirectional nn.GRU, whose outputs are the concatenated directions."""

    def __init__(
        self, d_model: int, hidden: int, direction: Direction = "bi", dropout: float = 0.0
    ) -> None:
        super().__init__()
        bi = _check_direction(direction)
        self.norm = RMSNorm(d_model)
        self.rnn = nn.GRU(d_model, hidden, 1, batch_first=True, bidirectional=bi)
        self.proj = nn.Linear(hidden * (2 if bi else 1), d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.norm(x))
        return x + self.drop(self.proj(h))


class BiGRUMixer(nn.Module):
    """GRU token mixer with the VisionBiMamba skeleton: n_layers BiGRULayers and a final
    RMSNorm. forward(x, diagnostics) returns (y, {}) like the Mamba mixers."""

    def __init__(
        self,
        d_model: int,
        hidden: int,
        n_layers: int = 2,
        direction: Direction = "bi",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _check_direction(direction)
        self.direction, self.hidden, self.n_layers = direction, hidden, n_layers
        self.layers = nn.ModuleList(
            BiGRULayer(d_model, hidden, direction, dropout) for _ in range(n_layers)
        )
        self.out_norm = RMSNorm(d_model)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(x)

    def forward(self, x: torch.Tensor, diagnostics: bool = False) -> tuple[torch.Tensor, dict]:
        return self.features(x), {}


TokenMixer = VisionBiMamba | BiMamba | BiGRUMixer


class MixedTokenEncoder(nn.Module):
    """VGG-BN CNN -> Linear(512, d_model) tokens -> optional 2-D positions -> token mixer.

    Output (B, h*w, d_model) in fp32, raster order (row-major), like TokenEncoder in Phase 1.
    """

    def __init__(
        self,
        d_model: int = 256,
        mixer: MixerName = "mamba",
        pos_emb: bool = True,
        grid: tuple[int, int] = (7, 7),
        dropout: float = 0.1,
        mamba: dict | None = None,
        gru_hidden: int = 272,
        gru_layers: int = 2,
        gru_direction: Direction = "bi",
        gru_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cnn = VggEncoder(dropout=dropout)
        self.grid, self.dim, self.mixer_name = tuple(grid), d_model, mixer
        self.proj = nn.Linear(self.cnn.out_channels, d_model)
        if pos_emb:
            self.row = nn.Parameter(torch.randn(grid[0], d_model) * 0.02)
            self.col = nn.Parameter(torch.randn(grid[1], d_model) * 0.02)
        else:
            self.row = self.col = None
        if mixer == "mamba":
            cfg = MAMBA_DEFAULTS | dict(mamba or {})
            fusion = cfg.pop("fusion")
            if fusion == "layer":
                self.mixer: TokenMixer = VisionBiMamba(d_model, **cfg)
            elif fusion == "late":
                self.mixer = BiMamba(d_model, **cfg)
            else:
                raise ValueError(f"unknown mamba fusion {fusion!r}")
        elif mixer == "gru":
            self.mixer = BiGRUMixer(d_model, gru_hidden, gru_layers, gru_direction, gru_dropout)
        else:
            raise ValueError(f"unknown token mixer {mixer!r}")

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Projected tokens with positions, before the mixer. (B, h*w, d_model), fp32."""
        f = self.cnn(images).float()
        _, _, h, w = f.shape
        if (h, w) != self.grid:
            raise ValueError(f"feature map {h}x{w} does not match the {self.grid} grid")
        with _fp32(f):
            t = self.proj(f.flatten(2).transpose(1, 2))
            if self.row is not None and self.col is not None:
                pos = self.row[:, None, :] + self.col[None, :, :]  # token i*w + j = row i, col j
                t = t + pos.reshape(1, h * w, self.dim)
        return t

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        t = self.tokens(images)
        with _fp32(t):
            return self.mixer(t)[0]

    def describe_mixer(self) -> str:
        d, m = self.dim, self.mixer
        if isinstance(m, BiGRUMixer):
            return (
                f"{m.n_layers} pre-norm residual layers, each a 1-layer "
                f"{'bi' if m.direction == 'bi' else 'uni'}directional GRU (hidden {m.hidden}) "
                f"+ Linear back to {d}, final RMSNorm"
            )
        sel = "static (input-independent dt, B, C)" if m.static else "selective"
        if isinstance(m, VisionBiMamba):
            blk = m.ssm_mixers()[0]
            fuse = (
                "forward and reverse mixers on the shared RMSNorm input, concat + "
                "Linear(2*d_model, d_model) inside every layer"
                if m.direction == "bi"
                else "forward mixer only"
            )
            return (
                f"VisionBiMamba direction={m.direction}, {m.n_layers} pre-norm residual layers "
                f"({fuse}), {type(blk).__name__} d_model {d} d_inner {blk.d_inner} "
                f"d_state {blk.d_state}, {sel}, final RMSNorm"
            )
        blk = cast(MambaBlock, m.fwd[0]).mixer
        return (
            f"BiMamba (late fusion) direction={m.direction}, {m.n_layers} pre-norm residual "
            f"blocks per direction, {type(blk).__name__} d_model {d} d_inner {blk.d_inner} "
            f"d_state {blk.d_state}, {sel}, final RMSNorm per direction"
            + (", concat + Linear(2*d_model, d_model)" if m.direction == "bi" else "")
        )

    def forward_diag(self, images: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Mixed tokens and the mixer diagnostics ({} for the GRU mixer)."""
        t = self.tokens(images)
        with _fp32(t):
            return self.mixer(t, diagnostics=True)


class MixerAttentionGRURecognizer(GRUDecoderRecognizer):
    """Phase 1 attentive GRU decoder over d_model-wide mixed visual tokens.

    All decoding (greedy, length-constrained and the batched beam search of any width that
    gives the reported results), loss and prediction code is inherited from
    GRUDecoderRecognizer. Only
    __init__ differs, because the encoder and the token width (d_model instead of 512) change.
    It builds the same decoder modules as the parent, in the same order.
    """

    attention = True
    mixer: MixerName = "mamba"

    def __init__(
        self,
        max_len: int = MAX_LEN,
        hidden: int = 256,
        layers: int = 2,
        attn_dim: int = 256,
        emb_dim: int = 64,
        length_loss_weight: float = 0.2,
        dropout: float = 0.1,
        d_model: int = 256,
        pos_emb: bool = True,
        **mixer_kw,
    ) -> None:
        nn.Module.__init__(self)  # GRUDecoderRecognizer.__init__ would build a 512-d encoder
        self.max_len, self.hidden, self.layers = max_len, hidden, layers
        self.length_loss_weight = length_loss_weight
        self.encoder = MixedTokenEncoder(
            d_model, self.mixer, pos_emb=pos_emb, dropout=dropout, **mixer_kw
        )
        c = self.encoder.dim
        self.attn = AdditiveAttention(hidden, c, attn_dim)
        self.emb = nn.Embedding(VOCAB, emb_dim)
        self.cells = nn.ModuleList(
            [nn.GRUCell(emb_dim + c if i == 0 else hidden, hidden) for i in range(layers)]
        )
        self.init_h = nn.Linear(c, hidden * layers)
        self.out = nn.Linear(hidden + c, VOCAB)
        self.length_head = nn.Linear(c, N_LENGTHS)

    @torch.no_grad()
    def mixer_diagnostics(self, images: torch.Tensor) -> dict:
        """Token-mixer diagnostics. For Mamba: dt_mean (B, n_dirs, n_layers, h*w) in raster
        order, directions, static and notes. For the GRU mixer: {}."""
        return self.encoder.forward_diag(images)[1]

    def param_breakdown(self) -> dict[str, int]:
        enc = self.encoder
        pos = sum(p.numel() for p in (enc.row, enc.col) if p is not None)
        total = count_params(self)
        parts = {
            "cnn": count_params(enc.cnn),
            "token_projection": count_params(enc.proj),
            "position_embeddings": pos,
            "token_mixer": count_params(enc.mixer),
        }
        parts["decoder_and_heads"] = total - sum(parts.values())
        return parts | {"total": total}


class MambaAttentionGRURecognizer(MixerAttentionGRURecognizer):
    """T3-M0 (and M2/M3/M4 through `mamba` and `pos_emb`): CNN + Vision-Mamba token mixer +
    Phase 1 attentive GRU decoder. `mamba` holds mixer kwargs over MAMBA_DEFAULTS."""

    mixer = "mamba"


class GRUMixerAttentionGRURecognizer(MixerAttentionGRURecognizer):
    """T3-M1: the same model with a parameter-matched bidirectional GRU token mixer."""

    mixer = "gru"


MIXER_KINDS = {
    "mamba_attention": MambaAttentionGRURecognizer,
    "gru_mixer_attention": GRUMixerAttentionGRURecognizer,
}


def build_mixer_recognizer(kind: str, **kw) -> MixerAttentionGRURecognizer:
    if kind not in MIXER_KINDS:
        raise ValueError(f"unknown mixer recognizer kind {kind}")
    return MIXER_KINDS[kind](**kw)


class GreedyDecodeExport(nn.Module):
    """images (B, 3, 54, 54) -> token ids (B, max_len + 1), unconstrained greedy decoding.

    The decode loop is unrolled to max_len + 1 steps. Every step runs for every image and EOS
    is forced at the last step, so there is no data-dependent control flow. A finished
    sequence keeps decoding after EOS, and `svhn.decode` stops at the first EOS.
    """

    def __init__(self, model: GRUDecoderRecognizer) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.greedy(images)[0]
