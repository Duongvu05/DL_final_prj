"""Phase 2 Mamba forecasters for Task 1 and Task 2 (plan Sections 2.1, 2.2 and 3).

Both models keep the Phase 1 data contract. They map an electricity Batch to
(pred (B, H) standardized seasonal residuals, aux dict), and the kW forecast is
ElectricityData.inverse(pred, origins). Only the temporal encoder changes.

- MambaGRUForecaster (Task 1, T1-M0..M4). Linear input projection 11 -> d_model, then a
  BiMamba encoder (Mamba-2, forward and reverse stacks) or, for the matched control T1-M1,
  a two-layer bidirectional GRU. A one-layer GRU readout runs over the encoder outputs. Its
  final state is the Bahdanau query over the readout outputs, and the Phase 1 head sees
  [final, context, target features].
- MambaSeq2SeqForecaster (Task 2, T2-M0..M4). Linear input projection 11 -> d_model, then a
  causal (forward-only) Mamba-2 encoder or, for T2-M1, a two-layer LSTM. The Phase 1
  LSTMCell decoder, horizon embedding, attention and output layer are kept exactly. The
  encoder outputs are projected to the decoder width for the attention memory, and the
  decoder (h, c) comes from a learned linear projection of the last encoder output.

forward(b, teacher_forcing=0.0, diagnostics=False). With diagnostics=True and a Mamba
encoder, aux also holds "dt_mean" (B, n_directions, n_layers, L), the step size averaged
over channels/heads at every input position in original time order, "state_influence"
(same shape), the mean |SSM weight| of each input position on the last scan output of that
layer and direction, plus "dt_directions" and "dt_static". This feeds the plan's
state-diagnostic figures. state_influence is informative for the static M2 too, where
dt_mean is constant by construction.

Run `python -m dl_final_prj.ts_mamba [configs/phase2.yaml]` to print the parameter counts,
the widths that match M1, M3 and M4 to M0 (plan Section 3, at most 5% apart) and the size
of M0 against the Phase 1 course-main models.
"""

import sys
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch import nn

from dl_final_prj.electricity import INPUT_CHANNELS, TARGET_FEATURES, Batch
from dl_final_prj.mamba import BiMamba
from dl_final_prj.ts_models import AdditiveAttention, count_params

N_FEATURES = len(INPUT_CHANNELS)
N_TARGET_FEATS = len(TARGET_FEATURES)
PARAM_TOLERANCE = 0.05


def _mamba_encoder(
    d_model: int,
    n_layers: int,
    direction: Literal["bi", "forward"],
    d_state: int,
    d_conv: int,
    expand: float,
    ssm_head_dim: int,
    static: bool,
    dropout: float,
    d_inner: int | None,
) -> BiMamba:
    return BiMamba(
        d_model,
        n_layers=n_layers,
        direction=direction,
        kind="mamba2",
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        head_dim=ssm_head_dim,
        static=static,
        dropout=dropout,
        d_inner=d_inner,  # None means expand * d_model
    )


def _diag_aux(diag: dict) -> dict:
    if not diag:
        return {}
    return {
        "dt_mean": diag["dt_mean"],
        "state_influence": diag["influence"],
        "dt_directions": list(diag["directions"]),
        "dt_static": bool(diag["static"]),
    }


class MambaGRUForecaster(nn.Module):
    """Task 1. Seasonal-residual Mamba-2-GRU hybrid (T1-M0) and its controls.

    encoder="mamba": BiMamba(d_model, n_layers, direction) over Linear(n_features, d_model).
    static=True gives T1-M2, d_conv=1 with a wider d_inner gives T1-M3 and
    direction="forward" gives T1-M4.
    encoder="bigru": two-layer bidirectional GRU with hidden `enc_hidden` per direction and
    dropout between layers, over the same input projection (T1-M1).

    A one-layer GRU readout (hidden `readout_hidden`) runs over the encoder outputs. With
    attention the Phase 1 AdditiveAttention scores the readout outputs with the final readout
    state as query. The head is the Phase 1 head LayerNorm -> Linear(head_dim) -> GELU ->
    Dropout -> Linear(H) on [final, context (if attention), target features flattened].
    """

    def __init__(
        self,
        n_features: int,
        horizon: int = 1,
        n_target_feats: int = 10,
        encoder: Literal["mamba", "bigru"] = "mamba",
        d_model: int = 96,
        n_layers: int = 2,
        direction: Literal["bi", "forward"] = "bi",
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2,
        ssm_head_dim: int = 32,
        d_inner: int | None = None,
        static: bool = False,
        enc_hidden: int = 64,
        readout_hidden: int = 64,
        dropout: float = 0.2,
        attention: bool = True,
        attn_dim: int = 128,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder_kind = encoder
        self.in_proj = nn.Linear(n_features, d_model)
        if encoder == "mamba":
            self.encoder = _mamba_encoder(
                d_model,
                n_layers,
                direction,
                d_state,
                d_conv,
                expand,
                ssm_head_dim,
                static,
                dropout,
                d_inner,
            )
            enc_dim = d_model
        elif encoder == "bigru":
            self.encoder = nn.GRU(
                d_model,
                enc_hidden,
                n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            enc_dim = 2 * enc_hidden
        else:
            raise ValueError(f"unknown encoder {encoder!r}")
        self.readout = nn.GRU(enc_dim, readout_hidden, 1, batch_first=True)
        self.attn = (
            AdditiveAttention(readout_hidden, readout_hidden, attn_dim) if attention else None
        )
        feat = readout_hidden * (2 if attention else 1) + horizon * n_target_feats
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, horizon),
        )

    @property
    def has_diagnostics(self) -> bool:
        return self.encoder_kind == "mamba"

    def encode(self, x: torch.Tensor, diagnostics: bool = False) -> tuple[torch.Tensor, dict]:
        """x (B, L, n_features) -> encoder outputs (B, L, enc_dim) and Mamba diagnostics."""
        u = self.in_proj(x)
        if isinstance(self.encoder, BiMamba):
            return self.encoder(u, diagnostics=diagnostics)
        return self.encoder(u)[0], {}

    def forward(self, b: Batch, teacher_forcing: float = 0.0, diagnostics: bool = False):
        enc, diag = self.encode(b.x, diagnostics)
        out, h_n = self.readout(enc)
        final = h_n[-1]
        aux: dict = {}
        parts = [final]
        if self.attn is not None:
            ctx, w = self.attn(final, out)
            parts.append(ctx)
            aux["attention"] = w
        parts.append(b.target_feats.flatten(1))
        aux.update(_diag_aux(diag))
        return self.head(torch.cat(parts, dim=-1)), aux


class MambaSeq2SeqForecaster(nn.Module):
    """Task 2. Causal Mamba-2 encoder + Phase 1 attentive LSTM decoder (T2-M0) and controls.

    encoder="mamba": forward-only BiMamba(d_model, n_layers) over Linear(n_features, d_model),
    so encoder output t depends on input steps <= t only. static=True gives T2-M2, d_conv=1
    with a wider d_inner gives T2-M3 and attention=False gives T2-M4.
    encoder="lstm": two-layer LSTM with hidden `enc_hidden` and dropout between layers over
    the same input projection (T2-M1).

    The decoder is the Phase 1 one. Its input at step k is [previous standardized residual,
    target features of hour t+k, horizon embedding k, attention context (if attention)]. The
    attention query is the decoder state before the update and the memory is
    Linear(enc_dim, hidden) of the encoder outputs. The first previous residual is r(t). The
    decoder (h, c) starts from Linear(enc_dim, 2 * hidden) of the last encoder output. The
    output Linear([h, context]) gives one residual per step. Teacher forcing with the ratio
    `teacher_forcing` happens in training mode only. Eval decoding is fully autoregressive.
    """

    def __init__(
        self,
        n_features: int,
        horizon: int = 5,
        n_target_feats: int = 10,
        encoder: Literal["mamba", "lstm"] = "mamba",
        d_model: int = 96,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2,
        ssm_head_dim: int = 32,
        d_inner: int | None = None,
        static: bool = False,
        enc_hidden: int = 64,
        hidden: int = 64,
        dropout: float = 0.2,
        attention: bool = True,
        attn_dim: int = 128,
        horizon_dim: int = 16,
    ) -> None:
        super().__init__()
        self.horizon, self.hidden = horizon, hidden
        self.encoder_kind = encoder
        self.in_proj = nn.Linear(n_features, d_model)
        if encoder == "mamba":
            self.encoder = _mamba_encoder(
                d_model,
                n_layers,
                "forward",
                d_state,
                d_conv,
                expand,
                ssm_head_dim,
                static,
                dropout,
                d_inner,
            )
            enc_dim = d_model
        elif encoder == "lstm":
            self.encoder = nn.LSTM(d_model, enc_hidden, n_layers, batch_first=True, dropout=dropout)
            enc_dim = enc_hidden
        else:
            raise ValueError(f"unknown encoder {encoder!r}")
        self.init_state = nn.Linear(enc_dim, 2 * hidden)
        self.mem_proj = nn.Linear(enc_dim, hidden) if attention else None
        self.attn = AdditiveAttention(hidden, hidden, attn_dim) if attention else None
        self.h_emb = nn.Embedding(horizon, horizon_dim)
        dec_in = 1 + n_target_feats + horizon_dim + (hidden if attention else 0)
        self.decoder = nn.LSTMCell(dec_in, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * (2 if attention else 1), 1)

    @property
    def has_diagnostics(self) -> bool:
        return self.encoder_kind == "mamba"

    def encode(self, x: torch.Tensor, diagnostics: bool = False) -> tuple[torch.Tensor, dict]:
        """x (B, L, n_features) -> causal encoder outputs (B, L, enc_dim) and diagnostics."""
        u = self.in_proj(x)
        if isinstance(self.encoder, BiMamba):
            return self.encoder(u, diagnostics=diagnostics)
        return self.encoder(u)[0], {}

    def forward(self, b: Batch, teacher_forcing: float = 0.0, diagnostics: bool = False):
        enc, diag = self.encode(b.x, diagnostics)
        h, c = self.init_state(enc[:, -1]).chunk(2, dim=-1)
        mem = self.mem_proj(enc) if self.mem_proj is not None else None
        kp = self.attn.w_k(mem) if self.attn is not None and mem is not None else None
        prev = b.r0.unsqueeze(-1)
        preds, weights = [], []
        bsz = b.x.shape[0]
        tf = teacher_forcing if self.training else 0.0
        for k in range(self.horizon):
            emb = self.h_emb.weight[k].expand(bsz, -1)
            parts = [prev, b.target_feats[:, k], emb]
            ctx = None
            if self.attn is not None and mem is not None:
                ctx, w = self.attn(h, mem, kp)
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
        aux: dict = {"attention": torch.stack(weights, dim=1)} if weights else {}
        aux.update(_diag_aux(diag))
        return torch.cat(preds, dim=-1), aux


# ---------------------------------------------------------------- parameter matching

MODEL_KINDS = {"mamba_gru": MambaGRUForecaster, "mamba_seq2seq": MambaSeq2SeqForecaster}


MAMBA_ONLY_ARGS = (
    "encoder",
    "direction",
    "d_state",
    "d_conv",
    "expand",
    "ssm_head_dim",
    "d_inner",
    "static",
)


def build(kind: str, args: dict, horizon: int) -> nn.Module:
    """Model of `kind` with the Phase 1/2 data dimensions (11 inputs, 10 target features)."""
    return MODEL_KINDS[kind](N_FEATURES, horizon=horizon, n_target_feats=N_TARGET_FEATS, **args)


def n_params(kind: str, args: dict, horizon: int) -> int:
    return count_params(build(kind, args, horizon))


def rel_diff(n: int, ref: int) -> float:
    return (n - ref) / ref


def match_rnn_hidden(
    kind: str, m0_args: dict, horizon: int, rnn: str, candidates: range = range(16, 257)
) -> tuple[int, int]:
    """enc_hidden of the RNN control (M1) whose total count is closest to M0's.

    The control keeps every M0 argument that is not Mamba-specific (input projection width,
    readout/decoder, attention, head) and swaps the encoder for `rnn`. Returns
    (enc_hidden, n_params).
    """
    target = n_params(kind, m0_args, horizon)
    base = {k: v for k, v in m0_args.items() if k not in MAMBA_ONLY_ARGS}
    best = min(
        candidates,
        key=lambda h: abs(
            n_params(kind, {**base, "encoder": rnn, "enc_hidden": h}, horizon) - target
        ),
    )
    return best, n_params(kind, {**base, "encoder": rnn, "enc_hidden": best}, horizon)


def match_noconv_width(kind: str, m0_args: dict, horizon: int) -> tuple[int, int, int]:
    """Compensating width for the d_conv=1 ablation (M3).

    The number of Mamba-2 heads stays that of M0 (d_inner / ssm_head_dim), so dt and A keep
    the same granularity, and the per-head width ssm_head_dim is chosen so the total count
    is closest to M0's. Returns (ssm_head_dim, d_inner, n_params).
    """
    target = n_params(kind, m0_args, horizon)
    d_inner0 = m0_args.get("d_inner") or int(m0_args.get("expand", 2) * m0_args["d_model"])
    heads = d_inner0 // m0_args.get("ssm_head_dim", 32)

    def count(p: int) -> int:
        args = {**m0_args, "d_conv": 1, "ssm_head_dim": p, "d_inner": heads * p}
        return n_params(kind, args, horizon)

    p = min(range(4, 129), key=lambda p: abs(count(p) - target))
    return p, heads * p, count(p)


def match_head_count(kind: str, m0_args: dict, horizon: int, change: dict) -> tuple[int, int]:
    """Compensating width for an ablation that removes parameters outside the SSM (M4).

    `change` is applied to the M0 args (e.g. direction "forward" or attention False). The
    per-head width ssm_head_dim stays that of M0 and the number of Mamba-2 heads is chosen
    so the total count is closest to M0's. For T1-M4 this restores roughly the 2 x 6 heads
    per layer that the bidirectional M0 has across its two stacks. Returns (d_inner,
    n_params).
    """
    target = n_params(kind, m0_args, horizon)
    p = m0_args.get("ssm_head_dim", 32)

    def count(h: int) -> int:
        return n_params(kind, {**m0_args, **change, "d_inner": h * p}, horizon)

    heads = min(range(1, 65), key=lambda h: abs(count(h) - target))
    return heads * p, count(heads)


M4_CHANGE = {1: {"direction": "forward"}, 2: {"attention": False}}


def matched_widths(cfg: dict) -> dict:
    """Widths that match M1, M3 and M4 to M0 for both tasks of a phase 2 config."""
    out = {}
    for t, (kind, rnn) in {1: ("mamba_gru", "bigru"), 2: ("mamba_seq2seq", "lstm")}.items():
        tcfg = cfg[f"task{t}"]
        horizon = len(tcfg["offsets"])
        m0 = tcfg["models"][f"t{t}_m0"]["args"]
        h, n1 = match_rnn_hidden(kind, m0, horizon, rnn)
        p, di, n3 = match_noconv_width(kind, m0, horizon)
        di4, n4 = match_head_count(kind, m0, horizon, M4_CHANGE[t])
        out[f"task{t}"] = {
            "m0_params": n_params(kind, m0, horizon),
            "m1_enc_hidden": h,
            "m1_params": n1,
            "m3_ssm_head_dim": p,
            "m3_d_inner": di,
            "m3_params": n3,
            "m4_d_inner": di4,
            "m4_params": n4,
        }
    return out


PHASE1_COURSE_MAIN = {1: "bigru_attention", 2: "seq2seq_attention"}


def phase1_budget(cfg: dict, phase1_cfg: dict) -> dict:
    """M0 against the Phase 1 course-main model of the same task (plan 2.1 budget note).

    The phase 2 config keeps d_model=96 and expand=2 from the plan, so M0 is larger than
    Phase 1 P0. This reports both counts so the report can state the gap.
    """
    from dl_final_prj.ts_models import BiRNNForecaster, Seq2SeqForecaster

    p1_kinds = {"birnn": BiRNNForecaster, "seq2seq": Seq2SeqForecaster}
    out = {}
    for t in (1, 2):
        horizon = len(cfg[f"task{t}"]["offsets"])
        m0 = cfg[f"task{t}"]["models"][f"t{t}_m0"]
        p1 = phase1_cfg[f"task{t}"]["models"][PHASE1_COURSE_MAIN[t]]
        n_p1 = count_params(
            p1_kinds[p1["kind"]](
                N_FEATURES, horizon=horizon, n_target_feats=N_TARGET_FEATS, **p1["args"]
            )
        )
        n_m0 = n_params(m0["kind"], m0["args"], horizon)
        out[f"task{t}"] = {"phase1_p0": n_p1, "m0": n_m0, "ratio": n_m0 / n_p1}
    return out


def param_table(cfg: dict) -> list[dict]:
    """One row per phase 2 model: count and relative difference to its task's M0."""
    rows = []
    for t in (1, 2):
        tcfg = cfg[f"task{t}"]
        horizon = len(tcfg["offsets"])
        models = tcfg["models"]
        ref = n_params(models[f"t{t}_m0"]["kind"], models[f"t{t}_m0"]["args"], horizon)
        for name, m in models.items():
            n = n_params(m["kind"], m["args"], horizon)
            rows.append({"model": name, "n_params": n, "rel_diff_vs_m0": rel_diff(n, ref)})
    return rows


def main(config: str = "configs/phase2.yaml", phase1: str = "configs/phase1.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    print(f"{'model':8s} {'params':>9s} {'vs M0':>8s}")
    for r in param_table(cfg):
        print(f"{r['model']:8s} {r['n_params']:9,d} {100 * r['rel_diff_vs_m0']:+7.2f}%")
    print("matched widths (searched from the M0 args):")
    for task, w in matched_widths(cfg).items():
        print(f"  {task}: {w}")
    if Path(phase1).exists():
        print("M0 vs Phase 1 course-main model:")
        for task, b in phase1_budget(cfg, yaml.safe_load(Path(phase1).read_text())).items():
            print(f"  {task}: {b}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
