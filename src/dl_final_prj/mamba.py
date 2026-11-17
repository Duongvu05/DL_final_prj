"""Pure-PyTorch Mamba-1 and Mamba-2 blocks for the Phase 2 models (plan Sections 2.1-2.3, 3).

No mamba-ssm CUDA kernels are used. The sequences are short (168 hours, 49 visual tokens),
the static-state ablation (M2) needs the scan internals, and the Task 3 model is exported to
ONNX. So every forward pass uses plain tensor ops with no in-place updates, no complex dtypes,
no custom ops and no Python branching on tensor values. Loops run over shapes only, which are
constants once traced.

- Mamba1Mixer is the selective SSM block of Gu and Dao (2023), as used by Vision Mamba.
  Parameter names match mambapy's MambaBlock so weights copy across for testing.
- Mamba2Mixer is the SSD block of Dao and Gu (2024) with one group (B and C shared by all
  heads). Its output uses the exact quadratic (dual) form, which is cheap at L <= 168.
- `static=True` gives the M2 control. dt, B and C become learned time-invariant parameters.
  The in_proj, conv, gate and out_proj path stay the same. The dropped selection
  projections are replaced by a low-rank residual adapter on x of the same size, so the
  trainable parameter count stays within 5% (see `static_adapter_rank`). The static
  Mamba-2 keeps a full-rank state. A, B and C are per head and per state (S4D-real
  diagonal A), so each head is a sum of d_state decaying modes, not a single EMA.
- Diagnostics. `forward_state_diag` also returns the state influence, |weight| of each
  input position on the last output of the scan, averaged over channels/heads.
- MambaBlock is the pre-norm residual block x + Dropout(Mixer(RMSNorm(x))).
- BiMamba stacks blocks over x and over x reversed in time (separate weights), then projects
  the concatenation back to d_model. `direction="forward"` keeps only the causal stack.

The SSM cores run in fp32 with autocast switched off. Their outputs go back to the dtype of
the gate branch before out_proj.
"""

import math
from typing import Literal, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

MixerKind = Literal["mamba1", "mamba2"]
ScanMode = Literal["parallel", "loop"]


def count_params(module: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _fp32_ctx(x: torch.Tensor):
    return torch.autocast(device_type=x.device.type, enabled=False)


def _inv_softplus_dt(
    n: int, dt_min: float = 1e-3, dt_max: float = 1e-1, floor: float = 1e-4
) -> torch.Tensor:
    """Standard Mamba dt bias init. dt ~ log-uniform on [dt_min, dt_max], then softplus^-1."""
    dt = torch.exp(torch.rand(n) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    dt = dt.clamp(min=floor)
    return dt + torch.log(-torch.expm1(-dt))


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight, computed in fp32."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


class CausalDepthwiseConv(nn.Module):
    """Depthwise Conv1d over time on (B, L, C) with left padding k-1. Identity when k == 1."""

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        # padding k-1 on both sides then crop to L equals left padding (mambapy layout)
        self.conv = (
            nn.Conv1d(channels, channels, kernel, groups=channels, padding=kernel - 1)
            if kernel > 1
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return x
        L = x.shape[1]
        return self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)


# ---------------------------------------------------------------- Mamba-1 selective scan


def selective_scan_loop(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Sequential Mamba-1 scan. h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t, y_t = C_t.h_t + D x_t.

    x, dt (b, L, d), A (d, N), B, C (b, L, N), D (d) -> y (b, L, d). The state is (b, d, N).
    """
    dA = torch.exp(dt.unsqueeze(-1) * A)  # (b, L, d, N)
    dBx = (dt * x).unsqueeze(-1) * B.unsqueeze(2)  # (b, L, d, N)
    h = torch.zeros_like(dBx[:, 0])
    ys = []
    for t in range(x.shape[1]):
        h = dA[:, t] * h + dBx[:, t]
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, dim=1) + D * x


def _shift(t: torch.Tensor, off: int) -> torch.Tensor:
    """Shift right by `off` steps along dim 1 and fill the front with zeros."""
    return torch.cat([torch.zeros_like(t[:, :off]), t[:, :-off]], dim=1)


def selective_scan_parallel(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Log-depth (Hillis-Steele) version of `selective_scan_loop`, same signature and result.

    Each step composes the affine map of window (t-2k, t-k] into the one of (t-k, t]. The
    decay product is carried in log space (sum of dt*A <= 0), so it never overflows, and
    exp() of it lies in [0, 1]. The input term dt*B*x is signed, so it stays in linear space
    (a fully log-space scan would need complex logs, which ONNX lacks). ceil(log2 L) steps,
    no in-place ops, so it exports and backpropagates like ordinary tensor code.
    """
    log_a = dt.unsqueeze(-1) * A  # (b, L, d, N), <= 0
    h = (dt * x).unsqueeze(-1) * B.unsqueeze(2)
    L = x.shape[1]
    off = 1
    while off < L:
        h = h + torch.exp(log_a) * _shift(h, off)
        log_a = log_a + _shift(log_a, off)
        off *= 2
    return (h * C.unsqueeze(2)).sum(-1) + D * x


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    mode: ScanMode = "parallel",
) -> torch.Tensor:
    if mode == "parallel":
        return selective_scan_parallel(x, dt, A, B, C, D)
    if mode == "loop":
        return selective_scan_loop(x, dt, A, B, C, D)
    raise ValueError(f"unknown scan mode {mode!r}")


# ---------------------------------------------------------------- Mamba-2 SSD


def segsum(a: torch.Tensor) -> torch.Tensor:
    """Stable segment sum. a (..., T) -> S (..., T, T), S_ij = sum_{k=j+1..i} a_k for j <= i.

    Entries above the diagonal are -inf so exp(S) is the causal decay matrix L. Each S_ij is
    a cumsum of the masked terms themselves, not a difference of two long prefix sums, so it
    has no cancellation error (as in the reference ssd_minimal).
    """
    T = a.shape[-1]
    strict = torch.tril(torch.ones(T, T, dtype=torch.bool, device=a.device), diagonal=-1)
    # S_ij = sum_k [j < k <= i] a_k. Build terms[..., k, j] = a_k [k > j], cumsum over k.
    terms = a.unsqueeze(-1).expand(*a.shape, T).masked_fill(~strict, 0.0)
    S = torch.cumsum(terms, dim=-2)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=a.device))
    return S.masked_fill(~causal, float("-inf"))


def ssd_quadratic(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Exact quadratic (dual) SSD form. y = (L o C B^T)(dt x) + D x.

    x (b, L, H, P), dt (b, L, H), A (H), B, C (b, L, N), D (H) -> y (b, L, H, P).
    L_ij = exp(sum_{k=j+1..i} dt_k A) for j <= i, else 0.
    """
    decay = torch.exp(segsum((dt * A).transpose(1, 2)))  # (b, H, L, S)
    cb = torch.einsum("bln,bsn->bls", C, B)  # (b, L, S)
    m = decay * cb.unsqueeze(1)
    y = torch.einsum("bhls,bshp->blhp", m, x * dt.unsqueeze(-1))
    return y + D.unsqueeze(-1) * x


def ssd_recurrent(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Step-by-step SSD recurrence, same signature as `ssd_quadratic` (test reference).

    h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T with h (b, H, P, N), y_t = h_t C_t + D x_t.
    """
    b, L, H, P = x.shape
    h = x.new_zeros(b, H, P, B.shape[-1])
    ys = []
    for t in range(L):
        a = torch.exp(dt[:, t] * A).view(b, H, 1, 1)
        u = (dt[:, t].unsqueeze(-1) * x[:, t]).unsqueeze(-1) * B[:, t].view(b, 1, 1, -1)
        h = a * h + u
        ys.append((h * C[:, t].view(b, 1, 1, -1)).sum(-1))
    return torch.stack(ys, dim=1) + D.unsqueeze(-1) * x


def lti_kernel(dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, L: int):
    """Causal Toeplitz kernel of a diagonal time-invariant SSM with one state set per head.

    dt (H), A (H, N) <= 0, B, C (H, N) -> K (H, L, L) with
    K[h, l, s] = dt_h sum_n C_hn B_hn exp((l - s) dt_h A_hn) for s <= l, else 0.
    The lag is clamped at 0 above the diagonal before exp, so nothing overflows.
    """
    idx = torch.arange(L, device=A.device)
    lag = idx.unsqueeze(1) - idx.unsqueeze(0)  # (L, S)
    causal = (lag >= 0).to(A.dtype)
    rate = (dt.unsqueeze(-1) * A).unsqueeze(-1).unsqueeze(-1)  # (H, N, 1, 1)
    modes = torch.exp(rate * lag.clamp(min=0).to(A.dtype))  # (H, N, L, S)
    K = torch.einsum("hn,hnls->hls", C * B, modes) * dt.view(-1, 1, 1)
    return K * causal


def ssd_static(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Time-invariant Mamba-2 SSM with a full-rank diagonal state per head.

    x (b, L, H, P), dt (H), A, B, C (H, N), D (H) -> y (b, L, H, P).
    h_t = exp(dt A) h_{t-1} + dt x_t B^T per head (state (P, N)), y_t = h_t C + D x_t.
    Computed as y = K x + D x with the (H, L, L) kernel of `lti_kernel`.
    """
    K = lti_kernel(dt, A, B, C, x.shape[1])
    return torch.einsum("hls,bshp->blhp", K, x) + D.unsqueeze(-1) * x


def ssd_static_recurrent(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Step-by-step version of `ssd_static`, same signature (test reference)."""
    b, L, H, P = x.shape
    a = torch.exp(dt.unsqueeze(-1) * A).view(1, H, 1, -1)  # (1, H, 1, N)
    h = x.new_zeros(b, H, P, A.shape[-1])
    ys = []
    for t in range(L):
        u = (dt.view(1, H, 1) * x[:, t]).unsqueeze(-1) * B.view(1, H, 1, -1)
        h = a * h + u
        ys.append((h * C.view(1, H, 1, -1)).sum(-1))
    return torch.stack(ys, dim=1) + D.unsqueeze(-1) * x


def _suffix_sum(a: torch.Tensor) -> torch.Tensor:
    """a (b, L, ...) -> s (b, L, ...) with s_t = sum_{k > t} a_k (0 at the last step)."""
    shifted = torch.cat([a[:, 1:], torch.zeros_like(a[:, :1])], dim=1)
    return torch.flip(torch.cumsum(torch.flip(shifted, dims=[1]), dim=1), dims=[1])


# ---------------------------------------------------------------- static-state helpers


def static_adapter_rank(deficit: int, d_inner: int) -> int:
    """Rank r of the static mixer's residual adapter x + up(down(x)).

    down is d_inner -> r without bias and up is r -> d_inner with bias, so the adapter has
    2*r*d_inner + d_inner parameters. r is chosen so this is closest to `deficit`, the
    number of selection parameters removed minus the static dt/B/C parameters added.

    r is capped at d_inner // 2. A linear d -> r -> d map with r >= d is just one d x d map,
    so a larger r would count parameters that add no capacity. When the cap binds, the
    caller must compensate elsewhere (e.g. a wider head) to stay parameter matched.
    """
    return max(1, min(d_inner // 2, round((deficit - d_inner) / (2 * d_inner))))


class _StaticAdapter(nn.Module):
    """x + up(down(x)), the capacity match for the static-state control.

    It acts only on the SSM input x (the value path), never on dt, B or C. `up` starts at
    zero, so at init the static mixer is a plain LTI SSM and the adapter learns from there.
    """

    def __init__(self, d_inner: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(d_inner, rank, bias=False)
        self.up = nn.Linear(rank, d_inner)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


# ---------------------------------------------------------------- mixers


class Mamba1Mixer(nn.Module):
    """Mamba-1 selective SSM mixer (B, L, d_model) -> (B, L, d_model).

    in_proj -> (x, z). x -> causal depthwise conv (d_conv) -> SiLU -> x_proj -> (dt_low, B, C).
    dt = softplus(dt_proj(dt_low)). Selective scan with A = -exp(A_log) (S4D-real init,
    A_log[d, n] = log(n+1)) and D skip. Output out_proj(y * SiLU(z)).

    static=True drops x_proj and dt_proj. dt = softplus(dt_param) per channel and B, C are
    learned vectors of size d_state, so the SSM is time invariant. A residual low-rank adapter
    on x of the same size as x_proj + dt_proj keeps the parameter count matched.
    `d_inner` overrides expand * d_model (for width matching, e.g. the d_conv=1 ablation).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2,
        dt_rank: int | None = None,
        static: bool = False,
        scan: ScanMode = "parallel",
        d_inner: int | None = None,
        bias: bool = False,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_model, self.d_state, self.static = d_model, d_state, static
        self.scan: ScanMode = scan
        self.d_inner = di = d_inner if d_inner is not None else int(expand * d_model)
        self.dt_rank = R = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * di, bias=bias)
        self.conv = CausalDepthwiseConv(di, d_conv)
        if static:
            self.dt_param = nn.Parameter(_inv_softplus_dt(di, dt_min, dt_max))
            self.B_param = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))
            self.C_param = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))
            removed = di * (R + 2 * d_state) + R * di + di
            added = di + 2 * d_state
            self.adapter = _StaticAdapter(di, static_adapter_rank(removed - added, di))
        else:
            self.x_proj = nn.Linear(di, R + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(R, di, bias=True)
            std = R**-0.5
            nn.init.uniform_(self.dt_proj.weight, -std, std)
            with torch.no_grad():
                self.dt_proj.bias.copy_(_inv_softplus_dt(di, dt_min, dt_max))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(di, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d_model, bias=bias)

    @property
    def conv1d(self) -> nn.Conv1d | None:
        """The depthwise conv (None when d_conv == 1), named as in mambapy."""
        return self.conv.conv

    def ssm_inputs(self, u: torch.Tensor):
        """u (B, L, d_model) -> x, z, dt (B, L, d_inner) and B, C (B, L, d_state)."""
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = F.silu(self.conv(x))
        b, L, _ = x.shape
        if self.static:
            x = self.adapter(x)
            dt = F.softplus(self.dt_param).expand(b, L, self.d_inner)
            Bm = self.B_param.expand(b, L, self.d_state)
            Cm = self.C_param.expand(b, L, self.d_state)
        else:
            dbc = self.x_proj(x)
            dt_low, Bm, Cm = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt_low))
        return x, z, dt, Bm, Cm

    def _forward(self, u: torch.Tensor, influence: bool):
        x, z, dt, Bm, Cm = self.ssm_inputs(u)
        infl = None
        with _fp32_ctx(u):
            A = -torch.exp(self.A_log.float())
            y = selective_scan(
                x.float(), dt.float(), A, Bm.float(), Cm.float(), self.D.float(), self.scan
            )
            if influence:
                # weight of x_s on y_{L-1}: sum_n C_{L-1,n} exp(sum_{k>s} dt_k A_n) dt_s B_sn
                dtf = dt.float()
                decay = torch.exp(_suffix_sum(dtf.unsqueeze(-1) * A))  # (b, L, d, N)
                w = decay * (Bm.float() * Cm.float()[:, -1:]).unsqueeze(2)
                infl = (w.sum(-1) * dtf).abs().mean(-1).detach()  # (b, L)
        y = y.to(z.dtype) * F.silu(z)
        return self.out_proj(y), dt, infl

    def forward_diag(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output (B, L, d_model), dt (B, L, d_inner))."""
        out, dt, _ = self._forward(u, influence=False)
        return out, dt

    def forward_state_diag(self, u: torch.Tensor):
        """Returns (output, dt (B, L, d_inner), influence (B, L)). influence[b, s] is the
        mean over channels of |weight of x_s on the last scan output y_{L-1}|."""
        return self._forward(u, influence=True)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.forward_diag(u)[0]


class Mamba2Mixer(nn.Module):
    """Mamba-2 (SSD) mixer (B, L, d_model) -> (B, L, d_model), one group of B and C.

    in_proj -> [z, x, B, C, dt]. [x, B, C] -> causal depthwise conv (d_conv) -> SiLU.
    dt = softplus(dt + dt_bias) per head. A = -exp(A_log) is a scalar per head, initialised
    uniform in [1, 16] as in the reference. y = SSD(x, dt, A, B, C) + D x per head, then the
    gated RMSNorm RMSNorm(y * SiLU(z)) (norm_before_gate=False, reference default) or plain
    y * SiLU(z) when gated_norm=False, then out_proj.

    static=True: in_proj -> [z, x] and the conv covers x only. dt = softplus(dt_param) per
    head. A (S4D-real init, linspace over A_init_range), B and C are learned (H, d_state)
    parameters, so every head keeps d_state distinct decay modes. A scalar A per head with
    shared fixed B, C would collapse each head to one exponential moving average (state rank
    1), which would confound selectivity with state size. The output is the exact
    convolution form of `ssd_static`. A residual low-rank adapter on x replaces the removed
    in_proj columns and conv channels (parameter match).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2,
        head_dim: int = 32,
        static: bool = False,
        gated_norm: bool = True,
        d_inner: int | None = None,
        bias: bool = False,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        A_init_range: tuple[float, float] = (1.0, 16.0),
    ) -> None:
        super().__init__()
        di = d_inner if d_inner is not None else int(expand * d_model)
        if di % head_dim:
            raise ValueError(f"d_inner {di} is not a multiple of head_dim {head_dim}")
        self.d_model, self.d_inner, self.d_state = d_model, di, d_state
        self.head_dim, self.n_heads = head_dim, di // head_dim
        self.static, self.gated_norm = static, gated_norm
        H, N = self.n_heads, d_state
        if static:
            self.in_proj = nn.Linear(d_model, 2 * di, bias=bias)
            self.conv = CausalDepthwiseConv(di, d_conv)
            self.dt_param = nn.Parameter(_inv_softplus_dt(H, dt_min, dt_max))
            # C.B per head has std about 1/sqrt(N), like the selective C_t.B_s at init
            self.B_param = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
            self.C_param = nn.Parameter(torch.randn(H, N) / math.sqrt(N))
            A = torch.linspace(*A_init_range, N).repeat(H, 1)  # (H, N), S4D-real
            # in_proj columns for B, C, dt, conv channels for B, C, and dt_bias
            removed = (d_model + int(bias)) * (2 * N + H) + H
            removed += 2 * N * (d_conv + 1) if d_conv > 1 else 0
            added = H + 2 * H * N + H * (N - 1)  # dt, B, C, and A beyond one per head
            self.adapter = _StaticAdapter(di, static_adapter_rank(removed - added, di))
        else:
            self.in_proj = nn.Linear(d_model, 2 * di + 2 * N + H, bias=bias)
            self.conv = CausalDepthwiseConv(di + 2 * N, d_conv)
            self.dt_bias = nn.Parameter(_inv_softplus_dt(H, dt_min, dt_max))
            A = torch.empty(H).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(H))
        self.norm = RMSNorm(di) if gated_norm else None
        self.out_proj = nn.Linear(di, d_model, bias=bias)

    def ssm_inputs(self, u: torch.Tensor):
        """u (B, L, d_model) -> x (B, L, d_inner), z (B, L, d_inner), dt (B, L, H), B, C.

        B and C are (B, L, N) when selective and (B, L, H, N) when static.
        """
        di, N, H = self.d_inner, self.d_state, self.n_heads
        b, L, _ = u.shape
        if self.static:
            z, x = self.in_proj(u).chunk(2, dim=-1)
            x = self.adapter(F.silu(self.conv(x)))
            dt = F.softplus(self.dt_param).expand(b, L, H)
            Bm = self.B_param.expand(b, L, H, N)
            Cm = self.C_param.expand(b, L, H, N)
        else:
            z, xbc, dt = torch.split(self.in_proj(u), [di, di + 2 * N, H], dim=-1)
            dt = F.softplus(dt + self.dt_bias)
            x, Bm, Cm = torch.split(F.silu(self.conv(xbc)), [di, N, N], dim=-1)
        return x, z, dt, Bm, Cm

    def _forward(self, u: torch.Tensor, influence: bool):
        x, z, dt, Bm, Cm = self.ssm_inputs(u)
        b, L, _ = x.shape
        infl = None
        with _fp32_ctx(u):
            A = -torch.exp(self.A_log.float())
            xh = x.float().reshape(b, L, self.n_heads, self.head_dim)
            if self.static:
                dt0 = F.softplus(self.dt_param.float())
                Bp, Cp = self.B_param.float(), self.C_param.float()
                y = ssd_static(xh, dt0, A, Bp, Cp, self.D.float())
                if influence:
                    K = lti_kernel(dt0, A, Bp, Cp, L)[:, -1]  # (H, S)
                    infl = K.abs().mean(0).expand(b, L).detach()
            else:
                y = ssd_quadratic(xh, dt.float(), A, Bm.float(), Cm.float(), self.D.float())
                if influence:
                    # weight of x_s on y_{L-1}: exp(sum_{k>s} dt_k A) (C_{L-1}.B_s) dt_s
                    dtf = dt.float()
                    decay = torch.exp(_suffix_sum(dtf * A))  # (b, L, H)
                    cb = (Bm.float() * Cm.float()[:, -1:]).sum(-1, keepdim=True)  # (b, L, 1)
                    infl = (decay * cb * dtf).abs().mean(-1).detach()
        y = y.reshape(b, L, self.d_inner).to(z.dtype) * F.silu(z)
        if self.norm is not None:
            y = self.norm(y)
        return self.out_proj(y), dt, infl

    def forward_diag(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output (B, L, d_model), dt (B, L, n_heads))."""
        out, dt, _ = self._forward(u, influence=False)
        return out, dt

    def forward_state_diag(self, u: torch.Tensor):
        """Returns (output, dt (B, L, n_heads), influence (B, L)). influence[b, s] is the
        mean over heads of |weight of x_s on the last SSM output y_{L-1}|."""
        return self._forward(u, influence=True)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.forward_diag(u)[0]


def build_mixer(kind: MixerKind, d_model: int, **kw) -> Mamba1Mixer | Mamba2Mixer:
    """kw goes to the mixer. head_dim and gated_norm are dropped for mamba1, scan and dt_rank
    for mamba2, so one config dict serves both kinds."""
    if kind == "mamba1":
        kw = {k: v for k, v in kw.items() if k not in ("head_dim", "gated_norm")}
        return Mamba1Mixer(d_model, **kw)
    if kind == "mamba2":
        kw = {k: v for k, v in kw.items() if k not in ("scan", "dt_rank")}
        return Mamba2Mixer(d_model, **kw)
    raise ValueError(f"unknown mixer kind {kind!r}")


def mixer_param_counts(kind: MixerKind, d_model: int, **kw) -> dict[str, float]:
    """Trainable parameters of the selective and static mixers for the same config.

    Returns {"selective": n, "static": m, "rel_diff": (m - n) / n}. Plan Section 3 requires
    |rel_diff| <= 0.05 for the M2 control.
    """
    kw = {k: v for k, v in kw.items() if k != "static"}
    n = count_params(build_mixer(kind, d_model, static=False, **kw))
    m = count_params(build_mixer(kind, d_model, static=True, **kw))
    return {"selective": n, "static": m, "rel_diff": (m - n) / n}


# ---------------------------------------------------------------- blocks


class MambaBlock(nn.Module):
    """Pre-norm residual block x + Dropout(Mixer(RMSNorm(x))) on (B, L, d_model)."""

    def __init__(
        self,
        d_model: int,
        kind: MixerKind = "mamba2",
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2,
        head_dim: int = 32,
        static: bool = False,
        dropout: float = 0.0,
        **mixer_kw,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.norm = RMSNorm(d_model)
        self.mixer = build_mixer(
            kind,
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            static=static,
            **mixer_kw,
        )
        self.drop = nn.Dropout(dropout)

    def forward_diag(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x + mixer output, mean dt over channels/heads per step (B, L), detached)."""
        y, dt = self.mixer.forward_diag(self.norm(x))
        return x + self.drop(y), dt.detach().float().mean(-1)

    def forward_state_diag(self, x: torch.Tensor):
        """Returns (x + mixer output, mean dt per step (B, L), state influence (B, L))."""
        y, dt, infl = self.mixer.forward_state_diag(self.norm(x))
        return x + self.drop(y), dt.detach().float().mean(-1), infl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mixer(self.norm(x)))


def _reverse_time(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[1])


class BiMamba(nn.Module):
    """Stack of MambaBlocks over a token sequence (B, L, d_model) -> (B, L, d_model).

    direction="bi": a forward stack on x and a separate reverse stack on x flipped in time.
    The reverse output is flipped back, each stack output gets a final RMSNorm, and
    Linear(2*d_model, d_model) mixes the concatenation (Vision Mamba style).
    direction="forward": only the forward stack and its final RMSNorm, so the output is
    causal (output t depends on inputs <= t). Same output width, fewer parameters.

    forward(x, diagnostics=False) returns (y, diag). diag is {} unless diagnostics=True, then
    it holds "dt_mean" (B, n_directions, n_layers, L) in original time order, "influence"
    (same shape, the mean |weight| of each input position on the last scan output of that
    layer and direction, so for the reverse stack on the output at original position 0),
    "directions", "static" and "notes". `features(x)` returns y alone (for ONNX export).
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        direction: Literal["bi", "forward"] = "bi",
        kind: MixerKind = "mamba2",
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2,
        head_dim: int = 32,
        static: bool = False,
        dropout: float = 0.0,
        final_norm: bool = True,
        **mixer_kw,
    ) -> None:
        super().__init__()
        if direction not in ("bi", "forward"):
            raise ValueError(f"unknown direction {direction!r}")
        self.direction, self.static, self.n_layers = direction, static, n_layers

        def stack() -> nn.ModuleList:
            return nn.ModuleList(
                MambaBlock(
                    d_model,
                    kind=kind,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    head_dim=head_dim,
                    static=static,
                    dropout=dropout,
                    **mixer_kw,
                )
                for _ in range(n_layers)
            )

        def norm() -> nn.Module:
            return RMSNorm(d_model) if final_norm else nn.Identity()

        self.fwd = stack()
        self.fwd_norm = norm()
        if direction == "bi":
            self.bwd = stack()
            self.bwd_norm = norm()
            self.proj = nn.Linear(2 * d_model, d_model)

    @staticmethod
    def _run(blocks: nn.ModuleList, x: torch.Tensor, diag: bool):
        dts, infls = [], []
        for blk in blocks:
            if diag:
                x, dt, infl = cast(MambaBlock, blk).forward_state_diag(x)
                dts.append(dt)
                infls.append(infl)
            else:
                x = blk(x)
        return x, dts, infls

    def _forward(self, x: torch.Tensor, diag: bool):
        f, dts_f, inf_f = self._run(self.fwd, x, diag)
        f = self.fwd_norm(f)
        if self.direction == "forward":
            return f, [dts_f], [inf_f]
        r, dts_r, inf_r = self._run(self.bwd, _reverse_time(x), diag)
        r = _reverse_time(self.bwd_norm(r))
        dts_r = [_reverse_time(d) for d in dts_r]
        inf_r = [_reverse_time(d) for d in inf_r]
        return self.proj(torch.cat([f, r], dim=-1)), [dts_f, dts_r], [inf_f, inf_r]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x, diag=False)[0]

    def forward(self, x: torch.Tensor, diagnostics: bool = False) -> tuple[torch.Tensor, dict]:
        y, dts, infls = self._forward(x, diagnostics)
        if not diagnostics:
            return y, {}
        diag: dict = {
            "dt_mean": torch.stack([torch.stack(d, dim=1) for d in dts], dim=1),
            "influence": torch.stack([torch.stack(d, dim=1) for d in infls], dim=1),
            "directions": ["forward", "reverse"][: len(dts)],
            "static": self.static,
            "notes": (
                "static-state control: dt, B and C are learned time-invariant parameters, "
                "so dt_mean is constant over time"
                if self.static
                else "selective: dt, B and C are computed from the input at every step"
            ),
        }
        return y, diag
