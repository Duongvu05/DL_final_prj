"""Tests for the pure-PyTorch Mamba blocks (src/dl_final_prj/mamba.py). CPU only."""

import importlib.util

import pytest
import torch

from dl_final_prj.mamba import (
    BiMamba,
    Mamba1Mixer,
    Mamba2Mixer,
    MambaBlock,
    RMSNorm,
    count_params,
    lti_kernel,
    mixer_param_counts,
    segsum,
    selective_scan_loop,
    selective_scan_parallel,
    ssd_quadratic,
    ssd_recurrent,
    ssd_static,
    ssd_static_recurrent,
)

KINDS = ("mamba1", "mamba2")


def scan_inputs(b=2, L=37, d=6, N=4, dt_scale=1.0, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, L, d, generator=g, dtype=dtype)
    dt = torch.nn.functional.softplus(torch.randn(b, L, d, generator=g, dtype=dtype)) * dt_scale
    A = -torch.exp(torch.randn(d, N, generator=g, dtype=dtype))
    B = torch.randn(b, L, N, generator=g, dtype=dtype)
    C = torch.randn(b, L, N, generator=g, dtype=dtype)
    D = torch.randn(d, generator=g, dtype=dtype)
    return x, dt, A, B, C, D


def naive_scan(x, dt, A, B, C, D):
    """Elementwise triple loop, independent of both implementations."""
    b, L, d = x.shape
    N = A.shape[1]
    y = torch.zeros_like(x)
    for i in range(b):
        for c in range(d):
            h = torch.zeros(N, dtype=x.dtype)
            for t in range(L):
                h = torch.exp(dt[i, t, c] * A[c]) * h + dt[i, t, c] * B[i, t] * x[i, t, c]
                y[i, t, c] = (h * C[i, t]).sum() + D[c] * x[i, t, c]
    return y


# ------------------------------------------------------------------ Mamba-1 scan


def test_mamba1_loop_scan_matches_naive():
    args = scan_inputs(b=2, L=9, d=3, N=4)
    torch.testing.assert_close(selective_scan_loop(*args), naive_scan(*args))


@pytest.mark.parametrize("L", [1, 2, 7, 49, 64, 168])
def test_mamba1_parallel_scan_equals_loop(L):
    args = scan_inputs(L=L)
    torch.testing.assert_close(selective_scan_parallel(*args), selective_scan_loop(*args))


def test_mamba1_parallel_scan_is_stable_with_large_decay():
    # sum of dt*A over 168 steps reaches about -1e4. exp(-cumsum) would overflow, our scan not.
    args = scan_inputs(L=168, dt_scale=20.0, dtype=torch.float32)
    yp, yl = selective_scan_parallel(*args), selective_scan_loop(*args)
    assert torch.isfinite(yp).all()
    torch.testing.assert_close(yp, yl, rtol=1e-4, atol=1e-4)


def test_mamba1_scan_gradients_agree():
    args = [a.clone().requires_grad_(True) for a in scan_inputs(L=21)]
    g_par = torch.autograd.grad(selective_scan_parallel(*args).square().sum(), args)
    g_loop = torch.autograd.grad(selective_scan_loop(*args).square().sum(), args)
    for a, b in zip(g_par, g_loop):
        torch.testing.assert_close(a, b)


def _mambapy_block(m: Mamba1Mixer, pscan: bool):
    mp = pytest.importorskip("mambapy.mamba")
    cfg = mp.MambaConfig(
        d_model=m.d_model,
        n_layers=1,
        d_state=m.d_state,
        expand_factor=m.d_inner // m.d_model,
        d_conv=m.conv.kernel,
        pscan=pscan,
    )
    ref = mp.MambaBlock(cfg)
    sd = {k.replace("conv.conv.", "conv1d."): v for k, v in m.state_dict().items()}
    ref.load_state_dict(sd, strict=True)
    return ref


@pytest.mark.parametrize("pscan", [True, False])
@pytest.mark.parametrize("scan", ["parallel", "loop"])
def test_mamba1_mixer_matches_mambapy(pscan, scan):
    torch.manual_seed(0)
    m = Mamba1Mixer(32, d_state=8, d_conv=4, expand=2, scan=scan)
    ref = _mambapy_block(m, pscan)
    u = torch.randn(3, 49, 32)
    with torch.no_grad():
        torch.testing.assert_close(m(u), ref(u), rtol=1e-4, atol=1e-5)


def test_mamba1_scan_matches_mambapy_selective_scan():
    """Call mambapy's own scans with the same A, B, C, delta, D tensors."""
    torch.manual_seed(1)
    m = Mamba1Mixer(16, d_state=4, d_conv=4, expand=2)
    x, dt, A, B, C, D = scan_inputs(b=2, L=29, d=m.d_inner, N=4, dtype=torch.float32)
    ref_par = _mambapy_block(m, pscan=True)
    ref_seq = _mambapy_block(m, pscan=False)
    ours = selective_scan_loop(x, dt, A, B, C, D)
    torch.testing.assert_close(ref_seq.selective_scan_seq(x, dt, A, B, C, D), ours)
    torch.testing.assert_close(
        ref_par.selective_scan(x, dt, A, B, C, D), ours, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(selective_scan_parallel(x, dt, A, B, C, D), ours)


# ------------------------------------------------------------------ Mamba-2 SSD


def test_segsum_matches_definition():
    a = torch.randn(2, 3, 6, dtype=torch.float64)
    S = segsum(a)
    for i in range(6):
        for j in range(6):
            if j <= i:
                torch.testing.assert_close(S[..., i, j], a[..., j + 1 : i + 1].sum(-1))
            else:
                assert torch.isneginf(S[..., i, j]).all()


@pytest.mark.parametrize("L", [1, 5, 49, 168])
def test_mamba2_quadratic_equals_recurrence(L):
    g = torch.Generator().manual_seed(L)
    b, H, P, N = 2, 3, 4, 5
    x = torch.randn(b, L, H, P, generator=g, dtype=torch.float64)
    dt = torch.nn.functional.softplus(torch.randn(b, L, H, generator=g, dtype=torch.float64))
    A = -torch.rand(H, generator=g, dtype=torch.float64) * 16
    B = torch.randn(b, L, N, generator=g, dtype=torch.float64)
    C = torch.randn(b, L, N, generator=g, dtype=torch.float64)
    D = torch.randn(H, generator=g, dtype=torch.float64)
    torch.testing.assert_close(ssd_quadratic(x, dt, A, B, C, D), ssd_recurrent(x, dt, A, B, C, D))


def test_mamba2_ssd_matches_mamba1_scan_with_shared_decay():
    """SSD with head_dim 1 is the Mamba-1 scan with A[d, n] = A_head[d] for all n."""
    x, dt, A1, B, C, D = scan_inputs(L=23, d=4, N=3)
    Ah = A1[:, 0]
    y1 = selective_scan_loop(x, dt, Ah.unsqueeze(-1).expand(4, 3), B, C, D)
    y2 = ssd_quadratic(x.unsqueeze(-1), dt, Ah, B, C, D).squeeze(-1)
    torch.testing.assert_close(y1, y2)


def test_mamba2_quadratic_backward_is_finite():
    x = torch.randn(2, 40, 2, 4, requires_grad=True)
    dt = torch.rand(2, 40, 2, requires_grad=True)
    y = ssd_quadratic(
        x, dt, -torch.ones(2), torch.randn(2, 40, 3), torch.randn(2, 40, 3), torch.ones(2)
    )
    y.sum().backward()
    assert x.grad is not None and dt.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(dt.grad).all()


@pytest.mark.parametrize("L", [1, 5, 49, 168])
def test_static_ssd_equals_recurrence(L):
    g = torch.Generator().manual_seed(0)
    b, H, P, N = 2, 3, 4, 5
    x = torch.randn(b, L, H, P, generator=g, dtype=torch.float64)
    dt = torch.nn.functional.softplus(torch.randn(H, generator=g, dtype=torch.float64))
    A = -torch.exp(torch.randn(H, N, generator=g, dtype=torch.float64))
    B, C = (torch.randn(H, N, generator=g, dtype=torch.float64) for _ in range(2))
    D = torch.randn(H, generator=g, dtype=torch.float64)
    torch.testing.assert_close(
        ssd_static(x, dt, A, B, C, D), ssd_static_recurrent(x, dt, A, B, C, D)
    )


def test_static_mamba2_state_is_full_rank():
    """The static control keeps d_state decay modes per head, not one EMA per head."""
    torch.manual_seed(0)
    m = Mamba2Mixer(96, d_state=16, head_dim=32, static=True).double()
    H = m.n_heads
    assert m.A_log.shape == m.B_param.shape == m.C_param.shape == (H, 16)
    A = -torch.exp(m.A_log)
    assert len(torch.unique(A[0])) == 16
    dt = torch.nn.functional.softplus(m.dt_param)
    k = lti_kernel(dt, A, m.B_param, m.C_param, 64)[:, -1].flip(-1)  # (H, lag)
    for h in range(H):
        hankel = torch.stack([k[h, i : i + 32] for i in range(32)])
        s = torch.linalg.svdvals(hankel)
        assert s[1] / s[0] > 1e-8  # a single exponential has Hankel rank 1
    # reference: one scalar decay per head is rank 1
    k1 = lti_kernel(dt, A[:, :1], m.B_param[:, :1], m.C_param[:, :1], 64)[:, -1].flip(-1)
    s1 = torch.linalg.svdvals(torch.stack([k1[0, i : i + 32] for i in range(32)]))
    assert s1[1] / s1[0] < 1e-10


# ------------------------------------------------------------------ mixers and blocks


def make_mixer(kind, static=False, d_conv=4, d_model=16, **kw):
    if kind == "mamba1":
        return Mamba1Mixer(d_model, d_state=8, d_conv=d_conv, static=static, **kw)
    return Mamba2Mixer(d_model, d_state=8, d_conv=d_conv, head_dim=8, static=static, **kw)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("d_conv", [1, 4])
def test_mixer_is_causal(kind, static, d_conv):
    torch.manual_seed(0)
    m = make_mixer(kind, static, d_conv).eval()
    u = torch.randn(2, 30, 16)
    u2 = u.clone()
    u2[:, 17:] = torch.randn(2, 13, 16)
    with torch.no_grad():
        y, y2 = m(u), m(u2)
    torch.testing.assert_close(y[:, :17], y2[:, :17])
    assert not torch.allclose(y[:, 17:], y2[:, 17:])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("d_conv", [1, 4])
def test_mixer_shapes_and_backward(kind, static, d_conv):
    torch.manual_seed(0)
    m = make_mixer(kind, static, d_conv)
    u = torch.randn(3, 11, 16, requires_grad=True)
    y, dt = m.forward_diag(u)
    assert y.shape == (3, 11, 16)
    assert dt.shape == (3, 11, m.d_inner if kind == "mamba1" else m.n_heads)
    y.square().mean().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_d_conv_1_has_no_conv():
    assert Mamba1Mixer(16, d_conv=1).conv1d is None
    assert Mamba2Mixer(16, d_conv=1, head_dim=8).conv.conv is None
    assert isinstance(Mamba1Mixer(16, d_conv=4).conv1d, torch.nn.Conv1d)


def test_mamba1_d_conv_1_matches_mambapy():
    torch.manual_seed(0)
    m = Mamba1Mixer(16, d_state=4, d_conv=1)
    mp = pytest.importorskip("mambapy.mamba")
    ref = mp.MambaBlock(mp.MambaConfig(d_model=16, n_layers=1, d_state=4, d_conv=1))
    sd = m.state_dict()
    sd["conv1d.weight"] = torch.ones(m.d_inner, 1, 1)
    sd["conv1d.bias"] = torch.zeros(m.d_inner)
    ref.load_state_dict(sd, strict=True)
    u = torch.randn(2, 13, 16)
    with torch.no_grad():
        torch.testing.assert_close(m(u), ref(u), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_static_selection_ignores_input(kind):
    torch.manual_seed(0)
    u = torch.randn(2, 12, 16, requires_grad=True)
    for static in (False, True):
        m = make_mixer(kind, static)
        x, _, dt, B, C = m.ssm_inputs(u)
        for t in (dt, B, C):
            (g,) = torch.autograd.grad(t.sum(), u, allow_unused=True, retain_graph=True)
            dependent = g is not None and bool(g.abs().sum() > 0)
            assert dependent is (not static)
        # the value path still depends on the input
        (gx,) = torch.autograd.grad(x.sum(), u)
        assert gx.abs().sum() > 0
        if static:
            assert torch.equal(dt[0, 0], dt[1, -1]) and torch.equal(B[0, 0], B[1, 5])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("static", [False, True])
def test_state_influence_is_the_last_output_weight(kind, static):
    """influence[b, s] = mean over channels/heads of |d y_{L-1} / d x_s| without the D skip."""
    torch.manual_seed(0)
    m = make_mixer(kind, static)
    u = torch.randn(2, 12, 16)
    with torch.no_grad():
        _, _, infl = m.forward_state_diag(u)
        x, _, dt, B, C = m.ssm_inputs(u)
    A = -torch.exp(m.A_log.detach())
    x = x.detach().requires_grad_(True)
    if kind == "mamba1":
        y = selective_scan_loop(x, dt, A, B, C, torch.zeros(m.d_inner))
        (g,) = torch.autograd.grad(y[:, -1].sum(), x)  # (b, L, d)
    else:
        xh = x.reshape(2, 12, m.n_heads, m.head_dim)
        if static:
            dt0 = torch.nn.functional.softplus(m.dt_param.detach())
            Bp, Cp = m.B_param.detach(), m.C_param.detach()
            y = ssd_static(xh, dt0, A, Bp, Cp, torch.zeros(m.n_heads))
        else:
            y = ssd_quadratic(xh, dt, A, B, C, torch.zeros(m.n_heads))
        (g,) = torch.autograd.grad(y[:, -1].sum(), x)
        g = g.reshape(2, 12, m.n_heads, m.head_dim)[..., 0]  # (b, L, H)
    assert infl.shape == (2, 12)
    torch.testing.assert_close(infl, g.abs().mean(-1), rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize(
    ("kind", "d_model", "kw"),
    [
        ("mamba1", 256, {}),
        ("mamba1", 96, {}),
        ("mamba1", 256, {"d_conv": 1}),
        ("mamba1", 64, {"d_state": 32}),
        ("mamba2", 96, {"head_dim": 32}),
        ("mamba2", 96, {"head_dim": 16}),
        ("mamba2", 96, {"head_dim": 64}),
        ("mamba2", 96, {"head_dim": 32, "d_conv": 1}),
        ("mamba2", 256, {"head_dim": 64}),
    ],
)
def test_static_parameter_count_within_5_percent(kind, d_model, kw):
    r = mixer_param_counts(kind, d_model, **kw)
    assert abs(r["rel_diff"]) <= 0.05, r
    # and the equal-size adapter actually carries the capacity (not a stray unused module)
    assert abs(r["rel_diff"]) <= 0.01, r


@pytest.mark.parametrize("kind", KINDS)
def test_block_is_residual_prenorm(kind):
    torch.manual_seed(0)
    blk = MambaBlock(16, kind=kind, head_dim=8, dropout=0.2).eval()
    x = torch.randn(2, 9, 16)
    with torch.no_grad():
        torch.testing.assert_close(blk(x), x + blk.mixer(blk.norm(x)))
        y, dt = blk.forward_diag(x)
    torch.testing.assert_close(y, blk(x))
    assert dt.shape == (2, 9)


def test_rmsnorm():
    n = RMSNorm(4)
    x = torch.tensor([[3.0, -3.0, 3.0, -3.0]])
    torch.testing.assert_close(n(x), torch.tensor([[1.0, -1.0, 1.0, -1.0]]), atol=1e-5, rtol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_bimamba_sees_both_directions(kind):
    torch.manual_seed(0)
    bi = BiMamba(16, n_layers=2, direction="bi", kind=kind, head_dim=8).eval()
    fw = BiMamba(16, n_layers=2, direction="forward", kind=kind, head_dim=8).eval()
    u = torch.randn(2, 20, 16)
    future, past = u.clone(), u.clone()
    future[:, 15:] += 1.0
    past[:, :5] += 1.0
    with torch.no_grad():
        y, _ = bi(u)
        yf, _ = bi(future)
        yp, _ = bi(past)
        f0, _ = fw(u)
        ff, _ = fw(future)
    assert y.shape == f0.shape == (2, 20, 16)
    assert not torch.allclose(y[:, :15], yf[:, :15])  # early tokens see the future
    assert not torch.allclose(y[:, 5:], yp[:, 5:])  # late tokens see the past
    torch.testing.assert_close(f0[:, :15], ff[:, :15])  # forward-only stays causal


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("direction", ["bi", "forward"])
def test_bimamba_diagnostics_and_backward(kind, static, direction):
    torch.manual_seed(0)
    m = BiMamba(16, n_layers=2, direction=direction, kind=kind, head_dim=8, static=static)
    u = torch.randn(3, 14, 16, requires_grad=True)
    y, diag = m(u, diagnostics=True)
    n_dir = 2 if direction == "bi" else 1
    assert y.shape == (3, 14, 16)
    assert diag["dt_mean"].shape == (3, n_dir, 2, 14)
    assert diag["influence"].shape == (3, n_dir, 2, 14)
    assert diag["static"] is static and len(diag["directions"]) == n_dir
    assert isinstance(diag["notes"], str)
    dtm = diag["dt_mean"]
    if static:
        torch.testing.assert_close(dtm, dtm[..., :1].expand_as(dtm))
    else:
        assert dtm.std(-1).max() > 0
    torch.testing.assert_close(m.features(u), y)
    y2, d2 = m(u)
    assert d2 == {}
    torch.testing.assert_close(y2, y)
    y.sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    assert all(p.grad is not None for p in m.parameters())


def test_bimamba_reverse_stack_runs_on_flipped_input():
    torch.manual_seed(0)
    m = BiMamba(8, n_layers=1, direction="bi", kind="mamba1", final_norm=False).eval()
    u = torch.randn(1, 6, 8)
    with torch.no_grad():
        f = m.fwd[0](u)
        r = m.bwd[0](u.flip(1)).flip(1)
        torch.testing.assert_close(m.features(u), m.proj(torch.cat([f, r], -1)))


def test_bf16_autocast_forward_runs_scan_in_fp32():
    torch.manual_seed(0)
    m = BiMamba(16, n_layers=1, kind="mamba2", head_dim=8)
    u = torch.randn(2, 10, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y, _ = m(u)
    ref, _ = m(u)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), ref, rtol=0.1, atol=0.1)


def test_param_counts_reported_for_plan_configs():
    blk2 = MambaBlock(96, kind="mamba2", d_state=16, d_conv=4, expand=2, head_dim=32)
    blk1 = MambaBlock(256, kind="mamba1", d_state=16, d_conv=4, expand=2)
    assert count_params(blk2) == count_params(blk2.mixer) + 96
    assert count_params(blk1) == count_params(blk1.mixer) + 256


# ------------------------------------------------------------------ export


class _Features(torch.nn.Module):
    def __init__(self, m: BiMamba) -> None:
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.features(x)


@pytest.mark.parametrize("kind", KINDS)
def test_torch_export_traces_without_graph_breaks(kind):
    torch.manual_seed(0)
    m = _Features(BiMamba(16, n_layers=1, direction="bi", kind=kind, head_dim=8)).eval()
    x = torch.randn(2, 49, 16)
    ep = torch.export.export(m, (x,))
    with torch.no_grad():
        torch.testing.assert_close(ep.module()(x), m(x))


@pytest.mark.parametrize("kind", KINDS)
def test_onnx_export_opset17(kind, tmp_path):
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("onnx is not installed, torch.onnx.export needs it")
    torch.manual_seed(0)
    m = _Features(BiMamba(16, n_layers=1, direction="bi", kind=kind, head_dim=8)).eval()
    x = torch.randn(2, 49, 16)
    path = tmp_path / f"bimamba_{kind}.onnx"
    torch.onnx.export(m, (x,), str(path), opset_version=17, dynamo=False)
    assert path.stat().st_size > 0
    if importlib.util.find_spec("onnxruntime") is None:
        return
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
    with torch.no_grad():
        torch.testing.assert_close(torch.from_numpy(out), m(x), rtol=1e-4, atol=1e-4)
