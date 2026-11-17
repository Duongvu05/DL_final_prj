"""Phase 2 Task 1/2 Mamba forecasters (src/dl_final_prj/ts_mamba.py, configs/phase2.yaml)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dl_final_prj.electricity import ElectricityData
from dl_final_prj.mamba import Mamba2Mixer
from dl_final_prj.ts_mamba import (
    M4_CHANGE,
    N_FEATURES,
    PARAM_TOLERANCE,
    MambaGRUForecaster,
    MambaSeq2SeqForecaster,
    build,
    matched_widths,
    n_params,
    phase1_budget,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "phase2.yaml").read_text())
T2 = (1, 2, 3, 4, 5)
T1_MODELS = [f"t1_m{i}" for i in range(5)]
T2_MODELS = [f"t2_m{i}" for i in range(5)]


def mcfg(name: str) -> dict:
    return CFG[f"task{name[1]}"]["models"][name]


def horizon(name: str) -> int:
    return len(CFG[f"task{name[1]}"]["offsets"])


def model_for(name: str, **override) -> torch.nn.Module:
    m = mcfg(name)
    return build(m["kind"], {**m["args"], **override}, horizon(name))


@pytest.fixture
def batches(synthetic_npz):
    d1 = ElectricityData(synthetic_npz, 168, (2,))
    d2 = ElectricityData(synthetic_npz, 168, T2)
    b1 = d1.batch(torch.as_tensor(d1.origins["train"][:3]))
    b2 = d2.batch(torch.as_tensor(d2.origins["train"][:3]))
    return {"1": b1, "2": b2}


# --- config and parameter matching ----------------------------------------------


def test_config_copies_phase1_task_settings():
    p1 = yaml.safe_load((ROOT / "configs" / "phase1.yaml").read_text())
    assert CFG["phase"] == "phase2" and CFG["seeds"] == [0, 1, 2]
    for t in ("task1", "task2"):
        # eval_batch_size is smaller in phase 2 (memory of the quadratic SSD form only)
        a = {k: v for k, v in CFG[t].items() if k not in ("models", "eval_batch_size")}
        b = {k: v for k, v in p1[t].items() if k not in ("models", "eval_batch_size")}
        assert a == b
        assert CFG[t]["eval_batch_size"] == 512
    assert sorted(CFG["task1"]["models"]) == T1_MODELS
    assert sorted(CFG["task2"]["models"]) == T2_MODELS


@pytest.mark.parametrize("task", [1, 2])
def test_matched_models_within_tolerance(task):
    ref = n_params(mcfg(f"t{task}_m0")["kind"], mcfg(f"t{task}_m0")["args"], horizon(f"t{task}"))
    for k in (1, 2, 3, 4):
        name = f"t{task}_m{k}"
        n = n_params(mcfg(name)["kind"], mcfg(name)["args"], horizon(name))
        assert abs(n - ref) / ref <= PARAM_TOLERANCE, (name, n, ref)


def test_config_widths_are_the_searched_ones():
    w = matched_widths(CFG)
    for t in (1, 2):
        m1 = mcfg(f"t{t}_m1")["args"]
        m3 = mcfg(f"t{t}_m3")["args"]
        assert m1["enc_hidden"] == w[f"task{t}"]["m1_enc_hidden"]
        assert m3["ssm_head_dim"] == w[f"task{t}"]["m3_ssm_head_dim"]
        assert m3["d_inner"] == w[f"task{t}"]["m3_d_inner"]
        assert m3["d_conv"] == 1
        m4 = mcfg(f"t{t}_m4")["args"]
        assert m4["d_inner"] == w[f"task{t}"]["m4_d_inner"]
        assert m4["ssm_head_dim"] == mcfg(f"t{t}_m0")["args"]["ssm_head_dim"]


def test_ablations_differ_from_m0_only_where_intended():
    for t, change in M4_CHANGE.items():
        m4_change = {**change, "d_inner": mcfg(f"t{t}_m4")["args"]["d_inner"]}
        m0 = mcfg(f"t{t}_m0")["args"]
        assert mcfg(f"t{t}_m2")["args"] == {**m0, "static": True}
        m3 = mcfg(f"t{t}_m3")["args"]
        width = ("d_conv", "ssm_head_dim", "d_inner")
        assert {k: v for k, v in m3.items() if k not in width} == {
            k: v for k, v in m0.items() if k not in width
        }
        m4 = {k: v for k, v in mcfg(f"t{t}_m4")["args"].items() if k != "attn_dim"}
        assert m4 == {**{k: v for k, v in m0.items() if k != "attn_dim"}, **m4_change}
        assert mcfg(f"t{t}_m1")["args"]["encoder"] == ("bigru" if t == 1 else "lstm")


def test_phase1_budget_gap_is_reported():
    """Plan 2.1 budget deviation: the config header must state the real counts."""
    p1 = yaml.safe_load((ROOT / "configs" / "phase1.yaml").read_text())
    b = phase1_budget(CFG, p1)
    assert b["task1"]["phase1_p0"] == 171_925 and b["task2"]["phase1_p0"] == 110_033
    header = (ROOT / "configs" / "phase2.yaml").read_text()
    assert "PLAN DEVIATION" in header
    for t in (1, 2):
        assert f"{b[f'task{t}']['m0']:,}" in header
        assert f"{b[f'task{t}']['phase1_p0']:,}" in header


def test_m0_follows_plan_sizes():
    m = model_for("t1_m0")
    assert isinstance(m, MambaGRUForecaster)
    assert m.in_proj.in_features == N_FEATURES and m.in_proj.out_features == 96
    mixers = [x for x in m.modules() if isinstance(x, Mamba2Mixer)]
    assert len(mixers) == 4  # 2 layers x (forward, reverse)
    assert all(x.d_state == 16 and x.d_inner == 192 and x.n_heads == 6 for x in mixers)
    assert all(x.conv.kernel == 4 for x in mixers)
    assert m.readout.num_layers == 1 and m.readout.hidden_size == 64
    assert m.head[1].in_features == 64 + 64 + 10 and m.head[1].out_features == 128
    m2 = model_for("t2_m0")
    assert isinstance(m2, MambaSeq2SeqForecaster)
    assert len([x for x in m2.modules() if isinstance(x, Mamba2Mixer)]) == 2
    assert m2.decoder.hidden_size == 64 and m2.decoder.input_size == 1 + 10 + 16 + 64
    assert m2.out.in_features == 128


# --- shapes, attention, diagnostics --------------------------------------------


@pytest.mark.parametrize("name", T1_MODELS + T2_MODELS)
def test_shapes_attention_and_backward(name, batches):
    b = batches[name[1]]
    H = horizon(name)
    m = model_for(name)
    m.train()
    p, aux = m(b, teacher_forcing=1.0)
    assert p.shape == (3, H)
    has_attn = mcfg(name)["args"]["attention"]
    assert ("attention" in aux) == has_attn
    if has_attn:
        w = aux["attention"]
        assert w.shape == ((3, 168) if H == 1 else (3, H, 168))
        torch.testing.assert_close(w.sum(-1), torch.ones(w.shape[:-1]))
    assert "dt_mean" not in aux and "state_influence" not in aux  # off by default
    p.pow(2).mean().backward()
    for n_, prm in m.named_parameters():
        assert prm.grad is not None, n_


@pytest.mark.parametrize("name", T1_MODELS + T2_MODELS)
def test_diagnostics(name, batches):
    b = batches[name[1]]
    m = model_for(name).eval()
    with torch.no_grad():
        p, aux = m(b, diagnostics=True)
        p0, _ = m(b)
    torch.testing.assert_close(p, p0)  # diagnostics do not change the prediction
    if mcfg(name)["args"]["encoder"] != "mamba":
        assert not m.has_diagnostics and "dt_mean" not in aux
        return
    assert m.has_diagnostics
    bi = mcfg(name)["args"].get("direction", "forward") == "bi"
    dt = aux["dt_mean"]
    assert dt.shape == (3, 2 if bi else 1, 2, 168)
    assert aux["dt_directions"] == (["forward", "reverse"] if bi else ["forward"])
    assert (dt > 0).all()
    static = mcfg(name)["args"]["static"]
    assert aux["dt_static"] == static
    # static dt is one value per layer/direction, the selective dt varies over time
    spread = (dt - dt[:1, :, :, :1]).abs().amax()
    assert (spread < 1e-6) if static else (spread > 1e-6)
    # the state influence varies over input positions for both M0 and the static M2
    infl = aux["state_influence"]
    assert infl.shape == dt.shape and torch.isfinite(infl).all() and (infl >= 0).all()
    assert (infl.amax(-1) > infl.amin(-1)).all()
    if static:  # input independent
        torch.testing.assert_close(infl, infl[:1].expand_as(infl))


@pytest.mark.parametrize("name", ["t1_m2", "t2_m2"])
def test_static_selection_ignores_the_input(name, batches):
    b = batches[name[1]]
    m = model_for(name).eval()
    mixers = [x for x in m.modules() if isinstance(x, Mamba2Mixer)]
    assert mixers and all(x.static for x in mixers)
    u1 = torch.randn(2, 168, 96)
    u2 = torch.randn(2, 168, 96) * 3 + 1
    for mx in mixers:
        x1, _, dt1, B1, C1 = mx.ssm_inputs(u1)
        x2, _, dt2, B2, C2 = mx.ssm_inputs(u2)
        torch.testing.assert_close(dt1, dt2)
        torch.testing.assert_close(B1, B2)
        torch.testing.assert_close(C1, C2)
        assert not torch.allclose(x1, x2)  # the value path still sees the input
    # the selective M0 does depend on the input
    sel = next(x for x in model_for(f"t{name[1]}_m0").modules() if isinstance(x, Mamba2Mixer))
    assert not torch.allclose(sel.ssm_inputs(u1)[2], sel.ssm_inputs(u2)[2])
    # and the static model still reacts to its input end to end
    b2 = type(b)(**{**vars(b), "x": b.x + 1.0})
    with torch.no_grad():
        assert not torch.allclose(m(b)[0], m(b2)[0])


@pytest.mark.parametrize("name", T2_MODELS)
def test_task2_encoder_is_causal(name, batches):
    """Changing inputs from step p on leaves encoder outputs before p unchanged."""
    m = model_for(name).eval()
    x = batches["2"].x
    for p in (1, 84, 167):
        x2 = x.clone()
        x2[:, p:] = torch.randn_like(x2[:, p:])
        with torch.no_grad():
            e1, _ = m.encode(x)
            e2, _ = m.encode(x2)
        torch.testing.assert_close(e1[:, :p], e2[:, :p], rtol=0, atol=1e-5)
        assert not torch.allclose(e1[:, p:], e2[:, p:])


def test_task1_bi_encoder_sees_the_future_and_forward_one_does_not(batches):
    x = batches["1"].x
    x2 = x.clone()
    x2[:, 100:] += 1.0
    with torch.no_grad():
        bi = model_for("t1_m0").eval()
        assert not torch.allclose(bi.encode(x)[0][:, :100], bi.encode(x2)[0][:, :100])
        fw = model_for("t1_m4").eval()
        torch.testing.assert_close(
            fw.encode(x)[0][:, :100], fw.encode(x2)[0][:, :100], rtol=0, atol=1e-5
        )


@pytest.mark.parametrize("name", ["t2_m0", "t2_m1", "t2_m2", "t2_m4"])
def test_teacher_forcing_only_in_training(name, synthetic_npz):
    d = ElectricityData(synthetic_npz, 168, T2)
    b = d.batch(torch.as_tensor(d.origins["val"][:4]))
    b_other = d.batch(torch.as_tensor(d.origins["val"][:4]))
    b_other.y = b_other.y + 5.0
    m = model_for(name, dropout=0.0)
    m.eval()
    with torch.no_grad():
        a, _ = m(b, teacher_forcing=1.0)
        c, _ = m(b, teacher_forcing=0.0)
        e, _ = m(b_other, teacher_forcing=1.0)
    torch.testing.assert_close(a, c)
    torch.testing.assert_close(a, e)  # eval never reads Batch.y
    m.train()  # dropout 0, so train mode is deterministic
    with torch.no_grad():
        f, _ = m(b, teacher_forcing=1.0)
        g, _ = m(b_other, teacher_forcing=1.0)
        h, _ = m(b_other, teacher_forcing=0.0)
    torch.testing.assert_close(f[:, 0], g[:, 0])  # step 1 is fed r(t), not Batch.y
    assert not torch.allclose(f[:, 1:], g[:, 1:])  # later steps are fed the true residual
    torch.testing.assert_close(h, a)  # ratio 0 in train mode equals eval decoding


def test_decoder_state_comes_from_last_encoder_output(batches):
    m = model_for("t2_m0", dropout=0.0).eval()
    with torch.no_grad():
        enc, _ = m.encode(batches["2"].x)
        h, c = m.init_state(enc[:, -1]).chunk(2, dim=-1)
    assert h.shape == c.shape == (3, 64)


# --- end to end ------------------------------------------------------------------


def _run(*args: str) -> None:
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "4"}
    cmd = [sys.executable, str(ROOT / "scripts" / "run_ts.py"), *args]
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-3000:]


@pytest.mark.parametrize(("task", "model"), [(1, "t1_m0"), (2, "t2_m0")])
def test_smoke_end_to_end_phase2(synthetic_npz, tmp_path, task, model):
    common = ["--task", str(task), "--npz", str(synthetic_npz), "--out", str(tmp_path)]
    common += ["--config", "configs/phase2.yaml", "--smoke"]
    _run(*common, "--model", model, "--seed", "0")
    run = tmp_path / "smoke" / "phase2" / "runs" / f"task{task}" / model / "seed0"
    assert not (tmp_path / "smoke" / "runs").exists()  # Phase 1 smoke folder untouched
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 2
    spec = json.loads((run / "spec.json").read_text())
    assert spec["phase"] == "phase2" and spec["kind"] == mcfg(model)["kind"]
    assert spec["mamba_diagnostics"] is True
    assert spec["n_params"] == n_params(spec["kind"], spec["architecture"], horizon(model))
    fp = json.loads((run / "fingerprint.json").read_text())
    assert fp["phase"] == "phase2" and fp["model"] == json.loads(json.dumps(mcfg(model)))
    metrics = json.loads((run / "metrics.json").read_text())
    test = metrics["test"] if task == 1 else metrics["test"]["pooled"]
    assert set(test) == {"R2", "NSE", "MAE", "RMSE"}
    d = ElectricityData(synthetic_npz, 168, (2,) if task == 1 else T2)
    n_test = len(d.origins["test"])
    for split in ("val", "test"):
        z = np.load(run / f"pred_{split}.npz")
        np.testing.assert_array_equal(z["origin"], d.origins[split])
        assert np.isfinite(z["y_pred"]).all()
        assert ("attention" in z.files) == ("dt_mean" in z.files) == (split == "test")
    z = np.load(run / "pred_test.npz")
    att = z["attention"].astype(np.float32)
    assert att.shape == ((n_test, 168) if task == 1 else (n_test, 5, 168))
    np.testing.assert_allclose(att.sum(-1), 1.0, atol=1e-2)
    n_dirs = 2 if task == 1 else 1
    assert z["dt_mean"].shape == (n_test, n_dirs, 2, 168)
    assert z["state_influence"].shape == (n_test, n_dirs, 2, 168)
    assert np.isfinite(z["state_influence"]).all()
    assert list(z["dt_directions"]) == ["forward", "reverse"][:n_dirs]
    assert not bool(z["dt_static"])
    # a finished run is skipped unless --force
    mtime = (run / "metrics.json").stat().st_mtime_ns
    _run(*common, "--model", model, "--seed", "0")
    assert (run / "metrics.json").stat().st_mtime_ns == mtime
