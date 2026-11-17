"""Phase 2 Task 3 token-mixer recognizers (ocr_mamba) and their run_ocr.py integration."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dl_final_prj.mamba import BiMamba, Mamba1Mixer, MambaBlock
from dl_final_prj.ocr_mamba import (
    P0_PARAMS,
    BiGRULayer,
    BiGRUMixer,
    BiMambaLayer,
    GreedyDecodeExport,
    GRUMixerAttentionGRURecognizer,
    MambaAttentionGRURecognizer,
    MixerAttentionGRURecognizer,
    VisionBiMamba,
)
from dl_final_prj.ocr_models import (
    BEAM_WIDTHS,
    AttentionGRURecognizer,
    build,
    primary_mode,
    search_widths,
)
from dl_final_prj.svhn import MAX_LEN, decode, prepare
from dl_final_prj.ts_models import count_params
from test_svhn import ocr_batch, write_raw

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "phase2_task3.yaml").read_text())
P1_CFG = yaml.safe_load((ROOT / "configs" / "phase1_task3.yaml").read_text())
MODELS = ["t3_m0", "t3_m1", "t3_m2", "t3_m3", "t3_m4"]
# exact trainable parameter counts, also recorded in configs/phase2_task3.yaml
N_PARAMS = {
    "t3_m0": 7_947_282,
    "t3_m1": 7_942_546,
    "t3_m2": 7_945_362,
    "t3_m3": 7_685_138,
    "t3_m4": 7_943_698,
}


def make(name: str, seed: int = 0) -> MixerAttentionGRURecognizer:
    torch.manual_seed(seed)
    m = CFG["models"][name]
    model = build(m["kind"], m["args"], CFG["max_len"], length_loss_weight=0.2)
    assert isinstance(model, MixerAttentionGRURecognizer)
    return model


def test_config_matches_phase1_except_models():
    skip = ("models", "phase")
    assert {k: v for k, v in CFG.items() if k not in skip} == {
        k: v for k, v in P1_CFG.items() if k not in skip
    }
    assert CFG["phase"] == "phase2" and "phase" not in P1_CFG
    assert sorted(CFG["models"]) == MODELS


@pytest.mark.parametrize("name", MODELS)
def test_shapes_backward_and_decoding_modes(name):
    model = make(name)
    b = ocr_batch(["123", "4", "56789"])
    logits, length_logits, att = model(b.images, b.tokens)
    assert logits.shape == (3, 6, 13) and length_logits.shape == (3, 5)
    assert att is not None and att.shape == (3, 6, 49)
    loss = model.loss(b)
    loss.backward()
    assert torch.isfinite(loss) and loss.dtype == torch.float32
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing
    model.eval()
    with torch.no_grad():
        assert model.modes == ("greedy", "length", "beam3", "beam5", "beam10")
        assert primary_mode(model, 5) == "beam5" and search_widths(model) == BEAM_WIDTHS
        for mode in model.modes:
            preds, aux = model.predict(b.images, mode)
            assert len(preds) == 3 and all(p.isdigit() for p in preds)
            assert all(1 <= len(p) <= MAX_LEN for p in preds)
            if mode != "length":  # greedy and every beam width return their attention
                torch.testing.assert_close(aux["attention"].sum(-1), torch.ones(3, 6))


@pytest.mark.parametrize("name", ["t3_m0", "t3_m1"])
def test_length_constraint_and_beam_width_one(name):
    model = make(name, seed=1).eval()
    b = ocr_batch(["1"] * 5, seed=2)
    with torch.no_grad():
        want = torch.tensor([1, 2, 3, 4, 5])
        ids, _ = model.greedy(b.images, constrain=True, lengths=want)
        assert [len(decode(s)) for s in ids.tolist()] == want.tolist()
        greedy, aux = model.predict(b.images, "greedy")
        ids, att = model.beam(b.images, 1)
        assert [decode(s) for s in ids.tolist()] == greedy
        torch.testing.assert_close(att, aux["attention"])
        ids5, att5 = model.beam(b.images, 5)
        _, _, forced = model(b.images, ids5)  # the attention of the chosen beam
        torch.testing.assert_close(att5, forced, atol=1e-5, rtol=1e-4)


def test_parameter_counts_match_m0_within_5_percent():
    counts = {n: count_params(make(n)) for n in MODELS}
    assert counts == N_PARAMS
    for n in MODELS[1:]:
        assert abs(counts[n] - counts["t3_m0"]) / counts["t3_m0"] < 0.05, n
    # the token mixers themselves are matched too (M1 and M2 within 1%)
    mix = {n: make(n).param_breakdown()["token_mixer"] for n in MODELS}
    assert abs(mix["t3_m1"] / mix["t3_m0"] - 1) < 0.01
    assert abs(mix["t3_m2"] / mix["t3_m0"] - 1) < 0.01
    # without the depth compensation (2 forward layers) M3 would be far outside 5%
    m3 = CFG["models"]["t3_m3"]["args"]
    plain = MambaAttentionGRURecognizer(**(m3 | {"mamba": m3["mamba"] | {"n_layers": 2}}))
    assert count_params(plain) / counts["t3_m0"] - 1 < -0.10
    # T3-P0 budget deviation (plan 2.3), recorded in the config and spec.json
    assert count_params(build("attention")) == P0_PARAMS
    assert counts["t3_m0"] / P0_PARAMS - 1 > 0.05


def test_m3_keeps_the_m0_mixer_config_and_mixer_count():
    """T3-M3 changes only the scan direction: same mixers (width, state), same count."""
    m0, m3 = make("t3_m0").encoder.mixer, make("t3_m3").encoder.mixer
    assert isinstance(m0, VisionBiMamba) and isinstance(m3, VisionBiMamba)
    mix0, mix3 = m0.ssm_mixers(), m3.ssm_mixers()
    assert len(mix0) == len(mix3) == 4
    shapes = lambda ms: [{n: p.shape for n, p in m.named_parameters()} for m in ms]  # noqa: E731
    assert shapes(mix0) == shapes(mix3)
    assert all(isinstance(m, Mamba1Mixer) and m.d_inner == 512 for m in mix3)
    assert m3.direction == "forward" and m3.n_layers == 4


def test_breakdown_and_mixer_types():
    m0, m1 = make("t3_m0"), make("t3_m1")
    assert isinstance(m0, MambaAttentionGRURecognizer)
    assert isinstance(m1, GRUMixerAttentionGRURecognizer)
    assert isinstance(m0.encoder.mixer, VisionBiMamba)
    assert isinstance(m1.encoder.mixer, BiGRUMixer) and m1.encoder.mixer.hidden == 272
    b0, b1 = m0.param_breakdown(), m1.param_breakdown()
    assert b0["total"] == count_params(m0) and b1["total"] == count_params(m1)
    for part in ("cnn", "token_projection", "position_embeddings", "decoder_and_heads"):
        assert b0[part] == b1[part], part
    assert b0["position_embeddings"] == 2 * 7 * 256
    assert "Mamba1Mixer" in m0.encoder.describe_mixer()
    assert "inside every layer" in m0.encoder.describe_mixer()
    assert "GRU" in m1.encoder.describe_mixer()


def test_m0_and_m1_share_the_per_layer_fusion_skeleton():
    """Both mixers fuse the two directions inside every residual layer (concat + Linear),
    so layer 2 of M0 sees both directions, as layer 2 of the M1 BiGRU does."""
    mx0, mx1 = make("t3_m0").encoder.mixer, make("t3_m1").encoder.mixer
    assert isinstance(mx0, VisionBiMamba) and isinstance(mx1, BiGRUMixer)
    assert len(mx0.layers) == len(mx1.layers) == 2
    for l0, l1 in zip(mx0.layers, mx1.layers, strict=True):
        assert isinstance(l0, BiMambaLayer) and isinstance(l1, BiGRULayer)
        assert l1.rnn.bidirectional and l1.rnn.num_layers == 1
        assert l0.proj.in_features == 2 * 256 and l0.proj.out_features == 256
        assert l1.proj.out_features == 256
    # layer 1's output at the first token already depends on the last token (reverse scan),
    # so layer 2's forward mixer reads both directions
    torch.manual_seed(4)
    t = torch.randn(1, 49, 256)
    t2 = t.clone()
    t2[:, -1] = -t2[:, -1]
    fwd_only = make("t3_m3").encoder.mixer
    assert isinstance(fwd_only, VisionBiMamba)
    with torch.no_grad():
        first, causal = mx0.layers[0], fwd_only.layers[0]
        assert (first(t)[:, 0] - first(t2)[:, 0]).abs().max() > 1e-6  # small dt at init
        torch.testing.assert_close(causal(t)[:, 0], causal(t2)[:, 0], rtol=0, atol=0)
    # the old late-fusion design stays available through `fusion: late`
    m0 = CFG["models"]["t3_m0"]["args"]
    late = MambaAttentionGRURecognizer(**(m0 | {"mamba": m0["mamba"] | {"fusion": "late"}}))
    assert isinstance(late.encoder.mixer, BiMamba) and count_params(late) == 7_816_722
    assert "late fusion" in late.encoder.describe_mixer()
    with pytest.raises(ValueError, match="fusion"):
        MambaAttentionGRURecognizer(mamba={"fusion": "middle"})


def test_mamba_layer_paths_agree_and_forward_layer_is_a_mamba_block():
    torch.manual_seed(5)
    x = torch.randn(2, 49, 32)
    bi = VisionBiMamba(32, n_layers=2, direction="bi", kind="mamba1").eval()
    with torch.no_grad():
        y, diag = bi(x, diagnostics=True)
        torch.testing.assert_close(y, bi(x)[0])
        torch.testing.assert_close(y, bi.features(x))
    assert diag["dt_mean"].shape == (2, 2, 2, 49) and diag["directions"] == ["forward", "reverse"]
    # the reverse dt is mapped back to original token order: layer 1 reverse == mixer on flip
    layer = bi.layers[0]
    assert isinstance(layer, BiMambaLayer)
    with torch.no_grad():
        dt_r = layer.bwd.forward_diag(torch.flip(layer.norm(x), dims=[1]))[1].mean(-1)
    torch.testing.assert_close(diag["dt_mean"][:, 1, 0], torch.flip(dt_r, dims=[1]))
    # a forward-only layer is exactly mamba.MambaBlock
    fl = BiMambaLayer(32, "forward", "mamba1", d_state=16, d_conv=4, expand=2).eval()
    blk = MambaBlock(32, kind="mamba1", d_state=16, d_conv=4, expand=2).eval()
    sd = {k.replace("fwd.", "mixer."): v for k, v in fl.state_dict().items()}
    blk.load_state_dict(sd)
    with torch.no_grad():
        torch.testing.assert_close(fl(x), blk(x))


def test_phase1_build_is_unchanged():
    m = build("attention")
    assert type(m) is AttentionGRURecognizer and count_params(m) == 6_202_898
    assert m.encoder.dim == 512


def test_static_selection_is_input_independent():
    m2, m0 = make("t3_m2").eval(), make("t3_m0").eval()
    u1, u2 = torch.randn(2, 49, 256), torch.randn(2, 49, 256)
    mixers = m2.encoder.mixer.ssm_mixers()  # type: ignore[union-attr]
    assert len(mixers) == 4
    for mixer in mixers:
        assert isinstance(mixer, Mamba1Mixer) and mixer.static
        _, _, dt1, b1, c1 = mixer.ssm_inputs(u1)
        _, _, dt2, b2, c2 = mixer.ssm_inputs(u2)
        for x, y in ((dt1, dt2), (b1, b2), (c1, c2)):
            torch.testing.assert_close(x, y)
        assert not hasattr(mixer, "x_proj") and not hasattr(mixer, "dt_proj")
    sel = m0.encoder.mixer.ssm_mixers()[0]  # type: ignore[union-attr]
    assert isinstance(sel, Mamba1Mixer) and not sel.static
    assert not torch.allclose(sel.ssm_inputs(u1)[2], sel.ssm_inputs(u2)[2])
    images = ocr_batch(["1", "2"]).images
    dt_static = m2.mixer_diagnostics(images)["dt_mean"]
    assert dt_static.shape == (2, 2, 2, 49)
    torch.testing.assert_close(dt_static, dt_static[:1, :, :, :1].expand_as(dt_static))
    dt_sel = m0.mixer_diagnostics(images)["dt_mean"]
    assert dt_sel.shape == (2, 2, 2, 49) and bool((dt_sel > 0).all())
    assert dt_sel.std() > 0


def test_forward_only_is_causal_and_bi_is_not():
    m0, m3 = make("t3_m0").eval(), make("t3_m3").eval()
    mx3 = m3.encoder.mixer
    assert isinstance(mx3, VisionBiMamba) and mx3.direction == "forward"
    assert not any(hasattr(layer, "bwd") or hasattr(layer, "proj") for layer in mx3.layers)
    torch.manual_seed(3)
    t = torch.randn(1, 49, 256)
    t2 = t.clone()
    t2[:, 30] += 1.0
    with torch.no_grad():
        f1, f2 = m3.encoder.mixer(t)[0], m3.encoder.mixer(t2)[0]
        b1, b2 = m0.encoder.mixer(t)[0], m0.encoder.mixer(t2)[0]
    torch.testing.assert_close(f1[:, :30], f2[:, :30])
    assert (f1[:, 30:] - f2[:, 30:]).abs().max() > 1e-4
    assert (b1[:, :30] - b2[:, :30]).abs().max() > 1e-4  # the reverse scan sees token 30
    images = ocr_batch(["12", "3"]).images
    diag = m3.mixer_diagnostics(images)
    assert diag["dt_mean"].shape == (2, 1, 4, 49) and diag["directions"] == ["forward"]


def test_position_embedding_removal():
    m0, m4 = make("t3_m0").eval(), make("t3_m4").eval()
    assert m4.encoder.row is None and m4.encoder.col is None
    assert count_params(m0) - count_params(m4) == 2 * 7 * 256
    missing, unexpected = m4.load_state_dict(m0.state_dict(), strict=False)
    assert not missing and sorted(unexpected) == ["encoder.col", "encoder.row"]
    images = ocr_batch(["12", "3"]).images
    with torch.no_grad():
        t0, t4 = m0.encoder.tokens(images), m4.encoder.tokens(images)
        assert m0.encoder.row is not None and m0.encoder.col is not None
        pos = (m0.encoder.row[:, None] + m0.encoder.col[None]).reshape(1, 49, 256)
        torch.testing.assert_close(t0 - pos, t4)
        assert not torch.allclose(m0.encoder(images), m4.encoder(images))
        m0.encoder.row.zero_()
        m0.encoder.col.zero_()
        torch.testing.assert_close(m0.encoder(images), m4.encoder(images))


def test_position_embeddings_follow_raster_order():
    m0 = make("t3_m0").eval()
    enc = m0.encoder
    assert enc.row is not None and enc.col is not None
    with torch.no_grad():
        enc.proj.weight.zero_()
        enc.proj.bias.zero_()
        t = enc.tokens(torch.zeros(1, 3, 54, 54))
    for i, j in ((0, 0), (2, 5), (6, 6)):
        torch.testing.assert_close(t[0, i * 7 + j], enc.row[i] + enc.col[j])


def test_gru_mixer_has_no_mamba_diagnostics():
    m1 = make("t3_m1").eval()
    assert m1.mixer_diagnostics(ocr_batch(["1"]).images) == {}


def test_bf16_autocast_keeps_mixer_in_fp32():
    m0 = make("t3_m0")
    b = ocr_batch(["12", "345"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        tokens = m0.encoder(b.images)
        loss = m0.loss(b)
    assert tokens.dtype == torch.float32 and loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("batch", [1, 4])  # batch 1 is the edge-profiling shape
@pytest.mark.parametrize("name", ["t3_m0"])
def test_onnx_export_greedy_decode_opset17(name, batch, tmp_path):
    """Exports CNN + Mamba mixer + the whole unrolled greedy decode as one graph.

    Needs onnx (and onnxruntime for the parity run), which are not in the dev group yet:
    uv run --with onnx --with onnxruntime pytest tests/test_ocr_mamba.py -k onnx
    """
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("onnx is not installed, torch.onnx.export needs it")
    model = make(name)
    with torch.no_grad():
        # BatchNorm statistics from data (at init the CNN output barely depends on the image)
        # and sharper logits, so the ids vary between images and steps
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.momentum = None
        model.train()
        model.encoder(ocr_batch(["1"] * 16, seed=6).images)
        model.eval()
        model.out.weight.mul_(30.0)
    wrapper = GreedyDecodeExport(model).eval()
    x4 = ocr_batch(["1", "22", "333", "4444"], seed=5).images
    x = x4[:batch]
    with torch.no_grad():
        assert len(set(wrapper(x4).flatten().tolist())) > 2
        ref = wrapper(x)
        ref_tokens = model.encoder(x)
    assert ref.shape == (batch, MAX_LEN + 1)
    path, enc_path = tmp_path / f"{name}_greedy.onnx", tmp_path / f"{name}_encoder.onnx"
    kw = {"opset_version": 17, "dynamo": False, "input_names": ["images"]}
    torch.onnx.export(wrapper, (x,), str(path), output_names=["ids"], **kw)
    torch.onnx.export(model.encoder, (x,), str(enc_path), output_names=["tokens"], **kw)
    import onnx

    for p in (path, enc_path):
        graph = onnx.load(str(p))
        onnx.checker.check_model(graph)
        # standard ONNX ops only: no custom domains, no complex tensors, no loops
        assert all(n.domain in ("", "ai.onnx") for n in graph.graph.node)
        assert not {"Loop", "Scan", "If"} & {n.op_type for n in graph.graph.node}
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime is not installed, export checked without a parity run")
    import onnxruntime as ort

    run = lambda p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(  # noqa: E731
        None, {"images": x.numpy()}
    )[0]
    np.testing.assert_array_equal(run(path), ref.numpy())
    np.testing.assert_allclose(run(enc_path), ref_tokens.numpy(), atol=1e-4, rtol=1e-4)


# --- run_ocr.py end to end ------------------------------------------------------------


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("svhn")
    raw = write_raw(tmp / "raw", {"train": 40, "extra": 40, "test": 12})
    prepare(raw, tmp / "cache", size=64, workers=1)
    return tmp / "cache"


def run_ocr(model: str, cache: Path, out: Path, config: Path, *extra: str, smoke: bool = True):
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "scripts/run_ocr.py", "--model", model, "--seed", "0"]
    cmd += ["--config", str(config), "--cache", str(cache), "--out", str(out), *extra]
    cmd += ["--smoke"] if smoke else []
    return subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("model", ["t3_m0", "t3_m1"])
def test_run_ocr_smoke(model, cache, tmp_path):
    run_ocr(model, cache, tmp_path, ROOT / "configs" / "phase2_task3.yaml")
    run = tmp_path / "smoke" / "runs" / "task3" / model / "seed0"
    metrics = json.loads((run / "metrics.json").read_text())
    spec = json.loads((run / "spec.json").read_text())
    history = json.loads((run / "history.json").read_text())
    want = {"val", "test", "val_greedy", "test_greedy", "test_length_constrained"}
    want |= {"test_whole", "test_whole_greedy"}
    assert set(metrics) == want | {"decoding"}
    dec = metrics["decoding"]
    assert dec["primary"] == "beam" and set(dec["val_by_width"]) == {"3", "5", "10"}
    assert metrics["val"]["exact_match"] == dec["val_by_width"][str(dec["width"])]
    assert len(history) == 2 and spec["epochs_run"] == 2
    assert (
        spec["n_params"] == N_PARAMS[model] == spec["architecture"]["n_params_breakdown"]["total"]
    )
    assert spec["phase"] == "phase2" and spec["teacher_forcing"] == "full"
    pm = spec["param_match"]
    assert pm["t3_p0_cnn_attention_gru_params"] == P0_PARAMS
    assert pm["rel_diff_vs_t3_p0"] == pytest.approx(N_PARAMS[model] / P0_PARAMS - 1)
    assert spec["kind"] == CFG["models"][model]["kind"]
    assert spec["vocabulary"]["size"] == 13 and spec["length_head"].startswith("Linear(256, 5)")
    assert spec["architecture"]["decoder_token_dim"] == 256
    assert spec["architecture"]["position_embeddings"].startswith("learned row/col")
    assert set(spec["decoding_modes"]) == want
    assert spec["decoding"]["mode"] == dec["mode"] == f"beam{dec['width']}"
    for key in want:
        preds = json.loads((run / f"pred_{key}.json").read_text())
        n = spec["n_images"]["val" if key.startswith("val") else "test"]
        assert len(preds) == n and all(len(r["pred"]) <= MAX_LEN for r in preds)
    att = np.load(run / "attention_test.npz")
    assert att["attention"].shape == att["attention_greedy"].shape == (12, 6, 49)
    assert str(att["mode"]) == dec["mode"]
    # --eval-only from ckpt.pt gives the same evaluation and leaves the checkpoint alone
    ck = (run / "ckpt.pt").read_bytes()
    out = run_ocr(model, cache, tmp_path, ROOT / "configs" / "phase2_task3.yaml", "--eval-only")
    assert "test (beam" in out.stderr and (run / "ckpt.pt").read_bytes() == ck
    again = json.loads((run / "metrics.json").read_text())
    assert {k: v for k, v in again.items() if k != "decoding"} == {
        k: v for k, v in metrics.items() if k != "decoding"
    }
    assert again["decoding"]["val_by_width"] == dec["val_by_width"]
    has_dt = (run / "mamba_dt_test.npz").exists()
    assert has_dt == (model == "t3_m0")
    if model == "t3_m0":
        d = np.load(run / "mamba_dt_test.npz")
        assert d["dt_mean"].shape == (12, 2, 2, 49) and (d["dt_mean"] > 0).all()
        assert d["directions"].tolist() == ["forward", "reverse"] and not bool(d["static"])
        assert spec["mamba_diagnostics"]["file"] == "mamba_dt_test.npz"
    else:
        assert spec["mamba_diagnostics"] is None
        assert "GRU" in spec["architecture"]["token_mixer"]


def test_run_ocr_full_run_goes_to_phase2(cache, tmp_path):
    cfg = dict(CFG)
    cfg["fit"] = CFG["fit"] | {"epochs": 1}
    cfg["models"] = {"t3_m2": CFG["models"]["t3_m2"]}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_ocr("t3_m2", cache, tmp_path / "results", path, smoke=False)
    run = tmp_path / "results" / "phase2" / "runs" / "task3" / "t3_m2" / "seed0"
    spec = json.loads((run / "spec.json").read_text())
    assert spec["smoke"] is False and spec["epochs_run"] == 1
    assert spec["n_images"]["train"] == 74  # no smoke limit
    assert not (tmp_path / "results" / "phase1").exists()
    dt = np.load(run / "mamba_dt_test.npz")
    assert bool(dt["static"]) and np.allclose(dt["dt_mean"], dt["dt_mean"][:1, :, :, :1])
    out = run_ocr("t3_m2", cache, tmp_path / "results", path, smoke=False)
    assert "already done" in out.stderr


@pytest.mark.parametrize("model,scan", [("t3_m3", "loop"), ("t3_m1", None)])
def test_run_ocr_bench_times_steps_and_writes_nothing(model, scan, cache, tmp_path):
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "scripts/run_ocr.py", "--model", model, "--seed", "0", "--config"]
    cmd += [str(ROOT / "configs" / "phase2_task3.yaml"), "--cache", str(cache)]
    cmd += ["--out", str(tmp_path), "--bench", "1", "--bench-warmup", "0"]
    cmd += ["--scan", scan] if scan else []
    out = subprocess.run(cmd, check=True, cwd=ROOT, env=env, capture_output=True, text=True)
    line = next(x for x in out.stderr.splitlines() if "bench {" in x)
    res = json.loads(line.split("bench ", 1)[1])
    assert res["model"] == model and res["scan"] == (scan if scan else None)
    assert res["ms_per_step"] > 0 and res["n_train"] == 74 and res["batch_size"] == 64
    assert res["train_h_per_run"] == pytest.approx(2 * res["ms_per_step"] / 1000 * 16 / 3600)
    assert not any(tmp_path.iterdir())  # no run directory
    bad = subprocess.run(
        [*cmd[: cmd.index("--bench")], "--scan", "loop"], cwd=ROOT, env=env, capture_output=True
    )
    assert bad.returncode != 0  # --scan without --bench is refused
