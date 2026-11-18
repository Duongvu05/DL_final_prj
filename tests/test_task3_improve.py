"""Task 3 improvement study (plan Section 3): RandomResizedCrop, label smoothing, STN, resolution."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dl_final_prj.ocr_models import AffineSTN, AttentionGRURecognizer, build, vgg_grid
from dl_final_prj.svhn import OcrBatch, SvhnSplit, encode, random_resized_crop

ROOT = Path(__file__).resolve().parents[1]


def _cache(tmp_path: Path, n: int = 6, size: int = 64) -> Path:
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, (n, size, size, 3), dtype=np.uint8)
    np.save(tmp_path / "train_images.npy", imgs)
    labels = [{"name": f"{i}.png", "dir": "train", "label": "12"[: 1 + i % 2]} for i in range(n)]
    import json

    (tmp_path / "train_labels.json").write_text(json.dumps(labels))
    return tmp_path


def test_random_resized_crop_shape_range_and_identity():
    x = torch.rand(4, 3, 64, 64)
    g = torch.Generator().manual_seed(0)
    y = random_resized_crop(x, 54, (0.7, 1.0), (0.75, 1.3333), g)
    assert y.shape == (4, 3, 54, 54)
    assert float(y.min()) >= 0 and float(y.max()) <= 1
    # area 1 and aspect 1 is the whole canvas resized to the crop size
    whole = random_resized_crop(x, 64, (1.0, 1.0), (1.0, 1.0), g)
    torch.testing.assert_close(whole, x, atol=1e-5, rtol=0)


def test_rrc_split_is_reproducible_and_eval_unchanged(tmp_path):
    cache = _cache(tmp_path)
    plain = SvhnSplit(cache, "train", crop=54)
    rrc = SvhnSplit(cache, "train", crop=54, rrc_scale=(0.7, 1.0))
    idx = torch.arange(6)
    a = rrc.batch(idx, train=True, g=torch.Generator().manual_seed(3))
    b = rrc.batch(idx, train=True, g=torch.Generator().manual_seed(3))
    torch.testing.assert_close(a.images, b.images)
    assert a.images.shape == (6, 3, 54, 54)
    torch.testing.assert_close(rrc.batch(idx).images, plain.batch(idx).images)  # eval: center crop
    # without rrc_scale the training path is the old translation crop, bit for bit
    p1 = plain.batch(idx, train=True, g=torch.Generator().manual_seed(3)).images
    p2 = (
        SvhnSplit(cache, "train", crop=54).batch(idx, True, torch.Generator().manual_seed(3)).images
    )
    torch.testing.assert_close(p1, p2)


def test_vgg_grid_and_token_counts():
    assert [vgg_grid(n) for n in (54, 64, 80)] == [7, 8, 10]
    for size, stn_out, tokens in ((54, None, 49), (54, 64, 64), (80, None, 100), (80, 80, 100)):
        m = AttentionGRURecognizer(input_size=size, stn_out=stn_out).eval()
        _, _, att = m(torch.randn(2, 3, size, size), torch.zeros(2, 6, dtype=torch.long))
        assert att.shape[-1] == tokens


def test_identity_stn_does_not_warp():
    stn = AffineSTN(out_size=32).eval()
    x = torch.rand(2, 3, 32, 32)
    torch.testing.assert_close(stn(x), x, atol=1e-5, rtol=0)
    assert torch.allclose(stn.theta(x), torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).expand(2, 2, 3))


def test_default_model_keeps_phase1_state_dict_keys():
    old = build("attention", {"hidden": 256, "layers": 2, "attn_dim": 256, "emb_dim": 64})
    assert not any(k.startswith("encoder.stn") for k in old.state_dict())
    assert old.encoder.grid == (7, 7)
    stn = build("attention", {"stn_out": 64})
    assert any(k.startswith("encoder.stn") for k in stn.state_dict())


def test_label_smoothing_changes_the_loss_only_when_set():
    torch.manual_seed(0)
    tokens = torch.tensor([encode("12", 5), encode("7", 5)])
    b = OcrBatch(torch.randn(2, 3, 54, 54), tokens, torch.tensor([1, 0]), torch.arange(2))
    m0 = AttentionGRURecognizer().eval()
    m1 = AttentionGRURecognizer(label_smoothing=0.05).eval()
    m1.load_state_dict(m0.state_dict())
    l0, l1 = m0.loss(b), m1.loss(b)
    assert torch.isfinite(l1) and not torch.isclose(l0, l1)


@pytest.mark.parametrize(
    "path,models",
    [
        ("configs/improve_task3.yaml", {"i2_rrc", "i3_ls"}),
        ("configs/improve_task3_whole.yaml", {"i2_rrc_whole", "i3_ls_whole", "i4_stn_whole"}),
        ("configs/improve_task3_whole96.yaml", {"i5_res96_whole", "i6_combo_whole"}),
    ],
)
def test_improvement_configs_change_one_factor(path, models):
    cfg = yaml.safe_load((ROOT / path).read_text())
    base_name = "phase1_task3.yaml" if "whole" not in path else "phase1_task3_whole.yaml"
    base = yaml.safe_load((ROOT / "configs" / base_name).read_text())
    assert set(cfg["models"]) == models and cfg["phase"] == "improve"
    p0 = base["models"]["cnn_attention_gru" if "whole" not in path else "cnn_attention_gru_whole"]
    for k in ("fit", "batch_size", "length_loss_weight", "color_jitter", "max_len", "seeds"):
        assert cfg[k] == base[k], k
    for m in cfg["models"].values():
        extra = {k: v for k, v in m["args"].items() if k not in p0["args"]}
        assert set(extra) <= {"label_smoothing", "stn_out", "input_size"}
        assert {k: v for k, v in m["args"].items() if k in p0["args"]} == p0["args"]
