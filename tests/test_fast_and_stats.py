import numpy as np
import pytest

from dl_final_prj import fast
from dl_final_prj.stats import bootstrap_forecast_diff, dm_test, holm, mcnemar


def test_rust_extension_is_built():
    assert fast.has_rust()


def test_parse_semicolon_table_matches_pandas(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(
        '"";"MT_001";"MT_002"\n"2011-01-01 00:15:00";0;1,5\n"2011-01-01 00:30:00";2,25;0\n'
    )
    names, stamps, v = fast.parse_semicolon_table(str(p))
    rn, rs, rv = fast.parse_semicolon_table_ref(str(p))
    assert names == rn == ["MT_001", "MT_002"]
    assert stamps == rs
    np.testing.assert_array_equal(v, rv)
    np.testing.assert_array_equal(v, [[0, 1.5], [2.25, 0]])


def test_block_means_rust_equals_numpy():
    rng = np.random.default_rng(0)
    cols = rng.normal(size=(103, 3))
    block = 10
    starts = rng.integers(0, 103 - block + 1, size=(50, 11))
    np.testing.assert_allclose(
        fast.block_means_from_starts(cols, block, starts),
        fast.block_means_from_starts_ref(cols, block, starts),
    )


def test_block_bootstrap_is_deterministic_and_centered():
    cols = np.random.default_rng(1).normal(size=(2000, 2))
    a = fast.block_bootstrap_means(cols, 24, 2000, seed=7)
    b = fast.block_bootstrap_means(cols, 24, 2000, seed=7)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2000, 2)
    np.testing.assert_allclose(a.mean(0), cols.mean(0), atol=0.01)
    # spread of the bootstrap mean is close to the i.i.d. standard error for white noise
    assert a.std(0) == pytest.approx(cols.std(0) / np.sqrt(2000), rel=0.15)


def test_levenshtein_rust_equals_python():
    a = ["123", "", "4567", "11", "908"]
    b = ["124", "12", "457", "11", "098"]
    np.testing.assert_array_equal(
        fast.levenshtein_batch(a, b), [fast.levenshtein_ref(x, y) for x, y in zip(a, b)]
    )


def test_letterbox_rust_matches_pillow(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    paths = []
    for i, (w, h) in enumerate([(100, 40), (30, 80), (256, 64)]):
        arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        arr = np.kron(arr[::10, ::10], np.ones((10, 10, 1), np.uint8))[:h, :w]  # smooth-ish blocks
        p = tmp_path / f"{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    crops = np.array([[10, 5, 90, 35], [0, 0, 30, 80], [0, 0, 256, 64]], np.float32)
    for c in (None, crops):
        im, g = fast.letterbox_images(paths, 64, 256, (1, 2, 3), c)
        rim, rg = fast.letterbox_images_ref(paths, 64, 256, [1, 2, 3], c)
        np.testing.assert_allclose(g, rg)
        assert im.shape == (3, 64, 256, 3) and im.dtype == np.uint8
        # same geometry, so the padding region is identical; resampling may differ slightly
        pad_mask = (rim == np.array([1, 2, 3], np.uint8)).all(-1)
        assert ((im == np.array([1, 2, 3], np.uint8)).all(-1) | ~pad_mask).all()
        assert np.abs(im.astype(int) - rim.astype(int)).mean() < 12
    np.testing.assert_allclose(fast.mean_rgb(paths), fast.mean_rgb_ref(paths))


def test_letterbox_stretch_rust_matches_pillow(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(1)
    paths = []
    for i, (w, h) in enumerate([(100, 40), (30, 80), (64, 64), (21, 9)]):
        arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        arr = np.kron(arr[::6, ::6], np.ones((6, 6, 1), np.uint8))[:h, :w]
        p = tmp_path / f"{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    # inside, partly outside (clamped), the whole image, and a fractional box
    crops = np.array(
        [[10, 5, 90, 35], [-12.5, 30, 41, 95], [0, 0, 64, 64], [2.4, 1.6, 15.2, 8.7]], np.float32
    )
    for c in (None, crops):
        im, g = fast.letterbox_images(paths, 64, 64, (1, 2, 3), c, stretch=True)
        rim, rg = fast.letterbox_images_ref(paths, 64, 64, [1, 2, 3], c, stretch=True)
        np.testing.assert_array_equal(g, rg)  # geometry is exact
        assert im.shape == (4, 64, 64, 3) and im.dtype == np.uint8
        assert np.all(g[:, 3:] == 0)
        np.testing.assert_allclose(g[:, 2], 64 / g[:, 0], rtol=1e-6)
        assert np.abs(im.astype(int) - rim.astype(int)).mean() < 12
    _, g = fast.letterbox_images(paths, 64, 64, (0, 0, 0), crops, stretch=True)
    np.testing.assert_array_equal(g[:, :2], [[80, 30], [30, 50], [64, 64], [14, 8]])
    for (w, h), box, (sw, sh) in zip([(100, 40), (30, 80), (64, 64), (21, 9)], crops, g[:, :2]):
        left, top, right, bottom = fast.clamp_box(box, w, h)
        assert (right - left, bottom - top) == (sw, sh)


def test_dm_test_detects_better_model():
    rng = np.random.default_rng(0)
    e = rng.normal(size=3000)
    loss_a = (0.8 * e) ** 2
    loss_b = e**2 + rng.normal(0, 0.1, 3000) ** 2
    r = dm_test(loss_a, loss_b, lag=1)
    assert r["mean_diff"] < 0 and r["p_value"] < 1e-3
    same = dm_test(loss_b, loss_b + rng.normal(0, 1e-3, 3000), lag=1)
    assert same["p_value"] > 0.001


def test_bootstrap_forecast_diff_sign():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(1000, 5)).cumsum(0)
    a = y + rng.normal(0, 0.5, y.shape)
    b = y + rng.normal(0, 1.0, y.shape)
    r = bootstrap_forecast_diff(y, a, b, block=24, reps=2000)
    assert r["RMSE"]["ci_high"] < 0 and r["MAE"]["ci_high"] < 0 and r["NSE"]["ci_low"] > 0


def test_mcnemar_and_holm():
    a = np.array([1] * 30 + [0] * 5, bool)
    b = np.array([0] * 20 + [1] * 10 + [0] * 5, bool)
    r = mcnemar(a, b)
    assert r["a_only"] == 20 and r["b_only"] == 0 and r["p_value"] < 1e-4
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_loss_diff_matches_seed_mean_metrics_not_ensemble():
    from dl_final_prj.stats import bootstrap_loss_diff

    rng = np.random.default_rng(3)
    y = rng.normal(size=(2000, 1)).cumsum(0)
    shared = rng.normal(0, 1.0, y.shape)
    seeds_a = [y + shared + rng.normal(0, 2.0, y.shape) for _ in range(3)]  # noisy per seed
    b = y + rng.normal(0, 1.6, y.shape)  # deterministic baseline
    se_a = np.mean([((p - y) ** 2).mean(1) for p in seeds_a], 0)
    ae_a = np.mean([np.abs(p - y).mean(1) for p in seeds_a], 0)
    r = bootstrap_loss_diff(
        y, se_a, ((b - y) ** 2).mean(1), ae_a, np.abs(b - y).mean(1), block=24, reps=2000
    )
    # A is worse per seed, so the interval must say so (an ensemble of A would look better)
    assert r["RMSE"]["diff"] == pytest.approx(np.sqrt(se_a.mean()) - np.sqrt(((b - y) ** 2).mean()))
    assert r["RMSE"]["ci_low"] > 0


def test_image_bootstrap_has_char_accuracy():
    from dl_final_prj.metrics import per_sample_losses
    from dl_final_prj.stats import bootstrap_image_diff

    t = ["12", "345", "7"] * 50
    a = per_sample_losses(t, ["12", "345", "7"] * 50)
    b = per_sample_losses(t, ["13", "34", "7"] * 50)
    np.testing.assert_allclose(b["char_correct"][:3], [1, 2, 1])
    r = bootstrap_image_diff(a, b, reps=500)
    assert r["char_accuracy"]["diff"] == pytest.approx((6 - 4) / 6)


def test_image_bootstrap_p_values_and_edit_distance():
    from dl_final_prj.metrics import per_sample_losses
    from dl_final_prj.stats import bootstrap_image_diff

    t = ["12", "345", "7"] * 200
    better = per_sample_losses(t, ["12", "345", "7"] * 200)
    worse = per_sample_losses(t, ["13", "34", "7"] * 200)
    r = bootstrap_image_diff(better, worse, reps=2000)
    assert r["exact_match"]["p_value"] < 0.01 and r["mean_edit_distance"]["diff"] < 0
    same = bootstrap_image_diff(worse, worse, reps=2000)
    assert same["exact_match"]["p_value"] == 1.0
