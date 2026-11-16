import pytest
import torch

from dl_final_prj.models.sota_forecasters import MODELS, SOTAForecaster

N_FEATURES, TARGET_IDX, LOOKBACK = 5, 2, 48
SMALL = {"d_model": 16, "n_heads": 2, "e_layers": 1, "d_ff": 32}


def _make(name: str, horizon: int) -> SOTAForecaster:
    return SOTAForecaster(name, N_FEATURES, TARGET_IDX, LOOKBACK, horizon, **SMALL)


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("horizon", [2, 5])
def test_shape_and_finite(name: str, horizon: int) -> None:
    model = _make(name, horizon).eval()
    out = model(torch.randn(4, LOOKBACK, N_FEATURES))
    assert out.shape == (4, horizon)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(MODELS))
def test_deterministic_with_seed(name: str) -> None:
    x = torch.randn(3, LOOKBACK, N_FEATURES)
    outs = []
    for _ in range(2):
        torch.manual_seed(7)
        outs.append(_make(name, 5).eval()(x))
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("name", sorted(MODELS))
def test_gradients_flow(name: str) -> None:
    model = _make(name, 2).train()
    model(torch.randn(4, LOOKBACK, N_FEATURES)).sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_future_rows_do_not_exist_in_input(name: str) -> None:
    """The output depends on the window only. Changing one window must not change another."""
    model = _make(name, 2).eval()
    x = torch.randn(2, LOOKBACK, N_FEATURES)
    base = model(x)
    x2 = x.clone()
    x2[1] += 5.0
    assert torch.allclose(model(x2)[0], base[0], atol=1e-6)


def test_unknown_model_and_short_lookback() -> None:
    with pytest.raises(ValueError):
        SOTAForecaster("nope", N_FEATURES, TARGET_IDX, LOOKBACK, 2)
    with pytest.raises(ValueError):
        SOTAForecaster("patchtst", N_FEATURES, TARGET_IDX, 8, 2)
