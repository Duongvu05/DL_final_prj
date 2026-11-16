import pytest
import torch
from torch import nn

from dl_final_prj.models.sota_forecasters.ts_seq2seq import (
    BahdanauAttention,
    LSTMCell,
    TSSeq2SeqForecaster,
)


def _model(**kw) -> TSSeq2SeqForecaster:
    """Small model. The repo init (std 0.01) makes outputs near constant, so widen the weights."""
    torch.manual_seed(0)
    model = TSSeq2SeqForecaster(3, target_idx=0, horizon=5, hidden=16, dropout=0.0, **kw)
    for p in model.parameters():
        nn.init.normal_(p, std=0.5)
    return model.eval()


def test_attention_is_softmax_over_encoder_steps() -> None:
    attn = BahdanauAttention(8)
    q, enc = torch.randn(4, 1, 8), torch.randn(4, 12, 8)
    context, w = attn(q, enc, enc)
    assert context.shape == (4, 1, 8)
    assert w.shape == (4, 12, 1)
    torch.testing.assert_close(w.sum(dim=1), torch.ones(4, 1))
    assert (w >= 0).all()
    torch.testing.assert_close(context, (w * enc).sum(dim=1, keepdim=True))


def test_lstm_cell_matches_torch_lstmcell() -> None:
    mine, ref = LSTMCell(4, 6), nn.LSTMCell(4, 6)
    ref.weight_ih.data, ref.weight_hh.data = mine.weight_ih.data, mine.weight_hh.data
    ref.bias_ih.data, ref.bias_hh.data = mine.bias_ih.data, mine.bias_hh.data
    x, h, c = torch.randn(3, 4), torch.randn(3, 6), torch.randn(3, 6)
    h1, c1 = mine(x, (h, c))
    h2, c2 = ref(x, (h, c))
    torch.testing.assert_close(h1, h2)
    torch.testing.assert_close(c1, c2)


def test_output_shape_and_attention_weights() -> None:
    model = _model()
    out, w = model(torch.randn(4, 12, 3), return_attention=True)
    assert out.shape == (4, 5, 1)
    assert w is not None
    assert w.shape == (4, 5, 12)
    torch.testing.assert_close(w.sum(-1), torch.ones(4, 5))


def test_reference_decoder_requires_attention() -> None:
    with pytest.raises(ValueError):
        TSSeq2SeqForecaster(3, target_idx=0, attention=False)


def test_free_run_feeds_back_predictions_and_ignores_targets() -> None:
    model = _model()
    x, y = torch.randn(4, 10, 3), torch.randn(4, 5)
    free = model(x)
    torch.testing.assert_close(model(x, y, teacher_forcing=0.0), free)
    assert not torch.allclose(model(x, y, teacher_forcing=1.0), free)
    # step 1 cannot depend on y even with full teacher forcing
    torch.testing.assert_close(model(x, y, teacher_forcing=1.0)[:, 0], free[:, 0])


def test_teacher_forcing_has_no_target_leak() -> None:
    """The step k output may only depend on y[:k]. The repo fed y[k] in to predict y[k]."""
    model = _model()
    x, y = torch.randn(4, 10, 3), torch.randn(4, 5)
    base = model(x, y, teacher_forcing=1.0)
    for k in range(5):
        y2 = y.clone()
        y2[:, k] += 10.0
        out = model(x, y2, teacher_forcing=1.0)
        torch.testing.assert_close(out[:, : k + 1], base[:, : k + 1])
        if k < 4:
            assert not torch.allclose(out[:, k + 1 :], base[:, k + 1 :])


def test_prediction_uses_only_the_window() -> None:
    model = _model()
    x = torch.randn(2, 10, 3)
    pair = torch.cat([x, x.flip(0)])
    torch.testing.assert_close(model(pair)[:2], model(x), atol=1e-6, rtol=1e-5)


def test_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = TSSeq2SeqForecaster(2, target_idx=0, horizon=5, hidden=16, dropout=0.0)
    x = torch.randn(32, 10, 2)
    y = x[:, -1, :1] + torch.linspace(0, 1, 5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x, y, 0.5).squeeze(-1), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
