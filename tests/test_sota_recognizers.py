"""Fast CPU tests for the clovaai based recognizers."""

import pytest
import torch

from dl_final_prj.data.image_sequences import BOS, EOS, PAD, VOCAB_SIZE, encode
from dl_final_prj.models.sota_recognizers import ClovaRecognizer
from dl_final_prj.models.sota_recognizers.clova_modules import Attention

H, W = 32, 192
TOKENS = torch.tensor([encode("123", 8), encode("45678", 8)])
N_DIGITS = torch.tensor([3, 5])


def make(feature: str, sequence: str, prediction: str) -> ClovaRecognizer:
    return ClovaRecognizer(feature, sequence, prediction, output_channel=32, hidden=16)  # type: ignore[arg-type]


@pytest.mark.parametrize(("feature", "frames"), [("vgg", W // 4 - 1), ("resnet", W // 4 + 1)])
def test_feature_extractor_frames(feature: str, frames: int) -> None:
    model = make(feature, "bilstm", "ctc").eval()
    feats = model.encode(torch.rand(2, 1, H, W))
    assert feats.shape == (2, frames, 16)
    assert make(feature, "none", "ctc").encode(torch.rand(2, 1, H, W)).shape == (2, frames, 32)


@pytest.mark.parametrize("feature", ["vgg", "resnet"])
def test_ctc_loss_and_decode(feature: str) -> None:
    model = make(feature, "bilstm", "ctc").eval()
    images = torch.rand(2, 1, H, W)
    log_probs = model(images)
    assert log_probs.shape[1:] == (2, VOCAB_SIZE)
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(log_probs.shape[:2]), atol=1e-4)
    loss = model.loss(images, TOKENS, N_DIGITS)
    assert loss.ndim == 0 and torch.isfinite(loss)
    preds = model.predict(images)
    assert len(preds) == 2 and all(p == "" or p.isdigit() for p in preds)


@pytest.mark.parametrize(("feature", "sequence"), [("vgg", "bilstm"), ("resnet", "none")])
def test_attn_shapes_loss_and_decode(feature: str, sequence: str) -> None:
    model = make(feature, sequence, "attn").eval()
    images = torch.rand(2, 1, H, W)
    logits = model(images, TOKENS[:, :7])
    assert logits.shape == (2, 7, VOCAB_SIZE)
    loss = model.loss(images, TOKENS, N_DIGITS)
    assert loss.ndim == 0 and torch.isfinite(loss)
    ids, alphas = model.greedy_decode(images)
    assert ids.shape == (2, 7) and alphas is not None and alphas.shape[:2] == (2, 7)
    preds = model.predict(images)
    assert len(preds) == 2 and all(p == "" or p.isdigit() for p in preds)


def test_attention_weights_sum_to_one() -> None:
    model = make("vgg", "bilstm", "attn").eval()
    _, alphas = model.greedy_decode(torch.rand(3, 1, H, W))
    assert alphas is not None
    assert torch.allclose(alphas.sum(dim=-1), torch.ones(3, 7), atol=1e-5)
    assert (alphas >= 0).all()


def test_attention_decoder_teacher_forcing_matches_free_run_first_step() -> None:
    att = Attention(8, 8, VOCAB_SIZE).eval()
    feats = torch.rand(2, 5, 8)
    forced, _ = att(feats, torch.full((2, 3), BOS), 3, BOS)
    free, _ = att(feats, None, 3, BOS)
    assert torch.allclose(forced[:, 0], free[:, 0])


def test_predict_removes_padding_and_stops_at_eos() -> None:
    model = make("vgg", "bilstm", "attn").eval()

    def fake(images: torch.Tensor) -> tuple[torch.Tensor, None]:
        row = [BOS, 5, 6, EOS, 7, PAD, PAD]  # digits 2 and 3, then junk after EOS
        return torch.tensor([row] * images.shape[0]), None

    model.greedy_decode = fake  # type: ignore[method-assign]
    assert model.predict(torch.rand(2, 1, H, W)) == ["23", "23"]


def test_bad_options_raise() -> None:
    with pytest.raises(ValueError):
        ClovaRecognizer("rcnn")  # type: ignore[arg-type]
