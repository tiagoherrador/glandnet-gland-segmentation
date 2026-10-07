import torch
from glandnet.losses import bce_border_weighted, combined_loss, dice_loss, tversky_loss
from glandnet.model import UNet4L


def test_unet_output_shape():
    out = UNet4L()(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 2, 64, 64)


def test_perfect_prediction_has_near_zero_dice_and_tversky():
    t = torch.zeros(1, 2, 16, 16); t[:, :, 4:12, 4:12] = 1
    logits = (t * 2 - 1) * 20  # confident, correct
    assert dice_loss(logits, t) < 1e-3
    assert tversky_loss(logits, t) < 1e-3


def test_wrong_prediction_has_high_loss():
    t = torch.zeros(1, 2, 16, 16); t[:, :, 4:12, 4:12] = 1
    assert dice_loss(-(t * 2 - 1) * 20, t) > 0.9


def test_border_channel_is_weighted_more():
    t = torch.zeros(1, 2, 8, 8)
    logits = torch.zeros(1, 2, 8, 8); logits[:, 1] = 5  # error only on the border channel
    assert bce_border_weighted(logits, t, w_border=8) > bce_border_weighted(logits, t, w_border=1)


def test_combined_loss_is_scalar_and_backpropagates():
    x = torch.randn(1, 2, 16, 16, requires_grad=True)
    loss = combined_loss(x, torch.randint(0, 2, (1, 2, 16, 16)).float())
    loss.backward()
    assert loss.ndim == 0 and x.grad is not None
