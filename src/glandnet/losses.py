"""Losses for the 2-channel (interior, border) output. All take logits and targets of shape [B,2,H,W]."""
import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def tversky_loss(logits, targets, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(dim=(2, 3))
    fp = (probs * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - probs) * targets).sum(dim=(2, 3))
    return 1 - ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def bce_border_weighted(logits, targets, w_border: float = 8.0):
    """BCE where the border channel (index 1) counts `w_border` times more."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.ones_like(targets)
    weights[:, 1:2] = w_border
    return (bce * weights).mean()


def combined_loss(logits, targets, w_border: float = 8.0):
    """0.3 * weighted BCE + 0.3 * Dice + 0.4 * Tversky (the mix used for the final model)."""
    return (0.3 * bce_border_weighted(logits, targets, w_border)
            + 0.3 * dice_loss(logits, targets)
            + 0.4 * tversky_loss(logits, targets))
