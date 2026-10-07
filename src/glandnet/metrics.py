"""Pixel Dice and object-level metrics (Dice, Hausdorff, F1 of matched glands)."""
import numpy as np
from skimage.measure import label, regionprops
from skimage.metrics import hausdorff_distance


def dice_pixel(pred, gt, eps: float = 1e-6) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float(2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + eps))


def object_metrics(pred_mask, gt_mask, match_dice: float = 0.5) -> dict:
    """Each ground-truth gland is matched to the predicted gland with the highest Dice;
    a match counts as TP when Dice > `match_dice`. F1 = TP / (TP + (FP + FN) / 2)."""
    lp, lg = label(pred_mask), label(gt_mask)
    pr, gr = regionprops(lp), regionprops(lg)
    if not pr or not gr:
        return {"dice_mean": 0.0, "hd_mean": 0.0, "F1": 0.0}
    pred_masks = [lp == r.label for r in pr]
    dices, hds, matched, tp, fn = [], [], set(), 0, 0
    for g in gr:
        gm = lg == g.label
        scores = [dice_pixel(pm, gm) for pm in pred_masks]
        j = int(np.argmax(scores))
        if scores[j] > match_dice:
            tp += 1
            matched.add(j)
            dices.append(scores[j])
            try:
                hds.append(hausdorff_distance(gm, pred_masks[j]))
            except ValueError:
                hds.append(np.nan)
        else:
            fn += 1
    fp = len(pr) - len(matched)
    return {
        "dice_mean": float(np.nanmean(dices)) if dices else 0.0,
        "hd_mean": float(np.nanmean(hds)) if hds else 0.0,
        "F1": float(tp / (tp + 0.5 * (fp + fn) + 1e-6)),
    }
