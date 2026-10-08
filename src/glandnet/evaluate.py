"""Evaluate a trained U-Net with a leakage-free protocol.

Post-processing parameters (probability threshold, min object size, watershed footprint) are
selected on the **validation** split only (same 80/20 split as training) and then applied
unchanged to test A and test B.

    glandnet-eval --data-root data --checkpoint models/unet4l.pth --out results/segmentation_metrics.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GlasEvalDataset, list_pairs, split_train_val
from .metrics import dice_pixel, object_metrics
from .model import UNet4L, pick_device
from skimage.measure import label as cc_label

from .postprocess import gland_instances, predict_prob_map, segment

THRESHOLDS = [round(t, 2) for t in np.linspace(0.3, 0.7, 9)]
# Minimum gland size in pixels at 512 x 512 (262,144 px): 25 px = 0.01 % of the image ... 4,000 px = 1.5 %.
# Predicted regions smaller than this are discarded as spurious detections (see docs/min_gland_size.md).
MIN_SIZES = [15, 25, 50, 100, 250, 500, 1000, 2000, 4000]
FOOTPRINT = 22  # watershed footprint: it does not change the gland mask, so it is not searched
POSTPROC = [(m, FOOTPRINT) for m in MIN_SIZES]  # (min_size, footprint_size)


def collect(model, pairs, device):
    """Forward pass once; returns list of (prob_map [H,W], gt [H,W], name)."""
    out = []
    for x, gt, name in DataLoader(GlasEvalDataset(pairs), batch_size=1):
        out.append((predict_prob_map(model, x.to(device))[0], gt[0].numpy(), name[0]))
    return out


def score(items, thr, min_size, footprint):
    """Mean object/pixel metrics over images, plus detection counts summed over images:
    n_gt (ground-truth glands), n_pred (predicted glands) and n_extra (predicted glands with < 10 % of their pixels
    inside the annotation, i.e. spurious detections)."""
    rows, n_gt = [], 0
    n_pred = n_extra = 0
    for prob, gt, _ in items:
        inst = gland_instances(segment(prob, thr, min_size, footprint))
        pred = (inst > 0).astype(np.uint8)
        m = object_metrics(pred, gt)
        m["dice_pixel"] = dice_pixel(pred, gt)
        rows.append(m)
        ids = [i for i in np.unique(inst) if i > 0]
        n_pred += len(ids)
        n_extra += sum(1 for i in ids if (gt[inst == i] > 0).mean() < 0.1)
        n_gt += int(cc_label(gt > 0).max())
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out.update(n_gt=n_gt, n_pred=n_pred, n_extra=n_extra)
    return out


def select_params(val_items):
    """Grid search maximising object-level F1 on validation. Returns (best score, best params, full sweep)."""
    best, sweep = None, []
    for thr, (ms, fp) in itertools.product(THRESHOLDS, POSTPROC):
        sc = score(val_items, thr, ms, fp)
        sweep.append({"thr": thr, "min_size": ms, "footprint": fp, **sc})
        if best is None or sc["F1"] > best[0]["F1"]:
            best = (sc, dict(thr=thr, min_size=ms, footprint=fp))
    return best[0], best[1], sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="results/segmentation_metrics.json")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    device = pick_device()
    model = UNet4L().to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    train_pairs = list_pairs(a.data_root, "train")
    _, va_idx = split_train_val(len(train_pairs), 0.2, a.seed)
    val_items = collect(model, [train_pairs[i] for i in va_idx], device)
    val_score, params, sweep = select_params(val_items)
    print("selected on validation:", params, val_score)

    res = {"selected_on_validation": params, "validation": val_score, "test": {}}
    allitems, per_set = [], {}
    for sub in ("A", "B"):
        items = collect(model, list_pairs(a.data_root, "test", sub), device)
        allitems += items
        per_set[sub] = items
        res["test"][sub] = score(items, params["thr"], params["min_size"], params["footprint"])
        print("test", sub, res["test"][sub])
    res["test"]["A+B"] = score(allitems, params["thr"], params["min_size"], params["footprint"])
    # Reference only (NOT used for selection): the same test sets with the previous, very permissive minimum size (25 px)
    old = dict(thr=params["thr"], min_size=25, footprint=params["footprint"])
    res["test_reference_min_size_25"] = {"params": old, "A": score(per_set["A"], **old), "B": score(per_set["B"], **old), "A+B": score(allitems, **old)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(sweep).to_csv(Path(a.out).parent / "validation_sweep.csv", index=False)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
