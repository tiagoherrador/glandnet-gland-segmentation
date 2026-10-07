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
from .postprocess import predict_prob_map, segment

THRESHOLDS = [round(t, 2) for t in np.linspace(0.3, 0.7, 9)]
POSTPROC = [(15, 15), (20, 20), (25, 22), (25, 25)]  # (min_size, footprint_size)


def collect(model, pairs, device):
    """Forward pass once; returns list of (prob_map [H,W], gt [H,W], name)."""
    out = []
    for x, gt, name in DataLoader(GlasEvalDataset(pairs), batch_size=1):
        out.append((predict_prob_map(model, x.to(device))[0], gt[0].numpy(), name[0]))
    return out


def score(items, thr, min_size, footprint):
    rows = []
    for prob, gt, _ in items:
        pred = (segment(prob, thr, min_size, footprint) > 0).astype(np.uint8)
        m = object_metrics(pred, gt)
        m["dice_pixel"] = dice_pixel(pred, gt)
        rows.append(m)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def select_params(val_items):
    """Grid search maximising object-level F1 on validation."""
    best = None
    for thr, (ms, fp) in itertools.product(THRESHOLDS, POSTPROC):
        s = score(val_items, thr, ms, fp)
        if best is None or s["F1"] > best[0]["F1"]:
            best = (s, dict(thr=thr, min_size=ms, footprint=fp))
    return best


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
    val_score, params = select_params(val_items)
    print("selected on validation:", params, val_score)

    res = {"selected_on_validation": params, "validation": val_score, "test": {}}
    allitems = []
    for sub in ("A", "B"):
        items = collect(model, list_pairs(a.data_root, "test", sub), device)
        allitems += items
        res["test"][sub] = score(items, params["thr"], params["min_size"], params["footprint"])
        print("test", sub, res["test"][sub])
    res["test"]["A+B"] = score(allitems, params["thr"], params["min_size"], params["footprint"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
