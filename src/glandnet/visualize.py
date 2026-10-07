"""Figures of the full pipeline: segmentation -> gland instances -> benign/malignant call per gland.

Gland boxes are drawn **green = predicted benign** and **red = predicted malignant**, labelled with p(malignant).

    glandnet-visualize --data-root data --seg-checkpoint models/unet4l.pth --cls-checkpoint models/gland_classifier_effb0.pth \
        --metrics results/segmentation_metrics.json --names testA_2 testB_8 --out assets
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from .classifier import crops_from_labels, load_classifier, predict, read_grades
from .data import IMG_SIZE, load_image, load_mask, normalize, resize_image
from .model import UNet4L, pick_device
from .postprocess import predict_prob_map, segment

BENIGN_COLOR, MALIGNANT_COLOR = "#00b050", "#e00000"
GRADE_NAMES = {0: "benign", 1: "malignant"}


def run_pipeline(img, seg_model, cls_model, device, params):
    """RGB image -> (instance labels at original size, list of per-gland dicts with bbox and prob_malignant)."""
    H, W = img.shape[:2]
    x = normalize(resize_image(img)).unsqueeze(0).to(device)
    labels512 = segment(predict_prob_map(seg_model, x)[0], params["thr"], params["min_size"], params["footprint"])
    labels = cv2.resize(labels512.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    glands = crops_from_labels(img, labels, "", 0, 0, "pred")
    if glands:
        for g, p in zip(glands, predict(cls_model, glands, device)):
            g["prob_malignant"] = float(p)
    return labels, glands


def image_verdict(glands, thr: float = 0.5):
    """Image-level call = mean gland probability (same rule as the evaluation)."""
    if not glands:
        return None, float("nan")
    p = float(np.mean([g["prob_malignant"] for g in glands]))
    return int(p >= thr), p


DISPLAY_MIN_AREA = 400  # px; tiny fragments are hidden in the figure only (the image verdict uses every gland)


def draw_gland_boxes(ax, glands, thr: float = 0.5, show_prob: bool = True, lw: float = 1.8,
                     min_area: int = DISPLAY_MIN_AREA):
    for g in glands:
        if g.get("area", min_area) < min_area:
            continue
        r0, c0, r1, c1 = g["bbox"]
        color = MALIGNANT_COLOR if g["prob_malignant"] >= thr else BENIGN_COLOR
        ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor=color, linewidth=lw))
        if show_prob:
            ax.text(c0 + 2, r0 + 2, f"{g['prob_malignant']:.2f}", color="white", fontsize=7, va="top",
                    bbox=dict(facecolor=color, edgecolor="none", pad=1.0, alpha=0.85))


def render_figure(img, gt_mask, labels, glands, true_label, name, out_path, thr: float = 0.5,
                  min_area: int = DISPLAY_MIN_AREA):
    pred_label, p_img = image_verdict(glands, thr)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = axes.ravel()
    ax[0].imshow(img); ax[0].set_title(f"Original image ({name})", fontsize=12)
    ax[1].imshow(gt_mask, cmap="gray"); ax[1].set_title("Ground truth (binary mask)", fontsize=12)
    shown = np.ma.masked_where(labels == 0, labels % 20 + 1)
    ax[2].imshow(img); ax[2].imshow(shown, cmap="tab20", alpha=0.55, interpolation="nearest")
    ax[2].set_title(f"Predicted glands ({len(glands)} instances)", fontsize=12)
    ax[3].imshow(img); draw_gland_boxes(ax[3], glands, thr, min_area=min_area)
    n_mal = sum(g["prob_malignant"] >= thr for g in glands)
    verdict = "n/a" if pred_label is None else f"{GRADE_NAMES[pred_label]} (mean p = {p_img:.2f})"
    ax[3].set_title(f"Gland classification: green = benign, red = malignant\n"
                    f"{n_mal}/{len(glands)} glands malignant -> image: {verdict}\n"
                    f"Ground-truth grade: {GRADE_NAMES[true_label]}", fontsize=12)
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.28)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return pred_label, p_img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--seg-checkpoint", required=True)
    ap.add_argument("--cls-checkpoint", required=True)
    ap.add_argument("--metrics", default="results/segmentation_metrics.json", help="holds the validation-selected post-processing params")
    ap.add_argument("--names", nargs="+", required=True, help="test image names, e.g. testA_2 testB_8")
    ap.add_argument("--out", default="assets")
    a = ap.parse_args()
    device = pick_device()
    params = json.load(open(a.metrics))["selected_on_validation"]
    seg = UNet4L().to(device); seg.load_state_dict(torch.load(a.seg_checkpoint, map_location=device)); seg.eval()
    cls = load_classifier(a.cls_checkpoint, device)
    grades = read_grades(a.data_root).set_index("name")
    root = Path(a.data_root)
    for n in a.names:
        d = "train" if n.startswith("train") else "test"
        img = load_image(root / d / "images" / f"{n}.bmp")
        gt = load_mask(root / d / "masks" / f"{n}_anno.bmp")
        labels, glands = run_pipeline(img, seg, cls, device, params)
        pred, p = render_figure(img, gt, labels, glands, int(grades.loc[n, "label"]), n, Path(a.out) / f"pipeline_{n}.png")
        print(n, "true:", GRADE_NAMES[int(grades.loc[n, 'label'])], "pred:", None if pred is None else GRADE_NAMES[pred], f"p={p:.2f}", f"glands={len(glands)}")


if __name__ == "__main__":
    main()
