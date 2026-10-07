"""Figures of the full pipeline: segmentation -> gland instances -> benign/malignant call per gland.

Gland boxes are drawn **green = predicted benign** and **red = predicted malignant**, numbered and labelled with p(malignant).
Besides the figure, every image gets a results table (one row per gland: size, bounding box, probability, call), shown under
the figure and saved as CSV, plus one `summary.csv` with the image-level verdicts.

    glandnet-visualize --data-root data --seg-checkpoint models/unet4l.pth --cls-checkpoint models/gland_classifier_effb0.pth \
        --metrics results/segmentation_metrics.json --names testA_2 testB_8 --out assets --tables results/examples

Works on any image: pass `--images path/to/slide.png` instead of `--names`. Expert annotations and pathologist grades are
optional extras; when they are missing the figure simply omits those lines.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from .classifier import crops_from_labels, load_classifier, predict, read_grades
from .data import IMG_SIZE, load_image, load_mask, normalize, resize_image
from .metrics import dice_pixel
from .model import UNet4L, pick_device
from .postprocess import gland_instances, predict_prob_map, segment

BENIGN_COLOR, MALIGNANT_COLOR = "#00b050", "#e00000"
GRADE_NAMES = {0: "benign", 1: "malignant"}


def run_pipeline(img, seg_model, cls_model, device, params):
    """RGB image -> (instance labels at original size, list of per-gland dicts with bbox and prob_malignant)."""
    H, W = img.shape[:2]
    x = normalize(resize_image(img)).unsqueeze(0).to(device)
    labels512 = gland_instances(segment(predict_prob_map(seg_model, x)[0], params["thr"], params["min_size"], params["footprint"]))
    labels = cv2.resize(labels512.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    glands = crops_from_labels(img, labels, "", 0, 0, "pred")
    if glands:
        for i, (g, p) in enumerate(zip(glands, predict(cls_model, glands, device)), start=1):
            g["id"], g["prob_malignant"] = i, float(p)
    return labels, glands


def image_verdict(glands, thr: float = 0.5):
    """Image-level call = mean gland probability (same rule as the evaluation)."""
    if not glands:
        return None, float("nan")
    p = float(np.mean([g["prob_malignant"] for g in glands]))
    return int(p >= thr), p


def glands_table(glands, name: str, thr: float = 0.5) -> pd.DataFrame:
    """One row per detected gland: size, bounding box (pixels), malignancy probability and call."""
    rows = []
    for g in glands:
        r0, c0, r1, c1 = g["bbox"]
        p = g["prob_malignant"]
        rows.append({"image": name, "gland": g["id"], "size_px2": g["area"], "row_min": r0, "col_min": c0,
                     "row_max": r1, "col_max": c1, "p_malignant": round(p, 4),
                     "prediction": GRADE_NAMES[int(p >= thr)]})
    return pd.DataFrame(rows)


def summary_row(glands, name: str, true_label: int | None = None, thr: float = 0.5) -> dict:
    pred, p = image_verdict(glands, thr)
    return {"image": name, "n_glands": len(glands), "n_glands_malignant": int(sum(g["prob_malignant"] >= thr for g in glands)),
            "mean_p_malignant": round(p, 4), "image_prediction": None if pred is None else GRADE_NAMES[pred],
            "true_grade": None if true_label is None else GRADE_NAMES[true_label],
            "correct": None if true_label is None else pred == true_label}


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
            ax.text(c0 + 2, r0 + 2, f"#{g['id']}  {g['prob_malignant']:.2f}", color="white", fontsize=7, va="top",
                    bbox=dict(facecolor=color, edgecolor="none", pad=1.0, alpha=0.85))


def draw_results_table(ax, glands, name, thr: float = 0.5, max_rows: int = 10, min_area: int = DISPLAY_MIN_AREA):
    shown = sorted([g for g in glands if g["area"] >= min_area], key=lambda g: -g["area"])[:max_rows]
    ax.axis("off")
    if not shown:
        return
    cols = ["Gland", "Size (px\u00b2)", "Probability of malignancy", "Prediction", "Bounding box (row_min, col_min, row_max, col_max)"]
    cells = [[f"#{g['id']}", f"{g['area']:,}", f"{g['prob_malignant']:.2f}",
              GRADE_NAMES[int(g["prob_malignant"] >= thr)].capitalize(), str(tuple(int(v) for v in g["bbox"]))] for g in shown]
    t = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center", colWidths=[0.08, 0.12, 0.2, 0.14, 0.4])
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.35)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#e8e8e8"); t[0, j].set_text_props(weight="bold")
    for i, g in enumerate(shown, start=1):
        t[i, 3].set_facecolor("#ffd6d6" if g["prob_malignant"] >= thr else "#d6f2de")
    ax.set_title(f"Results table for {name}: the {len(shown)} largest of {len(glands)} detected glands "
                 f"(full table in the CSV file)", fontsize=10, pad=4)


def render_figure(img, gt_mask, labels, glands, true_label, name, out_path, thr: float = 0.5,
                  min_area: int = DISPLAY_MIN_AREA):
    """`gt_mask` (expert annotation) and `true_label` (pathologist's grade) are optional (None)."""
    pred_label, p_img = image_verdict(glands, thr)
    fig = plt.figure(figsize=(13, 13.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.62])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax[0].imshow(img); ax[0].set_title(f"Original image ({name})", fontsize=12)
    pred_mask = labels > 0
    ax[1].imshow(pred_mask, cmap="gray")
    title = "Mask predicted by the model (white = gland)"
    if gt_mask is not None:
        title += f"\nOverlap with the expert annotation (Dice): {dice_pixel(pred_mask, gt_mask > 0):.2f}"
    ax[1].set_title(title, fontsize=12)
    shown = np.ma.masked_where(labels == 0, labels % 20 + 1)
    ax[2].imshow(img); ax[2].imshow(shown, cmap="tab20", alpha=0.55, interpolation="nearest")
    ax[2].set_title(f"Glands found by the model ({len(glands)})", fontsize=12)
    ax[3].imshow(img); draw_gland_boxes(ax[3], glands, thr, min_area=min_area)
    n_mal = sum(g["prob_malignant"] >= thr for g in glands)
    verdict = "n/a" if pred_label is None else f"{GRADE_NAMES[pred_label]} (average probability {p_img:.2f})"
    t3 = (f"Each gland classified: green = benign, red = malignant\n"
          f"{n_mal} of {len(glands)} glands malignant -> image called {verdict}")
    if true_label is not None:
        t3 += f"\nPathologist's grade: {GRADE_NAMES[true_label]}"
    ax[3].set_title(t3, fontsize=12)
    for a_ in ax:
        a_.axis("off")
    draw_results_table(fig.add_subplot(gs[2, :]), glands, name, thr, min_area=min_area)
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
    ap.add_argument("--names", nargs="*", default=[], help="GlaS image names, e.g. testA_2 testB_8 (annotation and grade are used if available)")
    ap.add_argument("--images", nargs="*", default=[], help="paths to any other images (no annotation / grade needed)")
    ap.add_argument("--out", default="assets", help="folder for the PNG figures")
    ap.add_argument("--tables", default="results/examples", help="folder for the per-image CSV tables and summary.csv")
    a = ap.parse_args()
    device = pick_device()
    params = json.load(open(a.metrics))["selected_on_validation"]
    seg = UNet4L().to(device); seg.load_state_dict(torch.load(a.seg_checkpoint, map_location=device)); seg.eval()
    cls = load_classifier(a.cls_checkpoint, device)
    grades = read_grades(a.data_root).set_index("name")
    root = Path(a.data_root)
    if not (a.names or a.images):
        ap.error("give at least one of --names or --images")
    items = []
    for n in a.names:
        d = "train" if n.startswith("train") else "test"
        mpath = root / d / "masks" / f"{n}_anno.bmp"
        items.append((n, load_image(root / d / "images" / f"{n}.bmp"), load_mask(mpath) if mpath.exists() else None,
                      int(grades.loc[n, "label"]) if n in grades.index else None))
    for pth in a.images:
        items.append((Path(pth).stem, load_image(pth), None, None))
    summary = []
    Path(a.tables).mkdir(parents=True, exist_ok=True)
    for n, img, gt, true in items:
        labels, glands = run_pipeline(img, seg, cls, device, params)
        pred, p = render_figure(img, gt, labels, glands, true, n, Path(a.out) / f"pipeline_{n}.png")
        glands_table(glands, n).to_csv(Path(a.tables) / f"{n}_glands.csv", index=False)
        summary.append(summary_row(glands, n, true))
        print(n, "true:", None if true is None else GRADE_NAMES[true], "pred:", None if pred is None else GRADE_NAMES[pred], f"p={p:.2f}", f"glands={len(glands)}")
    pd.DataFrame(summary).to_csv(Path(a.tables) / "summary.csv", index=False)


if __name__ == "__main__":
    main()