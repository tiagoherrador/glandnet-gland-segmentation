"""Run the trained GlandNet models on your own images.

    glandnet-predict --images slide.png other_folder/ --models models/ --out predictions/

For every image you get a figure (original, predicted mask, glands found, each gland coloured green/red), a table with one row
per gland (`<image>_glands.csv`) and one `summary.csv` with the image-level verdicts. No annotations or dataset needed.
The trained models are shared on request (see the README).
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from .classifier import DECISION_THRESHOLD, load_classifier
from .data import load_image
from .model import UNet4L, pick_device
from .visualize import glands_table, render_figure, run_pipeline, summary_row

# Post-processing parameters selected on the validation split (see results/segmentation_metrics.json).
DEFAULT_PARAMS = {"thr": 0.65, "min_size": 25, "footprint": 22}
UNET_FILE, CLASSIFIER_FILE = "unet4l.pth", "gland_classifier_effb0.pth"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(paths) -> list[Path]:
    """Expand files and folders into a sorted list of image files."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.suffix.lower() in IMAGE_EXTS and p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(f"not an image file or folder: {p}")
    return out


def load_models(models_dir, device):
    models_dir = Path(models_dir)
    missing = [f for f in (UNET_FILE, CLASSIFIER_FILE) if not (models_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {', '.join(missing)} in '{models_dir}'. The trained models are shared on request: see 'Get the trained models' in the README.")
    seg = UNet4L().to(device)
    seg.load_state_dict(torch.load(models_dir / UNET_FILE, map_location=device))
    return seg.eval(), load_classifier(models_dir / CLASSIFIER_FILE, device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="image files and/or folders (png, jpg, bmp, tif)")
    ap.add_argument("--models", default="models", help=f"folder with {UNET_FILE} and {CLASSIFIER_FILE}")
    ap.add_argument("--out", default="predictions", help="folder for figures and tables")
    ap.add_argument("--threshold", type=float, default=DECISION_THRESHOLD, help="call malignant if probability >= this (default 0.4)")
    a = ap.parse_args(argv)

    images = collect_images(a.images)
    if not images:
        ap.error("no images found")
    device = pick_device()
    seg, cls = load_models(a.models, device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in images:
        name = path.stem
        img = load_image(path)
        labels, glands = run_pipeline(img, seg, cls, device, DEFAULT_PARAMS)
        render_figure(img, None, labels, glands, None, name, out / f"{name}.png", thr=a.threshold)
        glands_table(glands, name, a.threshold).to_csv(out / f"{name}_glands.csv", index=False)
        row = summary_row(glands, name, None, a.threshold)
        summary.append(row)
        print(f"{name}: {row['n_glands']} glands, {row['n_glands_malignant']} malignant -> {row['image_prediction']} (mean p = {row['mean_p_malignant']})")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(f"results in {out}/")


if __name__ == "__main__":
    main()
