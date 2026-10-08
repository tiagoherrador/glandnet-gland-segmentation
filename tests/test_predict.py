import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from glandnet.classifier import load_classifier
from glandnet.model import UNet4L
from glandnet.predict import CLASSIFIER_FILE, UNET_FILE, collect_images, main


def test_collect_images_expands_folders_and_rejects_bad_paths(tmp_path):
    for n in ("a.png", "b.JPG", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in collect_images([tmp_path])] == ["a.png", "b.JPG"]
    with pytest.raises(FileNotFoundError):
        collect_images([tmp_path / "missing.png"])


def test_missing_models_gives_a_helpful_message(tmp_path):
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(tmp_path / "x.png")
    with pytest.raises(FileNotFoundError, match="shared on request"):
        main(["--images", str(tmp_path / "x.png"), "--models", str(tmp_path / "nomodels"), "--out", str(tmp_path / "o")])


def test_end_to_end_with_random_weights_writes_figure_and_tables(tmp_path):
    """Smoke test: random weights, so the numbers mean nothing; checks that every output is produced."""
    import timm
    torch.save(UNet4L().state_dict(), tmp_path / UNET_FILE)
    torch.save(timm.create_model("efficientnet_b0", pretrained=False, num_classes=2).state_dict(), tmp_path / CLASSIFIER_FILE)
    Image.fromarray(np.random.randint(0, 255, (200, 260, 3), np.uint8)).save(tmp_path / "slide.png")
    out = tmp_path / "out"
    main(["--images", str(tmp_path / "slide.png"), "--models", str(tmp_path), "--out", str(out)])
    assert (out / "slide.png").stat().st_size > 1000 and (out / "slide_glands.csv").exists()
    s = pd.read_csv(out / "summary.csv")
    assert list(s["image"]) == ["slide"] and pd.isna(s.loc[0, "true_grade"])


def test_default_params_match_the_validation_selected_ones():
    """Guards against the shipped defaults drifting away from results/segmentation_metrics.json."""
    import json
    from pathlib import Path
    from glandnet.predict import DEFAULT_PARAMS
    res = Path(__file__).resolve().parents[1] / "results" / "segmentation_metrics.json"
    assert json.load(open(res))["selected_on_validation"] == DEFAULT_PARAMS
