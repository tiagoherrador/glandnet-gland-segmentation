import numpy as np
import pandas as pd
from PIL import Image

from glandnet.classifier import crops_from_labels, read_grades
from glandnet.data import build_border, list_pairs, load_mask, split_train_val


def test_split_is_reproducible_and_disjoint():
    a1, b1 = split_train_val(85, 0.2, 42)
    a2, b2 = split_train_val(85, 0.2, 42)
    assert (a1, b1) == (a2, b2)
    assert not set(a1) & set(b1) and len(a1) + len(b1) == 85 and len(b1) == 17


def test_build_border_is_a_ring_around_the_mask():
    m = np.zeros((40, 40), bool); m[10:30, 10:30] = True
    b = build_border(m, iterations=2)
    assert b[20, 20] == 0 and b[10, 20] == 1 and b[0, 0] == 0


def test_list_pairs_and_masks(tmp_path):
    for split, names in (("train", ["train_1"]), ("test", ["testA_1", "testB_1"])):
        (tmp_path / split / "images").mkdir(parents=True); (tmp_path / split / "masks").mkdir(parents=True)
        for n in names:
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / split / "images" / f"{n}.bmp")
            Image.fromarray(np.eye(8, dtype=np.uint8) * 3).save(tmp_path / split / "masks" / f"{n}_anno.bmp")
    assert len(list_pairs(tmp_path, "train")) == 1
    assert [p[0].stem for p in list_pairs(tmp_path, "test", "B")] == ["testB_1"]
    assert load_mask(list_pairs(tmp_path, "train")[0][1]).sum() == 8


def test_read_grades_handles_gla_s_header_and_labels(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "Grade.csv").write_text(
        "name,patient ID, grade (GlaS), grade (Sirinukunwattana et al. 2015)\r\n"
        "testA_1,4, benign, adenomatous\r\ntestA_2,7, malignant, poorly differentiated\r\n")
    g = read_grades(tmp_path)
    assert g["label"].tolist() == [0, 1] and g["patient"].tolist() == [4, 7]


def test_crops_one_per_gland_and_small_ones_dropped():
    img = np.random.randint(0, 255, (100, 100, 3), np.uint8)
    lab = np.zeros((100, 100), np.int32); lab[10:40, 10:40] = 1; lab[60:90, 60:90] = 2; lab[0:3, 0:3] = 3
    crops = crops_from_labels(img, lab, "x", 1, 5, "gt")
    assert len(crops) == 2 and crops[0]["crop"].shape == (224, 224, 3)


def test_operating_point_counts_and_rates():
    from glandnet.classifier import operating_point
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.45, 0.35, 0.2, 0.3, 0.1, 0.55, 0.05]
    o = operating_point(y, p, 0.4)  # called malignant: 0.9, 0.45, 0.55
    assert (o["tp"], o["fn"], o["fp"], o["tn"]) == (2, 2, 1, 3)
    assert o["sensitivity"] == 0.5 and o["specificity"] == 0.75 and o["precision"] == 2 / 3


def test_lower_threshold_never_lowers_sensitivity():
    from glandnet.classifier import threshold_sweep
    rng = np.random.default_rng(0)
    img = pd.DataFrame({"label": rng.integers(0, 2, 200), "p": rng.random(200)})
    sw = threshold_sweep(img, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6))
    assert (np.diff(sw["sensitivity"]) <= 1e-12).all() and (np.diff(sw["specificity"]) >= -1e-12).all()


def test_default_threshold_is_the_cautious_one():
    from glandnet.classifier import DECISION_THRESHOLD
    assert DECISION_THRESHOLD == 0.4
