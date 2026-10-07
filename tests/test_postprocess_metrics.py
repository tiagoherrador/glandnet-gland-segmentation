import numpy as np
from glandnet.metrics import dice_pixel, object_metrics
from glandnet.postprocess import clean_mask, segment, split_glands_watershed


def two_blobs():
    m = np.zeros((64, 64), np.uint8)
    m[10:30, 10:30] = 1
    m[40:60, 40:60] = 1
    return m


def test_clean_mask_removes_specks():
    m = two_blobs(); m[0, 0] = 1
    assert clean_mask(m, min_size=5)[0, 0] == 0


def test_watershed_finds_two_separate_glands():
    labels = split_glands_watershed(two_blobs(), footprint_size=9)
    assert len(np.unique(labels)) - 1 == 2


def test_watershed_splits_two_touching_discs():
    yy, xx = np.mgrid[:80, :120]
    m = (((yy - 40) ** 2 + (xx - 35) ** 2) < 28 ** 2) | (((yy - 40) ** 2 + (xx - 85) ** 2) < 28 ** 2)
    assert len(np.unique(split_glands_watershed(m.astype(np.uint8), footprint_size=15))) - 1 == 2


def test_segment_thresholds_a_probability_map():
    prob = np.zeros((64, 64)); prob[10:30, 10:30] = 0.9; prob[40:60, 40:60] = 0.4
    assert len(np.unique(segment(prob, thr=0.5, min_size=5, footprint_size=9))) - 1 == 1


def test_dice_identical_and_disjoint():
    a = two_blobs()
    assert dice_pixel(a, a) > 0.999
    assert dice_pixel(a, 1 - a) < 1e-3


def test_object_metrics_perfect_and_empty():
    a = two_blobs()
    m = object_metrics(a, a)
    assert m["F1"] > 0.999 and m["dice_mean"] > 0.999
    assert object_metrics(np.zeros_like(a), a)["F1"] == 0.0


def test_object_metrics_counts_a_missed_gland():
    a = two_blobs(); pred = a.copy(); pred[40:60, 40:60] = 0
    assert 0.6 < object_metrics(pred, a)["F1"] < 0.7  # TP=1, FN=1, FP=0 -> 2/3
