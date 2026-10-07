import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from glandnet.visualize import BENIGN_COLOR, MALIGNANT_COLOR, draw_gland_boxes, image_verdict, render_figure

G = [dict(id=1, bbox=(10, 10, 40, 40), prob_malignant=0.9, area=900),
     dict(id=2, bbox=(50, 50, 90, 90), prob_malignant=0.1, area=1600),
     dict(id=3, bbox=(0, 0, 5, 5), prob_malignant=0.99, area=25)]


def test_boxes_are_red_for_malignant_green_for_benign_and_tiny_ones_are_hidden():
    fig, ax = plt.subplots()
    draw_gland_boxes(ax, G, thr=0.5, min_area=400)
    colors = [matplotlib.colors.to_hex(p.get_edgecolor()) for p in ax.patches]
    assert colors == [MALIGNANT_COLOR, BENIGN_COLOR]  # the 25 px fragment is not drawn
    plt.close(fig)


def test_threshold_moves_the_colour_boundary():
    fig, ax = plt.subplots()
    draw_gland_boxes(ax, G[:1], thr=0.95, min_area=0)
    assert matplotlib.colors.to_hex(ax.patches[0].get_edgecolor()) == BENIGN_COLOR
    plt.close(fig)


def test_image_verdict_uses_every_gland_including_hidden_ones():
    label, p = image_verdict(G)
    assert abs(p - np.mean([0.9, 0.1, 0.99])) < 1e-9 and label == 1
    assert image_verdict([])[0] is None


def test_render_figure_writes_a_file(tmp_path):
    img = np.random.randint(0, 255, (100, 120, 3), np.uint8)
    labels = np.zeros((100, 120), np.int32); labels[10:40, 10:40] = 1
    out = tmp_path / "fig.png"
    pred, p = render_figure(img, labels > 0, labels, [dict(g, id=i + 1) for i, g in enumerate(G)], 1, "testX", out)
    assert out.exists() and out.stat().st_size > 1000 and pred == 1


def test_glands_table_and_summary_row():
    from glandnet.visualize import glands_table, summary_row
    gl = [dict(G[0], id=1), dict(G[1], id=2), dict(G[2], id=3)]
    t = glands_table(gl, "testX")
    assert list(t["prediction"]) == ["malignant", "benign", "malignant"]
    assert {"image", "gland", "size_px2", "row_min", "col_max", "p_malignant"} <= set(t.columns)
    s = summary_row(gl, "testX", true_label=1)
    assert s["n_glands"] == 3 and s["n_glands_malignant"] == 2 and s["correct"] is True and s["true_grade"] == "malignant"
