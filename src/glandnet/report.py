"""Sensitivity / specificity trade-off plot for the decision threshold."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_threshold_tradeoff(sweep: pd.DataFrame, out_path, chosen: float, title: str = ""):
    """`sweep`: columns threshold, sensitivity, specificity (e.g. results/threshold_sweep_oof.csv)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sweep["threshold"], sweep["sensitivity"] * 100, "o-", color="#e00000", label="Sensitivity (malignant images caught)")
    ax.plot(sweep["threshold"], sweep["specificity"] * 100, "o-", color="#00b050", label="Specificity (benign images correctly cleared)")
    ax.axvline(chosen, color="k", ls="--", lw=1)
    ax.text(chosen + 0.005, 52, f"chosen cut-off = {chosen}", fontsize=9)
    ax.set_xlabel("Cut-off: call an image malignant if its probability is at least ...")
    ax.set_ylabel("%"); ax.set_ylim(40, 102); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title or "Lowering the cut-off catches more cancers but flags more healthy images")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
