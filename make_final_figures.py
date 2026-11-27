"""Figures added for the final report: the project pipeline and predicted-versus-observed throughput.

Usage (from the analysis folder, after run_models.py):  python make_final_figures.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from make_figures import BLUE, INK, INK2, MUTED, ORANGE, save
from tput import config as C

STAGES = [
    ("Data", "Public 5G dataset\n83 CSV traces\nSQLite + Parquet", True),
    ("Preprocessing", "Merge duplicates\nSentinels to missing\nEligibility rule", True),
    ("EDA", "Summaries, figures\nIdle share, gaps\nInsights per RQ", True),
    ("Feature\nEngineering", "Lags, 10 s mean/SD\nMissing flags, context\nImpute, scale, one-hot", True),
    ("Model", "Persistence baseline\nRidge, RF, GB\nNested grouped CV", True),
    ("Deployment", "Prediction service\nfor players and\ntest benches", False),
]


def pipeline():
    """Two rows of three stages, read left to right: Data → Preprocessing → EDA, then Feature Engineering → Model → Deployment."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    w, h, gap = 1.62, 1.45, 0.42
    xs = [0.12 + i * (w + gap) for i in range(3)]
    ys = [2.35, 0.45]
    box = {}
    for i, (name, detail, done) in enumerate(STAGES):
        x, y = xs[i % 3], ys[i // 3]
        box[i] = (x, y)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#E8EEF6" if done else "white", ec=BLUE if done else ORANGE, lw=1.5,
                                    ls="-" if done else "--"))
        ax.text(x + w / 2, y + h - 0.26, f"{i + 1}. {name.replace(chr(10), ' ')}", ha="center", va="center",
                fontsize=9.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + 0.58, detail, ha="center", va="center", fontsize=8.5, color=INK2, linespacing=1.35)
    arrow = dict(arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.3)
    for i in (0, 1, 3, 4):  # within a row
        (x, y) = box[i]
        ax.add_patch(FancyArrowPatch((x + w + 0.03, y + h / 2), (x + w + gap - 0.03, y + h / 2), **arrow))
    # EDA (end of row 1) to Feature Engineering (start of row 2), routed through the gap between the rows
    (x2, y2), (x3, y3) = box[2], box[3]
    mid = (y2 + y3 + h) / 2
    ax.plot([x2 + w / 2, x2 + w / 2, x3 + w / 2], [y2 - 0.03, mid, mid], color=MUTED, lw=1.3, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((x3 + w / 2, mid), (x3 + w / 2, y3 + h + 0.03), **arrow))
    ax.legend(handles=[Line2D([0], [0], color=BLUE, lw=1.6, label="Implemented and evaluated"),
                       Line2D([0], [0], color=ORANGE, lw=1.6, ls="--", label="Proposed (not implemented)")],
              loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9, frameon=False)
    ax.text(3.0, 0.12, "Sensitivity schemes and feature importance feed back into the interpretation of each RQ.",
            ha="center", fontsize=8.5, color=MUTED)
    save(fig, "fig9_pipeline.png")


def predicted_vs_observed():
    p = pd.read_parquet(C.ROOT / "outputs" / "models" / "predictions_outer_fold.parquet")
    y = p[C.TARGET].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), sharex=True, sharey=True)
    lim = np.log10(1 + 600)
    for ax, (col, title) in zip(axes, [("persistence", "Persistence (baseline)"),
                                       ("base_selected", "Gradient boosting (selected)")]):
        x = np.log10(1 + np.clip(p[col].to_numpy(), 0, None))
        hb = ax.hexbin(np.log10(1 + y), x, gridsize=45, bins="log", mincnt=1, cmap="Blues", linewidths=0)
        ax.plot([0, lim], [0, lim], color=ORANGE, lw=1, ls="--")
        mae = np.mean(np.abs(y - p[col].to_numpy()))
        ax.set_title(f"{title}\nrow MAE {mae:.2f} Mbps", fontsize=9, color=INK2, loc="left")
        ticks = [0, 1, 10, 100, 500]
        ax.set_xticks(np.log10(1 + np.array(ticks)), ticks)
        ax.set_yticks(np.log10(1 + np.array(ticks)), ticks)
        ax.grid(alpha=0.3)
    fig.supxlabel("Observed throughput 5 s ahead (Mbps)", fontsize=10, color=INK2, y=-0.04)
    axes[0].set_ylabel("Predicted (Mbps)")
    fig.colorbar(hb, ax=axes, shrink=0.85, label="Prediction origins (log count)")
    save(fig, "fig10_predicted_vs_observed.png")


if __name__ == "__main__":
    pipeline()
    predicted_vs_observed()
