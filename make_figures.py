"""Weeks 7-8: report figures (static PNG, 300 dpi) from saved results.

Usage (after run_pipeline.py, rq1 tables, run_models.py and analyze_models.py):
  python make_figures.py
Writes outputs/figures/fig*.png. Captions and notes are in outputs/figures/captions.md.
Colours: validated categorical slots 1-2 (blue, orange) on the light surface; reference baselines in neutral grey.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tput import config as C
from tput.stats import bootstrap_mean

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e0", "#fcfcfb"
OUT = C.FIGURES
SPLIT = "outer_fold"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "text.color": INK,
})
LABEL = {"persistence": "Persistence", "base_ridge": "Ridge", "base_rf": "Random forest",
         "base_hgb": "Gradient boosting", "context_selected": "Gradient boosting + context",
         "ref_zero": "Always 0 (reference)", "ref_mean_10s": "10 s mean (reference)"}
rng = np.random.default_rng(C.SEED)


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def jitter(n, width=0.18):
    return rng.uniform(-width, width, n)


def fig1_rq1():
    corr = pd.read_csv(C.TABLES / "rq1_trace_correlations.csv")
    corr = corr[corr.eligible]
    summ = pd.read_csv(C.TABLES / "rq1_summary.csv").set_index("measurement")   # same t-intervals as the table
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for i, m in enumerate(C.RADIO):
        c = corr[corr.measurement == m]
        for mob, col in (("Driving", BLUE), ("Static", ORANGE)):
            x = c[c.mobility_condition == mob].rho
            ax.scatter(x, i + jitter(len(x)), s=14, color=col, alpha=0.75, linewidths=0.4, edgecolors=SURFACE,
                       label=mob if i == 0 else None)
        s = summ.loc[m]
        b = {"mean": s.mean_rho, "ci_low": s.ci95_low, "ci_high": s.ci95_high}
        ax.errorbar(b["mean"], i, xerr=[[b["mean"] - b["ci_low"]], [b["ci_high"] - b["mean"]]], fmt="D",
                    color=INK, ms=5, capsize=3, lw=1.2, zorder=5, label="Mean, 95% CI" if i == 0 else None)
        ax.annotate(f"mean {b['mean']:+.2f}\n[{b['ci_low']:+.2f}, {b['ci_high']:+.2f}]", (1.01, i), va="center",
                    fontsize=9, color=INK2, xycoords=("axes fraction", "data"), annotation_clip=False)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(C.RADIO)), C.RADIO)
    ax.set_ylim(-0.6, len(C.RADIO) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Within-trace Spearman correlation with contemporaneous throughput")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.16), ncol=3, fontsize=9, handletextpad=0.3)
    save(fig, "fig1_rq1_correlations.png")


def _paired(errs, model, ref="persistence"):
    w = errs.pivot_table(index=["trace_id", "application_type", "mobility_condition", "recording_date"],
                         columns="model", values="mae").reset_index()
    return w.assign(delta=w[ref] - w[model], rel=100 * (w[ref] - w[model]) / w[ref])


def fig2_rq2(errs):
    rows = ["base_ridge", "base_rf", "base_hgb", "context_selected", "ref_mean_10s", "ref_zero"]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for i, m in enumerate(rows):
        d = _paired(errs, m)
        b = bootstrap_mean(d.delta)
        ref = m.startswith("ref_")
        col = MUTED if ref else BLUE
        ax.errorbar(b["mean"], i, xerr=[[b["mean"] - b["ci_low"]], [b["ci_high"] - b["mean"]]], fmt="o",
                    color=col, ms=6, capsize=3, lw=1.4)
        ax.annotate(f"{b['mean']:+.2f} Mbps · better on {100 * (d.delta > 0).mean():.0f}% of traces",
                    (b["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axhline(3.5, color=GRID, lw=1)
    ax.set_yticks(range(len(rows)), [LABEL[m] for m in rows])
    for t, m in zip(ax.get_yticklabels(), rows):
        t.set_color(MUTED if m.startswith("ref_") else INK)
    ax.invert_yaxis()
    ax.set_xlim(-4.5, 6)
    ax.set_xlabel("Mean reduction in trace-level MAE vs persistence (Mbps), 95% CI  →  better")
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_rq2_improvement.png")


def fig3_rq3(errs):
    d = _paired(errs, "context_selected", ref="base_selected")
    b = bootstrap_mean(d.delta)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    lim = np.ceil(np.abs(d.delta).max() * 10) / 10
    ax.hist(d.delta, bins=np.linspace(-lim, lim, 41), color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(b["mean"], color=INK, lw=1.2, ls="--")
    ax.annotate(f"mean {b['mean']:+.2f} Mbps (95% CI {b['ci_low']:.2f} to {b['ci_high']:.2f})\n"
                f"context better on {100 * (d.delta > 0).mean():.0f}% of {len(d)} traces",
                (b["mean"], 0.95), xycoords=("data", "axes fraction"), xytext=(8, 0), textcoords="offset points",
                va="top", fontsize=9, color=INK2)
    ax.set_xlabel("Context benefit per trace: base MAE − context MAE (Mbps)  →  context better")
    ax.set_ylabel("Traces")
    ax.grid(axis="x", visible=False)
    save(fig, "fig3_rq3_context_benefit.png")


def fig4_rq4(errs):
    d = _paired(errs, "context_selected")
    panels = [("mobility_condition", ["Static", "Driving"], "Mobility (primary)"),
              ("application_type", ["Amazon Prime", "Netflix", "Download"], "Application (exploratory)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), sharey=True, gridspec_kw={"width_ratios": [2, 3]})
    for ax, (col, groups, title) in zip(axes, panels):
        for i, g in enumerate(groups):
            x = d[d[col] == g]
            ax.scatter(i + jitter(len(x)), x.rel, s=14, color=BLUE, alpha=0.7, linewidths=0.4, edgecolors=SURFACE)
            med = x.rel.median()
            ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.6)
            ax.annotate(f"median {med:.0f}%\nn = {len(x)}", (i, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=9, color=INK2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(groups)), groups)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_title(title, pad=26, color=INK2)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE reduction vs persistence (%)")
    save(fig, "fig4_rq4_groups.png")


def fig5_coverage():
    f = pd.read_parquet(C.PROCESSED / "features.parquet")
    lags = [f"throughput_lag_{k}" for k in C.LAGS_S]
    strict = f[C.TARGET].notna() & f[lags].notna().all(axis=1) & (f.window_n_obs == C.WINDOW_S)
    g = f.application_type + " · " + f.mobility_condition
    cov = pd.DataFrame({"strict": 100 * strict.groupby(g).mean(), "used": 100 * f.eligible.groupby(g).mean()})
    cov = cov.sort_values("used")
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    y = np.arange(len(cov))
    h = 0.36
    ax.barh(y + h / 2 + 0.01, cov.used, h, color=BLUE, label=f"Rule used (window ≥ {C.WINDOW_MIN_OBS} of 10 s)")
    ax.barh(y - h / 2 - 0.01, cov.strict, h, color=ORANGE, label="Strict rule (all 10 s and every lag)")
    for yy, v in zip(y, cov.used):
        ax.annotate(f"{v:.0f}%", (v, yy + h / 2), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK2)
    for yy, v in zip(y, cov.strict):
        ax.annotate(f"{v:.0f}%", (v, yy - h / 2), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK2)
    ax.set_yticks(y, cov.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Prediction origins kept (% of logged seconds)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.18), ncol=2, fontsize=9)
    save(fig, "fig5_coverage_rules.png")


def fig6_example():
    p = pd.read_parquet(C.ROOT / "outputs" / "models" / f"predictions_{SPLIT}.parquet")
    p = p[(p.application_type == "Download") & (p.mobility_condition == "Driving")].sort_values(["trace_id", "t"])
    # Most variable 180 s stretch of observed throughput among the held-out driving download traces
    best = None
    for tid, d in p.groupby("trace_id"):
        d = d.reset_index(drop=True)
        if len(d) < 200:
            continue
        sd = d[C.TARGET].rolling(180).std()
        i = int(sd.idxmax())
        if best is None or sd[i] > best[0]:
            best = (sd[i], d.iloc[i - 179:i + 1])
    d = best[1]
    sec = (d.t - d.t.iloc[0]).dt.total_seconds() + C.HORIZON_S          # time of the predicted value
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    series = [(C.TARGET, "Observed", INK, 1.6), ("persistence", "Persistence", ORANGE, 1.3),
              ("context_selected", "Gradient boosting + context", BLUE, 1.3)]
    for col, name, colr, lw in series:
        ax.plot(sec, d[col], color=colr, lw=lw, label=name)
    ax.set_xlabel("Time within excerpt (s)")
    ax.set_ylabel("Downlink throughput (Mbps)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.16), ncol=3, fontsize=9)
    save(fig, "fig6_example_trace.png")


def fig7_sensitivity():
    """Each finding under the three evaluation schemes: does its interval stay clear of zero?"""
    schemes = [("outer_fold", "Trace folds (primary)"), ("date_fold", "Recording dates held out"),
               ("chrono_test", "Latest dates held out")]
    panels = [("rq2_improvement_over_persistence.csv", "comparison", "persistence - base_selected",
               ("mean_delta_mbps", "ci95_low", "ci95_high"), "RQ2\nImprovement over persistence"),
              ("rq3_context_benefit.csv", "comparison", "base_selected - context_selected",
               ("mean_delta_mbps", "ci95_low", "ci95_high"), "RQ3\nContext benefit"),
              ("rq4_group_differences.csv", "comparison", "Driving - Static",
               ("diff", "ci_low", "ci_high"), "RQ4a\nDriving minus static")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0), sharey=True, gridspec_kw={"wspace": 0.25})
    for ax, (fname, key, row, (m, lo, hi), title) in zip(axes, panels):
        for i, (split, _) in enumerate(schemes):
            t = pd.read_csv(C.TABLES / split / fname)
            r = t[t[key] == row].iloc[0]
            col = BLUE if split == "outer_fold" else MUTED
            ax.errorbar(r[m], i, xerr=[[r[m] - r[lo]], [r[hi] - r[m]]], fmt="o", color=col, ms=5, capsize=2.5,
                        lw=1.3)
            ax.annotate(f"{r[m]:+.2f}", (r[m], i), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=9, color=INK2)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=9, color=INK2, loc="left")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.15)
    axes[0].set_yticks(range(len(schemes)), [s[1] for s in schemes])
    axes[0].set_ylim(-0.6, len(schemes) - 0.4)
    axes[0].invert_yaxis()
    fig.supxlabel("Mbps, 95% CI (positive = AI/ML or context better; RQ4a: driving gains more)", fontsize=9,
                  color=INK2, y=-0.04)
    save(fig, "fig7_sensitivity.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    errs = pd.read_csv(C.TABLES / SPLIT / "trace_errors.csv")
    fig1_rq1()
    fig2_rq2(errs)
    fig3_rq3(errs)
    fig4_rq4(errs)
    fig5_coverage()
    fig6_example()
    if all((C.TABLES / s / "rq3_context_benefit.csv").exists() for s in ("date_fold", "chrono_test")):
        fig7_sensitivity()
    else:
        print("skipped fig7_sensitivity: run run_models.py + analyze_models.py for date_fold and chrono_test")


if __name__ == "__main__":
    main()
