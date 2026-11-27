"""Exploratory data analysis: summary tables and figures for the report (run after run_pipeline.py).

Usage:  python make_eda.py
Writes outputs/tables/eda_*.csv and outputs/figures/eda*.png.
Raw-row statistics use the unmerged CSVs (what the dataset contains); time-series statistics use the
cleaned one-row-per-second table (what the models see).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, save
from tput import config as C
from tput.clean import load_trace

GROUPS = [("Amazon Prime", "Static"), ("Amazon Prime", "Driving"), ("Netflix", "Static"),
          ("Netflix", "Driving"), ("Download", "Static"), ("Download", "Driving")]
MOB_COL = {"Driving": BLUE, "Static": ORANGE}


def raw_rows():
    m = pd.read_csv(C.PROCESSED / "trace_manifest.csv")
    frames = [load_trace(C.RAW_DIR / t).assign(trace_id=t) for t in m.trace_id]
    return pd.concat(frames, ignore_index=True)


def tables(raw, clean):
    # 1. Overview by application x mobility (cleaned seconds)
    g = clean.groupby(["application_type", "mobility_condition"])
    tr = clean.groupby("trace_id").agg(app=("application_type", "first"), mob=("mobility_condition", "first"),
                                       secs=("t", "size"))
    over = pd.DataFrame({
        "traces": g.trace_id.nunique(),
        "seconds": g.size(),
        "median_trace_minutes": tr.groupby(["app", "mob"]).secs.median().div(60).round(1).rename_axis(
            ["application_type", "mobility_condition"]),
        "mean_mbps": g.throughput_mbps.mean().round(2),
        "median_mbps": g.throughput_mbps.median().round(2),
        "p90_mbps": g.throughput_mbps.quantile(0.9).round(2),
        "zero_share_pct": g.throughput_mbps.apply(lambda s: round(100 * (s == 0).mean(), 1)),
        "share_5g_pct": g.NetworkMode.apply(lambda s: round(100 * (s == "5G").mean(), 1)),
        "mean_speed_kmh": g.Speed.mean().round(1),
    }).reindex(pd.MultiIndex.from_tuples(GROUPS, names=["application_type", "mobility_condition"]))
    over.to_csv(C.TABLES / "eda_overview_by_group.csv")

    # 2. Descriptive statistics of the modelling variables (cleaned seconds)
    cols = {"throughput_mbps": "Throughput (Mbps)", "RSRP": "RSRP (dBm)", "RSRQ": "RSRQ (dB)", "SNR": "SNR (dB)",
            "CQI": "CQI", "Speed": "Speed (km/h)"}
    desc = clean[list(cols)].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.assign(missing_pct=(100 * clean[list(cols)].isna().mean()).round(1))
    desc.index = [cols[c] for c in desc.index]
    desc = desc[["count", "missing_pct", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    desc.to_csv(C.TABLES / "eda_descriptive_stats.csv")

    # 3. Missingness and data-quality issues in the raw rows
    num = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ", "PINGAVG", "Speed", "DL_bitrate"]
    miss = (100 * raw[num].isna().mean()).round(1).rename("missing_pct_raw_rows")
    miss.to_csv(C.TABLES / "eda_missingness.csv")

    steps = clean.groupby("trace_id").t.diff().dt.total_seconds().dropna()
    quality = pd.Series({
        "raw_rows": len(raw),
        "rows_with_repeated_timestamp": int(raw.duplicated(["trace_id", "t"], keep=False).sum()),
        "rsrp_sentinel_minus200": int((raw.RSRP == -200).sum()),
        "neighbour_rsrp_sentinel_minus205": int((raw.NRxRSRP == -205).sum()),
        "snr_below_minus23": int((raw.SNR < -23).sum()),
        "cqi_zero": int((raw.CQI == 0).sum()),
        "seconds_after_merge": len(clean),
        "gaps_2s": int((steps == 2).sum()),
        "gaps_3s_plus": int((steps >= 3).sum()),
        "max_dl_mbps": round(raw.DL_bitrate.max() / 1000, 1),
        "max_speed_kmh": raw.Speed.max(),
    })
    quality.to_csv(C.TABLES / "eda_data_quality.csv", header=["value"])

    # 4. Network-mode composition by mobility (cleaned seconds)
    mode = clean.NetworkMode.where(clean.NetworkMode.isin(["5G", "LTE"]), "3G/2G")
    comp = (pd.crosstab(clean.mobility_condition, mode, normalize="index") * 100).round(1)[["5G", "LTE", "3G/2G"]]
    comp.to_csv(C.TABLES / "eda_network_mode_by_mobility.csv")
    return over, desc, miss, quality, comp


def fig_throughput(clean):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), gridspec_kw={"width_ratios": [3.4, 2]})
    ax = axes[0]
    for i, (app, mob) in enumerate(GROUPS):
        s = clean[(clean.application_type == app) & (clean.mobility_condition == mob)].throughput_mbps
        nz = s[s > 0]
        ax.boxplot(nz, positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=MOB_COL[mob], alpha=0.75, edgecolor=MOB_COL[mob]),
                   medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    ax.set_yscale("log")
    ax.set_xticks(range(len(GROUPS)), [f"{a.replace('Amazon ', '').replace('Download', 'Down-' + chr(10) + 'load')}\n{m}" for a, m in GROUPS],
                  fontsize=9)
    ax.set_ylabel("Non-zero throughput (Mbps, log scale)")
    ax.set_title("Throughput when data is flowing", fontsize=9.5, color=INK2, loc="left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    zs = [100 * (clean[(clean.application_type == a) & (clean.mobility_condition == m)].throughput_mbps == 0).mean()
          for a, m in GROUPS]
    y = np.arange(len(GROUPS))
    ax.barh(y, zs, color=[MOB_COL[m] for _, m in GROUPS], height=0.6)
    for yy, v in zip(y, zs):
        ax.annotate(f"{v:.0f}%", (v, yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK2)
    ax.set_yticks(y, [f"{a.replace('Amazon ', '')} · {m}" for a, m in GROUPS], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Zero-throughput seconds (%)")
    ax.set_title("Idle seconds", fontsize=9.5, color=INK2, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "eda1_throughput_by_group.png")


def fig_radio(clean):
    cols = [("RSRP", "RSRP (dBm)"), ("RSRQ", "RSRQ (dB)"), ("SNR", "SNR (dB)"), ("CQI", "CQI")]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.6))
    for ax, (c, lab) in zip(axes, cols):
        lo, hi = clean[c].quantile([0.005, 0.995])
        bins = np.arange(-0.5, 16, 1) if c == "CQI" else np.linspace(lo, hi, 30)
        for mob in ("Static", "Driving"):
            ax.hist(clean.loc[clean.mobility_condition == mob, c].dropna(), bins=bins, density=True,
                    histtype="step", lw=1.4, color=MOB_COL[mob], label=mob)
        ax.set_xlabel(lab, fontsize=9)
        ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.32), ncol=2, fontsize=9)
    fig.tight_layout()
    save(fig, "eda2_radio_distributions.png")


def fig_mode_and_gaps(clean, comp):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    shades = {"5G": BLUE, "LTE": "#86b6ef", "3G/2G": "#cde2fb"}
    for i, mob in enumerate(["Static", "Driving"]):
        left = 0
        for mode in ["5G", "LTE", "3G/2G"]:
            v = comp.loc[mob, mode]
            ax.barh(i, v, left=left, color=shades[mode], edgecolor=SURFACE, lw=1.5, height=0.55,
                    label=mode if i == 0 else None)
            if v >= 6:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if mode == "5G" else INK)
            left += v
    ax.set_yticks([0, 1], ["Static", "Driving"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of seconds by network mode (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.3), ncol=3, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    steps = clean.groupby("trace_id").t.diff().dt.total_seconds().dropna()
    cats = ["1 s", "2 s", "3 s+"]
    vals = [100 * (steps == 1).mean(), 100 * (steps == 2).mean(), 100 * (steps >= 3).mean()]
    ax.bar(cats, vals, color=[BLUE, ORANGE, ORANGE], width=0.6)
    for x, v in zip(cats, vals):
        ax.annotate(f"{v:.1f}%", (x, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=9,
                    color=INK2)
    ax.set_ylabel("Share of consecutive records (%)")
    ax.set_xlabel("Time between records")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "eda3_network_mode_and_gaps.png")


def fig_example_traces(clean):
    picks = [("Netflix", "Static"), ("Download", "Driving")]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.8), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for j, (app, mob) in enumerate(picks):
        sub = clean[(clean.application_type == app) & (clean.mobility_condition == mob)]
        tid = sub.groupby("trace_id").size().sort_values().index[len(sub.trace_id.unique()) // 2]   # median length
        d = sub[sub.trace_id == tid].sort_values("t").head(300)
        sec = (d.t - d.t.iloc[0]).dt.total_seconds()
        axes[0, j].plot(sec, d.throughput_mbps, color=MOB_COL[mob], lw=1.1)
        axes[0, j].set_title(f"{app}, {mob.lower()} (300 records of one trace)", fontsize=9, color=INK2, loc="left")
        axes[1, j].plot(sec, d.CQI, color=INK2, lw=1.0, drawstyle="steps-post")
        axes[1, j].set_ylim(0, 15.5)
        axes[1, j].set_xlabel("Time (s)", fontsize=9)
    axes[0, 0].set_ylabel("Throughput (Mbps)", fontsize=9)
    axes[1, 0].set_ylabel("CQI", fontsize=9)
    fig.tight_layout()
    save(fig, "eda4_example_traces.png")


def main():
    clean = pd.read_parquet(C.INTERIM / "clean_1s.parquet")
    raw = raw_rows()
    over, desc, miss, quality, comp = tables(raw, clean)
    fig_throughput(clean)
    fig_radio(clean)
    fig_mode_and_gaps(clean, comp)
    fig_example_traces(clean)
    pd.set_option("display.width", 200)
    print(over.to_string(), "\n\n", desc.to_string(), "\n\n", miss.to_string(), "\n\n", quality.to_string(),
          "\n\n", comp.to_string())


if __name__ == "__main__":
    main()
