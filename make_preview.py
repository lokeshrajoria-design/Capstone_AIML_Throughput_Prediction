"""Render a preview of the first records of one raw trace file (report evidence figure).

Usage (from the analysis folder):  python make_preview.py
Coordinates, cell and node identifiers, ping fields and neighbor-cell fields are left out of the preview.
"""
import matplotlib.pyplot as plt
import pandas as pd

from make_figures import INK2, MUTED, save
from tput import config as C

TRACE = "Download/Driving/B_2019.12.14_10.16.30.csv"
COLS = ["Timestamp", "Speed", "NetworkMode", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "DL_bitrate", "UL_bitrate", "State"]


def main():
    files = sorted(C.RAW_DIR.rglob("*.csv"))
    path = next((f for f in files if f.as_posix().endswith(TRACE)), None) or next(f for f in files if "Download" in f.parts)
    df = pd.read_csv(path, dtype=str, usecols=COLS)[COLS].iloc[100:112]
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=COLS, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.25)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor(MUTED)
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK2)
            cell.set_facecolor("#E8EEF6")
    t.auto_set_column_width(list(range(len(COLS))))
    save(fig, "fig0_dataset_preview.png")


if __name__ == "__main__":
    main()
