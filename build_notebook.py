"""Generate the Colab demonstration notebook from the current analysis sources.

Usage (from the analysis folder):  python build_notebook.py
Writes Capstone_5G_Throughput_Colab.ipynb. The notebook writes this package into the Colab runtime with
%%writefile cells, downloads the public dataset, and runs the analysis in the order of the report.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Capstone_5G_Throughput_Colab.ipynb"
PACKAGE = ["tput/__init__.py", "tput/config.py", "tput/manifest.py", "tput/clean.py", "tput/features.py",
           "tput/splits.py", "tput/evaluate.py", "tput/checks.py", "tput/stats.py", "tput/rq1.py", "tput/models.py"]
SCRIPTS = ["run_pipeline.py", "run_rq1.py", "run_models.py", "analyze_models.py", "make_figures.py",
           "make_eda.py", "run_importance.py"]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# AI/ML-Based Prediction of 5G Downlink Throughput Using Channel and Context Metrics\n"
       "**QM640 Data Analytics Capstone, Lokesh Rajoria, Walsh College**\n\n"
       "This notebook reproduces the analysis in the interim report, in the same order: data acquisition, "
       "cleaning, exploratory analysis, RQ1 correlations, model comparison (RQ2-RQ4) and feature importance.\n\n"
       "**How to run:** *Runtime > Run all*. Everything runs in Colab; the public dataset is downloaded from the "
       "authors' GitHub repository (Raca et al., 2020).\n\n"
       "**Run time:** about 10-15 minutes in quick mode (default). Quick mode searches only the model settings that "
       "the full grid selected most often; set `QUICK = False` below to run the full nested grid (about 45-60 "
       "minutes on a standard Colab CPU). Numbers in quick mode can differ slightly from the report."),
    code("QUICK = True                 # False = full nested grid search, as in the report\n"
         "RUN_SENSITIVITY = False      # True = also hold out whole recording dates and the latest dates (slow)\n"
         "RUN_IMPORTANCE = True        # permutation feature importance (about 3-5 minutes)\n"
         "\n"
         "import os, sys, logging\n"
         "os.makedirs('/content/capstone', exist_ok=True)\n"
         "%cd /content/capstone\n"
         "sys.path.insert(0, os.getcwd())   # make the tput package importable\n"
         "os.environ['TPUT_QUICK'] = '1' if QUICK else '0'\n"
         "logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)\n"
         "import pandas as pd, sklearn\n"
         "pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)\n"
         "print('Python', sys.version.split()[0], '| pandas', pd.__version__, '| scikit-learn', sklearn.__version__)"),
    md("## 1. Analysis package\n"
       "The next cells write the project's Python modules into the runtime (`tput/` package and the run scripts). "
       "They are identical to the files in the submitted analysis folder. You can collapse this section."),
    code("os.makedirs('tput', exist_ok=True)"),
]
for f in PACKAGE + SCRIPTS:
    cells.append(code(f"%%writefile {f}\n" + (ROOT / f).read_text(encoding="utf-8")))

cells += [
    md("## 2. Data acquisition\n"
       "Downloads `5G-production-dataset.zip` from https://github.com/uccmisl/5Gdataset, verifies its SHA-256 "
       "checksum against the version used in the report, and extracts only the 83 CSV traces (the archive's "
       "`__MACOSX` files are skipped)."),
    code("import hashlib, urllib.request, zipfile\n"
         "from tput import config as C\n"
         "C.RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)\n"
         "if not C.RAW_ZIP.exists():\n"
         "    urllib.request.urlretrieve('https://raw.githubusercontent.com/uccmisl/5Gdataset/master/5G-production-dataset.zip', C.RAW_ZIP)\n"
         "digest = hashlib.sha256(C.RAW_ZIP.read_bytes()).hexdigest()\n"
         "print('SHA-256 matches the report version:', digest == C.RAW_ZIP_SHA256)\n"
         "with zipfile.ZipFile(C.RAW_ZIP) as z:\n"
         "    members = [n for n in z.namelist() if n.startswith('5G-production-dataset/') and n.endswith('.csv')]\n"
         "    z.extractall(C.RAW_ZIP.parent, members)\n"
         "print(len(members), 'CSV traces extracted')"),
    md("## 3. Data preparation\n"
       "Builds the trace manifest, cleans the data (missing-value markers, the RSRP -200 sentinel, repeated "
       "timestamps), constructs the 5-second target and history features, assigns whole traces to evaluation folds, "
       "checks for leakage, and scores the persistence baseline."),
    code("!python run_pipeline.py"),
    md("## 4. Exploratory data analysis"),
    code("!python -W ignore make_eda.py > /dev/null\n"
         "from IPython.display import Image, display\n"
         "T, F = C.TABLES, C.FIGURES\n"
         "display(pd.read_csv(T / 'eda_overview_by_group.csv'))\n"
         "display(pd.read_csv(T / 'eda_descriptive_stats.csv', index_col=0))\n"
         "display(pd.read_csv(T / 'eda_data_quality.csv', index_col=0))\n"
         "for f in ['eda1_throughput_by_group.png', 'eda3_network_mode_and_gaps.png', 'eda2_radio_distributions.png', 'eda4_example_traces.png']:\n"
         "    display(Image(filename=str(F / f), width=820))"),
    md("**Insights.** Video throughput is governed by the player's on/off pattern (Netflix is idle in about three "
       "quarters of seconds), driving traces spend much of their time outside 5G, and about 10% of seconds are "
       "missing from the logs. These shape RQ1 (weak radio-throughput association expected) and RQ2 (history "
       "should dominate)."),
    md("## 5. RQ1: radio measurements and throughput\n"
       "Within-trace Spearman correlations, one-sample t tests across traces, Holm adjustment."),
    code("!python run_rq1.py > /dev/null\n"
         "display(pd.read_csv(T / 'rq1_summary.csv')[['measurement', 'eligible_traces', 'mean_rho', 'ci95_low', 'ci95_high', 'p_holm', 'date_p_holm']].round(4))"),
    md("## 6. Models: nested grouped cross-validation (RQ2-RQ4)\n"
       "Persistence vs ridge regression, random forest and gradient boosting. Five outer folds hold out whole "
       "traces; three inner folds choose the settings; the context model adds speed, network mode and application "
       "type."),
    code("!python -W ignore run_models.py --split outer_fold\n"
         "!python -W ignore analyze_models.py --split outer_fold > /dev/null\n"
         "if RUN_SENSITIVITY:\n"
         "    for s in ['date_fold', 'chrono_test']:\n"
         "        !python -W ignore run_models.py --split {s}\n"
         "        !python -W ignore analyze_models.py --split {s} > /dev/null"),
    md("### RQ2: AI/ML compared with persistence"),
    code("O = T / 'outer_fold'\n"
         "display(pd.read_csv(O / 'errors_overall.csv'))\n"
         "display(pd.read_csv(O / 'rq2_improvement_over_persistence.csv')[['comparison', 'mean_delta_mbps', 'ci95_low', 'ci95_high', 'traces_improved_pct']])\n"
         "display(pd.read_csv(O / 'robustness_vs_reference_baselines.csv')[['comparison', 'mean_delta_mbps', 'ci95_low', 'ci95_high', 'traces_improved_pct']])"),
    md("### RQ3: added value of context, and RQ4: mobility and application"),
    code("display(pd.read_csv(O / 'rq3_context_benefit.csv')[['comparison', 'mean_delta_mbps', 'ci95_low', 'ci95_high', 'traces_improved_pct']])\n"
         "display(pd.read_csv(O / 'rq4_by_group.csv')[['comparison', 'traces', 'mean_mae_persistence', 'mean_mae_model', 'mean_delta_mbps', 'ci95_low', 'ci95_high', 'median_rel_improvement_pct']])\n"
         "display(pd.read_csv(O / 'rq4_group_differences.csv'))"),
    md("## 7. Figures"),
    code("!python -W ignore make_figures.py > /dev/null\n"
         "for f in ['fig1_rq1_correlations.png', 'fig2_rq2_improvement.png', 'fig6_example_trace.png', 'fig3_rq3_context_benefit.png', 'fig4_rq4_groups.png', 'fig7_sensitivity.png']:\n"
         "    p = F / f\n"
         "    if p.exists(): display(Image(filename=str(p), width=820))"),
    md("## 8. Feature importance\n"
       "Permutation importance of the selected context model on held-out traces, averaged over the five folds."),
    code("if RUN_IMPORTANCE:\n"
         "    !python -W ignore run_importance.py > /dev/null\n"
         "    display(pd.read_csv(T / 'feature_importance.csv'))\n"
         "    display(Image(filename=str(F / 'fig8_feature_importance.png'), width=760))"),
    md("## 9. Summary\n"
       "| RQ | Result |\n|---|---|\n"
       "| RQ1 | No radio measurement is associated with throughput within traces after Holm adjustment. |\n"
       "| RQ2 | Gradient boosting with an absolute-error loss beats persistence on held-out traces, and also beats a 10 s moving average. |\n"
       "| RQ3 | Context adds a small benefit that does not hold when whole recording days are held out. |\n"
       "| RQ4 | No clear difference between stationary and driving traces; application differences follow throughput scale. |\n\n"
       "Throughput history dominates feature importance; the four radio measurements contribute almost nothing. "
       "See the interim report for interpretation and limitations."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "colab": {"name": OUT.name, "provenance": []},
})
nbf.write(nb, OUT)
print("wrote", OUT, f"({len(cells)} cells)")
