# AI/ML-Based Prediction of 5G Downlink Throughput Using Channel and Context Metrics

QM640 Data Analytics Capstone, Walsh College (Term 3). Student: Lokesh Rajoria.

The project predicts 5G downlink throughput five seconds ahead from what a phone records (throughput history,
radio measurements RSRP, RSRQ, SNR, CQI, and context: speed, network mode, application type) and compares
ridge regression, random forest, and gradient boosting with a persistence baseline. Models are tested on whole
held-out traces with nested grouped cross-validation, and again with whole recording days held out.

**Final result (final report and presentation):** with nested cross-validation that withholds whole traces, gradient
boosting with an absolute-error loss reduced mean trace MAE from 9,521 to 8,046 kbps (15.5%; mean improvement
1,475 kbps, 95% CI [845, 2,106], p < .001) and beat the persistence baseline on 82 of 83 traces. Context features
(speed, network mode, application) added a further 235 kbps (95% CI [51, 419], p = .006). A history-only model gained
1,586 kbps over persistence, and adding radio measurements changed MAE by 111 kbps (slightly worse), so the gain comes
from throughput history. Code, settings and results are in `final/` (seed 0; run `python final/final_analysis.py`).

**Earlier interim run:** the code in `tput/`, `outputs/` and the Colab notebook reproduces the interim-report analysis
(seed 640, different evaluation design; mean MAE 9.49 to 7.73 Mbps). Those numbers are kept for reference and are not
the final results.

## Data

Raca, D., Leahy, D., Sreenan, C. J., & Quinlan, J. J. (2020). *Beyond throughput, the next generation: A 5G dataset
with channel and context metrics.* ACM MMSys. Source: https://github.com/uccmisl/5Gdataset

The raw data are redistributed here unchanged under the dataset's GPL-3.0 license (`data/raw/LICENSE`, with the
authors' `data/raw/README.md`). Please cite the paper above if you use them.

| Path | Content |
|---|---|
| `data/raw/5G-production-dataset.zip` | Original archive (SHA-256 checked on every run) |
| `data/raw/5G-production-dataset/` | The 83 CSV traces extracted from it |
| `data/interim/clean_1s.parquet` | Cleaned one-second records (180,743) |
| `data/processed/` | `features.parquet`, `splits.csv`, `trace_manifest.csv` |
| `data/capstone.sqlite.zip` | Zipped SQLite database (unzip to `data/capstone.sqlite`, or rebuild with `python build_database.py`) of the same data (tables `traces`, `records`, `features`, `splits`; view `v_group_summary`); coordinates and cell identifiers are not stored |
| `outputs/` | Model predictions, result tables, and all figures |

## Run

Python 3.13 with the packages in `requirements.txt` (`pip install -r requirements.txt`). Run from this folder:

```
python run_pipeline.py        # checksum, cleaning, features, folds, persistence baseline (~10 s)
python -m pytest tests -q     # timing, leakage, and split rules on synthetic traces
python run_rq1.py             # RQ1 correlation tables
python run_models.py --split outer_fold     # nested CV, primary scheme (~18 min on a laptop CPU)
python run_models.py --split date_fold      # whole recording dates held out (~20 min)
python run_models.py --split chrono_test    # latest dates held out (~5 min)
python analyze_models.py --split outer_fold # RQ2-RQ4 tables (repeat for the other two splits)
python run_train_errors.py    # training versus test error of every model
python run_importance.py      # permutation feature importance
python make_eda.py            # EDA tables and figures
python make_figures.py        # result figures
python make_preview.py        # dataset preview figure
python make_final_figures.py  # pipeline diagram and predicted-versus-observed plot
python build_database.py      # SQLite database (~7 s)
```

All runs use random seed 640 and run on the CPU only (developed on an Intel Core i7-1165G7 with 24 GB of memory).

**Colab:** `Capstone_5G_Throughput_Colab.ipynb` runs the whole analysis in Google Colab (*Runtime > Run all*).

**Reports:** `python report/build_report_data.py`, then `node report/build_final_report.js` (or `build_report.js`
for the interim report) writes the .docx into a local `docs/` folder (not tracked in this repository). This needs Node.js and `npm install` in `report/`.

## Repository layout

```
final/        final analysis (RQ1 to RQ4): scripts, settings, results
tput/         analysis package: config, cleaning, features, models, statistics
tests/        unit tests for timing and leakage rules
report/       report builders (docx) and report_data.json
data/         raw, interim, processed data and the SQLite database
outputs/      predictions, tables, figures
```
