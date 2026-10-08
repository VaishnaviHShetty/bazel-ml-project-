# Bazel Build CPU-Time Prediction

UE24CS352A – Machine Learning Mini-Project

Predicts the CPU time (ms) of a Bazel build from the **most common file prefix** and **most common file type** changed in a commit, using linear (Ridge) regression with and without a **feature cross**. The approach follows the reference paper *"Predicting Bazel Build Times Using Machine Learning"* (Nadeem & Raza, Stanford).

**Team:** Vaishnavi H Shetty(PES1UG24CS641), Sami Shahapuri(PES1UG24CS611)

---

## Problem Statement

Given a commit to the Bazel repository, predict how much CPU time the build will take, using only the kind of files the commit touched. This is a regression problem (target: `cpu_time_ms`). The reference paper is in [`References/predicting_bazel_build_times.pdf`](References/predicting_bazel_build_times.pdf).

## Dataset

The training set is built from two sources:

| Source | Rows | Where it is from |
|--------|-----:|------------------|
| Our own real builds (`model/realdataset.csv`) | 100 | Collected by us by building historical Bazel commits (method below) |
| Data from the reference paper | 400 | Public dataset released by the paper's authors: https://github.com/umar-nadeem/CS229-Final-Project |
| **Combined (`model/combined_dataset.csv`)** | **500** | **Used for all training, tuning and plots** |

`model/data.csv` (300 rows) is synthetic data from `make_fake_data.py`, used only to test the pipeline early on. It is **not** used for any reported result.

**Columns:** `prefix`, `file_type`, `cpu_time_ms`

- `prefix`: one of `bazelci/`, `examples/`, `scripts/`, `site/`, `src/conditions`, `src/java_tools`, `src/main`, `src/test`, `src/tools`, `third_party/`, `tools/`
- `file_type`: one of `JAVA`, `C/C++`, `Starlark`, `python`, `HTML/CSS/JS`
- `cpu_time_ms`: build CPU time in milliseconds

**Caveat:** the two sources were measured on different machines and at different times, so their CPU-time distributions differ (our 100 rows have a higher median than the paper's rows). This adds noise to the combined data.

### How our 100 real records were collected

For each historical commit of the open-source Bazel repository:

1. Identify the changed files.
2. Classify supported files by Bazel path prefix and file type.
3. Select the most common prefix and file type as the input features.
4. Build the target `//src:bazel-dev`.
5. Generate Bazel Build Event Protocol (BEP) JSON for the build.
6. Extract the CPU time from `buildMetrics -> timingMetrics -> cpuTimeInMs`.
7. Append one row: `prefix, file_type, cpu_time_ms`.

`model/validate_data.py` checks that required columns exist, no values are missing, CPU times are positive, and prefixes and file types are from the allowed categories.

## Approach

- **Features:** `prefix` and `file_type`, one-hot encoded.
- **Feature cross:** `prefix + "_" + file_type`, one-hot encoded.
- **Split:** 60% train / 20% validation / 20% test (`random_state=42`).
- **Models:** mean baseline; linear regression and Ridge, each with and without the cross.
- **Tuning:** Ridge `alpha` chosen on validation RMSE from `[0.01, 0.1, 1, 3, 10, 30]`; the test set is evaluated once with the chosen alpha.
- **Metric:** RMSE (ms).

## Results (test set)

| Model | Test RMSE (ms) |
|-------|---------------:|
| Baseline (training mean) | 7049 |
| Linear (no cross) | 7197 |
| Linear (with cross) | 7005 |
| Ridge (no cross, alpha=30) | 6967 |
| Ridge (with cross, alpha=30) | 6950 |

**Note:** the models only slightly improve on the mean baseline. The target is heavy-tailed (up to ~195,000 ms), so RMSE is dominated by a few outliers; the data is concentrated in `src/main` + `JAVA`; the validation and test sets are small (100 rows each); and the two data sources differ in distribution. Prefix and file type alone carry limited signal about build time.

Plots are saved in `model/plots/`: `rmse_comparison.png`, `predicted_vs_actual.png`, `combined_dataset_distribution.png`.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── References/
│   └── predicting_bazel_build_times.pdf
└── model/
    ├── combined_dataset.csv     # training data (500 rows)
    ├── realdataset.csv          # our real collected data (100 rows)
    ├── data.csv                 # synthetic test data (not used for results)
    ├── make_fake_data.py        # generates data.csv
    ├── validate_data.py         # validates a dataset CSV
    ├── train.py                 # baseline + linear models, saves model_*.joblib
    ├── tune.py                  # Ridge alpha tuning, saves best_*.joblib
    ├── plots.py                 # generates plots in model/plots/
    ├── app.py                   # Streamlit demo
    └── plots/
```

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/VaishnaviHShetty/bazel-ml-project-.git
cd bazel-ml-project-
pip install -r requirements.txt
```

## How to Run

**Run every command from the repository root** (the scripts use paths like `model/combined_dataset.csv`).

```bash
python model/train.py                  # baseline + linear models, prints val/test RMSE
python model/tune.py                   # tunes Ridge alpha, saves best_cross / best_no_cross models
python model/plots.py                  # writes plots to model/plots/
python -m streamlit run model/app.py   # launches the demo app
```

Run `tune.py` before `plots.py` and `app.py`; both load the tuned models it saves. If loading a `.joblib` file fails with a scikit-learn version error, rerun `train.py` and `tune.py` to regenerate the models.

## Demo App

The Streamlit app lets you pick a changed-file prefix and file type, and shows the predicted build CPU time from the Ridge model without the cross and with the cross, side by side.

## References

- M. U. Nadeem and S. Raza, *Predicting Bazel Build Times Using Machine Learning*, Stanford University (CS229). PDF in `References/`.
- Paper's public data and code: https://github.com/umar-nadeem/CS229-Final-Project

## Known Limitations

- Only two coarse features are used (most common changed prefix and file type), so they carry limited information about how long a build actually takes.
- The 500-row dataset combines 100 rows we collected with 400 rows from the reference paper's authors. The two sources were measured on different machines and years apart, so their CPU-time distributions differ.
- The data is heavily skewed toward `src/main` + `JAVA`, so other prefix and file-type combinations have very few examples.
- CPU time is heavy-tailed (up to ~195,000 ms), so RMSE is dominated by a few outlier builds.
- The validation and test sets are small (100 rows each) and results come from a single random split, so differences between models are within noise.
- The split is random rather than chronological, so the model may see commits from the same period in both training and test sets.
