\# Bazel Build CPU-Time Prediction (Model Part)



Predicts the CPU time of a Bazel build from the most common file prefix and

file type changed in a commit, using linear regression with feature crossing.



\## Setup



Requires Python 3.9+.



&#x20;   pip install pandas numpy scikit-learn matplotlib streamlit joblib



\## Files



| File | Purpose |

|------|---------|

| `make\_fake\_data.py` | Generates a fake `data.csv` for testing the pipeline |

| `data.csv` | Dataset with columns `prefix, file\_type, cpu\_time\_ms` |

| `train.py` | Trains baseline, linear (no cross) and linear (with cross) models |

| `tune.py` | Tunes the Ridge alpha on the validation set, saves the best models |

| `plots.py` | Creates the result plots in `plots/` |

| `app.py` | Streamlit demo app |



\## How to run (from the repository root)



&#x20;   python model/train.py

&#x20;   python model/tune.py

&#x20;   python model/plots.py

&#x20;   python -m streamlit run model/app.py



To use the real data, replace `model/data.csv` with the collected file

(same column names) and rerun the commands above.



\## Method



\- Features: most common changed file prefix and file type, one-hot encoded.

\- Feature cross: `prefix + "\_" + file\_type`, one-hot encoded.

\- Split: 60% train, 20% validation, 20% test.

\- Models: mean baseline, Ridge regression with and without the cross.

\- Metric: RMSE (ms). Ridge alpha is chosen on validation RMSE.

## Real Dataset Collection

The real dataset was collected from the open-source Bazel repository by evaluating historical Bazel commits.

For each commit:

1. The changed files were identified from the commit.
2. Supported files were classified by their Bazel path prefix and file type.
3. The most common changed prefix and file type were selected as the input features.
4. The Bazel target `//src:bazel-dev` was built.
5. Bazel Build Event Protocol (BEP) JSON was generated for the build.
6. The build CPU time was extracted from `buildMetrics -> timingMetrics -> cpuTimeInMs`.
7. One row was added to the dataset with the columns:
   `prefix, file_type, cpu_time_ms`.

The collected dataset contains **100 real Bazel build records**.

The dataset was validated to ensure:
- The required columns are present.
- No values are missing.
- CPU times are positive.
- Prefixes and file types use the allowed project categories.

The resulting dataset is stored in `model/realdataset.csv`.

