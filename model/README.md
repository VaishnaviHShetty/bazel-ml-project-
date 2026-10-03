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

