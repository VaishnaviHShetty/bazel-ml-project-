# Bazel Build CPU-Time Prediction (Model Part)

This project predicts Bazel build CPU time using the most common file-path prefix and file type changed in a commit. It implements the approach from the reference paper using Linear Regression with feature crossing and compares it with Ridge Regression and Random Forest Regression.

## Setup

Requires Python 3.9+.

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is located in the repository root, the command above installs the required dependencies.

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Lists the Python dependencies required by the project |
| `combined_dataset.csv` | Combined dataset containing 500 records: 400 reference records and 100 real Bazel build records |
| `realdataset.csv` | Dataset containing 100 real Bazel build records |
| `data.csv` | Original sample dataset used for testing |
| `make_fake_data.py` | Generates sample data for testing |
| `train.py` | Trains the baseline and paper-based Linear Regression models |
| `tune.py` | Tunes Ridge Regression using validation RMSE and saves the best models |
| `random_forest.py` | Trains and evaluates Random Forest Regression |
| `plots.py` | Generates model-comparison, predicted-versus-actual, and CPU-time distribution plots |
| `app.py` | Streamlit application for interactive CPU-time predictions |
| `best_no_cross.joblib` | Saved best Ridge model without feature crossing |
| `best_cross.joblib` | Saved best Ridge model with feature crossing |
| `random_forest.joblib` | Saved Random Forest model |
| `plots/` | Generated evaluation plots |

## How to Run

Run commands from the repository root.

### 1. Train the Linear Regression models

```bash
python model/train.py
```

### 2. Tune Ridge Regression

```bash
python model/tune.py
```

### 3. Train and evaluate Random Forest

```bash
python model/random_forest.py
```

This saves the selected model to `model/random_forest.joblib`.

### 4. Generate the plots

Run this after generating the saved models:

```bash
python model/plots.py
```

The plots are saved in `model/plots/`:

- `rmse_comparison.png`
- `predicted_vs_actual.png`
- `combined_dataset_distribution.png`

### 5. Launch the Streamlit application

```bash
python -m streamlit run model/app.py
```

The application lets users select a file prefix and file type and compare CPU-time predictions from the saved Ridge and Random Forest models.

## Dataset

The combined dataset contains 500 records:

- 400 records from the reference project's dataset.
- 100 real Bazel build records collected from historical commits.

Each record contains three columns:

| Column | Description |
|---|---|
| `prefix` | Most common supported file-path prefix changed in the commit |
| `file_type` | Most common supported file type changed in the commit |
| `cpu_time_ms` | Bazel build CPU time in milliseconds |

The models use the prefix and file type as categorical input features and predict `cpu_time_ms`.

## Real Dataset Collection

The real dataset was collected from the open-source Bazel repository by evaluating historical commits.

For each commit:

1. The changed files were identified.
2. Supported files were classified by their Bazel path prefix and file type.
3. The most common changed prefix and file type were selected as input features.
4. The Bazel target `//src:bazel-dev` was built.
5. Bazel Build Event Protocol (BEP) JSON was generated for the build.
6. CPU time was extracted from `buildMetrics -> timingMetrics -> cpuTimeInMs`.
7. One row was added to the dataset using the columns `prefix`, `file_type`, and `cpu_time_ms`.

The collected dataset contains 100 real Bazel build records. The dataset was validated for required columns, missing values, positive CPU times, and supported prefix and file-type categories.

The real dataset is stored in `model/realdataset.csv`.

## Methodology

### Feature Engineering

The categorical features are transformed using One-Hot Encoding.

Feature crossing combines the prefix and file type into an additional categorical feature:

`cross = prefix + "_" + file_type`

For example, `src/main` and `JAVA` produce `src/main_JAVA`.

This allows the model to learn the interaction between the prefix and file type.

### Train, Validation, and Test Split

The dataset is split into:

- 60% training data
- 20% validation data
- 20% test data

The split uses `random_state=42` for reproducibility.

Validation RMSE is used to select model configurations. Test RMSE is used for final evaluation.

### Models

The project evaluates the following approaches:

1. **Mean baseline:** predicts the mean CPU time from the training data.
2. **Linear Regression:** implements the reference-paper approach, with and without feature crossing.
3. **Ridge Regression:** uses L2 regularization, with the regularization parameter selected using validation RMSE.
4. **Random Forest Regression:** combines multiple decision trees and selects a configuration using validation RMSE.

### Evaluation Metric

Root Mean Squared Error (RMSE), measured in milliseconds, is used to compare model predictions with actual CPU times.

Lower RMSE indicates better performance on the evaluated dataset.

## Results

The following test RMSE values were obtained using the current 500-record dataset and the fixed test split.

| Model | Test RMSE (ms) |
|---|---:|
| Mean baseline | 7049.0 |
| Linear Regression without feature crossing | 7196.7 |
| Linear Regression with feature crossing | 7004.5 |
| Ridge Regression without feature crossing | 6967.2 |
| Ridge Regression with feature crossing | 6950.0 |
| Random Forest Regression | **6928.9** |

Random Forest achieved the lowest test RMSE among these experiments. Its result is slightly better than Ridge Regression with feature crossing on this test split. The difference is small, so the results do not establish that Random Forest will perform better on every dataset or split.

## Prediction Analysis

The CPU-time distribution is strongly right-skewed: most builds have relatively low CPU times, while a small number have much higher CPU times.

The predicted-versus-actual plots show that the models tend to predict within the typical CPU-time range and underpredict some expensive builds. This is a limitation of using only the changed file prefix and file type as input features. The model does not directly observe the number of changed files, build actions, dependency changes, or other factors affecting build complexity.

## Limitations

- The dataset contains only 500 records.
- The input features provide limited information about build complexity.
- CPU times are strongly right-skewed and include expensive outliers.
- Results are based on one fixed train/validation/test split.
- A lower test RMSE on this split does not guarantee better generalization to future builds.

## Conclusion

This project implements the reference-paper approach using Linear Regression and feature crossing, then compares it with Ridge Regression and Random Forest Regression. Random Forest achieved the lowest test RMSE of 6928.9 ms in the current experiments. The results also demonstrate that accurate prediction of extreme build times remains challenging with the available categorical features.