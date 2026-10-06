import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

CSV_PATH = "model/combined_dataset.csv"   # change this when the real data arrives


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# 1. Load data and build the crossed feature
df = pd.read_csv(CSV_PATH)
df["cross"] = df["prefix"] + "_" + df["file_type"]

X = df[["prefix", "file_type", "cross"]]
y = df["cpu_time_ms"]

# 2. Split 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# 3. Model builder: one-hot encode the chosen columns, then linear regression
def make_model(columns):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), columns)]
    )
    return Pipeline([("pre", pre), ("reg", LinearRegression())])


model_a = make_model(["prefix", "file_type"])             # no feature cross
model_b = make_model(["prefix", "file_type", "cross"])    # with feature cross

model_a.fit(X_train, y_train)
model_b.fit(X_train, y_train)

# 4. Baseline: always predict the training mean
baseline = y_train.mean()

# 5. Evaluate
results = {
    "Baseline (mean)": (
        rmse(y_val, np.full(len(y_val), baseline)),
        rmse(y_test, np.full(len(y_test), baseline)),
    ),
    "Linear (no cross)": (
        rmse(y_val, model_a.predict(X_val)),
        rmse(y_test, model_a.predict(X_test)),
    ),
    "Linear (with cross)": (
        rmse(y_val, model_b.predict(X_val)),
        rmse(y_test, model_b.predict(X_test)),
    ),
}

print(f"{'Model':25s} {'Val RMSE':>10s} {'Test RMSE':>10s}")
for name, (v, t) in results.items():
    print(f"{name:25s} {v:10.1f} {t:10.1f}")

# 6. Save models for the demo
joblib.dump(model_a, "model/model_no_cross.joblib")
joblib.dump(model_b, "model/model_cross.joblib")
print("Models saved.")
