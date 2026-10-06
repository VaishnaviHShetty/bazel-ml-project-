import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

CSV_PATH = "model/combined_dataset.csv"
ALPHAS = [0.01, 0.1, 1, 3, 10, 30]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


df = pd.read_csv(CSV_PATH)
df["cross"] = df["prefix"] + "_" + df["file_type"]
X = df[["prefix", "file_type", "cross"]]
y = df["cpu_time_ms"]

# Same split as train.py (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


def make_model(columns, alpha):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), columns)])
    return Pipeline([("pre", pre), ("reg", Ridge(alpha=alpha))])


configs = {
    "Ridge (no cross)": ["prefix", "file_type"],
    "Ridge (with cross)": ["prefix", "file_type", "cross"],
}

for name, cols in configs.items():
    best_alpha, best_val = None, float("inf")
    print(f"\n{name}")
    for a in ALPHAS:
        m = make_model(cols, a).fit(X_train, y_train)
        v = rmse(y_val, m.predict(X_val))
        print(f"  alpha={a:<6} val RMSE={v:.1f}")
        if v < best_val:
            best_alpha, best_val = a, v

    # Retrain with the best alpha and check the test set once
    best_model = make_model(cols, best_alpha).fit(X_train, y_train)
    test = rmse(y_test, best_model.predict(X_test))
    print(f"  BEST alpha={best_alpha}  val={best_val:.1f}  test={test:.1f}")

    tag = "cross" if "cross" in cols else "no_cross"
    joblib.dump(best_model, f"model/best_{tag}.joblib")

print("\nBest models saved.")
