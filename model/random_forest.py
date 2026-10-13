
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

CSV_PATH = "model/combined_dataset.csv"

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load dataset
df = pd.read_csv(CSV_PATH)
df["cross"] = df["prefix"] + "_" + df["file_type"]

X = df[["prefix", "file_type", "cross"]]
y = df["cpu_time_ms"]

# Same split as the existing experiments
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# One-hot encode categorical features
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     ["prefix", "file_type", "cross"])
])

# Try different tree configurations
configs = [
    (100,  None, 2),
    (100,  5,    2),
    (200,  5,    2),
    (200,  8,    2),
    (200,  5,    5),
    (200,  None, 5),
]

best_model = None
best_config = None
best_val_rmse = float("inf")

print("Testing Random Forest configurations...\n")

for n_trees, depth, min_leaf in configs:
    model = Pipeline([
        ("pre", preprocessor),
        ("reg", RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=depth,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_score = rmse(y_val, val_pred)

    print(
        f"Trees={n_trees:3d} | "
        f"Depth={str(depth):4s} | "
        f"Min leaf={min_leaf} | "
        f"Validation RMSE={val_score:.1f} ms"
    )

    if val_score < best_val_rmse:
        best_val_rmse = val_score
        best_model = model
        best_config = (n_trees, depth, min_leaf)

# Evaluate the selected configuration on the test set
test_pred = best_model.predict(X_test)
test_score = rmse(y_test, test_pred)

print("\n--- Best Random Forest ---")
print(f"Trees: {best_config[0]}")
print(f"Max depth: {best_config[1]}")
print(f"Minimum samples per leaf: {best_config[2]}")
print(f"Validation RMSE: {best_val_rmse:.1f} ms")
print(f"Test RMSE: {test_score:.1f} ms")

# Save separately
joblib.dump(best_model, "model/random_forest.joblib")
print("\nModel saved to model/random_forest.joblib")
