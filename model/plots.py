
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

CSV_PATH = "model/combined_dataset.csv"
PLOT_DIR = "model/plots"

os.makedirs(PLOT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# --------------------------------------------------
# 1. Load dataset and create the same train/test split
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)
df["cross"] = df["prefix"] + "_" + df["file_type"]

X = df[["prefix", "file_type", "cross"]]
y = df["cpu_time_ms"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# --------------------------------------------------
# 2. Load trained models
# --------------------------------------------------

m_no = joblib.load("model/best_no_cross.joblib")
m_cr = joblib.load("model/best_cross.joblib")
m_rf = joblib.load("model/random_forest.joblib")


# --------------------------------------------------
# 3. Generate predictions
# --------------------------------------------------

baseline_pred = np.full(len(y_test), y_train.mean())

pred_no = m_no.predict(X_test)
pred_cr = m_cr.predict(X_test)
pred_rf = m_rf.predict(X_test)


# --------------------------------------------------
# 4. Calculate and print test RMSE
# --------------------------------------------------

names = [
    "Baseline (mean)",
    "Ridge (no cross)",
    "Ridge (with cross)",
    "Random Forest",
]

predictions = [
    baseline_pred,
    pred_no,
    pred_cr,
    pred_rf,
]

vals = [
    rmse(y_test, pred)
    for pred in predictions
]

print("\nTest RMSE comparison")
for name, value in zip(names, vals):
    print(f"{name:25s}: {value:.1f} ms")


# --------------------------------------------------
# 5. Plot RMSE comparison
# --------------------------------------------------

plt.figure(figsize=(8, 4))

bars = plt.bar(names, vals)

for bar, value in zip(bars, vals):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.ylabel("Test RMSE (ms)")
plt.title("Test RMSE by Model")
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "rmse_comparison.png"),
    dpi=150,
)
plt.close()


# --------------------------------------------------
# 6. Plot predicted vs actual CPU time
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 3, figsize=(15, 4), sharex=True, sharey=True
)

model_predictions = [
    (pred_no, "Ridge (no cross)"),
    (pred_cr, "Ridge (with cross)"),
    (pred_rf, "Random Forest"),
]

lo = min(y_test.min(), *(p.min() for p, _ in model_predictions))
hi = max(y_test.max(), *(p.max() for p, _ in model_predictions))

for ax, (pred, title) in zip(axes, model_predictions):
    ax.scatter(y_test, pred, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_title(title)
    ax.set_xlabel("Actual CPU time (ms)")
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Predicted CPU time (ms)")

plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "predicted_vs_actual.png"),
    dpi=150,
)
plt.close()


# --------------------------------------------------
# 7. Plot combined dataset CPU-time distribution
# --------------------------------------------------

plt.figure(figsize=(7, 4))

plt.hist(df["cpu_time_ms"], bins=30)

plt.xlabel("CPU time (ms)")
plt.ylabel("Frequency")
plt.title("CPU Time Distribution - Combined Dataset")

plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "combined_dataset_distribution.png"),
    dpi=150,
)
plt.close()

print("\nAll plots saved to model/plots/")
