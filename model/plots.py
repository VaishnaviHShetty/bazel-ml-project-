import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

CSV_PATH = "model/combined_dataset.csv"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


df = pd.read_csv(CSV_PATH)
df["cross"] = df["prefix"] + "_" + df["file_type"]
X = df[["prefix", "file_type", "cross"]]
y = df["cpu_time_ms"]

# Same split as train.py and tune.py
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

m_no = joblib.load("model/best_no_cross.joblib")
m_cr = joblib.load("model/best_cross.joblib")

baseline_pred = np.full(len(y_test), y_train.mean())
pred_no = m_no.predict(X_test)
pred_cr = m_cr.predict(X_test)

# Plot 1: RMSE comparison on the test set
names = ["Baseline\n(mean)", "Ridge\n(no cross)", "Ridge\n(with cross)"]
vals = [rmse(y_test, baseline_pred), rmse(y_test, pred_no), rmse(y_test, pred_cr)]

plt.figure(figsize=(6, 4))
bars = plt.bar(names, vals, color=["gray", "steelblue", "orange"])
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")
plt.ylabel("Test RMSE (ms)")
plt.title("Test RMSE by model")
plt.tight_layout()
plt.savefig("model/plots/rmse_comparison.png", dpi=150)
plt.close()

# Plot 2: predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, pred, title in zip(axes, [pred_no, pred_cr], ["No cross", "With cross"]):
    ax.scatter(y_test, pred, alpha=0.7)
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_title(title)
    ax.set_xlabel("Actual CPU time (ms)")
axes[0].set_ylabel("Predicted CPU time (ms)")
plt.tight_layout()
plt.savefig("model/plots/predicted_vs_actual.png", dpi=150)
plt.close()

print("Saved plots to model/plots/")
for n, v in zip(names, vals):
    print(n.replace("\n", " "), round(v, 1))
    # Plot 3: Combined dataset CPU time distribution
plt.figure(figsize=(7, 4))
plt.hist(df["cpu_time_ms"], bins=30)
plt.xlabel("CPU time (ms)")
plt.ylabel("Frequency")
plt.title("CPU Time Distribution - Combined Dataset")
plt.tight_layout()

plt.savefig("model/plots/combined_dataset_distribution.png", dpi=150)
plt.close()

print("COMBINED DATASET PLOT CREATED")
