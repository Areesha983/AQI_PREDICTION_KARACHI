"""
diagnose.py
-----------
Run this before retraining to understand why CV folds vary so wildly.
Prints per-fold AQI statistics so you can see what each fold "looks like".

Run:
    python diagnose.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "feature_pipeline" / "data" / "processed" / "featured_dataset.csv"

df = pd.read_csv(data_path)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

target = "target_aqi_24h"
df = df.dropna(subset=[target]).reset_index(drop=True)
y = df[target]

print("=" * 60)
print("OVERALL TARGET DISTRIBUTION")
print("=" * 60)
print(y.describe())
print(f"\nSkewness : {y.skew():.3f}")
print(f"% above 150 (Unhealthy): {(y > 150).mean()*100:.1f}%")
print(f"% above 200 (Very Unhealthy): {(y > 200).mean()*100:.1f}%")
print(f"% above 300 (Hazardous): {(y > 300).mean()*100:.1f}%")

print("\n" + "=" * 60)
print("AQI BY MONTH (seasonal patterns)")
print("=" * 60)
df["month"] = df["datetime"].dt.month
monthly = df.groupby("month")[target].agg(["mean", "std", "median"])
print(monthly.round(1))

print("\n" + "=" * 60)
print("AQI BY YEAR")
print("=" * 60)
df["year"] = df["datetime"].dt.year
yearly = df.groupby("year")[target].agg(["mean", "std", "min", "max", "count"])
print(yearly.round(1))

print("\n" + "=" * 60)
print("CV FOLD STATISTICS  (what each fold actually sees)")
print("=" * 60)
tscv = TimeSeriesSplit(n_splits=5, gap=72)
for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    t_start = df["datetime"].iloc[val_idx[0]]
    t_end   = df["datetime"].iloc[val_idx[-1]]
    print(f"\nFold {fold}  [{t_start.date()} → {t_end.date()}]")
    print(f"  Train : n={len(y_tr):,}  mean={y_tr.mean():.1f}  std={y_tr.std():.1f}  "
          f"max={y_tr.max():.0f}")
    print(f"  Val   : n={len(y_val):,}  mean={y_val.mean():.1f}  std={y_val.std():.1f}  "
          f"max={y_val.max():.0f}")
    mean_shift = abs(y_val.mean() - y_tr.mean())
    print(f"  Mean shift train→val: {mean_shift:.1f}  "
          f"{'⚠ HIGH' if mean_shift > 30 else 'OK'}")

print("\n" + "=" * 60)
print("SO2 COLUMN CHECK  (top feature — could be a data issue)")
print("=" * 60)
so2 = df["so2"]
print(so2.describe())
print(f"Zero values : {(so2 == 0).mean()*100:.1f}%")
print(f"Correlation with target: {so2.corr(y):.3f}")

print("\n" + "=" * 60)
print("MISSING DATA SUMMARY")
print("=" * 60)
missing = df.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0].to_string())

# Plot AQI over time to visually spot regime shifts
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(df["datetime"], y, linewidth=0.4, alpha=0.7, color="steelblue")
axes[0].set_title("Target AQI (24h ahead) over time")
axes[0].set_ylabel("AQI")
axes[0].axhline(150, color="orange", linestyle="--", linewidth=0.8, label="Unhealthy (150)")
axes[0].axhline(300, color="red",    linestyle="--", linewidth=0.8, label="Hazardous (300)")
axes[0].legend()

# Rolling 30-day mean to see trend
rolling_mean = y.rolling(24 * 30, min_periods=24).mean()
axes[1].plot(df["datetime"], rolling_mean, color="darkorange", linewidth=1.5)
axes[1].set_title("30-day rolling mean AQI (trend)")
axes[1].set_ylabel("AQI")

plt.tight_layout()
out = Path("aqi_diagnosis.png")
plt.savefig(out, dpi=120)
print(f"\nPlot saved → {out.resolve()}")