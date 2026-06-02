from pathlib import Path
import numpy as np
import pandas as pd

# Absolute pointer to the root of KARACHI_AQI_PREDICTOR/ workspace
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# ── Columns always excluded ────────────────────────────────────────────────
LEAKY_RAW_COLS = ["aqi"]

REDUNDANT_TIME_COLS = [
    "hour", "day", "month", "weekday",
    "day_of_year", "week_of_year", "hour_of_week",
]

ALL_TARGETS = [
    "target_aqi_12h", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    "target_aqi_12h_log", "target_aqi_24h_log", "target_aqi_48h_log", "target_aqi_72h_log",
    "target_cat_12h", "target_cat_24h", "target_cat_48h", "target_cat_72h",
]

BASE_DROP = ["datetime"] + LEAKY_RAW_COLS + REDUNDANT_TIME_COLS + ALL_TARGETS
LEAKAGE_EXACT = frozenset(LEAKY_RAW_COLS + ALL_TARGETS)


def load_xy(horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load (X, y) for a given forecast horizon with integrated volatility trends.
    """
    assert horizon in (12, 24, 48, 72), "Horizon must be 12, 24, 48, or 72"
    
    # Check new global data folder structure first, then fallback to local directory
    data_path = BASE_DIR / "data" / "processed" / "featured_dataset.csv"
    if not data_path.exists():
        data_path = BASE_DIR / "feature_pipeline" / "data" / "processed" / "featured_dataset.csv"
    if not data_path.exists():
        data_path = SCRIPT_DIR / "featured_dataset.csv"
        
    print(f"\nLoading dataset from : {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Target data file asset missing at path: {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Raw dataset shape    : {df.shape}")

    target_col = f"target_aqi_{horizon}h"
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: '{target_col}'")

    # ── ADVANCED FEATURE ENGINEERING: VOLATILITY & ACCELERATION ───────────
    if "pm25" in df.columns:
        df["pm25_diff_1h"] = df["pm25"].diff()
        df["pm25_roll_std_24h"] = df["pm25"].rolling(window=24, min_periods=1).std()
        df["pm25_roll_std_6h"] = df["pm25"].rolling(window=6, min_periods=1).std()
    if "pm10" in df.columns:
        df["pm10_diff_1h"] = df["pm10"].diff()
        df["pm10_roll_std_12h"] = df["pm10"].rolling(window=12, min_periods=1).std()
    if "temperature" in df.columns:
        df["temp_diff_24h"] = df["temperature"].diff(24)
        
    # Weather-Pollution Interaction Features to combat PM2.5 single-lag reliance
    if "pm25" in df.columns and "humidity" in df.columns:
        df["interaction_pm25_humidity"] = df["pm25"] * df["humidity"]
    if "pm25" in df.columns and "wind_speed" in df.columns:
        df["interaction_pm25_wind_inverse"] = df["pm25"] / (df["wind_speed"] + 0.1)

    # Drop rows with no valid target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    y = df[target_col].copy()

    # Drop all excluded columns
    X = df.drop(columns=[c for c in BASE_DROP if c in df.columns], errors="ignore")

    # Drop high-missing columns
    missing_frac = X.isna().mean()
    high_missing = missing_frac[missing_frac > 0.10].index.tolist()
    if high_missing:
        print(f"Dropping {len(high_missing)} cols with >10% NaN: {high_missing}")
        X = X.drop(columns=high_missing)

    # Fill any residual gaps safely
    X = X.ffill().bfill()

    # Leakage guard
    leaky_found = [c for c in X.columns if c in LEAKAGE_EXACT]
    if leaky_found:
        raise ValueError(f"Leakage columns still present in X after dropping: {leaky_found}")

    if X.isna().any().any():
        still_nan = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs remain in X after cleaning: {still_nan}")

    X = X.select_dtypes(include=[np.number])

    # Diagnostics
    print(f"\nTarget: {target_col}")
    print(y.describe().round(1).to_string())
    print(f"\nFeature count        : {X.shape[1]}")
    print(f"Rows                 : {X.shape[0]:,}")
    print(f"AQI > 200 (spikes)   : {(y > 200).sum():,}  ({(y > 200).mean() * 100:.1f}%)")

    lag_cols = [c for c in X.columns if any(k in c for k in ("lag", "roll", "ewm", "diff", "interaction"))]
    print(f"\nSample engineered/interaction features ({len(lag_cols)} total):")
    print(lag_cols[:20])

    return X, y


def get_chronological_splits(X: pd.DataFrame, y: pd.Series, horizon: int):
    n = len(X)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.85)
    
    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end]
    
    X_cal = X.iloc[train_end + horizon : cal_end].copy()
    y_cal = y.iloc[train_end + horizon : cal_end]
    
    X_test = X.iloc[cal_end:].copy()
    y_test = y.iloc[cal_end:]
    
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def apply_leakage_free_correlation_filter(X_train: pd.DataFrame, X_test: pd.DataFrame, X_cal: pd.DataFrame = None, threshold: float = 0.90) -> tuple:
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    protected_features = {
        "aqi_lag_24", "aqi_lag_48", "aqi_lag_72", 
        "pm25_roll_std_24h", "interaction_pm25_humidity", "interaction_pm25_wind_inverse"
    }
    
    to_drop = [
        col for col in upper.columns 
        if any(upper[col] > threshold) and col not in protected_features
    ]
    
    X_train_clean = X_train.drop(columns=to_drop)
    X_test_clean = X_test.drop(columns=to_drop)
    
    if X_cal is not None:
        X_cal_clean = X_cal.drop(columns=to_drop)
        return X_train_clean, X_cal_clean, X_test_clean, to_drop
        
    return X_train_clean, X_test_clean, to_drop


def calculate_conformal_margin(abs_residuals: np.ndarray, alpha: float = 0.05) -> float:
    n_cal = len(abs_residuals)
    if n_cal == 0:
        return 0.0
    q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
    return float(np.quantile(abs_residuals, q_level))


def compute_aqi_event_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: int) -> dict:
    true_event = (y_true > threshold).astype(int)
    pred_event = (y_pred > threshold).astype(int)
    
    tp = np.sum((true_event == 1) & (pred_event == 1))
    fp = np.sum((true_event == 0) & (pred_event == 1))
    fn = np.sum((true_event == 1) & (pred_event == 0))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {f"precision_gt{threshold}": precision, f"recall_gt{threshold}": recall, f"f1_gt{threshold}": f1}


def export_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, horizon: int, model_name: str, metrics_dir: Path, models_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    residuals = y_true - y_pred
    res_df = pd.DataFrame({"actual": y_true, "predicted": y_pred, "residual": residuals})
    res_df.to_csv(metrics_dir / f"{model_name.lower()}_residuals_{horizon}h.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, color="teal", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{model_name} Residuals Distribution Histogram ({horizon}h)")
    ax.set_xlabel("Residual Error (Actual - Predicted)")
    ax.set_ylabel("Frequency Count")
    
    # FIXED: Direct plot routing straight to metrics_dir instead of models_dir
    plt.savefig(metrics_dir / f"{model_name.lower()}_residual_hist_{horizon}h.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── RESTORED RUNTIME EXECUTION GATEWAY ─────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("    LOAD_DATA ENGINE: PIPELINE INTEGRITY & SPLIT VERIFICATION LOOP")
    print("=" * 80)

    for h in (24, 48, 72):
        print("\n" + "#" * 60)
        print(f" TESTING TIMELINE PROFILE FOR FORECAST HORIZON: {h}h")
        print("#" * 60)
        
        try:
            X, y = load_xy(h)
            X_train, y_train, X_cal, y_cal, X_test, y_test = get_chronological_splits(X, y, horizon=h)
            
            print(f"\n[{h}h] Core Layout Verification:")
            print(f"  ├─ Complete Arrays : X Dim -> {X.shape} | y Dim -> {y.shape}")
            print(f"  ├─ Train Segment   : {X_train.shape[0]:,} rows")
            print(f"  ├─ Validation Gap  : {h} hours")
            print(f"  ├─ Calibration Set : {X_cal.shape[0]:,} rows")
            print(f"  └─ Evaluation Test : {X_test.shape[0]:,} rows")

            X_train_f, X_test_f, dropped_cols = apply_leakage_free_correlation_filter(X_train, X_test, threshold=0.90)
            print(f"  └─ Feature Filtering: Dropped {len(dropped_cols)} highly correlated columns.")
            print(f"   Final Feature Matrix Dimensions: {X_train_f.shape[1]}")

        except Exception as e:
            print(f"CRITICAL ERROR testing horizon {h}h: {str(e)}")