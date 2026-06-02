"""
Production-locked Ridge Regression baseline forecasting pipeline for AQI.
Establishes a rigorous, linear, time-aware statistical baseline to evaluate 
against advanced non-linear ensemble models.
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Route exact components from your modular loader setup
from load_data import (
    load_xy, 
    get_chronological_splits, 
    apply_leakage_free_correlation_filter, 
    calculate_conformal_margin, 
    compute_aqi_event_metrics,
    export_residual_diagnostics
)
from monitoring import run_data_drift_monitoring

# ── FIXED: PRODUCTION PATH SETUP RESOLVING UPWARD TO ROOT ──────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def train_ridge(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n Ridge Regression Baseline — {horizon}h Forecast Horizon\n{'=' * 75}")
    
    # 1. Feature Consumption via load_data base
    X, y = load_xy(horizon)

    # FIXED: Unpack directly to targets without '_raw' suffixes to satisfy downstream dependencies
    X_train_raw, y_train, X_cal_raw, y_cal, X_test_raw, y_test = get_chronological_splits(X, y, horizon)
    
    # Apply Advanced Correlation Filtering (Shields distinct summaries, handles adjacent lag chains)
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train_raw, X_test_raw, X_cal_raw, threshold=0.95
    )
    
    # Save Dropped Features Log
    pd.DataFrame({"dropped_feature": dropped_cols}).to_csv(METRICS_DIR / f"ridge_dropped_features_{horizon}h.csv", index=False)
    
    # Save Feature Manifest to prevent inference schema mismatch
    pd.DataFrame({"feature": X_train.columns}).to_csv(METRICS_DIR / f"ridge_features_{horizon}h.csv", index=False)
    print(f"Features retained after advanced filtering: {X_train.shape[1]} (Dropped {len(dropped_cols)})")

    # 2. TimeSeries Cross-Validation Phase via Clean Training Space
    gap = horizon
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_metrics = []
    
    print(f"Validating architectural stability over {n_splits} historical validation folds...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        fold_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20)))
        ])
        fold_pipeline.fit(X_fold_train, y_fold_train)
        
        fold_preds = np.clip(fold_pipeline.predict(X_fold_val), 0, 500)
        fold_rmse = root_mean_squared_error(y_fold_val.values, fold_preds)
        fold_metrics.append(fold_rmse)

    # 3. Fit Final Production-Grade Pipeline (Alpha search via nested TimeSeriesSplit)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 30), cv=TimeSeriesSplit(n_splits=5))),
    ])
    print("\nTraining Ridge Regression Baseline Pipeline...")
    model.fit(X_train, y_train)
    best_alpha = model.named_steps["ridge"].alpha_
    print(f"Training complete. Optimized Regularization Strength Alpha: {best_alpha:.4f}")

    # 4. Inductive Conformal Formatting via Isolated Calibration Partition
    cal_preds = np.clip(model.predict(X_cal), 0, 500)
    margin_of_error = calculate_conformal_margin(np.abs(y_cal.values - cal_preds))

    # Test Set Inference
    preds = np.clip(model.predict(X_test), 0, 500)
    y_arr = y_test.values
    
    pi_lower = np.clip(preds - margin_of_error, 0, 500)
    pi_upper = np.clip(preds + margin_of_error, 0, 500)

    # Package Complete Production Deployment Dictionary Artifact
    joblib.dump({
        "model": model,
        "feature_names": list(X_train.columns),
        "conformal_margin": float(margin_of_error)
    }, MODEL_DIR / f"ridge_{horizon}h.pkl")

    # Save Predictions DataFrame Manifest
    pd.DataFrame({
        "actual": y_arr, "predicted": preds, "lower_bound": pi_lower, "upper_bound": pi_upper
    }, index=X_test.index).to_csv(METRICS_DIR / f"ridge_predictions_{horizon}h.csv", index=False)

    # Run Complete Residual Analysis Suite Plots (load_data handles the metrics routing)
    export_residual_diagnostics(y_arr, preds, horizon, "Ridge", METRICS_DIR, MODEL_DIR)

    # 5. Core Performance Telemetry Calculations
    test_rmse = root_mean_squared_error(y_arr, preds)
    test_mae  = mean_absolute_error(y_arr, preds)
    test_r2   = r2_score(y_arr, preds)
    test_evs  = explained_variance_score(y_arr, preds)
    test_mape = float(np.mean(np.abs((y_arr - preds) / np.maximum(np.abs(y_arr), 1.0))) * 100)
    
    # Stratified Coverage Metrics with Hard Statistical Sample Counts
    observed_coverage = np.mean((y_arr >= pi_lower) & (y_arr <= pi_upper))
    avg_interval_width = np.mean(pi_upper - pi_lower)
    
    mask_150 = y_arr > 150
    n_gt150 = int(mask_150.sum())
    coverage_above_150 = float(np.mean((y_arr[mask_150] >= pi_lower[mask_150]) & (y_arr[mask_150] <= pi_upper[mask_150]))) if n_gt150 > 0 else np.nan
    
    mask_200 = y_arr > 200
    n_gt200 = int(mask_200.sum())
    coverage_above_200 = float(np.mean((y_arr[mask_200] >= pi_lower[mask_200]) & (y_arr[mask_200] <= pi_upper[mask_200]))) if n_gt200 > 0 else np.nan

    # Target Pollution Spike Detections
    events_150 = compute_aqi_event_metrics(y_arr, preds, 150)
    events_200 = compute_aqi_event_metrics(y_arr, preds, 200)

    # Naive Persistence Baseline Validation
    lag_col = "aqi_lag_1"
    p_mae, p_r2, forecast_skill_score, r2_improvement_vs_baseline = 0.0, 0.0, 0.0, 0.0
    if lag_col in X_test.columns:
        p_mae = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2 = r2_score(y_arr, X_test[lag_col].values)
        forecast_skill_score = float(1.0 - (test_mae / p_mae))
        r2_improvement_vs_baseline = test_r2 - p_r2

    # 6. Feature Coefficient Feature Sorting & Drift Evaluation
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "abs_coef": np.abs(model.named_steps["ridge"].coef_)
    }).sort_values("abs_coef", ascending=False)
    
    coef_df.to_csv(METRICS_DIR / f"ridge_coefficients_{horizon}h.csv", index=False)
    top_20_linear_features = coef_df.head(20)["feature"].tolist()
    
    # Trigger distribution stability monitoring suite with high-variance features only
    run_data_drift_monitoring(X_train, X_test, horizon, "Ridge", top_20_linear_features)

    # 7. Package Final Payload Dictionary
    metrics = {
        "model": "Ridge", 
        "horizon": f"{horizon}h", 
        "best_alpha": float(best_alpha),
        "cv_mean_val_rmse": float(np.mean(fold_metrics)),
        "cv_std_val_rmse": float(np.std(fold_metrics)),
        "test_rmse": test_rmse, 
        "test_mae": test_mae, 
        "test_mape": test_mape, 
        "test_r2": test_r2,
        "test_explained_variance": float(test_evs),
        "baseline_horizon_mae": p_mae, 
        "baseline_horizon_r2": p_r2, 
        "forecast_skill_score": forecast_skill_score,
        "r2_improvement_vs_baseline": float(r2_improvement_vs_baseline),
        "conformal_margin_width": float(margin_of_error), 
        "conformal_global_coverage": float(observed_coverage), 
        "conformal_average_width": float(avg_interval_width),
        "conformal_coverage_gt150": float(coverage_above_150),
        "conformal_n_gt150": n_gt150,
        "conformal_coverage_gt200": float(coverage_above_200),
        "conformal_n_gt200": n_gt200,
        **events_150, 
        **events_200
    }

    with open(METRICS_DIR / f"ridge_metrics_{horizon}h.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics


if __name__ == "__main__":
    results = {}
    for h in (24, 48, 72):
        results[f"{h}h"] = train_ridge(h)

    print("\n" + "=" * 145)
    print("FINAL SUMMARY REPORT — TIME-SERIES CV RIDGE REGRESSION BASELINES")
    print("=" * 145)
    print(f"{'Horizon':<8}{'CV Val RMSE':>14}{'Test MAE':>10}{'Test MAPE':>11}{'Test R²':>10}{'Test EVS':>10}{'Obs Coverage':>14}{'Coverage >200':>15}")
    print("-" * 145)
    for horizon, m in results.items():
        print(
            f"{horizon:<8}"
            f"{m['cv_mean_val_rmse']:>14.3f}"
            f"{m['test_mae']:>10.1f}"
            f"{m['test_mape']:>10.2f}%"
            f"{m['test_r2']:>10.3f}"
            f"{m['test_explained_variance']:>10.3f}"
            f"{m['conformal_global_coverage']*100:>13.1f}%"
            f"{m['conformal_coverage_gt200']*100:>14.1f}%"
        )