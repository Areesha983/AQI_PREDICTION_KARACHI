"""
Enterprise-grade time-series forecasting pipeline for AQI using XGBoost.
Combines rigorous TimeSeries Cross-Validation with absolute MLOps deployment readiness.
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import shap

# Route exact components from your modular load setup
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

print("INITIALIZING SUPREME HYBRID FORECASTING ENGINE (PRODUCTION CODES LOCK)")

# ── Stratified Metric Analyzers ─────────────────────────────────────────────

def error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates MAE breakdowns partitioned by EPA Air Quality index severity tiers."""
    results = {}
    bands = {
        "all":                 (0,    9999),
        "good_moderate":       (0,    100),
        "unhealthy_sensitive": (101,  150),
        "unhealthy":           (151,  200),
        "very_unhealthy":      (201,  300),
        "hazardous":           (301,  9999),
    }
    for label, (lo, hi) in bands.items():
        mask = (y_true >= lo) & (y_true <= hi)
        if mask.sum() < 5:
            results[label] = {"n": int(mask.sum()), "mae": None}
            continue
        results[label] = {
            "n":   int(mask.sum()),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        }
    return results


def quantile_error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates MAE on the upper tail percentiles to measure extreme event accuracy."""
    quantiles = [90, 95, 99]
    errors = {}
    for q in quantiles:
        threshold = np.percentile(y_true, q)
        mask = y_true >= threshold
        if mask.sum() < 5:
            errors[f"mae_top_{q}"] = None
            continue
        errors[f"mae_top_{q}"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    return errors


# ── Core Engine Pipeline ────────────────────────────────────────────────────

def train_xgboost(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n XGBoost Engine — {horizon}h Forecast Horizon\n{'=' * 75}")
    
    # 1. Feature Consumption via load_data base
    X, y = load_xy(horizon)

    # FIXED: Unpack directly to targets without '_raw' suffixes to satisfy downstream dependencies
    X_train_raw, y_train, X_cal_raw, y_cal, X_test_raw, y_test = get_chronological_splits(X, y, horizon)
    
    # Apply Advanced Correlation Filtering
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train_raw, X_test_raw, X_cal_raw, threshold=0.95
    )
    
    # Save Feature Manifests
    pd.DataFrame({"dropped_feature": dropped_cols}).to_csv(METRICS_DIR / f"xgb_dropped_features_{horizon}h.csv", index=False)
    pd.DataFrame({"feature": X_train.columns}).to_csv(METRICS_DIR / f"xgb_features_{horizon}h.csv", index=False)
    print(f"Features retained after filtering: {X_train.shape[1]} (Dropped {len(dropped_cols)})")

    # 2. TimeSeries Cross-Validation Phase via Clean Training Space
    gap = horizon
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_metrics = []
    
    print(f"Validating stability over {n_splits} historical folds...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        fold_weights = np.ones(len(y_fold_train))
        fold_weights[y_fold_train > 150] = 2.5
        fold_weights[y_fold_train > 200] = 5.0
        
        fold_model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42 + fold, n_jobs=-1
        )
        fold_model.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)
        
        fold_preds = np.clip(fold_model.predict(X_fold_val), 0, 500)
        fold_rmse = root_mean_squared_error(y_fold_val.values, fold_preds)
        fold_metrics.append(fold_rmse)

    # 3. Fit Final Production-Grade Model
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train.values > 150] = 2.5
    sample_weights[y_train.values > 200] = 5.0

    model = xgb.XGBRegressor(
        n_estimators=1200, max_depth=7, learning_rate=0.02,
        subsample=0.85, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    print("\nTraining Final XGBoost Model...")
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("Training complete.")

    # 4. Inductive Conformal Formatting
    cal_preds = np.clip(model.predict(X_cal), 0, 500)
    margin_of_error = calculate_conformal_margin(np.abs(y_cal.values - cal_preds))

    # Test Set Inference
    preds = np.clip(model.predict(X_test), 0, 500)
    y_arr = y_test.values
    
    pi_lower = np.clip(preds - margin_of_error, 0, 500)
    pi_upper = np.clip(preds + margin_of_error, 0, 500)

    # Package Deployment Dictionary
    joblib.dump({
        "model": model,
        "feature_names": list(X_train.columns),
        "conformal_margin": float(margin_of_error)
    }, MODEL_DIR / f"xgboost_{horizon}h.pkl")

    # 5. Core Performance Telemetry Calculations
    test_rmse = root_mean_squared_error(y_arr, preds)
    test_mae  = mean_absolute_error(y_arr, preds)
    test_r2   = r2_score(y_arr, preds)
    test_evs  = explained_variance_score(y_arr, preds)
    test_mape = float(np.mean(np.abs((y_arr - preds) / np.maximum(np.abs(y_arr), 1.0))) * 100)
    
    observed_coverage = np.mean((y_arr >= pi_lower) & (y_arr <= pi_upper))
    avg_interval_width = np.mean(pi_upper - pi_lower)
    
    mask_150 = y_arr > 150
    n_gt150 = int(mask_150.sum())
    coverage_above_150 = float(np.mean((y_arr[mask_150] >= pi_lower[mask_150]) & (y_arr[mask_150] <= pi_upper[mask_150]))) if n_gt150 > 0 else np.nan
    
    mask_200 = y_arr > 200
    n_gt200 = int(mask_200.sum())
    coverage_above_200 = float(np.mean((y_arr[mask_200] >= pi_lower[mask_200]) & (y_arr[mask_200] <= pi_upper[mask_200]))) if n_gt200 > 0 else np.nan

    events_150 = compute_aqi_event_metrics(y_arr, preds, 150)
    events_200 = compute_aqi_event_metrics(y_arr, preds, 200)
    stratified_bands = error_analysis(y_arr, preds)
    quantile_errors = quantile_error_analysis(y_arr, preds)

    # Naive Persistence Benchmark
    lag_horizon = f"aqi_lag_{horizon}"
    p_mae, p_r2, forecast_skill_score, r2_improvement_vs_baseline = 0.0, 0.0, 0.0, 0.0
    if lag_horizon in X_test.columns:
        p_mae = mean_absolute_error(y_arr, X_test[lag_horizon].values)
        p_r2 = r2_score(y_arr, X_test[lag_horizon].values)
        forecast_skill_score = float(1.0 - (test_mae / p_mae))
        r2_improvement_vs_baseline = test_r2 - p_r2

    # Export Predictions Manifest
    pd.DataFrame({
        "actual": y_arr, "predicted": preds, "lower_bound": pi_lower, "upper_bound": pi_upper
    }, index=X_test.index).to_csv(METRICS_DIR / f"xgb_predictions_{horizon}h.csv", index=False)

    # Residual Graphics Tracing
    export_residual_diagnostics(y_arr, preds, horizon, "XGB", METRICS_DIR, MODEL_DIR)

    # 6. SHAP Calculations & Interpretability Suite
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(200, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_sample)
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)

    shap_imp = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(METRICS_DIR / f"xgb_top_features_{horizon}h.csv", index=False)

    # Drift Checking Pipeline
    top_20_shap_features = shap_imp.head(20)["feature"].tolist()
    run_data_drift_monitoring(X_train, X_test, horizon, "XGB", top_20_shap_features)

    # 7. Package Final Payload Dictionary
    metrics = {
        "model": "XGBoost", 
        "horizon": f"{horizon}h", 
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
        "error_by_band": stratified_bands,
        "quantile_errors": quantile_errors,
        **events_150, 
        **events_200
    }
    
    with open(METRICS_DIR / f"xgb_metrics_{horizon}h.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics


if __name__ == "__main__":
    results = {}
    for h in (24, 48, 72):
        results[f"{h}h"] = train_xgboost(h)

    print("\n" + "=" * 145)
    print("FINAL SUMMARY REPORT — TIME-SERIES CV XGBOOST FORECASTING ENGINES")
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