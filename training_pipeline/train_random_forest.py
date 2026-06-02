"""
Enterprise-grade Random Forest forecasting pipeline for AQI.
Enhanced with non-linear exponential spike weighting and SHAP explainability.
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import shap

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

print("INITIALIZING SUPREME RANDOM FOREST ENGINE (PRODUCTION CODES LOCK)")


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def train_rf(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n Random Forest Engine — {horizon}h Forecast Horizon\n{'=' * 75}")
    
    X, y = load_xy(horizon)
    X_train, y_train, X_cal, y_cal, X_test, y_test = get_chronological_splits(X, y, horizon)
    
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.90
    )
    
    pd.DataFrame({"dropped_feature": dropped_cols}).to_csv(METRICS_DIR / f"rf_dropped_features_{horizon}h.csv", index=False)
    pd.DataFrame({"feature": X_train.columns}).to_csv(METRICS_DIR / f"rf_features_{horizon}h.csv", index=False)
    print(f"Features retained after advanced filtering: {X_train.shape[1]} (Dropped {len(dropped_cols)})")

    print(f"\nExecuting Hyperparameter Optimization Pass via RandomizedSearchCV (9 total fits)...")
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20],
        "min_samples_leaf": [5, 10],
        "min_samples_split": [8, 12],
        "max_features": ["sqrt"]
    }
    
    base_rf = RandomForestRegressor(random_state=42, n_jobs=1)
    tuning_cv = TimeSeriesSplit(n_splits=3, gap=horizon)
    
    search_engine = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=3,
        cv=tuning_cv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # ── FIXED SAMPLE WEIGHTS: NON-LINEAR EXPONENTIAL SCALING ──────────────
    sample_weights = 1.0 + np.exp(np.minimum(y_train.values, 350) / 110.0) - np.exp(0)
    sample_weights = np.clip(sample_weights, 1.0, 25.0)
    
    search_engine.fit(X_train, y_train, sample_weight=sample_weights)
    model = search_engine.best_estimator_
    print(f"Optimal Hyperparameters Discovered: {search_engine.best_params_}")

    cv_results = search_engine.cv_results_
    best_idx = search_engine.best_index_
    cv_mean_rmse = -cv_results["mean_test_score"][best_idx]
    cv_std_rmse = cv_results["std_test_score"][best_idx]

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    feature_importance.to_csv(METRICS_DIR / f"rf_feature_importance_{horizon}h.csv", index=False)

    # Inductive Conformal Calibration
    cal_preds = np.clip(model.predict(X_cal), 0, 500)
    margin_of_error = calculate_conformal_margin(np.abs(y_cal.values - cal_preds))

    # Test Set Inference
    preds = np.clip(model.predict(X_test), 0, 500)
    y_arr = y_test.values
    
    pi_lower = np.clip(preds - margin_of_error, 0, 500)
    pi_upper = np.clip(preds + margin_of_error, 0, 500)

    joblib.dump({
        "model": model,
        "feature_names": list(X_train.columns),
        "conformal_margin": float(margin_of_error)
    }, MODEL_DIR / f"random_forest_{horizon}h.pkl")

    test_rmse = root_mean_squared_error(y_arr, preds)
    test_mae  = mean_absolute_error(y_arr, preds)
    test_median_ae = median_absolute_error(y_arr, preds)
    test_r2   = r2_score(y_arr, preds)
    test_evs  = explained_variance_score(y_arr, preds)
    test_mape = float(np.mean(np.abs((y_arr - preds) / np.maximum(np.abs(y_arr), 1.0))) * 100)
    
    observed_coverage = np.mean((y_arr >= pi_lower) & (y_arr <= pi_upper))
    avg_interval_width = np.mean(pi_upper - pi_lower)
    
    mask_150 = y_arr > 150
    n_gt150 = int(mask_150.sum())
    coverage_above_150 = float(np.mean((y_arr[mask_150] >= pi_lower[mask_150]) & (y_arr[mask_150] <= pi_upper[mask_150]))) if n_gt150 >= 5 else 0.0
    
    mask_200 = y_arr > 200
    n_gt200 = int(mask_200.sum())
    coverage_above_200 = float(np.mean((y_arr[mask_200] >= pi_lower[mask_200]) & (y_arr[mask_200] <= pi_upper[mask_200]))) if n_gt200 >= 5 else 0.0

    events_150 = compute_aqi_event_metrics(y_arr, preds, 150)
    events_200 = compute_aqi_event_metrics(y_arr, preds, 200)
    stratified_bands = error_analysis(y_arr, preds)
    quantile_errors = quantile_error_analysis(y_arr, preds)

    # Baseline Comparisons
    lag_horizon = f"aqi_lag_{horizon}"
    p_mae, p_r2, forecast_skill_score, r2_improvement_vs_baseline = 0.0, 0.0, 0.0, 0.0
    if lag_horizon in X_test.columns:
        p_mae = mean_absolute_error(y_arr, X_test[lag_horizon].values)
        p_r2 = r2_score(y_arr, X_test[lag_horizon].values)
        forecast_skill_score = float(1.0 - (test_mae / p_mae)) if p_mae > 0 else 0.0
        r2_improvement_vs_baseline = test_r2 - p_r2

    pd.DataFrame({
        "actual": y_arr, "predicted": preds, "lower_bound": pi_lower, "upper_bound": pi_upper
    }, index=X_test.index).to_csv(METRICS_DIR / f"rf_predictions_{horizon}h.csv", index=False)

    export_residual_diagnostics(y_arr, preds, horizon, "RF", METRICS_DIR, MODEL_DIR)

    # ── FIXED INTERPRETABILITY: ENABLED SHAP RUNS ─────────────────────────
    GENERATE_SHAP = True  

    if GENERATE_SHAP:
        print("Calculating SHAP explainability mapping via localized test subsample...")
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(X_test), size=min(150, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_idx]
        
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(X_sample)
        mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)

        shap_imp = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
        shap_imp.to_csv(METRICS_DIR / f"rf_top_features_{horizon}h.csv", index=False)

        top_20_shap_features = shap_imp.head(20)["feature"].tolist()
        try:
            run_data_drift_monitoring(X_train, X_test, horizon, "RF", top_20_shap_features)
        except Exception as e:
            print(f"Drift monitoring log bypass: {e}")

        fig = plt.figure(figsize=(10, 8))
        try:
            shap.plots.beeswarm(shap_explanation, show=False)
        except Exception:
            shap.summary_plot(shap_explanation.values, X_sample, show=False)
            
        # FIXED: Routing SHAP summary visualization charts straight to METRICS_DIR
        plt.savefig(METRICS_DIR / f"rf_shap_summary_{horizon}h.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    metrics = {
        "model": "RandomForest", 
        "horizon": f"{horizon}h", 
        "cv_mean_val_rmse": float(cv_mean_rmse),
        "cv_std_val_rmse": float(cv_std_rmse),
        "test_rmse": float(test_rmse), 
        "test_mae": float(test_mae), 
        "test_median_ae": float(test_median_ae),
        "test_mape": float(test_mape), 
        "test_r2": float(test_r2),
        "test_explained_variance": float(test_evs),
        "baseline_horizon_mae": float(p_mae), 
        "baseline_horizon_r2": float(p_r2), 
        "forecast_skill_score": float(forecast_skill_score),
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
    
    with open(METRICS_DIR / f"rf_metrics_{horizon}h.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics


if __name__ == "__main__":
    results = {}
    for h in (24, 48, 72):
        results[f"{h}h"] = train_rf(h)

    print("\n" + "=" * 155)
    print("FINAL SUMMARY REPORT — TIME-SERIES CV RANDOM FOREST PRODUCTION ENGINES")
    print("=" * 155)
    print(f"{'Horizon':<8}{'CV Mean RMSE':>14}{'CV Std RMSE':>13}{'Test MAE':>10}{'Test MedAE':>12}{'Test R²':>10}{'Obs Coverage':>14}{'Coverage >200':>15}")
    print("-" * 155)
    for horizon, m in results.items():
        print(
            f"{horizon:<8}"
            f"{m['cv_mean_val_rmse']:>14.3f}"
            f"{m['cv_std_val_rmse']:>13.3f}"
            f"{m['test_mae']:>10.1f}"
            f"{m['test_median_ae']:>12.1f}"
            f"{m['test_r2']:>10.3f}"
            f"{m['conformal_global_coverage']*100:>13.1f}%"
            f"{m['conformal_coverage_gt200']*100:>14.1f}%"
        )