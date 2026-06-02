from pathlib import Path
import json
import pandas as pd
from scipy.stats import ks_2samp

# Route directories cleanly to baseline execution scopes
METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def run_data_drift_monitoring(X_train: pd.DataFrame, X_test: pd.DataFrame, horizon: int, model_name: str, top_features_list: list) -> dict:
    """
    Production-Grade SHAP-Driven Population Stability & Data Drift Monitoring Module.
    
    Mathematical & Operational Mechanics:
    ─────────────────────────────────────────────
    1. Feature Target Filtering: Accepts the pre-computed feature importance array. It ignores 
       high-dimensional tracking noise by dynamically tracking only the Top 20 most impactful 
       predictors responsible for model variance.
       
    2. Non-Parametric Distribution Verification: Executes a Two-Sample Kolmogorov-Smirnov (KS) test. 
       It tests the null hypothesis (H₀) that the training baseline sample and live test stream 
       sample are drawn from identical continuous distributions.
       
    3. Alpha-Level Statistical Flagging: Implements a strict confidence constraint (Alpha = 0.05). 
       If the asymptotic p-value drops below 0.05, H₀ is rejected, proving that a statistically 
       significant population distribution shift has occurred.
       
    4. MLOps Retraining Trigger: Exports a machine-readable JSON telemetry file. Production orchestration 
       layers (e.g., Airflow, Prefect) can instantly consume this artifact to trigger automated model 
       pipeline retraining before silent accuracy degradation sets in.
    """
    print(f"\nExecuting SHAP-Driven Kolmogorov-Smirnov Drift Diagnostics for {model_name} ({horizon}h)...")
    
    drift_report = {}
    
    # Isolate checking strictly to the top 20 most critical tracking features to avoid false alerts
    features_to_check = top_features_list[:20]
    
    if not features_to_check:
        print(" [WARNING]: Drift monitor received an empty features array. Defaulting to first 10 columns.")
        features_to_check = list(X_train.columns[:10])

    for feature in features_to_check:
        if feature not in X_train.columns or feature not in X_test.columns:
            print(f"   ↳ [Skipped] Feature '{feature}' missing from split matrices.")
            continue
            
        # Strip tracking NaNs to isolate continuous sample densities securely
        train_sample = X_train[feature].dropna().values
        test_sample = X_test[feature].dropna().values
        
        # Enforce minimum critical sample threshold constraint for reliable statistical inference
        if len(train_sample) < 10 or len(test_sample) < 10:
            drift_report[feature] = {
                "ks_statistic": None,
                "p_value": None,
                "drift_detected": False,
                "status": "Insufficient Sample Length"
            }
            continue
            
        # Execute the non-parametric statistical check
        ks_stat, p_value = ks_2samp(train_sample, test_sample)
        
        # Determine drift detection flag via significance threshold boundary alpha = 0.05
        drift_detected = bool(p_value < 0.05)
        
        drift_report[feature] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "status": "Success"
        }
        
        if drift_detected:
            print(f"   🚨 [DRIFT DETECTED] Feature: {feature:<28} | KS-Stat: {ks_stat:.3f} | p-value: {p_value:.5f}")
        else:
            print(f"   ✅ [Stable]         Feature: {feature:<28} | KS-Stat: {ks_stat:.3f} | p-value: {p_value:.5f}")

    # Compile final telemetry metadata summary payload
    total_checked = len(drift_report)
    total_drifted = sum(1 for f in drift_report.values() if f.get("drift_detected", False))
    
    summary_report = {
        "metadata": {
            "model_name": model_name,
            "horizon_hours": horizon,
            "total_features_evaluated": total_checked,
            "total_features_drifted": total_drifted,
            "system_status": "CRITICAL_RETRAIN_REQUIRED" if total_drifted > (total_checked * 0.3) else "HEALTHY"
        },
        "feature_metrics": drift_report
    }

    # Persist artifact file to disc path
    output_filename = f"{model_name.lower()}_drift_report_{horizon}h.json"
    output_path = METRICS_DIR / output_filename
    
    with open(output_path, "w") as json_file:
        json.dump(summary_report, json_file, indent=2)
        
    print(f"Drift monitoring log written cleanly to: {output_path}")
    return summary_report