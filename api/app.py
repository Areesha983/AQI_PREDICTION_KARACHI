"""
Enterprise MLOps Prediction API Service Layer.
Exposes real-time endpoints for 24h, 48h, and 72h AQI forecasting.
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd

# ── SYSTEM CONSTANTS & LOCAL PATH LAYOUT ───────────────────────────────────
API_DIR = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent  # Points to KARACHI_AQI_PREDICTOR/

MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"

app = Flask(__name__)

# Cache models in memory to ensure sub-10ms response times
MODEL_CACHE = {}


def load_prediction_artifacts(horizon: int):
    """Retrieves serialized model instances and conformal parameters from disk."""
    if horizon in MODEL_CACHE:
        return MODEL_CACHE[horizon]

    model_path = MODELS_DIR / f"random_forest_{horizon}h.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact asset missing on server disk: {model_path.name}")

    artifacts = joblib.load(model_path)
    MODEL_CACHE[horizon] = artifacts
    return artifacts


@app.route("/health", methods=["GET"])
def health_check():
    """System liveness and readiness probe for cloud monitoring gateways."""
    available_horizons = []
    for h in [24, 48, 72]:
        if (MODELS_DIR / f"random_forest_{h}h.pkl").exists():
            available_horizons.append(f"{h}h")

    return jsonify({
        "status": "healthy",
        "service": "aqi-forecasting-api",
        "warmed_models_in_cache": list(MODEL_CACHE.keys()),
        "discovered_artifacts_on_disk": available_horizons
    }), 200


@app.route("/predict/<int:horizon>", methods=["POST"])
def predict_aqi(horizon: int):
    """
    Main inference routing portal.
    Accepts a JSON feature vector array payload and outputs bounded forecasts.
    """
    if horizon not in (24, 48, 72):
        return jsonify({"error": "Invalid forecast horizon request route. Choose 24, 48, or 72 hours."}), 400

    payload = request.get_json(silent=True)
    if not payload or "features" not in payload:
        return jsonify({"error": "Malformed request. JSON payload must contain a valid 'features' key dictionary."}), 400

    try:
        # Load cached model architecture
        artifacts = load_prediction_artifacts(horizon)
        model = artifacts["model"]
        expected_features = artifacts["feature_names"]
        conformal_margin = artifacts["conformal_margin"]

        # Convert incoming JSON vector array directly into a Pandas DataFrame
        input_data = pd.DataFrame([payload["features"]])

        # Schema Alignment: Fill missing columns or align column orders
        for col in expected_features:
            if col not in input_data.columns:
                # Use a zero-fill fallback strategy for missing features at inference time
                input_data[col] = 0.0

        # Re-index to match the exact training column matrix layout
        input_data = input_data[expected_features]

        # Execute model inference
        raw_prediction = float(model.predict(input_data)[0])

        # Apply Conformal Calibration Safety Boundaries
        lower_bound = max(0.0, raw_prediction - conformal_margin)
        upper_bound = min(500.0, raw_prediction + conformal_margin)  # Cap at standard EPA AQI max index limit

        # Categorize the prediction using standard EPA AQI breakpoints
        if raw_prediction <= 50:
            category = "Good"
        elif raw_prediction <= 100:
            category = "Moderate"
        elif raw_prediction <= 150:
            category = "Unhealthy for Sensitive Groups"
        elif raw_prediction <= 200:
            category = "Unhealthy"
        elif raw_prediction <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"

        return jsonify({
            "horizon_hours": horizon,
            "aqi_prediction": round(raw_prediction, 1),
            "lower_bound_95ci": round(lower_bound, 1),
            "upper_bound_95ci": round(upper_bound, 1),
            "conformal_margin_applied": round(conformal_margin, 1),
            "epa_category": category,
            "status": "success"
        }), 200

    except FileNotFoundError as fnf:
        return jsonify({"error": str(fnf), "hint": "Run training pipeline to generate model weights before calling inference endpoints."}), 500
    except Exception as e:
        return jsonify({"error": f"Internal inference pipeline failure: {str(e)}"}), 500


if __name__ == "__main__":
    # Internal dev-server bindings loop
    port = int(os.environ.get("PORT", 5000))
    print(f"\nLaunching Flask Inference Engine Server Instance on Port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)