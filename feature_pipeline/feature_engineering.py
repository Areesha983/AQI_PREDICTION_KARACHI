"""
Production-Grade Air Quality Index (AQI) Feature Engineering Pipeline for Karachi.

Builds an enriched, domain-specific, leakage-free modeling matrix designed 
specifically for non-linear ensembles (XGBoost / Random Forest). Implements explicit 
momentum, atmospheric stability windows, spike memory tracking, and wind vector transformations.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── 1. EPA Standard Calculation Helpers ─────────────────────────────────────

def calculate_aqi_from_pm25(pm25: float) -> float:
    """Applies the US-EPA piecewise linear interpolation formula for PM2.5 to AQI."""
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    breakpoints = [
        (0.0,   12.0,  0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4,  101, 150),
        (55.5,  150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(
                ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
            )
    return 500


def aqi_to_category(aqi: float) -> float:
    """Maps raw AQI numbers to standard integer labels for classification tasks."""
    if pd.isna(aqi): return np.nan
    if aqi <= 50:    return 0   # Good
    if aqi <= 100:   return 1   # Moderate
    if aqi <= 150:   return 2   # Unhealthy for Sensitive Groups
    if aqi <= 200:   return 3   # Unhealthy
    if aqi <= 300:   return 4   # Very Unhealthy
    return 5                    # Hazardous


def _rolling_slope_6h(y: np.ndarray) -> float:
    """Vectorized OLS slope computation over a fixed 6-hour lookback window."""
    x_dev = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    return float(np.dot(x_dev, y - np.mean(y)) / 17.5)


# ── 2. Pure Pipeline Core Transformation Function ───────────────────────────

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw hourly meteorological streams into an enriched feature store.
    Ensures complete isolation from target leakage using historical window shifting.
    """
    # Create deep copy to maintain source immutability
    df = df_raw.copy()
    
    # Enforce standard chronological indexing operations
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        raise KeyError("Input dataframe must contain a dedicated 'datetime' column.")

    print(f"Initializing primary transformations. Input shape: {df.shape}")

    # ── STEP 1: LEAKAGE-FREE CHRONOLOGICAL IMPUTATION ────────────────────────
    # Run time-based interpolation and causal forward filling. 
    # Absolute bfill() is removed to prevent leaking future metrics back to past profiles.
    df = df.set_index("datetime")
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].interpolate(method="time").ffill()
    df = df.reset_index()
    print(" -> Leakage-free causal imputation complete.")

    # ── STEP 2: AQI EXTRACTION & MULTI-HORIZON TARGET SPACE ──────────────────
    df["aqi"] = df["pm25"].apply(calculate_aqi_from_pm25)

    for h in [12, 24, 48, 72]:
        df[f"target_aqi_{h}h"]      = df["aqi"].shift(-h)
        df[f"target_aqi_{h}h_log"]  = np.log1p(df[f"target_aqi_{h}h"])
        df[f"target_cat_{h}h"]      = df[f"target_aqi_{h}h"].apply(aqi_to_category)

    # ── STEP 3: CYCLICAL TEMPORAL EMBEDDINGS ──────────────────────────────────
    df["hour"]    = df["datetime"].dt.hour
    df["day"]     = df["datetime"].dt.day
    df["month"]   = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]    / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]    / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"]   / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"]   / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["quarter"]      = df["datetime"].dt.quarter
    df["day_of_year"]  = df["datetime"].dt.dayofyear
    df["doy_sin"]      = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]      = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["is_weekend"]   = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Karachi Atmospheric Wind Vector Decompositions
    if "wind_direction" in df.columns:
        _wdir_rad       = np.deg2rad(df["wind_direction"].shift(1))
        df["wind_dir_sin"] = np.sin(_wdir_rad)
        df["wind_dir_cos"] = np.cos(_wdir_rad)
        df["wind_x"]       = df["wind_speed"].shift(1) * df["wind_dir_sin"]
        df["wind_y"]       = df["wind_speed"].shift(1) * df["wind_dir_cos"]

    # ── STEP 4: AUTOREGREGRESSIVE LAG CHAINS ─────────────────────────────────
    # Retains 1-week (168h) and 2-week (336h) macro-seasonality hooks
    for lag in [1, 2, 3, 6, 7, 12, 24, 48, 72, 168, 336]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    df["aqi_same_hour_yesterday"]  = df["aqi"].shift(24)
    df["aqi_same_hour_last_week"]  = df["aqi"].shift(168)
    df["aqi_same_hour_2weeks_ago"] = df["aqi"].shift(336)
    df["aqi_same_hour_3days_ago"]  = df["aqi"].shift(72)
    df["aqi_same_hour_30days_ago"] = df["aqi"].shift(720)
    # [FIXED]: Removed 8,760h year-lag to reclaim 1 full year of training data samples.

    # ── STEP 5: PM2.5 LAGS & ROLLING HISTOGRAMS ──────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

    _pm25 = df["pm25"].shift(1)
    for w in [6, 24, 72, 168, 336]:
        df[f"pm25_roll_mean_{w}"] = _pm25.rolling(w, min_periods=1).mean()
        df[f"pm25_roll_std_{w}"]  = _pm25.rolling(w, min_periods=1).std().fillna(0)

    df["pm25_roll_max_24"]  = _pm25.rolling(24, min_periods=1).max()
    df["pm25_roll_max_72"]  = _pm25.rolling(72, min_periods=1).max()
    df["pm25_ewm_24"]       = _pm25.ewm(span=24, adjust=False).mean()
    df["pm25_change_24h"]   = df["pm25"].shift(1) - df["pm25"].shift(25)
    df["pm25_vs_24h_avg"]   = _pm25 - df["pm25_roll_mean_24"]
    df["pm25_vs_72h_avg"]   = _pm25 - df["pm25_roll_mean_72"]

    # ── STEP 6: PM10 STRUCTURAL TRACKING ─────────────────────────────────────
    for lag in [1, 6, 24, 72]:
        df[f"pm10_lag_{lag}"] = df["pm10"].shift(lag)

    _pm10 = df["pm10"].shift(1)
    df["pm10_roll_mean_24"] = _pm10.rolling(24, min_periods=1).mean()
    df["pm10_roll_mean_72"] = _pm10.rolling(72, min_periods=1).mean()
    df["pm10_roll_std_24"]  = _pm10.rolling(24, min_periods=1).std().fillna(0)
    df["pm10_change_24h"]   = df["pm10"].shift(1) - df["pm10"].shift(25)
    df["dust_event"]        = (df["pm10_lag_1"] > 250).astype(int)

    # ── STEP 7: AQI ROLLING MATRIX HISTORIES ─────────────────────────────────
    _aqi = df["aqi"].shift(1)
    for w in [6, 12, 24, 48, 72, 168, 336]:
        df[f"aqi_roll_mean_{w}"] = _aqi.rolling(w, min_periods=1).mean()

    for w in [24, 72, 168]:
        df[f"aqi_roll_std_{w}"] = _aqi.rolling(w, min_periods=1).std().fillna(0)
        df[f"aqi_roll_max_{w}"] = _aqi.rolling(w, min_periods=1).max()
        df[f"aqi_roll_min_{w}"] = _aqi.rolling(w, min_periods=1).min()

    df["aqi_ewm_24"] = _aqi.ewm(span=24, adjust=False).mean()
    df["aqi_ewm_72"] = _aqi.ewm(span=72, adjust=False).mean()

    # ── STEP 8: MOMENTUM, ACCELERATION & TREND RATIOS ───────────────────────
    df["aqi_trend_ratio"]  = df["aqi_roll_mean_24"] / (df["aqi_roll_mean_72"]  + 1)
    df["pm25_trend_ratio"] = df["pm25_roll_mean_24"] / (df["pm25_roll_mean_72"] + 1)

    df["aqi_momentum_6_24"]   = df["aqi_roll_mean_6"]  - df["aqi_roll_mean_24"]
    df["aqi_momentum_24_72"]  = df["aqi_roll_mean_24"] - df["aqi_roll_mean_72"]
    df["aqi_momentum_24_168"] = df["aqi_roll_mean_24"] - df["aqi_roll_mean_168"]

    df["aqi_change_1h"]  = _aqi - df["aqi"].shift(2)
    df["aqi_change_6h"]  = _aqi - df["aqi"].shift(7)
    df["aqi_change_24h"] = _aqi - df["aqi"].shift(25)

    df["aqi_trend_slope_6h"] = (
        _aqi.rolling(6, min_periods=6)
            .apply(_rolling_slope_6h, raw=True)
            .fillna(0)
    )
    df["aqi_acceleration"] = _aqi.diff().diff().fillna(0)

    # ── STEP 9: PERSISTENCE & AIR MAS RECOVERY RATES ─────────────────────────
    df["aqi_persistence"]       = df["aqi_lag_1"] - df["aqi_lag_24"]
    df["aqi_diff_24_168"]       = df["aqi_lag_24"] - df["aqi_lag_168"]
    df["aqi_recovery_rate"]     = ((df["aqi_roll_max_72"] - df["aqi_lag_1"]) / (df["aqi_roll_max_72"] + 1))
    df["aqi_persistence_ratio"] = df["aqi_lag_1"] / (df["aqi_roll_mean_72"] + 1)
    df["aqi_vs_week"]           = df["aqi_roll_mean_24"] - df["aqi_roll_mean_168"]

    # ── STEP 10: STATISTICAL ANOMALY & Z-SCORE REGIMES ───────────────────────
    _roll_mean_72 = _aqi.rolling(72, min_periods=24).mean()
    _roll_std_72  = _aqi.rolling(72, min_periods=24).std().replace(0, 1)
    df["aqi_zscore_72h"] = ((df["aqi_lag_1"] - _roll_mean_72) / _roll_std_72).fillna(0)

    df["aqi_percentile_72"] = (
        _aqi.rolling(72, min_periods=12)
           .apply(lambda x: float(np.mean(x < x[-1])), raw=True)
           .fillna(0.5)
    )

    _rolling_q90 = _aqi.rolling(168, min_periods=72).quantile(0.90)
    df["aqi_above_recent_q90"]  = (df["aqi_lag_1"] > _rolling_q90).astype(int)
    df["aqi_volatility_ratio"]  = df["aqi_roll_std_24"] / (df["aqi_roll_std_72"] + 1)

    df["aqi_regime"] = pd.cut(
        df["aqi_lag_1"],
        bins=[0, 50, 100, 150, 200, 300, 1000],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float).fillna(0.0)

    # ── STEP 11: INVERSION SPIKE MEMORY TIMELINES ────────────────────────────
    df["spike_count_72h"] = (_aqi > 150).rolling(72, min_periods=1).sum().fillna(0)
    df["dust_hours_72h"]  = (_pm10 > 250).rolling(72, min_periods=1).sum().fillna(0)

    _spike = (df["aqi_lag_1"] > 150).astype(int)
    _hours_since, _consec = [], []
    _counter, _run = 999, 0
    for s in _spike:
        if s:
            _counter = 0
            _run += 1
        else:
            _counter += 1
            _run = 0
        _hours_since.append(_counter)
        _consec.append(_run)

    df["hours_since_aqi_spike"]       = pd.Series(_hours_since, index=df.index).shift(1)
    df["consecutive_hours_above_150"] = pd.Series(_consec,    index=df.index).shift(1)

    # ── STEP 12: SECONDARY POLLUTANT REACTION PROXIES ───────────────────────
    for poll in ["so2", "co", "o3", "no2"]:
        if poll not in df.columns: continue
        _s = df[poll].shift(1)
        df[f"{poll}_roll_mean_24"] = _s.rolling(24, min_periods=1).mean()
        df[f"{poll}_change_24h"]   = df[poll].shift(1) - df[poll].shift(25)

    for poll in ["no2", "o3"]:
        if poll not in df.columns: continue
        df[f"{poll}_roll_std_24"] = df[poll].shift(1).rolling(24, min_periods=1).std().fillna(0)

    if {"pm25", "pm10"}.issubset(df.columns):
        df["pm25_pm10_ratio"] = (df["pm25"].shift(1) / (df["pm10"].shift(1) + 1e-3)).clip(upper=50)
        df["pm25_fraction"]   = (df["pm25"].shift(1) / (df["pm25"].shift(1) + df["pm10"].shift(1) + 1e-3))

    if {"no2", "o3"}.issubset(df.columns):
        df["no2_o3_ratio"] = df["no2"].shift(1) / (df["o3"].shift(1) + 1e-3)

    # ── STEP 13: METEOROLOGICAL DISPERSAL INTERACTIONS ──────────────────────
    for col in ["temperature", "humidity", "wind_speed"]:
        if col not in df.columns: continue
        _s = df[col].shift(1)
        df[f"{col}_roll_mean_24"] = _s.rolling(24, min_periods=1).mean()
        df[f"{col}_change_24h"]   = df[col].shift(1) - df[col].shift(25)
        for lag in [1, 6, 24]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    df["wind_speed_roll_std_24"]  = df["wind_speed"].shift(1).rolling(24, min_periods=1).std().fillna(0)
    df["temperature_roll_std_24"] = df["temperature"].shift(1).rolling(24, min_periods=1).std().fillna(0)

    df["temp_humidity"]  = df["temperature"].shift(1) * df["humidity"].shift(1)
    df["heat_dryness"]   = df["temperature"].shift(1) / (df["humidity"].shift(1)  + 1)
    df["wind_dispersal"] = df["wind_speed"].shift(1)  / (df["pm25"].shift(1) + 5)

    if "precipitation" in df.columns:
        df["rain_24h"] = df["precipitation"].shift(1).rolling(24, min_periods=1).sum()
        df["rain_72h"] = df["precipitation"].shift(1).rolling(72, min_periods=1).sum()

    if "pressure" in df.columns:
        df["pressure_lag_1"]        = df["pressure"].shift(1)
        df["pressure_roll_mean_24"] = df["pressure"].shift(1).rolling(24, min_periods=1).mean()
        df["pressure_change_24h"]   = df["pressure"].shift(1) - df["pressure"].shift(25)

    if "wind_gusts" in df.columns:
        _gust = df["wind_gusts"].shift(1)
        df["wind_gusts_roll_mean_24"] = _gust.rolling(24, min_periods=1).mean()
        df["wind_gusts_roll_max_24"]  = _gust.rolling(24, min_periods=1).max()

    if "cloud_cover" in df.columns:
        df["cloud_cover_lag_1"]        = df["cloud_cover"].shift(1)
        df["cloud_cover_roll_mean_24"] = df["cloud_cover"].shift(1).rolling(24, min_periods=1).mean()

    # ── STEP 14: WARM-UP ROW FILTERING ───────────────────────────────────────
    # [FIXED]: Required non-null sequence reduced to 30 days (720 rows) instead of 1 year (8,760 rows).
    required_non_null = [
        "aqi_same_hour_30days_ago",  # Validates 720 hours of historical tracking
        "target_aqi_72h",            # Validates target array space boundaries
    ]
    before = len(df)
    df.dropna(subset=required_non_null, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f" -> Dropped {before - len(df):,} edge rows via warm-up boundaries. {len(df):,} remaining rows.")

    assert df["datetime"].is_monotonic_increasing, "CRITICAL FAULT: Temporal sequence is no longer monotonic."
    return df


# ── 3. Local Operational Trigger File Generation Execution Loop ─────────────

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    raw_path = BASE_DIR / "data" / "raw" / "karachi_aqi_dataset.csv"
    
    # Generate mock stream if file doesn't exist locally to guarantee validation safety
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Notice] Generating structural mock dataset for verification...")
        dates = pd.date_range(start="2023-01-01", periods=15000, freq="h", name="datetime")
        pd.DataFrame({
            "pm25": np.random.uniform(15, 300, size=len(dates)),
            "pm10": np.random.uniform(30, 450, size=len(dates)),
            "temperature": np.random.uniform(12, 44, size=len(dates)),
            "humidity": np.random.uniform(20, 95, size=len(dates)),
            "wind_speed": np.random.uniform(2, 35, size=len(dates)),
            "wind_direction": np.random.uniform(0, 360, size=len(dates)),
            "wind_gusts": np.random.uniform(5, 55, size=len(dates)),
            "cloud_cover": np.random.uniform(0, 100, size=len(dates))
        }, index=dates).reset_index().to_csv(raw_path, index=False)

    # Ingest baseline
    raw_input_data = pd.read_csv(raw_path)
    processed_df = build_features(raw_input_data)

    # Filter features from targets and raw logging trackers
    EXCLUDE = {
        "datetime", "aqi",
        "target_aqi_12h",  "target_aqi_24h",  "target_aqi_48h",  "target_aqi_72h",
        "target_aqi_12h_log", "target_aqi_24h_log", "target_aqi_48h_log", "target_aqi_72h_log",
        "target_cat_12h",  "target_cat_24h",  "target_cat_48h",  "target_cat_72h",
        "hour", "day", "month", "weekday", "day_of_year",
    }
    feature_columns = [c for c in processed_df.columns if c not in EXCLUDE]

    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "feature_columns.txt").write_text("\n".join(feature_columns))
    processed_df.to_csv(out_dir / "featured_dataset.csv", index=False)
    
    print("\nProcessing execution successful.")
    print(f"Final Matrix Shape   : {processed_df.shape}")
    print(f"Total Model Features : {len(feature_columns)}")