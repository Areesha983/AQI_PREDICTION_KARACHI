"""
build_dataset.py
-----------------
Fetches raw weather + air-quality data for Karachi and safely merges them.
Executes precise missingness profiling across all pollutants and cleanly 
drops unmeasured initial rows to avoid target imputation or synthetic leakage.

Run:
    python build_dataset.py
"""

from pathlib import Path
import pandas as pd

from fetch_weather import fetch_weather
from fetch_air_quality import fetch_air_quality


def main():
    print("Fetching weather data...")
    weather_df = fetch_weather()

    print("\nFetching air quality data...")
    aq_df = fetch_air_quality()

    # 1. Normalize datetime footprints to ensure an accurate structural merge
    if pd.api.types.is_datetime64_any_dtype(weather_df["datetime"]):
        weather_df["datetime"] = weather_df["datetime"].dt.tz_localize(None)
    else:
        weather_df["datetime"] = pd.to_datetime(weather_df["datetime"]).dt.tz_localize(None)

    if pd.api.types.is_datetime64_any_dtype(aq_df["datetime"]):
        aq_df["datetime"] = aq_df["datetime"].dt.tz_localize(None)
    else:
        aq_df["datetime"] = pd.to_datetime(aq_df["datetime"]).dt.tz_localize(None)

    print("\nExecuting synchronized datetime alignment join...")
    dataset = pd.merge(
        weather_df,
        aq_df,
        on="datetime",
        how="inner"
    )
    dataset = dataset.sort_values("datetime").reset_index(drop=True)

    # 2. Precise Missingness Profiling Suite (No blind assumptions)
    print("\n" + "=" * 60)
    print("      GRANULAR POLLUTANT MISSINGNESS PROFILE")
    print("=" * 60)
    
    pollutants = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    for col in pollutants:
        if col in dataset.columns:
            nan_indices = dataset[dataset[col].isna()]
            print(f"\nTarget: {col} | Total NaNs: {len(nan_indices)}")
            if len(nan_indices) > 0:
                print(nan_indices[["datetime", col]].head(10).to_string(index=True))
        else:
            print(f"\nCRITICAL WARNING: Pollutant column '{col}' is missing entirely!")

    # 3. Clean and Defensible Data Pruning
    initial_shape = dataset.shape[0]
    
    print("\nDropping rows with missing PM2.5 target observations...")
    dataset = dataset.dropna(subset=["pm25"]).reset_index(drop=True)
    
    final_shape = dataset.shape[0]
    dropped_rows = initial_shape - final_shape
    dropped_pct = (dropped_rows / initial_shape) * 100

    print(f"Dropped {dropped_rows} rows out of {initial_shape:,} ({dropped_pct:.4f}% of dataset).")

    # 4. Save Clean Dataset Manifest
    output_path = (
        Path(__file__).resolve().parent
        / "data" / "raw" / "karachi_aqi_dataset.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved successfully → {output_path}")
    print(f"Final Dataset Shape : {dataset.shape}")
    print("\nFirst 5 Remaining Rows Preview:")
    print(dataset[["datetime", "temperature", "humidity", "pm25", "co"]].head())


if __name__ == "__main__":
    main()