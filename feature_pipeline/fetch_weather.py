import requests
import pandas as pd
from pathlib import Path

from config import (
    LATITUDE,
    LONGITUDE,
    START_DATE,
    END_DATE
)


def fetch_weather():

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        "&hourly="
        "temperature_2m,"
        "relative_humidity_2m,"
        "pressure_msl,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "wind_gusts_10m,"
        "precipitation,"
        "cloud_cover"
        "&timezone=auto"
    )

    print("\n" + "=" * 80)
    print("WEATHER REQUEST URL")
    print("=" * 80)
    print(url)

    response = requests.get(url, timeout=60)

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    data = response.json()

    hourly = data.get("hourly")

    if hourly is None:
        raise ValueError(
            f"'hourly' section missing.\nResponse:\n{data}"
        )

    weather_df = pd.DataFrame({
        "datetime": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["pressure_msl"],
        "wind_speed": hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
        "wind_gusts": hourly["wind_gusts_10m"],
        "precipitation": hourly["precipitation"],
        "cloud_cover": hourly["cloud_cover"]
    })

    weather_df["datetime"] = pd.to_datetime(
        weather_df["datetime"]
    )

    weather_df = (
        weather_df
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"\nRows Retrieved: {len(weather_df):,}")

    print("\nWeather Summary")
    print(weather_df.describe())

    return weather_df


if __name__ == "__main__":

    weather_df = fetch_weather()

    output_path = (
        Path(__file__).resolve().parent
        / "data"
        / "raw"
        / "weather_karachi.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    weather_df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")