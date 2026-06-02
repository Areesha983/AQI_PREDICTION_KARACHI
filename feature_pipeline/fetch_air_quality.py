import requests
import pandas as pd
from pathlib import Path

from config import (
    LATITUDE,
    LONGITUDE,
    START_DATE,
    END_DATE
)


def fetch_air_quality() -> pd.DataFrame:

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        "&hourly="
        "pm2_5,"
        "pm10,"
        "carbon_monoxide,"
        "nitrogen_dioxide,"
        "sulphur_dioxide,"
        "ozone"
        "&timezone=auto"
    )

    print("\n" + "=" * 80)
    print("AIR QUALITY REQUEST URL")
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

    aq_df = pd.DataFrame({
        "datetime": hourly["time"],
        "pm25": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "co": hourly["carbon_monoxide"],
        "no2": hourly["nitrogen_dioxide"],
        "so2": hourly["sulphur_dioxide"],
        "o3": hourly["ozone"]
    })

    aq_df["datetime"] = pd.to_datetime(aq_df["datetime"])

    aq_df = (
        aq_df
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"\nRows Retrieved: {len(aq_df):,}")

    print("\nPM2.5 Summary")
    print(aq_df["pm25"].describe())

    return aq_df


if __name__ == "__main__":

    aq_df = fetch_air_quality()

    output_path = (
        Path(__file__).resolve().parent
        / "data"
        / "raw"
        / "air_quality_karachi.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    aq_df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")