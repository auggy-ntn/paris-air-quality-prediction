import requests
import pandas as pd
from datetime import datetime

def get_paris_weather(start_date: str, end_date: str, output_csv: str = "paris_weather.csv"):
    """
    Fetch hourly weather data for Paris from Open-Meteo API between start_date and end_date,
    and save it to a CSV file.
    """

    # Validate date format
    for date_str in [start_date, end_date]:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format for {date_str}. Use YYYY-MM-DD.")

    # Paris coordinates
    latitude = 48.8566
    longitude = 2.3522

    # Define the API endpoint and parameters
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "windspeed_10m"
        ],
        "timezone": "Europe/Paris"
    }

    print(f"Fetching data from {start_date} to {end_date} for Paris...")

    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error for bad responses

    data = response.json()

    # Convert to DataFrame
    hourly_data = data.get("hourly", {})
    if not hourly_data:
        raise ValueError("No hourly data returned from API.")

    df = pd.DataFrame(hourly_data)
    df["time"] = pd.to_datetime(df["time"])

    # Save to CSV
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    start = "2019-01-01"
    end = "2024-09-03"
    get_paris_weather(start, end, output_csv="./data/paris_weather.csv")
