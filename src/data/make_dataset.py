from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/yellow_tripdata_sample.parquet")
PROCESSED_PATH = Path("data/processed/cleaned_taxi_data.csv")


def main():
    df = pd.read_parquet(RAW_PATH)

    # Keep only columns needed for the first version
    keep_cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "payment_type",
    ]

    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].copy()

    # Drop rows with missing key fields
    required_cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
    ]
    required_existing = [col for col in required_cols if col in df.columns]
    df = df.dropna(subset=required_existing)

    # Build target variable: trip duration in minutes
    df["trip_duration_min"] = (
        pd.to_datetime(df["tpep_dropoff_datetime"])
        - pd.to_datetime(df["tpep_pickup_datetime"])
    ).dt.total_seconds() / 60

    # Basic filtering
    df = df[df["trip_duration_min"] > 0]
    df = df[df["trip_distance"] > 0]

    # Remove extreme outliers for the first study version
    df = df[df["trip_duration_min"].between(1, 180)]
    df = df[df["trip_distance"].between(0.1, 100)]

    # Save cleaned data
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Saved cleaned data to:", PROCESSED_PATH)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()
