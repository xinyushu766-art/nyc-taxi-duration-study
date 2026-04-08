from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/processed/cleaned_taxi_data.csv")
OUTPUT_PATH = Path("data/processed/model_data.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved feature dataset to:", OUTPUT_PATH)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()
