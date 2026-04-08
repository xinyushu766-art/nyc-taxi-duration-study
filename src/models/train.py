from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/processed/model_data.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    feature_cols = [
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "RatecodeID",
        "payment_type",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]
    target_col = "trip_duration_min"

    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols]
    y = df[target_col]

    categorical_cols = [
        col for col in ["PULocationID", "DOLocationID", "RatecodeID", "payment_type"]
        if col in X.columns
    ]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print("Baseline Linear Regression Results")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
