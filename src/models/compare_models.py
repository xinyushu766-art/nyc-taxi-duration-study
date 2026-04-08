from pathlib import Path
import time
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/model_data.csv")
OUTPUT_PATH = Path("reports/tables/model_comparison.csv")

RANDOM_STATE = 42
SAMPLE_SIZE = 200000


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start_time = time.perf_counter()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    end_time = time.perf_counter()

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    train_seconds = end_time - start_time

    return {
        "model": name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "train_seconds": round(train_seconds, 2),
    }


def main():
    # 读取模型数据
    df = pd.read_csv(DATA_PATH)

    # 定义特征列和目标列
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
    target_col = "trip_duration_min"

    # 删除缺失值
    df = df.dropna(subset=feature_cols + [target_col])

    # 为了让本地机器更稳地完成模型比较，先做一个可复现样本
    sample_size = min(SAMPLE_SIZE, len(df))
    df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    # 切分训练集和测试集
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # 指定类别特征和数值特征
    categorical_cols = [
        "PULocationID",
        "DOLocationID",
        "RatecodeID",
        "payment_type",
    ]
    numeric_cols = [
        "trip_distance",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
    ]

    # 预处理器：类别特征 one-hot，数值特征原样保留
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # 定义三个模型
    linear_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=16,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    xgb_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 统一评估
    results = []
    results.append(
        evaluate_model(
            "LinearRegression",
            linear_model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )
    results.append(
        evaluate_model(
            "RandomForest",
            rf_model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )
    results.append(
        evaluate_model(
            "XGBoost",
            xgb_model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    # 保存结果
    result_df = pd.DataFrame(results).sort_values("rmse")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("Model comparison finished.")
    print(result_df)
    print("Saved comparison table to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()