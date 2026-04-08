from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/model_data.csv")
TABLE_DIR = Path("reports/tables")
FIGURE_DIR = Path("reports/figures")

RANDOM_STATE = 42
SAMPLE_SIZE = 200000


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

    # 和 compare_models.py 保持一致：使用可复现子样本
    sample_size = min(SAMPLE_SIZE, len(df))
    df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    # 切分训练集和测试集
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 区分类别特征和数值特征
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

    # 预处理
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # 最佳模型：XGBoost
    model = Pipeline(
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

    # 训练并预测
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # 记录预测值和绝对误差
    test_df = test_df.copy()
    test_df["prediction"] = preds
    test_df["abs_error"] = (test_df[target_col] - test_df["prediction"]).abs()

    # 创建输出目录
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 整体指标 ----------
    overall_mae = mean_absolute_error(y_test, preds)
    overall_rmse = mean_squared_error(y_test, preds) ** 0.5

    with open(TABLE_DIR / "xgboost_overall_metrics.txt", "w", encoding="utf-8") as f:
        f.write("XGBoost Evaluation\n")
        f.write(f"MAE: {overall_mae:.4f}\n")
        f.write(f"RMSE: {overall_rmse:.4f}\n")

    # ---------- 按小时分析误差 ----------
    hour_error = (
        test_df.groupby("pickup_hour", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
        .sort_values("pickup_hour")
    )
    hour_error.to_csv(TABLE_DIR / "xgboost_mae_by_hour.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(hour_error["pickup_hour"], hour_error["mae"])
    plt.xlabel("Pickup hour")
    plt.ylabel("Mean absolute error")
    plt.title("XGBoost MAE by Pickup Hour")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_mae_by_hour.png")
    plt.close()

    # ---------- 按距离区间分析误差 ----------
    distance_bins = [0, 1, 2, 5, 10, 20, 50, 100]
    distance_labels = ["0-1", "1-2", "2-5", "5-10", "10-20", "20-50", "50-100"]

    test_df["distance_bucket"] = pd.cut(
        test_df["trip_distance"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
    )

    distance_error = (
        test_df.groupby("distance_bucket", as_index=False, observed=True)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
    )
    distance_error.to_csv(TABLE_DIR / "xgboost_mae_by_distance_bucket.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(distance_error["distance_bucket"].astype(str), distance_error["mae"])
    plt.xlabel("Trip distance bucket")
    plt.ylabel("Mean absolute error")
    plt.title("XGBoost MAE by Distance Bucket")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_mae_by_distance_bucket.png")
    plt.close()

    # ---------- 按上车区域 ID 分析误差 ----------
    zone_error = (
        test_df.groupby("PULocationID", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
        .sort_values("mae", ascending=False)
        .head(15)
    )
    zone_error.to_csv(TABLE_DIR / "xgboost_top15_pickup_zone_mae.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(zone_error["PULocationID"].astype(str), zone_error["mae"])
    plt.xlabel("Pickup zone ID")
    plt.ylabel("Mean absolute error")
    plt.title("XGBoost Top 15 Pickup Zone IDs by MAE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_top15_pickup_zone_mae.png")
    plt.close()

    # ---------- 高误差样本 ----------
    high_error_cases = (
        test_df[
            [
                "tpep_pickup_datetime",
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "pickup_hour",
                "trip_duration_min",
                "prediction",
                "abs_error",
            ]
        ]
        .sort_values("abs_error", ascending=False)
        .head(100)
    )
    high_error_cases.to_csv(TABLE_DIR / "xgboost_high_error_cases_top100.csv", index=False)

    print("XGBoost evaluation finished.")
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    print("Saved tables to:", TABLE_DIR)
    print("Saved figures to:", FIGURE_DIR)


if __name__ == "__main__":
    main()