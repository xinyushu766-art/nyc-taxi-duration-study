from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/model_data.csv")
TABLE_DIR = Path("reports/tables")
FIGURE_DIR = Path("reports/figures")

RANDOM_STATE = 42
SAMPLE_SIZE = 200000


def get_feature_family(feature_name: str) -> str:
    """
    把 one-hot 之后的特征名，映射回更容易理解的“原始特征家族”
    例如：
    cat__PULocationID_132 -> PULocationID
    num__trip_distance -> trip_distance
    """
    if feature_name.startswith("cat__PULocationID"):
        return "PULocationID"
    if feature_name.startswith("cat__DOLocationID"):
        return "DOLocationID"
    if feature_name.startswith("cat__RatecodeID"):
        return "RatecodeID"
    if feature_name.startswith("cat__payment_type"):
        return "payment_type"
    if feature_name == "num__trip_distance":
        return "trip_distance"
    if feature_name == "num__pickup_hour":
        return "pickup_hour"
    if feature_name == "num__pickup_dayofweek":
        return "pickup_dayofweek"
    if feature_name == "num__pickup_month":
        return "pickup_month"
    return "other"


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

    # 使用和前面模型比较一致的可复现子样本
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

    # 预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # 和当前最佳模型保持一致的 XGBoost 配置
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

    # 训练模型
    model.fit(X_train, y_train)

    # 取出 one-hot 之后的特征名
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # 取出 XGBoost 的特征重要性
    importances = model.named_steps["regressor"].feature_importances_

    # 保存“展开后的特征重要性”
    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    # 给每个展开后的特征映射回原始特征家族
    importance_df["feature_family"] = importance_df["feature_name"].apply(get_feature_family)

    # 聚合成“原始特征家族重要性”
    family_importance_df = (
        importance_df.groupby("feature_family", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    # 创建输出目录
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 保存完整表
    importance_df.to_csv(TABLE_DIR / "xgboost_feature_importance_full.csv", index=False)
    family_importance_df.to_csv(TABLE_DIR / "xgboost_feature_family_importance.csv", index=False)

    # 保存 top20 展开特征
    top20_df = importance_df.head(20).copy()
    top20_df.to_csv(TABLE_DIR / "xgboost_top20_feature_importance.csv", index=False)

    # 图 1：原始特征家族重要性
    plt.figure(figsize=(8, 5))
    plt.barh(
        family_importance_df["feature_family"][::-1],
        family_importance_df["importance"][::-1]
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature family")
    plt.title("XGBoost Feature Family Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_feature_family_importance.png")
    plt.close()

    # 图 2：Top 20 展开特征重要性
    top20_plot_df = top20_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top20_plot_df["feature_name"], top20_plot_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Expanded feature")
    plt.title("XGBoost Top 20 Expanded Feature Importances")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_top20_feature_importance.png")
    plt.close()

    print("Feature importance analysis finished.")
    print("Saved tables to:", TABLE_DIR)
    print("Saved figures to:", FIGURE_DIR)


if __name__ == "__main__":
    main()