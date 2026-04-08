from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = Path("data/processed/cleaned_taxi_data.csv")
FIGURE_DIR = Path("reports/figures")


def main():
    # 读取清洗后的数据
    df = pd.read_csv(INPUT_PATH)

    # 把上车时间转换成真正的时间格式
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    # 创建图表输出文件夹
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 图 1：行程时长分布 ----------
    plt.figure(figsize=(8, 5))
    plt.hist(df["trip_duration_min"], bins=50)
    plt.xlabel("Trip duration (minutes)")
    plt.ylabel("Count")
    plt.title("Distribution of Trip Duration")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "duration_distribution.png")
    plt.close()

    # ---------- 图 2：距离 vs 时长 ----------
    sample_df = df.sample(n=5000, random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(sample_df["trip_distance"], sample_df["trip_duration_min"], alpha=0.4)
    plt.xlabel("Trip distance")
    plt.ylabel("Trip duration (minutes)")
    plt.title("Trip Distance vs Trip Duration")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "distance_vs_duration.png")
    plt.close()

    # ---------- 图 3：不同小时的平均时长 ----------
    hourly_avg = (
        df.groupby(df["tpep_pickup_datetime"].dt.hour)["trip_duration_min"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(hourly_avg["tpep_pickup_datetime"], hourly_avg["trip_duration_min"])
    plt.xlabel("Pickup hour")
    plt.ylabel("Average trip duration (minutes)")
    plt.title("Average Trip Duration by Pickup Hour")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "avg_duration_by_hour.png")
    plt.close()

    print("EDA figures saved to:", FIGURE_DIR)
    print("Finished generating 3 figures.")


if __name__ == "__main__":
    main()