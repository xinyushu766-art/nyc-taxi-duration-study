# NYC Taxi Duration Study
### Interpretable Trip Duration Prediction and Error Analysis on Official NYC TLC Taxi Data

A research-style data science project that studies whether NYC taxi trip duration can be predicted using realistic trip-level structured features, and analyzes where prediction errors occur across time, distance, and spatial zones.

---

## Overview

This repository investigates taxi trip-duration prediction using one month of official NYC Taxi & Limousine Commission (TLC) Yellow Taxi trip records.

Rather than treating the task as a simple regression exercise, this project is structured as a small research-style repository with:

- official public data
- reproducible preprocessing
- baseline and strong-model benchmarking
- grouped error analysis
- feature-importance interpretation
- explicit discussion of limitations and next steps

The goal is not only to obtain a prediction score, but also to understand **which models work better**, **under what conditions prediction fails**, and **which feature groups drive model behavior**.

---

## Research Questions

This project is organized around four core questions:

1. **How well can taxi trip duration be predicted using structured trip-level variables?**
2. **How do linear, tree-based, and boosted-tree models compare on this task?**
3. **In which pickup hours, trip-distance ranges, and pickup zones do prediction errors become larger?**
4. **Which feature groups contribute most to model performance?**

---

## Why This Project

This project is designed as a compact but research-oriented tabular data science study.

Compared with common beginner datasets such as Titanic, churn prediction, or house prices, the NYC TLC trip dataset is more realistic and better suited for demonstrating:

- data cleaning on a real public dataset
- target construction from timestamps
- structured feature engineering
- rigorous model comparison
- grouped error analysis
- interpretability-oriented reporting

It also has a natural urban-data storytelling angle: trip duration is influenced by **distance, spatial heterogeneity, and time-of-day traffic dynamics**, which makes the task more analytically interesting than a generic classroom regression problem.

---

## Data Source

**Source:** Official NYC Taxi & Limousine Commission (TLC) Trip Record Data  
**Current scope:** one month of Yellow Taxi trip data for a reproducible first-stage version

The current version uses raw TLC trip records and constructs the target variable as:

- **Trip duration (minutes)**  
  = dropoff timestamp − pickup timestamp

### Core variables used in the current version

- `tpep_pickup_datetime`
- `tpep_dropoff_datetime`
- `PULocationID`
- `DOLocationID`
- `trip_distance`
- `RatecodeID`
- `payment_type`
- `pickup_hour`
- `pickup_dayofweek`
- `pickup_month`

---

## Repository Structure

```text
nyc-taxi-duration-study/
├── README.md
├── CITATION.cff
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── analysis/
├── reports/
│   ├── figures/
│   └── tables/
```

### Main scripts

- `src/data/make_dataset.py`  
  Cleans raw TLC data and constructs trip duration.

- `src/features/build_features.py`  
  Creates model-ready features such as pickup hour and pickup day of week.

- `src/models/train.py`  
  Trains the first baseline model (Linear Regression).

- `src/models/compare_models.py`  
  Compares Linear Regression, Random Forest, and XGBoost on a reproducible sampled dataset.

- `src/models/evaluate.py`  
  Runs grouped error analysis using the best-performing model.

- `src/analysis/eda.py`  
  Generates exploratory plots.

- `src/analysis/feature_importance.py`  
  Produces XGBoost feature-importance outputs at both expanded-feature and feature-family levels.

---

## Methodology

### 1. Data Cleaning
The preprocessing stage removes invalid or noisy records to create a stable first-stage benchmark dataset.

Current filtering rules include:

- remove records with missing key variables
- remove non-positive trip durations
- remove non-positive trip distances
- keep only trips with duration between 1 and 180 minutes
- keep only trips with distance between 0.1 and 100

### 2. Feature Engineering
The current version uses lightweight but realistic structured features:

- spatial identifiers (`PULocationID`, `DOLocationID`)
- trip distance
- rate code
- payment type
- pickup hour
- pickup day of week
- pickup month

### 3. Model Benchmarking
Three models are currently compared under a unified evaluation setup:

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

To ensure reproducibility and reasonable local runtime, model comparison is currently conducted on a fixed random sample of the processed dataset.

### 4. Error Analysis
Model performance is analyzed beyond a single score. The repository includes grouped error analysis by:

- pickup hour
- trip-distance bucket
- pickup zone ID

High-error cases are also exported for qualitative inspection.

### 5. Interpretability
The project includes XGBoost feature-importance analysis at two levels:

- **expanded one-hot feature importance**
- **aggregated feature-family importance**

This helps distinguish between local category-specific effects and broader feature-group influence.

---

## Model Comparison Results

Current benchmark results on the reproducible sampled dataset:

| Model | MAE | RMSE | Train Time (s) |
|---|---:|---:|---:|
| XGBoost | 3.7965 | 6.4358 | 2.59 |
| RandomForest | 3.9557 | 6.6480 | 29.02 |
| LinearRegression | 5.1454 | 7.9691 | 1.03 |

### Interpretation
- **XGBoost achieved the best overall performance** in both MAE and RMSE.
- **Random Forest also improved substantially over the linear baseline**, suggesting strong non-linear structure in the data.
- **Linear Regression underperformed the tree-based methods**, indicating that trip duration cannot be adequately modeled as a simple linear function of the current structured variables.
- Under the current setup, **XGBoost was not only more accurate than Random Forest but also much faster**, making it the strongest model in the present benchmark.

---

## Exploratory Data Analysis

The current repository includes three core EDA figures:

1. **Distribution of trip duration**
2. **Trip distance vs trip duration**
3. **Average trip duration by pickup hour**

These plots help establish several useful intuitions:

- trip duration is right-skewed
- longer trips generally take longer, but with large variance
- travel-time dynamics vary substantially across hours of the day

---

## Error Analysis

Error analysis is conducted using the current best model (**XGBoost**).

The repository exports grouped MAE summaries and figures for:

- **pickup hour**
- **distance bucket**
- **pickup zone**
- **top high-error cases**

This makes it possible to move from “what score did the model get?” to more meaningful questions such as:

- When is prediction harder?
- Are short trips or long trips more error-prone?
- Which pickup zones remain difficult to model?
- Do some failure cases reflect missing contextual information?

This is one of the most research-oriented parts of the project.

---

## Feature Importance

The current XGBoost feature-family importance results are:

| Feature Family | Importance |
|---|---:|
| DOLocationID | 0.384671 |
| PULocationID | 0.317879 |
| RatecodeID | 0.246348 |
| trip_distance | 0.040639 |
| pickup_hour | 0.004193 |
| payment_type | 0.003419 |
| pickup_dayofweek | 0.002849 |
| pickup_month | 0.000000 |

### Interpretation
Several observations stand out:

1. **Spatial information dominates the model.**  
   Dropoff and pickup zone IDs together account for most of the importance mass, suggesting that trip duration depends heavily on origin–destination structure.

2. **Rate code information is more important than expected.**  
   This may reflect route type or trip-pattern differences that correlate with duration.

3. **Trip distance matters, but is not the whole story.**  
   Although distance is clearly important, it is less dominant than the spatial identifiers in the current model.

4. **Time-only features play a relatively smaller role in the current setup.**  
   Pickup hour has some contribution, but weekday and month are much weaker.

5. **The current feature space likely captures route heterogeneity better than temporal traffic complexity.**  
   This suggests there may still be room to improve performance with richer temporal or contextual features.

### Top expanded features
Among expanded one-hot features, the most influential ones include:

- `RatecodeID_1.0`
- `RatecodeID_2.0`
- `RatecodeID_99.0`
- `trip_distance`
- specific pickup and dropoff zone IDs such as `PULocationID_138`, `PULocationID_132`, `DOLocationID_1`, and `DOLocationID_138`

This indicates that both **specific categorical states** and **continuous trip distance** contribute to the model’s behavior.

---

## Key Findings

1. **Boosted tree models outperform linear baselines on this task.**  
   XGBoost clearly outperformed Linear Regression, indicating that NYC taxi trip duration is governed by important non-linear relationships.

2. **Spatial heterogeneity is central to prediction.**  
   Pickup and dropoff zone identifiers are the strongest feature groups in the current model, suggesting that “where the trip starts and ends” matters more than simple time indicators alone.

3. **Trip duration prediction difficulty is not uniform across the dataset.**  
   Error analysis shows that performance varies across hours, distance ranges, and pickup zones, meaning that some travel contexts are structurally harder to predict.

4. **This task supports a full research-style tabular workflow.**  
   The project demonstrates a complete pipeline from official public data acquisition to preprocessing, model comparison, grouped error analysis, and interpretability-oriented reporting.

5. **Remaining error likely reflects missing external context.**  
   Weather, special events, holidays, traffic disruptions, and road conditions are not included in the current version and likely explain part of the residual error.

---

## Reproducibility

### Environment
A dedicated Python virtual environment and dependency file are used for reproducible execution.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Typical workflow

```bash
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train.py
python src/models/compare_models.py
python src/models/evaluate.py
python src/analysis/eda.py
python src/analysis/feature_importance.py
```

---

## Current Limitations

The current repository is intentionally a first research-style version rather than a final production system.

Main limitations include:

- only one month of TLC data is currently used
- no weather, holiday, or event features are included
- zone IDs are used directly without richer geographic metadata
- no route path or real traffic information is available
- the current version focuses on tabular learning rather than sequential trajectory modeling
- rate-code importance should be interpreted carefully because some variables may partly reflect downstream trip characteristics

---

## Future Work

Several directions would meaningfully strengthen the project:

### Data expansion
- extend from one month to multiple months of Yellow Taxi data
- compare seasonal effects across time

### Richer features
- add weather data
- add holiday indicators
- add airport / borough / zone-category metadata
- map zone IDs to human-readable area labels

### Modeling
- test log-transformed duration targets
- compare LightGBM or CatBoost
- explore more robust hyperparameter tuning

### Analysis
- deepen grouped error analysis
- inspect high-error trips qualitatively
- compare performance under different distance regimes
- study possible information-leakage risks more explicitly

### Packaging
- add a formal project report
- create a GitHub release
- optionally connect the repository to Zenodo for DOI archiving

---

## Project Status

**Current status:** research-style first version completed and being refined for public presentation.

At this stage, the project already includes:

- official public data
- a reproducible preprocessing pipeline
- benchmarked tabular models
- grouped error analysis
- XGBoost-based interpretability outputs

The current version is therefore suitable as a **compact but serious data-science portfolio project** and a strong foundation for further extension.

---

## Citation

If you use this repository, please cite it via the included `CITATION.cff` file.

---

## License

This project is released under the **MIT License**.

---

## Contact

- Author: **Xinyu Shu**
- GitHub: `https://github.com/xinyushu766-art`
- Repository: `https://github.com/xinyushu766-art/nyc-taxi-duration-study`
