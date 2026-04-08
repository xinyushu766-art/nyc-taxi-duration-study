# NYC Taxi Duration Study

Interpretable trip duration prediction and error analysis on official NYC TLC taxi data.

## Project Goal
This project studies whether trip duration can be predicted using realistic trip-level features such as pickup time, dropoff zone, trip distance, rate type, and payment type.

## Research Questions
1. How well can we predict taxi trip duration using structured tabular features?
2. How do linear models, tree models, and boosting models compare?
3. In which time windows, zones, and trip ranges do models fail more often?
4. Which features drive predictions, and what are the interpretability risks?

## Data Source
Official NYC Taxi & Limousine Commission (TLC) trip record data.

## Project Structure
- `data/raw/`: raw downloaded data
- `data/processed/`: cleaned datasets
- `notebooks/`: exploratory analysis and baseline experiments
- `src/data/`: data cleaning scripts
- `src/features/`: feature engineering scripts
- `src/models/`: training and evaluation scripts
- `reports/figures/`: final figures for README/report

## First Milestone
- [ ] Create repository structure
- [ ] Download one month of Yellow Taxi data
- [ ] Build a data dictionary
- [ ] Filter invalid durations and distances
- [ ] Plot 3 basic EDA figures
- [ ] Train 1 baseline model

## Reproducibility
Environment and dependency details will be added as the project develops.

## Status
Work in progress.
