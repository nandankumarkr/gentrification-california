# Predicting Gentrification in California Census Tracts

A machine learning pipeline that predicts whether a California census tract experienced a rise in inflation-adjusted housing costs between 2012 and 2016, using tract-level American Community Survey (ACS) data from 2010–2011 as predictors.

The repo covers the full workflow: downloading Census data, building and cleaning the dataset, constructing the target label, exploratory analysis, training and tuning classifiers, and running predictions with the saved model.

---

## Problem Statement

**Task:** Binary classification at the census-tract level.

**Target (`gentrification`):**

| Value | Meaning |
|-------|---------|
| `1` | Inflation-adjusted housing cost increased from 2012 to 2016 |
| `0` | Otherwise |

2012 housing costs are converted into 2016 dollars using CPI-U annual averages (`229.594` for 2012, `240.007` for 2016) before computing the change. Census missing-value sentinels (`-666666666`) are treated as missing and dropped.

**Features (8, all available before the prediction period):**

| Feature | Description |
|---------|-------------|
| `occupied_units_2011` | Occupied housing units |
| `owner_units_2011` | Owner-occupied units |
| `renter_units_2011` | Renter-occupied units |
| `median_monthly_cost_2010` | Median monthly housing cost, 2010 |
| `median_monthly_cost_2011` | Median monthly housing cost, 2011 |
| `renter_share_2011` | Share of occupied units that are rented |
| `housing_cost_change_x` | Housing cost change (pre-period) |
| `housing_cost_pct_change_x` | Housing cost percentage change (pre-period) |

---

## Repository Structure

```
gentrification-california/
├── data/                       # Raw, intermediate and final datasets
├── models/                     # Saved model + feature names (.pkl)
├── results/                    # EDA outputs and ML results
│   └── ml/                     # Metrics, confusion matrices, feature importance
│
├── download_data.py            # Download predictor data
├── download_b25085.py          # Download ACS table B25085 (current period)
├── download_b25085_2010.py     # Download ACS table B25085 (2010)
├── download_s2503.py           # Download ACS table S2503
├── download_s2503_2010.py      # Download ACS table S2503 (2010)
├── download_outcome_data.py    # Download outcome-period data (2012, 2016)
├── inspect_metadata.py         # Inspect Census variable metadata
│
├── build_dataset.py            # Assemble predictor dataset
├── merge_dataset.py            # Merge sources on tract ID
├── clean_dataset.py            # Initial cleaning
├── build_target.py             # Construct the gentrification label
├── clean_final_dataset.py      # Select features, drop missing rows
│
├── eda.py                      # Exploratory data analysis
├── modeling.py                 # Train, tune, evaluate, save models
└── predict.py                  # Run predictions with the saved model
```

---

## Pipeline

1. **Download** – the `download_*.py` scripts pull tract-level ACS tables from the U.S. Census Bureau into `data/raw/`.
2. **Build & merge** – `build_dataset.py` and `merge_dataset.py` combine the tables using the 11-digit tract ID (the `1400000US` prefix is stripped from `GEO_ID`).
3. **Clean** – `clean_dataset.py` handles initial cleaning.
4. **Target** – `build_target.py` computes inflation-adjusted housing cost change from 2012 to 2016 and writes `data/housing_cost_target.csv`.
5. **Final dataset** – `clean_final_dataset.py` keeps the 8 features plus the target, drops rows with missing values, and writes `data/analysis_dataset.csv`.
6. **EDA** – `eda.py` produces exploratory plots and summaries.
7. **Modeling** – `modeling.py` trains and compares models (below) and saves the best one.
8. **Prediction** – `predict.py` loads the saved model for inference.

---

## Modeling

- **Split:** 80/20 train/test, stratified on the target, `random_state=42`
- **Models:**
  - Logistic Regression (with `StandardScaler`, balanced class weights)
  - Random Forest (200 trees, balanced class weights)
  - XGBoost (class imbalance handled via `scale_pos_weight`)
  - Tuned Random Forest and Tuned XGBoost via `GridSearchCV` (5-fold CV, optimizing F1)
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices
- **Model selection:** the model with the highest test-set F1 is saved as the final model, along with its feature names.

### Outputs

| File | Contents |
|------|----------|
| `results/ml/model_comparison.csv` / `.png` | Metrics for all five models |
| `results/ml/*_confusion_matrix.png` | One confusion matrix per model |
| `results/ml/feature_importance.csv` / `.png` | Feature importances of the best tree model |
| `results/ml/final_model_summary.json` | Dataset size, split sizes, best model and its scores |
| `models/gentrification_model.pkl` | Final trained model |
| `models/feature_names.pkl` | Ordered feature list expected by the model |

### Results

> Fill in after running `modeling.py` (see `results/ml/model_comparison.csv`).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| _best model_ | – | – | – | – | – |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `joblib`, `requests`

```bash
git clone https://github.com/nandankumarkr/gentrification-california.git
cd gentrification-california
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost matplotlib joblib requests
```

### Run the pipeline

Run everything from the project root:

```bash
# 1. Download data
python download_data.py
python download_b25085.py
python download_b25085_2010.py
python download_s2503.py
python download_s2503_2010.py
python download_outcome_data.py

# 2. Build the dataset
python build_dataset.py
python merge_dataset.py
python clean_dataset.py
python build_target.py
python clean_final_dataset.py

# 3. Explore, train, predict
python eda.py
python modeling.py
python predict.py
```

### Making predictions

`predict.py` opens a small menu:

1. Test the model on a real row from `data/analysis_dataset.csv` (shows actual vs. predicted label and probability)
2. Enter the 8 feature values manually and get a predicted class with probabilities
3. Exit

---

## Data Sources

- U.S. Census Bureau, American Community Survey (ACS) 5-year estimates, census-tract level, California
  - Table S2503 (financial characteristics of occupied housing units)
  - Table B25085
- CPI-U annual averages (U.S. Bureau of Labor Statistics) for inflation adjustment

---

## Limitations

- The label captures a **rise in real housing costs**, not the full definition of gentrification (which usually also involves demographic, income or education shifts).
- A simple "cost increased" threshold of `> 0` can produce an imbalanced target; class weights are used to compensate.
- Predictors come from a narrow window (2010–2011), so the model may not generalize to other periods or regions.
- Model quality should be judged on F1 and ROC-AUC, not accuracy alone.

---

## Author

Nandan Kumar K R
Madisetty Roshini
