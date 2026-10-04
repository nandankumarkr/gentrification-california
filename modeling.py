# ============================================================
# GENTRIFICATION CALIFORNIA
# MACHINE LEARNING MODELING PIPELINE
#
# Person 2:
# - Dataset verification
# - Train/test split
# - Logistic Regression
# - Random Forest
# - XGBoost
# - Hyperparameter tuning
# - Model evaluation
# - Confusion matrices
# - Feature importance
# - Final model selection
# - Model saving
# ============================================================
import os

# Prevent Windows multiprocessing/threading issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import warnings

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/analysis_dataset.csv"
RESULTS_DIR = "results/ml"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("1. LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["gentrification"].value_counts())

print("\nTarget percentage:")
print(df["gentrification"].value_counts(normalize=True) * 100)


# ============================================================
# 3. VERIFY TARGET
# ============================================================

TARGET = "gentrification"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in the dataset."
    )

# Make sure target is binary
unique_targets = sorted(df[TARGET].dropna().unique())

print("\nUnique target values:", unique_targets)

if len(unique_targets) != 2:
    raise ValueError(
        "The target column must contain exactly two classes."
    )

# If target is not already 0/1, convert it safely
if set(unique_targets) != {0, 1}:
    target_mapping = {
        unique_targets[0]: 0,
        unique_targets[1]: 1
    }

    df[TARGET] = df[TARGET].map(target_mapping)

    print("\nTarget mapping used:")
    print(target_mapping)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\n" + "=" * 70)
print("2. FEATURES AND TARGET")
print("=" * 70)

print("Number of input features:", X.shape[1])
print("Feature names:")

for feature in X.columns:
    print(" -", feature)

print("\nTarget:", TARGET)


# ============================================================
# 5. HANDLE DATA TYPES
# ============================================================

# All eight expected ML features should be usable numerically.
# Convert numeric-looking columns to numeric values.

for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

if X.isnull().sum().sum() > 0:
    print("\nWARNING: Missing/non-numeric values detected after conversion.")
    print(X.isnull().sum())

    raise ValueError(
        "Input features contain missing/non-numeric values. "
        "Please inspect analysis_dataset.csv before continuing."
    )


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("3. TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X_test, y_test):
    """
    Evaluate a trained classification model.
    """

    predictions = model.predict(X_test)

    # Probability for positive class
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = None

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    if probabilities is not None:
        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )
    else:
        roc_auc = np.nan

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    print("Confusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Gentrified", "Gentrified"]
    )

    display.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(RESULTS_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# 8. MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("4. LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])

logistic_model.fit(X_train, y_train)

logistic_results = evaluate_model(
    "Logistic Regression",
    logistic_model,
    X_test,
    y_test
)


# ============================================================
# 9. MODEL 2 — RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("5. RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=1
)

rf_model.fit(X_train, y_train)

rf_results = evaluate_model(
    "Random Forest",
    rf_model,
    X_test,
    y_test
)


# ============================================================
# 10. MODEL 3 — XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("6. XGBOOST")
print("=" * 70)

# Calculate class imbalance ratio
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
    if positive_count > 0
    else 1
)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=1
)

xgb_model.fit(X_train, y_train)

xgb_results = evaluate_model(
    "XGBoost",
    xgb_model,
    X_test,
    y_test
)


# ============================================================
# 11. HYPERPARAMETER TUNING — RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("7. RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    ),
    param_grid=rf_param_grid,
    scoring="f1",
    cv=5,
    n_jobs=1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

print("\nBest Random Forest parameters:")
print(rf_grid.best_params_)

print(
    f"Best cross-validation F1: "
    f"{rf_grid.best_score_:.4f}"
)

tuned_rf_results = evaluate_model(
    "Tuned Random Forest",
    best_rf,
    X_test,
    y_test
)


# ============================================================
# 12. HYPERPARAMETER TUNING — XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("8. XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

xgb_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1]
}

xgb_grid = GridSearchCV(
    estimator=XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1
    ),
    param_grid=xgb_param_grid,
    scoring="f1",
    cv=5,
    n_jobs=1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_

print("\nBest XGBoost parameters:")
print(xgb_grid.best_params_)

print(
    f"Best cross-validation F1: "
    f"{xgb_grid.best_score_:.4f}"
)

tuned_xgb_results = evaluate_model(
    "Tuned XGBoost",
    best_xgb,
    X_test,
    y_test
)


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("9. FINAL MODEL COMPARISON")
print("=" * 70)

results = pd.DataFrame([
    logistic_results,
    rf_results,
    xgb_results,
    tuned_rf_results,
    tuned_xgb_results
])

results = results.sort_values(
    by="F1 Score",
    ascending=False
)

results = results.reset_index(drop=True)

print("\n")
print(results.to_string(index=False))

# Save results
results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 14. MODEL COMPARISON GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

plot_data = results.set_index("Model")[metrics]

ax = plot_data.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Gentrification Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. SELECT FINAL MODEL
# ============================================================

best_model_name = results.iloc[0]["Model"]

print("\n" + "=" * 70)
print("10. BEST MODEL")
print("=" * 70)

print("Best model based on F1 Score:")
print(best_model_name)


model_dictionary = {
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Tuned Random Forest": best_rf,
    "Tuned XGBoost": best_xgb
}

final_model = model_dictionary[best_model_name]


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODELS_DIR,
    "gentrification_model.pkl"
)

joblib.dump(
    final_model,
    final_model_path
)

# Save feature names
feature_names_path = os.path.join(
    MODELS_DIR,
    "feature_names.pkl"
)

joblib.dump(
    list(X.columns),
    feature_names_path
)

print("\nFinal model saved to:")
print(final_model_path)

print("\nFeature names saved to:")
print(feature_names_path)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("11. FEATURE IMPORTANCE")
print("=" * 70)


def get_tree_model(model):
    """
    Extract the underlying tree model.
    """

    if isinstance(model, Pipeline):
        return model.named_steps["model"]

    return model


tree_models = {
    "Random Forest": rf_model,
    "Tuned Random Forest": best_rf,
    "XGBoost": xgb_model,
    "Tuned XGBoost": best_xgb
}

# Use the best tree-based model available
if best_model_name in tree_models:
    importance_model = tree_models[best_model_name]
else:
    # Use tuned XGBoost for feature interpretation
    importance_model = best_xgb

importance_model = get_tree_model(importance_model)

if hasattr(importance_model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    print("\nFeature importance:")
    print(importance_df.to_string(index=False))

    importance_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Feature Importance - {best_model_name}"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 18. SAVE FINAL INFORMATION
# ============================================================

summary = {
    "dataset": DATA_PATH,
    "samples": int(df.shape[0]),
    "input_features": int(X.shape[1]),
    "target": TARGET,
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "best_model": best_model_name,
    "best_f1_score": float(results.iloc[0]["F1 Score"]),
    "best_accuracy": float(results.iloc[0]["Accuracy"]),
    "best_precision": float(results.iloc[0]["Precision"]),
    "best_recall": float(results.iloc[0]["Recall"]),
    "best_roc_auc": float(results.iloc[0]["ROC-AUC"])
}

with open(
    os.path.join(
        RESULTS_DIR,
        "final_model_summary.json"
    ),
    "w"
) as file:
    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# 19. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("ML PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCreated files:")
print("results/ml/model_comparison.csv")
print("results/ml/model_comparison.png")
print("results/ml/feature_importance.csv")
print("results/ml/feature_importance.png")
print("results/ml/final_model_summary.json")
print("results/ml/*_confusion_matrix.png")
print("models/gentrification_model.pkl")
print("models/feature_names.pkl")

print("\nBest model:", best_model_name)
print(
    f"Best F1 Score: "
    f"{results.iloc[0]['F1 Score']:.4f}"
)

print("\nYou can now run predict.py.")