"""
Gentrification Prediction Script
--------------------------------
Loads the final trained model and feature names, then:
1. Tests the model on a real row from data/analysis_dataset.csv
2. Optionally allows manual prediction using the 8 required features

Run from the project root:
    python predict.py
"""

from pathlib import Path
import joblib
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "gentrification_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "feature_names.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "analysis_dataset.csv"


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

for path in [MODEL_PATH, FEATURES_PATH, DATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
            f"\nMake sure this file exists in the correct project folder."
        )


# ============================================================
# 3. LOAD MODEL, FEATURES AND DATASET
# ============================================================

print("=" * 70)
print("GENTRIFICATION PREDICTION")
print("=" * 70)

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
df = pd.read_csv(DATA_PATH)

# Convert feature names to a normal Python list
feature_names = list(feature_names)

print("\nFinal model loaded successfully.")
print("Model:", type(model).__name__)

print("\nExpected features:")
for i, feature in enumerate(feature_names, start=1):
    print(f"{i}. {feature}")


# ============================================================
# 4. VERIFY DATASET
# ============================================================

missing_features = [f for f in feature_names if f not in df.columns]

if missing_features:
    raise ValueError(
        "\nThe following required features are missing from "
        "analysis_dataset.csv:\n"
        + "\n".join(missing_features)
    )

if "gentrification" not in df.columns:
    raise ValueError(
        "\nTarget column 'gentrification' was not found in "
        "analysis_dataset.csv."
    )

# Make sure all model features are numeric
X = df[feature_names].apply(pd.to_numeric, errors="coerce")

if X.isnull().any().any():
    bad_columns = X.columns[X.isnull().any()].tolist()
    raise ValueError(
        "\nNon-numeric or missing values were found in these "
        f"feature columns: {bad_columns}"
    )


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_row(row_number: int) -> None:
    """Predict one existing row from the dataset."""

    if row_number < 0 or row_number >= len(df):
        raise ValueError(
            f"Row number must be between 0 and {len(df) - 1}."
        )

    sample = X.iloc[[row_number]]

    prediction = model.predict(sample)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        predicted_probability = float(probabilities[int(prediction)])
    else:
        predicted_probability = None

    actual = df.iloc[row_number]["gentrification"]

    print("\n" + "=" * 70)
    print("REAL DATASET ROW PREDICTION")
    print("=" * 70)

    print(f"\nDataset row: {row_number}")

    print("\nInput values:")
    for feature in feature_names:
        print(f"{feature}: {sample.iloc[0][feature]}")

    print("\nActual gentrification:", actual)
    print("Predicted gentrification:", prediction)

    if predicted_probability is not None:
        print(
            f"Prediction probability for class {prediction}: "
            f"{predicted_probability:.4f} "
            f"({predicted_probability * 100:.2f}%)"
        )

    if prediction == actual:
        print("\nPrediction result: CORRECT")
    else:
        print("\nPrediction result: INCORRECT")

    print("=" * 70)


# ============================================================
# 6. MANUAL PREDICTION FUNCTION
# ============================================================

def manual_prediction() -> None:
    """Ask the user for the 8 feature values and make a prediction."""

    print("\n" + "=" * 70)
    print("MANUAL GENTRIFICATION PREDICTION")
    print("=" * 70)
    print("\nEnter the value for each feature.")

    values = []

    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                values.append(value)
                break
            except ValueError:
                print("Please enter a numeric value.")

    sample = pd.DataFrame([values], columns=feature_names)

    prediction = model.predict(sample)[0]

    print("\n" + "-" * 70)
    print("PREDICTION RESULT")
    print("-" * 70)

    print("Predicted gentrification:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]

        print("\nClass probabilities:")
        for class_value, probability in zip(model.classes_, probabilities):
            print(
                f"Class {class_value}: "
                f"{probability:.4f} ({probability * 100:.2f}%)"
            )

    print("-" * 70)


# ============================================================
# 7. MAIN MENU
# ============================================================

if __name__ == "__main__":

    print("\nChoose an option:")
    print("1. Test the model using a real dataset row")
    print("2. Enter feature values manually")
    print("3. Exit")

    while True:
        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            row_input = input(
                f"Enter dataset row number (0-{len(df) - 1}) "
                "[default: 0]: "
            ).strip()

            if row_input == "":
                row_number = 0
            else:
                try:
                    row_number = int(row_input)
                except ValueError:
                    print("Please enter a valid integer row number.")
                    continue

            try:
                predict_row(row_number)
            except Exception as exc:
                print(f"\nPrediction failed: {exc}")

            break

        elif choice == "2":
            try:
                manual_prediction()
            except Exception as exc:
                print(f"\nPrediction failed: {exc}")

            break

        elif choice == "3":
            print("\nExiting.")
            break

        else:
            print("Invalid choice. Enter 1, 2, or 3.")
