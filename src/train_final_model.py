from pathlib import Path

import json
import joblib
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "real_gas_sensor" / "Dataset"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "gas_svm_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "gas_svm_metadata.json"

FEATURE_COUNT = 128


# --------------------------------------------------
# DATASET LOADER
# --------------------------------------------------

def read_batch(file_path):
    X_rows = []
    y_rows = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            tokens = line.split()

            try:
                label = int(tokens[0])

                features = np.full(
                    FEATURE_COUNT,
                    np.nan,
                    dtype=np.float64,
                )

                for token in tokens[1:]:
                    index_text, value_text = token.split(":", 1)

                    index = int(index_text)
                    value = float(value_text)

                    if not 1 <= index <= FEATURE_COUNT:
                        raise ValueError(
                            f"Invalid feature index: {index}"
                        )

                    features[index - 1] = value

                X_rows.append(features)
                y_rows.append(label)

            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Error in {file_path.name}, "
                    f"line {line_number}: {exc}"
                ) from exc

    return (
        np.asarray(X_rows, dtype=np.float64),
        np.asarray(y_rows, dtype=np.int64),
    )


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

def load_training_data():

    X_batches = []
    y_batches = []

    for batch_number in range(1, 9):

        file_path = DATA_DIR / f"batch{batch_number}.dat"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {file_path}"
            )

        X_batch, y_batch = read_batch(file_path)

        X_batches.append(X_batch)
        y_batches.append(y_batch)

        print(
            f"Loaded batch {batch_number}: "
            f"{len(y_batch)} records"
        )

    X = np.vstack(X_batches)
    y = np.concatenate(y_batches)

    return X, y


# --------------------------------------------------
# TRAIN FINAL CANDIDATE
# --------------------------------------------------

def main():

    print("=" * 60)
    print("TRAINING GAS CLASSIFICATION MODEL")
    print("=" * 60)

    X, y = load_training_data()

    print(f"\nTraining records: {len(y)}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Classes: {np.unique(y).tolist()}")

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    C=1.0,
                    gamma="scale",
                ),
            ),
        ]
    )

    print("\nTraining the SVM...")
    model.fit(X, y)

    # Save the entire pipeline, including imputation and scaling.
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "Standardized SVM",
        "algorithm": "Support Vector Classifier",
        "kernel": "rbf",
        "training_batches": list(range(1, 9)),
        "feature_count": FEATURE_COUNT,
        "classes": [int(label) for label in model.classes_],
        "dataset": "UCI Gas Sensor Array Drift Dataset",
        "purpose": "Experimental gas-class classification",
        "safety_status": (
            "Not validated for hazardous concentration detection"
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nModel training completed.")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")

    print(
        "\nNote: This model classifies dataset patterns. "
        "It does not independently determine whether a "
        "gas concentration is dangerous."
    )


if __name__ == "__main__":
    main()
