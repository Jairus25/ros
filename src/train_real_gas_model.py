from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real_gas_sensor" / "Dataset"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_batch(file_path):
    """Read a LIBSVM-style batch file with indexed sensor features."""
    rows = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            fields = line.split()

            try:
                label = int(float(fields[0]))
                features = {}

                for item in fields[1:]:
                    index_text, value_text = item.split(":", 1)
                    index = int(index_text)
                    value = float(value_text)

                    if index < 1:
                        raise ValueError("Feature indexes must start at 1")

                    features[index - 1] = value

                rows.append((label, features))

            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Could not parse {file_path.name}, "
                    f"line {line_number}: {exc}"
                ) from exc

    return rows


def main():
    batch_files = sorted(
        DATA_DIR.glob("batch*.dat"),
        key=lambda path: int(re.search(r"batch(\d+)", path.stem).group(1)),
    )

    if not batch_files:
        raise FileNotFoundError(
            f"No batch files found in {DATA_DIR}"
        )

    records = []
    batch_ids = []

    for file_path in batch_files:
        batch_number = int(
            re.search(r"batch(\d+)", file_path.stem).group(1)
        )
        batch_records = read_batch(file_path)

        records.extend(batch_records)
        batch_ids.extend([batch_number] * len(batch_records))

        print(
            f"Loaded {file_path.name}: "
            f"{len(batch_records)} records"
        )

    # Construct a dense feature matrix from indexed features.
    max_feature_index = max(
        max(features.keys())
        for _, features in records
        if features
    )

    X = np.full(
        (len(records), max_feature_index + 1),
        np.nan,
        dtype=np.float64,
    )
    y = np.empty(len(records), dtype=np.int64)

    for row_index, (label, features) in enumerate(records):
        y[row_index] = label

        for feature_index, value in features.items():
            X[row_index, feature_index] = value

    batch_ids = np.asarray(batch_ids)

    if not np.isfinite(X[~np.isnan(X)]).all():
        raise ValueError("The dataset contains infinite feature values.")

    # The class labels are dataset labels, not safe/dangerous labels.
    label_values, label_counts = np.unique(y, return_counts=True)

    print("\nDataset summary")
    print("----------------")
    print(f"Total records: {len(y)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(label_values)}")
    print(f"Class counts: {dict(zip(label_values.tolist(), label_counts.tolist()))}")
    print(f"Batches: {sorted(np.unique(batch_ids).tolist())}")

    # Stratified holdout baseline. This evaluates a random record split,
    # not generalization to a completely unseen batch.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining Random Forest classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nHoldout accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    report = classification_report(
        y_test,
        predictions,
        labels=label_values,
        zero_division=0,
    )
    print(report)

    # Save model and feature metadata.
    model_path = MODEL_DIR / "real_gas_classifier.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "dataset": "UCI Gas Sensor Array Drift Dataset",
        "target": "original gas-class label",
        "feature_count": int(X.shape[1]),
        "class_labels": [int(label) for label in label_values],
        "training_records": int(len(y_train)),
        "test_records": int(len(y_test)),
        "random_split_accuracy": float(accuracy),
        "split_random_state": 42,
        "warning": (
            "Gas-class classification only. This model does not predict "
            "safe/dangerous concentration or replace calibrated safety sensors."
        ),
    }

    with (MODEL_DIR / "real_gas_classifier_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    # Save a human-readable report.
    with (RESULTS_DIR / "real_gas_classification_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("UCI Gas Sensor Array Drift Dataset\n")
        file.write("Random 80/20 stratified holdout evaluation\n\n")
        file.write(f"Records: {len(y)}\n")
        file.write(f"Features: {X.shape[1]}\n")
        file.write(f"Classes: {label_values.tolist()}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write(report)
        file.write(
            "\n\nLimitations: This is gas-class classification, not "
            "hazard-concentration detection. Randomly split records may "
            "share batch-related characteristics between train and test.\n"
        )

    # Save confusion matrix.
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=label_values,
        display_labels=[str(label) for label in label_values],
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Gas-Class Classification — Random Holdout")
    fig.tight_layout()

    plot_path = RESULTS_DIR / "real_gas_confusion_matrix.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("\nSaved files:")
    print(f"Model: {model_path}")
    print(f"Metadata: {MODEL_DIR / 'real_gas_classifier_metadata.json'}")
    print(f"Report: {RESULTS_DIR / 'real_gas_classification_report.txt'}")
    print(f"Confusion matrix: {plot_path}")
    print(
        "\nReminder: the reported accuracy is a random-split baseline, "
        "not proof of performance on new batches or real rover sensors."
    )


if __name__ == "__main__":
    main()
