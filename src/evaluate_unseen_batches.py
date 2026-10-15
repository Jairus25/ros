from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real_gas_sensor" / "Dataset"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_batch(path):
    features = []
    labels = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()

            if not parts:
                continue

            labels.append(int(float(parts[0])))

            row = np.full(128, np.nan)

            for item in parts[1:]:
                index, value = item.split(":")
                row[int(index) - 1] = float(value)

            features.append(row)

    return np.asarray(features), np.asarray(labels)


def main():
    train_batches = range(1, 9)
    test_batches = [9, 10]

    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []

    for batch in train_batches:
        X, y = read_batch(DATA_DIR / f"batch{batch}.dat")
        X_train_parts.append(X)
        y_train_parts.append(y)
        print(f"Training data: batch{batch}.dat — {len(y)} records")

    for batch in test_batches:
        X, y = read_batch(DATA_DIR / f"batch{batch}.dat")
        X_test_parts.append(X)
        y_test_parts.append(y)
        print(f"Unseen test data: batch{batch}.dat — {len(y)} records")

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    X_test = np.vstack(X_test_parts)
    y_test = np.concatenate(y_test_parts)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    print("\nTraining on batches 1–8...")
    model.fit(X_train, y_train)

    print("Evaluating on batches 9–10...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nUnseen-batch accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
    print(report)

    with (RESULTS_DIR / "unseen_batch_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("Train batches: 1–8\n")
        file.write("Test batches: 9–10\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write(report)
        file.write(
            "\nThis evaluates gas-class classification on held-out batches. "
            "It does not validate dangerous-concentration detection or "
            "guarantee performance on physical rover sensors.\n"
        )

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=sorted(np.unique(np.concatenate([y_train, y_test]))),
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Unseen-Batch Gas Classification")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "unseen_batch_confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\nSaved:")
    print(RESULTS_DIR / "unseen_batch_report.txt")
    print(RESULTS_DIR / "unseen_batch_confusion_matrix.png")


if __name__ == "__main__":
    main()
