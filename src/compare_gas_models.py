from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "real_gas_sensor" / "Dataset"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COUNT = 128

# Batches 1-8 are used for training and cross-validation.
# Batches 9-10 are held out for final evaluation.
TRAIN_BATCHES = list(range(1, 9))
TEST_BATCHES = [9, 10]

RANDOM_STATE = 42
N_SPLITS = 4


# ============================================================
# DATA LOADING
# ============================================================

def read_batch(file_path):
    """
    Read one UCI Gas Sensor Array Drift dataset batch.

    Expected row format:
        class_label feature_index:value feature_index:value ...

    Example:
        1 1:0.23 2:0.51 5:0.18

    The dataset contains sparse LIBSVM-style records.
    Missing feature values are represented as NaN.
    """

    X_rows = []
    y_rows = []

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset batch not found: {file_path}"
        )

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
                    if ":" not in token:
                        continue

                    index_text, value_text = token.split(":", 1)

                    feature_index = int(index_text)
                    feature_value = float(value_text)

                    # Dataset feature indices start at 1.
                    if not 1 <= feature_index <= FEATURE_COUNT:
                        raise ValueError(
                            f"Feature index {feature_index} is "
                            f"outside 1-{FEATURE_COUNT}"
                        )

                    features[feature_index - 1] = feature_value

                X_rows.append(features)
                y_rows.append(label)

            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Could not parse {file_path.name}, "
                    f"line {line_number}: {line}\n"
                    f"Reason: {exc}"
                ) from exc

    if not X_rows:
        raise ValueError(
            f"No records were read from {file_path}"
        )

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)

    return X, y


def load_dataset():
    """
    Load all ten batches and combine them into one dataset.

    Returns:
        X: Feature matrix
        y: Gas-class labels
        groups: Batch number for each record
    """

    all_X = []
    all_y = []
    all_groups = []

    print("\nLoading gas sensor dataset...")
    print("-" * 65)

    for batch_number in range(1, 11):

        batch_path = DATA_DIR / f"batch{batch_number}.dat"

        X_batch, y_batch = read_batch(batch_path)

        all_X.append(X_batch)
        all_y.append(y_batch)

        all_groups.append(
            np.full(len(y_batch), batch_number, dtype=np.int64)
        )

        print(
            f"Batch {batch_number:2d}: "
            f"{len(y_batch):5d} records | "
            f"{X_batch.shape[1]} features | "
            f"Classes: {np.unique(y_batch).tolist()}"
        )

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)

    print("-" * 65)
    print(f"Total records: {len(y)}")
    print(f"Total features: {X.shape[1]}")
    print(f"Gas classes: {np.unique(y).tolist()}")

    return X, y, groups


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models():
    """
    Create the three models to compare.

    Every model uses median imputation to handle missing values.
    The standardized SVM also scales the features.
    """

    random_forest_raw = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    random_forest_standardized = Pipeline(
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
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    svm_standardized = Pipeline(
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

    return {
        "SVM - standardized": svm_standardized,
        "Random Forest - raw": random_forest_raw,
        "Random Forest - standardized": random_forest_standardized,
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(models, X_dev, y_dev, groups_dev):
    """
    Perform GroupKFold cross-validation on batches 1-8.

    Each validation fold contains whole batches that are excluded
    from that fold's training data.

    Macro F1 gives equal importance to each class represented in
    the validation fold.
    """

    print("\n")
    print("=" * 75)
    print("GROUP CROSS-VALIDATION: BATCHES 1-8")
    print("=" * 75)

    unique_groups = np.unique(groups_dev)

    n_splits = min(N_SPLITS, len(unique_groups))

    if n_splits < 2:
        raise ValueError(
            "At least two distinct development batches are required."
        )

    splitter = GroupKFold(n_splits=n_splits)

    cv_results = []

    for model_name, model in models.items():

        print(f"\nModel: {model_name}")
        print("-" * 65)

        fold_accuracies = []
        fold_macro_f1 = []

        for fold_number, (train_idx, val_idx) in enumerate(
            splitter.split(X_dev, y_dev, groups_dev),
            start=1,
        ):

            X_train = X_dev[train_idx]
            y_train = y_dev[train_idx]

            X_val = X_dev[val_idx]
            y_val = y_dev[val_idx]

            train_groups = np.unique(groups_dev[train_idx])
            val_groups = np.unique(groups_dev[val_idx])

            # A class present in validation but absent from training
            # cannot be learned by that fold's classifier.
            unseen_classes = sorted(
                set(np.unique(y_val))
                - set(np.unique(y_train))
            )

            if unseen_classes:
                print(
                    f"Warning: Fold {fold_number} contains "
                    f"validation classes absent from training: "
                    f"{unseen_classes}"
                )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            fold_accuracy = accuracy_score(y_val, y_pred)

            fold_f1 = f1_score(
                y_val,
                y_pred,
                average="macro",
                zero_division=0,
            )

            fold_accuracies.append(fold_accuracy)
            fold_macro_f1.append(fold_f1)

            print(
                f"Fold {fold_number}: "
                f"Accuracy={fold_accuracy:.4f}, "
                f"Macro F1={fold_f1:.4f}, "
                f"Train batches={train_groups.tolist()}, "
                f"Validation batches={val_groups.tolist()}"
            )

        mean_accuracy = float(np.mean(fold_accuracies))
        std_accuracy = float(np.std(fold_accuracies))
        mean_macro_f1 = float(np.mean(fold_macro_f1))

        print("\nCross-validation summary:")
        print(f"Mean accuracy : {mean_accuracy:.4f}")
        print(f"Std accuracy  : {std_accuracy:.4f}")
        print(f"Mean macro F1 : {mean_macro_f1:.4f}")

        cv_results.append(
            {
                "model": model_name,
                "mean_cv_accuracy": mean_accuracy,
                "std_cv_accuracy": std_accuracy,
                "mean_cv_macro_f1": mean_macro_f1,
            }
        )

    cv_df = pd.DataFrame(cv_results)

    cv_df = cv_df.sort_values(
        by="mean_cv_macro_f1",
        ascending=False,
    )

    output_path = RESULTS_DIR / "gas_model_comparison.csv"

    cv_df.to_csv(output_path, index=False)

    print(f"\nSaved cross-validation results to:\n{output_path}")

    return cv_df


# ============================================================
# UNSEEN-BATCH EVALUATION
# ============================================================

def evaluate_unseen_batches(
    models,
    X_dev,
    y_dev,
    X_test,
    y_test,
):
    """
    Train each model on batches 1-8 and evaluate it on batches 9-10.

    Prints accuracy, macro F1, per-class classification metrics,
    and a confusion matrix for every model.

    Saves:
        results/gas_model_unseen_batch_comparison.csv
        results/<model>_unseen_batch_report.txt
        results/<model>_unseen_batch_confusion_matrix.csv
    """

    print("\n")
    print("=" * 75)
    print("UNSEEN-BATCH EVALUATION: TRAIN 1-8, TEST 9-10")
    print("=" * 75)

    test_classes = np.unique(y_test)

    test_results = []

    for model_name, model in models.items():

        print("\n")
        print("#" * 75)
        print(f"MODEL: {model_name}")
        print("#" * 75)

        # Train using development batches only.
        model.fit(X_dev, y_dev)

        # Evaluate using batches that were excluded from training.
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        macro_f1 = f1_score(
            y_test,
            y_pred,
            labels=test_classes,
            average="macro",
            zero_division=0,
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=test_classes,
            zero_division=0,
            digits=4,
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=test_classes,
        )

        print(f"\nUnseen-batch accuracy: {accuracy:.4f}")
        print(f"Unseen-batch macro F1: {macro_f1:.4f}")

        print("\nPer-class classification report:")
        print(report)

        print("Confusion matrix:")
        print(cm)

        # Use filesystem-friendly names.
        safe_name = (
            model_name.lower()
            .replace(" - ", "_")
            .replace(" ", "_")
        )

        report_path = (
            RESULTS_DIR
            / f"{safe_name}_unseen_batch_report.txt"
        )

        confusion_path = (
            RESULTS_DIR
            / f"{safe_name}_unseen_batch_confusion_matrix.csv"
        )

        with report_path.open("w", encoding="utf-8") as file:

            file.write(f"Model: {model_name}\n")
            file.write("Training batches: 1-8\n")
            file.write("Testing batches: 9-10\n\n")

            file.write(f"Accuracy: {accuracy:.6f}\n")
            file.write(f"Macro F1: {macro_f1:.6f}\n\n")

            file.write("Classification report:\n")
            file.write(report)

            file.write("\nConfusion matrix:\n")
            file.write(np.array2string(cm))

        cm_df = pd.DataFrame(
            cm,
            index=[
                f"Actual class {label}"
                for label in test_classes
            ],
            columns=[
                f"Predicted class {label}"
                for label in test_classes
            ],
        )

        cm_df.to_csv(confusion_path)

        print(f"\nSaved classification report:\n{report_path}")
        print(f"Saved confusion matrix:\n{confusion_path}")

        test_results.append(
            {
                "model": model_name,
                "unseen_batch_accuracy": accuracy,
                "unseen_batch_macro_f1": macro_f1,
            }
        )

    test_df = pd.DataFrame(test_results)

    test_df = test_df.sort_values(
        by="unseen_batch_macro_f1",
        ascending=False,
    )

    output_path = (
        RESULTS_DIR / "gas_model_unseen_batch_comparison.csv"
    )

    test_df.to_csv(output_path, index=False)

    print("\n")
    print("=" * 75)
    print("FINAL UNSEEN-BATCH COMPARISON")
    print("=" * 75)

    print(test_df.to_string(index=False))

    print(f"\nSaved comparison to:\n{output_path}")

    return test_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("UCI GAS SENSOR ARRAY DRIFT - MODEL COMPARISON")
    print("=" * 75)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{DATA_DIR}\n\n"
            "Check that the dataset was extracted into "
            "data/real_gas_sensor/Dataset/."
        )

    # Some sklearn versions emit non-fatal parallelism warnings.
    # We leave warnings enabled so meaningful warnings remain visible.

    X, y, groups = load_dataset()

    # Development data: batches 1-8.
    dev_mask = np.isin(groups, TRAIN_BATCHES)

    # Unseen test data: batches 9-10.
    test_mask = np.isin(groups, TEST_BATCHES)

    X_dev = X[dev_mask]
    y_dev = y[dev_mask]
    groups_dev = groups[dev_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("\n")
    print("=" * 75)
    print("DATA SPLIT")
    print("=" * 75)

    print(f"Development records: {len(y_dev)}")
    print(f"Test records:        {len(y_test)}")

    print(
        "Development batches:",
        np.unique(groups_dev).tolist(),
    )

    print(
        "Test batches:",
        np.unique(groups[test_mask]).tolist(),
    )

    print("\nDevelopment class counts:")
    print(
        pd.Series(y_dev)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTest class counts:")
    print(
        pd.Series(y_test)
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Build fresh model instances.
    models = build_models()

    # Step 1: Cross-validation on batches 1-8.
    cv_results = run_cross_validation(
        models,
        X_dev,
        y_dev,
        groups_dev,
    )

    # Step 2: Evaluate all models on batches 9-10.
    test_results = evaluate_unseen_batches(
        models,
        X_dev,
        y_dev,
        X_test,
        y_test,
    )

    # Step 3: Summarize the results.
    print("\n")
    print("=" * 75)
    print("OVERALL SUMMARY")
    print("=" * 75)

    summary = cv_results.merge(
        test_results,
        on="model",
        how="inner",
    )

    summary = summary.sort_values(
        by="unseen_batch_macro_f1",
        ascending=False,
    )

    print(summary.to_string(index=False))

    summary_path = (
        RESULTS_DIR / "gas_model_full_comparison.csv"
    )

    summary.to_csv(summary_path, index=False)

    print(f"\nSaved complete comparison to:\n{summary_path}")

    # Recommend a candidate based on unseen-batch macro F1.
    best_row = summary.iloc[0]

    print("\n")
    print("=" * 75)
    print("CURRENT EXPERIMENTAL MODEL RECOMMENDATION")
    print("=" * 75)

    print(f"Candidate model: {best_row['model']}")
    print(
        "Unseen-batch accuracy: "
        f"{best_row['unseen_batch_accuracy']:.4f}"
    )
    print(
        "Unseen-batch macro F1: "
        f"{best_row['unseen_batch_macro_f1']:.4f}"
    )

    print(
        "\nThis recommendation is based on classification "
        "performance on batches 9-10 only."
    )

    print(
        "It does not establish that the model is suitable "
        "for real-world hazard detection."
    )

    print("\nModel comparison completed successfully.")


if __name__ == "__main__":
    main()
