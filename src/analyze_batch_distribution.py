from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/real_gas_sensor/Dataset")
OUTPUT = Path("results")
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_batch(path):
    X = []
    y = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            fields = line.split()

            if not fields:
                continue

            y.append(int(float(fields[0])))

            row = np.full(128, np.nan)

            for item in fields[1:]:
                index, value = item.split(":")
                row[int(index) - 1] = float(value)

            X.append(row)

    return np.asarray(X), np.asarray(y)


summary = []
class_counts = []

for batch_number in range(1, 11):
    path = DATA_DIR / f"batch{batch_number}.dat"
    X, y = load_batch(path)

    row = {
        "batch": batch_number,
        "records": len(y),
        "features": X.shape[1],
    }

    for label in range(1, 7):
        row[f"class_{label}_count"] = int(np.sum(y == label))

    for feature in range(X.shape[1]):
        row[f"feature_{feature + 1}_median"] = float(
            np.nanmedian(X[:, feature])
        )

    summary.append(row)

    for label in range(1, 7):
        class_counts.append({
            "batch": batch_number,
            "class_label": label,
            "count": int(np.sum(y == label)),
        })

summary_df = pd.DataFrame(summary)
counts_df = pd.DataFrame(class_counts)

summary_df.to_csv(OUTPUT / "batch_feature_summary.csv", index=False)
counts_df.to_csv(OUTPUT / "batch_class_counts.csv", index=False)

print("Records and class counts by batch:")
print(
    counts_df.pivot(
        index="batch",
        columns="class_label",
        values="count",
    ).fillna(0).astype(int)
)

print("\nMedian of selected sensor features by batch:")
selected = [
    "batch",
    "records",
    "feature_1_median",
    "feature_9_median",
    "feature_65_median",
    "feature_128_median",
]
print(summary_df[selected].to_string(index=False))

print("\nSaved:")
print(OUTPUT / "batch_feature_summary.csv")
print(OUTPUT / "batch_class_counts.csv")
