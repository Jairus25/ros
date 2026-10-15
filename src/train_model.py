import os

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Reproducible demonstration data
rng = np.random.default_rng(42)

n_samples = 2000

# Simulated sensor features, in arbitrary units.
# These are NOT real gas concentrations or calibrated sensor readings.
sensor_a = rng.normal(300, 35, n_samples)
sensor_b = rng.normal(250, 30, n_samples)
sensor_c = rng.normal(200, 25, n_samples)

# Synthetic pattern used only to demonstrate classification.
hazard_score = (
    (sensor_a > 355).astype(int)
    + (sensor_b > 295).astype(int)
    + (sensor_c > 240).astype(int)
)

labels = (hazard_score >= 2).astype(int)

data = pd.DataFrame(
    {
        "sensor_a": sensor_a,
        "sensor_b": sensor_b,
        "sensor_c": sensor_c,
        "label": labels,
    }
)

dataset_path = os.path.join(DATA_DIR, "synthetic_sensor_data.csv")
data.to_csv(dataset_path, index=False)

print("=" * 55)
print("INDUSTRIAL HAZARD CLASSIFICATION")
print("SYNTHETIC DEMONSTRATION DATA")
print("=" * 55)

print(f"\nDataset saved to: {dataset_path}")
print(f"Number of samples: {len(data)}")

print("\nClass distribution:")
print(data["label"].map({0: "Normal", 1: "Simulated hazard"}).value_counts())

# Features and target
X = data[["sensor_a", "sensor_b", "sensor_c"]]
y = data["label"]

# Stratified split keeps class proportions similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Train the classifier
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# Evaluate on held-out synthetic samples
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel evaluation on held-out synthetic data")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Simulated hazard"],
        zero_division=0,
    )
)

# Save confusion matrix plot
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Simulated hazard"],
)

display.plot(values_format="d")

plt.title("Synthetic Demonstration: Confusion Matrix")
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(plot_path, dpi=150)
plt.close()

# Save trained model
model_path = os.path.join(MODEL_DIR, "hazard_model.joblib")
joblib.dump(model, model_path)

print(f"\nTrained model saved to: {model_path}")
print(f"Confusion matrix saved to: {plot_path}")

print("\nTraining completed successfully.")
print("This model uses synthetic data and is not validated for real hazards.")
