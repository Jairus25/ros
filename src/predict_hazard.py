import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "hazard_model.joblib")

model = joblib.load(MODEL_PATH)


def predict_hazard(sensor_a, sensor_b, sensor_c):
    readings = pd.DataFrame(
        [[sensor_a, sensor_b, sensor_c]],
        columns=["sensor_a", "sensor_b", "sensor_c"],
    )

    prediction = int(model.predict(readings)[0])

    probabilities = model.predict_proba(readings)[0]

    if prediction == 1:
        result = "SIMULATED HAZARD PATTERN"
    else:
        result = "NORMAL PATTERN"

    print("\nSensor readings")
    print(f"Sensor A: {sensor_a}")
    print(f"Sensor B: {sensor_b}")
    print(f"Sensor C: {sensor_c}")

    print(f"\nClassification: {result}")
    print(f"Model probability for simulated hazard: {probabilities[1]:.2%}")

    print(
        "\nNote: This is a synthetic-data demonstration, "
        "not a real-world safety assessment."
    )


if __name__ == "__main__":
    print("Industrial Hazard Rover - ML Demonstration")
    print("Enter three simulated sensor readings.")

    try:
        a = float(input("Sensor A: "))
        b = float(input("Sensor B: "))
        c = float(input("Sensor C: "))

        predict_hazard(a, b, c)

    except ValueError:
        print("Invalid input. Please enter numeric values.")
