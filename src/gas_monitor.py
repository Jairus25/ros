import time
from dataclasses import dataclass
from datetime import datetime


# ==========================================================
# CONFIGURATION
# ==========================================================

# Example threshold for software testing only.
# Replace this with a gas-specific, validated safety limit
# when integrating a real sensor.
TEST_ALARM_THRESHOLD_PPM = 100.0

READING_INTERVAL_SECONDS = 2


# ==========================================================
# SENSOR DATA
# ==========================================================

@dataclass
class GasReading:
    gas_name: str
    concentration_ppm: float
    timestamp: str


# ==========================================================
# GAS MONITOR
# ==========================================================

class GasMonitor:

    def __init__(self, alarm_threshold_ppm):
        if alarm_threshold_ppm <= 0:
            raise ValueError(
                "Alarm threshold must be greater than zero."
            )

        self.alarm_threshold_ppm = float(
            alarm_threshold_ppm
        )

    def process_reading(self, reading):
        """
        Validate one reading and determine its monitoring status.

        This function checks the configured threshold.
        It does not identify the gas or validate sensor accuracy.
        """

        if not reading.gas_name.strip():
            raise ValueError("Gas name cannot be empty.")

        concentration = reading.concentration_ppm

        if concentration < 0:
            raise ValueError(
                "Gas concentration cannot be negative."
            )

        if concentration >= self.alarm_threshold_ppm:
            status = "ALARM"
            message = (
                "Configured threshold reached or exceeded."
            )
        else:
            status = "NORMAL"
            message = (
                "Reading is below the configured test threshold."
            )

        return {
            "timestamp": reading.timestamp,
            "gas_name": reading.gas_name,
            "concentration_ppm": concentration,
            "threshold_ppm": self.alarm_threshold_ppm,
            "status": status,
            "message": message,
        }


# ==========================================================
# SIMULATED SENSOR
# ==========================================================

class SimulatedGasSensor:
    """
    Generates artificial readings for software testing.

    These values are not measurements from a real sensor.
    """

    def __init__(self, readings):
        self.readings = list(readings)
        self.index = 0

    def read(self):

        if self.index >= len(self.readings):
            return None

        concentration = self.readings[self.index]
        self.index += 1

        return GasReading(
            gas_name="SIMULATED_GAS",
            concentration_ppm=float(concentration),
            timestamp=datetime.now().isoformat(
                timespec="seconds"
            ),
        )


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():

    print("=" * 65)
    print("AUTONOMOUS INDUSTRIAL HAZARD DETECTION ROVER")
    print("Gas Monitoring Prototype")
    print("=" * 65)

    print("\nMode: SIMULATION")
    print(
        "Warning: displayed readings are artificial test values."
    )

    monitor = GasMonitor(
        alarm_threshold_ppm=TEST_ALARM_THRESHOLD_PPM
    )

    # Artificial values chosen to demonstrate normal and alarm
    # states. They are not calibrated sensor measurements.
    simulated_readings = [
        12.0,
        18.0,
        25.0,
        45.0,
        75.0,
        105.0,
        135.0,
        60.0,
        30.0,
    ]

    sensor = SimulatedGasSensor(
        readings=simulated_readings
    )

    print(
        f"\nExample software threshold: "
        f"{TEST_ALARM_THRESHOLD_PPM:.1f} ppm"
    )

    print("-" * 65)

    try:
        while True:

            reading = sensor.read()

            if reading is None:
                print("\nSimulation completed.")
                break

            result = monitor.process_reading(reading)

            print(f"\nTime: {result['timestamp']}")
            print(f"Gas: {result['gas_name']}")
            print(
                "Concentration: "
                f"{result['concentration_ppm']:.2f} ppm"
            )
            print(
                "Test threshold: "
                f"{result['threshold_ppm']:.2f} ppm"
            )
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")

            if result["status"] == "ALARM":
                print("!!! GAS THRESHOLD ALERT !!!")

            print("-" * 65)

            time.sleep(READING_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

    print("\nProgram terminated.")


if __name__ == "__main__":
    main()
