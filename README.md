# Autonomous Industrial Hazard Detection Rover
## Machine Learning Module

An ML module for an autonomous rover designed to assist in identifying
potential hazards in industrial environments, factories, chemical plants,
and other areas where manual inspection may be dangerous.

The project aims to combine sensor data, machine learning, and robotics
to support hazard detection and future integration with ROS 2 Humble.

---

## Table of Contents

1. Project Overview
2. Objectives
3. Features
4. System Architecture
5. Repository Structure
6. Requirements
7. Installation on a New Computer
8. Running the ML Module
9. Trained Models
10. Dataset and Training
11. ROS 2 Humble Integration
12. Troubleshooting
13. Safety and Limitations
14. Future Improvements
15. Contributing

---

## 1. Project Overview

The Autonomous Industrial Hazard Detection Rover is a robotics project
intended to inspect hazardous environments and identify potential dangers
using sensors and machine learning.

The rover is intended to reduce the need for people to enter potentially
dangerous areas during initial inspections.

The machine learning module processes input data and generates predictions
according to the models implemented in this repository.

Potential applications include:

- Gas-related hazard detection
- Sensor-based hazard classification
- Analysis of environmental measurements
- Hazard alerts for a rover operator
- Future integration with autonomous navigation
- Future integration with ROS 2 Humble

Actual detection capabilities depend on the available sensors, training
data, and models implemented in the project.

## 2. Objectives

- Develop a machine learning module for hazard classification.
- Process sensor readings and identify potentially abnormal conditions.
- Reuse trained models without retraining them on every computer.
- Make the software reproducible on another Ubuntu computer.
- Prepare the module for integration with a ROS 2-based rover.
- Establish a foundation for future real-world testing.

## 3. Features

The repository is intended to provide:

- Python-based machine learning functionality.
- Source code for prediction and classification.
- Storage for trained machine learning models.
- Dependency management through requirements.txt.
- A documented installation process.
- A foundation for integration with a robotic platform.

The features that are operational depend on the source files and models
currently included in the repository.

## 4. System Architecture

The intended processing workflow is:

1. Sensors collect environmental measurements.
2. Sensor readings are formatted into the expected model inputs.
3. The ML module processes the input data.
4. The trained model produces a prediction.
5. The prediction can be passed to the rover software.
6. The rover software can report the result to an operator.

A future ROS 2 implementation can use nodes and topics to connect the
sensor drivers, ML module, navigation system, and alert system.

## 5. Repository Structure

The project is organized around the following directories:

    hazard_rover/
    ├── src/
    │   ├── predict_hazard.py
    │   ├── gas_classifier.py
    │   └── other Python source files
    ├── data/
    │   └── datasets, if included
    ├── models/
    │   └── trained model files
    ├── results/
    │   └── evaluation results, if included
    ├── requirements.txt
    ├── .gitignore
    └── README.md

The exact filenames may vary as the project develops.

## 6. Requirements

Recommended development environment:

- Ubuntu Linux
- Python 3
- Git
- Python virtual environment support
- pip
- Compatible versions of the libraries used to train the models

Typical Python dependencies may include:

- NumPy
- pandas
- scikit-learn
- SciPy
- joblib

The actual dependencies are specified in requirements.txt.

ROS 2 is not required to run a standalone Python ML script unless the script
imports ROS 2 packages. ROS 2 must be installed separately when running
the module as part of a ROS 2 application.

## 7. Installation on a New Computer

Follow these steps on a new Ubuntu computer.

### Step 1: Install the basic tools

Open a terminal and run:

    sudo apt update
    sudo apt install -y git python3 python3-pip python3-venv

### Step 2: Clone the GitHub repository

Replace YOUR_GITHUB_USERNAME and YOUR_REPOSITORY with the correct GitHub
account and repository name.

    git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY

If the repository is private, you must have permission to access it.

### Step 3: Create a virtual environment

Run:

    python3 -m venv .venv

Activate it:

    source .venv/bin/activate

A successful activation normally displays (.venv) in the terminal prompt.

### Step 4: Upgrade pip

    python -m pip install --upgrade pip

### Step 5: Install Python dependencies

    python -m pip install -r requirements.txt

This installs the packages listed in requirements.txt into the virtual
environment.

### Step 6: Check the project files

List the Python source files:

    find src -maxdepth 2 -type f -name '*.py' -print

List the model files:

    find models -maxdepth 2 -type f -print

Check the installed packages:

    python -m pip list

### Step 7: Activate the environment whenever you return

Whenever you open a new terminal to work on the project, run:

    cd YOUR_REPOSITORY
    source .venv/bin/activate

The virtual environment must be activated before running scripts that
depend on the installed project packages.

### Quick Installation Summary

After installing Git and Python, the normal setup sequence is:

    git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Replace the example GitHub URL with the actual repository URL before use.

## 8. Running the ML Module

The repository may contain scripts such as:

- src/predict_hazard.py
- src/gas_classifier.py

Inspect the available scripts:

    find src -maxdepth 2 -type f -name '*.py' -print

Before running a script, check its documentation or source code to determine
the expected arguments and input format.

If the script implements command-line help, you can try:

    python src/predict_hazard.py --help

For a gas classification script that implements command-line help:

    python src/gas_classifier.py --help

These commands are applicable only if the corresponding scripts exist and
support the --help option.

The required prediction inputs depend on the model implementation.
Do not assume that arbitrary sensor readings can be passed directly to
a model without preprocessing.

## 9. Trained Models

The models/ directory stores trained models used by the ML module.

Keeping trained models in the repository allows another computer to reuse
them without repeating the training process, provided the required model
files are included and accessible.

To list the available models:

    find models -maxdepth 2 -type f -print

### Model compatibility

Models saved using joblib or pickle may depend on specific versions of
Python libraries, especially scikit-learn.

For reliable model loading:

1. Use a compatible Python version.
2. Install compatible library versions.
3. Preserve the original model files.
4. Use the correct model path in the prediction script.
5. Test that the model loads before processing sensor data.

Models should only be loaded from trusted sources.

If a model fails to load, check the dependency versions and model path.

## 10. Dataset and Training

The data/ directory is intended to contain datasets used for training,
testing, or evaluating the machine learning models.

To inspect the available files:

    find data -maxdepth 3 -type f -print

Training and evaluation commands depend on the scripts included in src/.

Before training, verify:

- The dataset exists.
- The expected input features are present.
- Labels are formatted correctly.
- Training and testing data are separated appropriately.
- The training script uses the intended dataset path.

If a model is already trained and included in models/, retraining should
not be necessary for normal inference.

Large datasets may be stored separately if they are unsuitable for GitHub.
If so, document where they can be obtained and how they should be arranged.

## 11. ROS 2 Humble Integration

ROS 2 Humble is the intended robotics integration environment for this
project.

The standalone ML module and a ROS 2 package are separate components unless
the repository explicitly implements both.

### ROS 2 setup

On a computer where ROS 2 Humble is installed, open a terminal and source
the ROS 2 environment:

    source /opt/ros/humble/setup.bash

Verify the installation:

    printenv ROS_DISTRO

The expected distribution name is:

    humble

If ROS 2 Humble is not installed, install it using the official ROS 2
installation instructions for a supported operating system.

Official documentation:

https://docs.ros.org/en/humble/

### Integration workflow

A ROS 2 integration may follow this architecture:

1. A sensor driver publishes sensor readings.
2. An ML node subscribes to the appropriate sensor topic.
3. The node preprocesses incoming measurements.
4. The trained model generates a prediction.
5. The node publishes the prediction or an alert.
6. The rover software uses the result for monitoring or further action.

The actual topic names, message types, node names, and launch commands must
match the ROS 2 implementation.

### Python environment compatibility

ROS 2 Humble is commonly used with Ubuntu 22.04 and its system Python.

Creating a virtual environment with an incompatible Python interpreter
or installing conflicting dependencies can prevent ROS 2 Python packages
from loading correctly.

Before combining the ML environment with ROS 2, verify compatibility
between the Python interpreter, installed ROS 2 packages, and ML libraries.

Do not assume that a standalone ML script is already a working ROS 2 node.

## 12. Troubleshooting

### Problem: git command not found

Install Git:

    sudo apt update
    sudo apt install -y git

### Problem: python3 or venv is missing

Run:

    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv

### Problem: Permission denied when cloning

Check the repository URL and verify that your GitHub account has permission
to access the repository.

For private repositories, configure GitHub authentication.

Never place a personal access token directly into a README or commit it
to the repository.

### Problem: No module named ...

Activate the virtual environment:

    source .venv/bin/activate

Then install dependencies:

    python -m pip install -r requirements.txt

### Problem: requirements installation fails

Check your Python version:

    python3 --version

Upgrade pip:

    python -m pip install --upgrade pip

Review the package installation error and check whether the package versions
support your Python version and operating system.

### Problem: model file not found

List the model files:

    find models -maxdepth 3 -type f -print

Verify that the required model exists and that the script uses the correct
relative or absolute path.

### Problem: model loading error

Check that the model file is intact and that the installed versions of
scikit-learn, joblib, and other relevant libraries are compatible with
the versions used during training.

### Problem: script fails because an input file is missing

Check the script's expected input format and confirm that the required
dataset or configuration file is available.

### Problem: ROS 2 commands are unavailable

Source the ROS 2 environment:

    source /opt/ros/humble/setup.bash

If this file does not exist, verify that ROS 2 Humble is installed correctly.

## 13. Safety and Limitations

This project is intended for educational research and prototyping.

Machine learning predictions alone are not sufficient to establish that
an industrial environment is safe.

Important limitations include:

- Accuracy depends on the quality and coverage of the training data.
- Synthetic demonstration data does not establish real-world accuracy.
- Gas detection requires suitable physical sensors and calibration.
- Electrical hazard detection requires appropriate sensing hardware.
- Radioactive material detection requires suitable radiation detection
  equipment.
- Pothole and terrain detection require suitable sensors and validated
  perception methods.
- Sensor faults, environmental conditions, and unexpected inputs can affect
  predictions.
- Model outputs must not be treated as certified safety measurements.

Before real-world deployment, validate the system using appropriate equipment,
independent testing, fail-safe mechanisms, and qualified safety procedures.

Do not use this prototype as the sole protection against hazardous conditions.

## 14. Future Improvements

Potential future developments include:

- ROS 2 Humble node integration.
- Real-time sensor data processing.
- Gas sensor integration and calibration.
- Improved model evaluation using real measurements.
- Terrain and pothole detection.
- Obstacle detection and autonomous navigation.
- Hazard alerts and event logging.
- A monitoring dashboard.
- Robust error handling and sensor fault detection.
- Reproducible dependency and model version management.

These are planned improvements and should not be interpreted as features
that are already implemented.

## 15. Contributing

Contributions and improvements are welcome.

Suggested workflow:

1. Fork or clone the repository.
2. Create a separate branch for your changes.
3. Test the changes in a suitable Python environment.
4. Update the documentation when behavior changes.
5. Submit a pull request for review.

Keep trained model files, datasets, and dependency specifications organized.

Do not commit credentials, access tokens, private configuration files,
or unrelated virtual environment files.

---

## Project Status

This repository contains the machine learning components currently included
in the project.

The actual capabilities depend on the source code, trained models, datasets,
and connected sensors available in the repository.

Installation instructions should be updated whenever dependencies, model
formats, script interfaces, or ROS 2 integration requirements change.
