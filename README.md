# Data Confidentiality Master Thesis

This project investigates and compares advanced Privacy-Preserving Machine Learning (PPML) techniques, focusing specifically on the trade-offs between **Differential Privacy (DP)** and **Fully Homomorphic Encryption (FHE)**.

## Project Overview

The primary goal of this research is to analyze the relationship between utility (Accuracy/MSE) and performance (Execution Time) using tabular datasets such as the Adult and Communities datasets.

### Technologies Implemented

* **Differential Privacy:** Utilizes **IBM Diffprivlib** and **Opacus** to protect training data.
* **Homomorphic Encryption:** Utilizes **Zama Concrete ML** to protect data during the inference phase.

## System Requirements

> **Important Note:** Due to dependencies on specific Linux libraries required by `concrete-ml`, this project **does not** run natively on Windows.

To run this project, you must use one of the following environments:

* **Operating System:** Linux (Ubuntu 20.04 or higher) or **WSL2** (Windows Subsystem for Linux).
* **Python Version:** Python **3.10** or **3.11**.
  * *Note: While tested on Python 3.12, strict compatibility with `numpy < 2.0` is required.*

## Installation and Usage

Follow the steps below to set up the environment and run the experiments.

### 1. Clone the Repository

```bash
git clone [https://github.com/pirjolmihaita/data-confidentiality-thesis.git](https://github.com/pirjolmihaita/data-confidentiality-thesis.git)
cd data-confidentiality-thesis

2. Create a Virtual Environment
It is highly recommended to use a virtual environment within your Linux or WSL terminal.

Bash

python3 -m venv venv_wsl
source venv_wsl/bin/activate
3. Install Dependencies
Install the required Python packages listed in the requirements file.

Bash

pip install -r requirements.txt
4. Run Experiments
Execute the main script to start the analysis and generate results.

Bash

python3 main.py
Project Structure
src/: Contains the source code for the machine learning models and experimental logic.

results/: Destination folder for the generated CSV files containing performance metrics.

main.py: The entry point for the application.
