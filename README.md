# Trinity-Mobility-Assignment

# Sensor Data Classification Pipeline

This repository contains a machine learning pipeline designed to classify sensor data into different machine statuses: `BROKEN`, `NORMAL`, and `RECOVERING`. The project follows a systematic approach, from data loading and cleaning to model training and evaluation.

## Project Overview

The pipeline consists of the following major steps:
1. **Data Loading**: Loads sensor data from a CSV file.
2. **Data Cleaning**: Cleans the dataset by handling missing values, ensuring proper column order, and filtering invalid targets.
3. **Feature Scaling**: Standardizes sensor data to ensure uniformity across features.
4. **Model Training**: Trains and evaluates multiple machine learning models, including Random Forest, Logistic Regression, K-Nearest Neighbors, and Decision Tree.
5. **Results & Evaluation**: Compares model performance using metrics such as accuracy, precision, recall, and F1-score.
6. **Saving Artifacts**: Saves the best-performing model and its associated components for future use.

## Installation

### Prerequisites
Ensure you have the following Python packages installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `joblib`

You can install them using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
