# Trinity-Mobility-Assignment

# Machine Learning Model Training Pipeline

This repository contains a machine learning pipeline designed to train models for predicting machine status based on sensor data. Below is a breakdown of each step involved in the process.

## ================= DATA LOADING =================

The data is loaded from a CSV file containing sensor readings and machine status. The pipeline ensures that the data is properly loaded, and the initial shape of the dataset is displayed.

- **Data Loading**: The dataset is read from the path `/content/drive/MyDrive/Colab Notebooks/Trinity Mobility/sensor.csv`.
- **Error Handling**: If the data fails to load, an error message is raised.

## ================= DATA CLEANING =================

Data cleaning involves preparing the dataset for model training by:
- Cleaning column names by removing extra spaces and converting them to lowercase.
- Enforcing the correct column order.
- Removing rows with missing target values (`NaN`) in the 'machine_status' column.
- Filling any remaining missing sensor data with forward and backward filling.

This ensures the data is complete and consistent before model training.

## ================= TARGET PROCESSING =================

The target variable 'machine_status' is processed by:
- Stripping any extra whitespace.
- Converting values to uppercase for consistency.
- Filtering out invalid target classes such as empty strings or 'NAN'.
- Encoding the target labels into numerical values using `LabelEncoder`.

Finally, classes with insufficient samples (less than 2) are removed to avoid class imbalance issues.

## ================= FEATURE EXTRACTION =================

The feature extraction step involves separating the timestamp column from the sensor features. 
- **Features (X)**: All sensor readings (52 columns).
- **Target (y)**: The machine status, which is the encoded label.

The dataset is checked to ensure it contains exactly 52 sensor features before proceeding to the next step.

## ================= TRAIN-TEST SPLIT =================

The dataset is split into training and testing sets:
- 70% of the data is used for training, and 30% is reserved for testing.
- The `stratify=y` parameter ensures that the class distribution is preserved in both the training and testing sets.

This ensures that both sets are representative of the overall class distribution.

## ================= FEATURE SCALING =================

Feature scaling is performed using `StandardScaler` to standardize the sensor data:
- The scaler is fitted to the training data and used to transform both training and testing data.
- Any `NaN` or infinite values in the scaled data are replaced using `np.nan_to_num()`.

This ensures that the sensor data is on a comparable scale and ready for model training.

## ================= MODEL TRAINING =================

The following machine learning models are trained:
1. **Random Forest** 
2. **Logistic Regression**
3. **K-Nearest Neighbors**
4. **Decision Tree**

Each model is trained using the scaled training data and evaluated using the test set. The performance of each model is measured by:
- Accuracy
- Precision
- Recall
- F1-score

Additionally, the confusion matrix for each model is plotted to visualize the performance.

## ================= RESULTS & SAVING =================

After training all models, the results are compiled into a DataFrame and sorted by accuracy to provide a clear comparison. The best-performing model is saved along with:
- The scaler used for feature scaling.
- The label encoder used for target encoding.
- The feature order.
- The timestamp column.

These artifacts are saved in a `.joblib` file for future use.

## ================= SAVING TRAINED ARTIFACTS =================

The best model (based on accuracy) is selected and saved with essential components using `joblib`:
- **Model**: The best-performing trained model.
- **Scaler**: The scaler used for feature transformation.
- **Label Encoder**: The encoder used for target label encoding.
- **Feature Order**: The list of features in the dataset.
- **Timestamp Column**: The column used to track the timestamp of sensor readings.

Finally, a summary of the saved artifacts is printed to confirm that the pipeline has executed successfully.

## 🎉 Pipeline executed successfully!

This project provides an end-to-end pipeline for training and evaluating machine learning models on sensor data. The best-performing model and its related components are saved for future predictions.


You can install them using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
