# Fraud Detection Project  
## 10 Academy – AI Mastery Week 8 & 9 Challenge

## Overview

This project implements a robust fraud detection system for e-commerce and banking transactions using advanced machine learning techniques. It covers data preprocessing, feature engineering, model training, evaluation, and explainability using SHAP.

---

## Objectives

- Build accurate fraud detection models for e-commerce and bank datasets  
- Address class imbalance using appropriate resampling techniques  
- Perform geolocation and behavioral transaction analysis  
- Provide interpretable results using SHAP  
- Optimize for both fraud detection and user experience

---

## Datasets

1. **Fraud_Data.csv** – E-commerce transaction data  
   - Features: `user_id`, `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`, `class`  
   - Challenge: Highly imbalanced

2. **IpAddress_to_Country.csv** – IP-to-country mapping  
   - Features: `lower_bound_ip_address`, `upper_bound_ip_address`, `country`

3. **creditcard.csv** – Bank transactions with PCA-transformed features  
   - Features: `Time`, `V1`–`V28` (anonymized), `Amount`, `Class`  
   - Challenge: Extremely imbalanced

---

## Project Structure

```

fraud-detection-project/
├── data/                         # Dataset files
├── notebooks/                    # Development notebooks
│   ├── 01\_data\_preprocessing\_and\_eda.py
│   ├── 02\_model\_building\_and\_evaluation.py
│   └── 03\_model\_explainability\_shap.py
├── src/                          # Core modules
│   ├── data\_preprocessing.py
│   ├── model\_trainer.py
│   ├── explainability.py
│   └── utils.py
├── models/                       # Trained models
├── plots/                        # Visualizations
├── results/                      # Output reports and metrics
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── main.py                       # Main pipeline runner

````

---

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` package manager

### Installation

1. Clone the repo:
```bash
git clone https://github.com/your-username/fraud-detection-project.git
cd fraud-detection-project
````

2. Create and activate a virtual environment:

```bash
python -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # On Windows: fraud_detection_env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create folders for storing outputs (if not already present):

```bash
mkdir -p data models plots results
```

---

## Running the Analysis

1. **Preprocessing and Exploratory Data Analysis**

```bash
python notebooks/01_data_preprocessing_and_eda.py
```

2. **Model Training and Evaluation**

```bash
python notebooks/02_model_building_and_evaluation.py
```

3. **Model Explainability (SHAP)**

```bash
python notebooks/03_model_explainability_shap.py
```

4. **Full Pipeline Execution**

```bash
python main.py
```

---

## Key Features

### Data Preprocessing

* Handle missing values and clean inputs
* IP-to-country mapping
* Feature engineering based on time, frequency, velocity
* Analyze and visualize class imbalance

### Feature Engineering

* Time-based: Hour of day, day of week, time since signup
* User behavior: Transaction count, average value
* Device patterns: Unique users per device
* Geolocation: Risk by country
* Risk flags: New user, high-value transaction

### Model Building

* Logistic Regression – Interpretable baseline
* Random Forest – Robust ensemble model
* SMOTE – Synthetic oversampling
* Stratified 5-fold Cross-Validation

### Evaluation Metrics

* AUC-ROC and AUC-PR
* Precision, Recall, F1-score
* Confusion matrix
* Business impact estimation

### Model Explainability

* SHAP summary and force plots
* Feature contributions globally and locally
* Waterfall charts for individual predictions
* Actionable insights for fraud teams

---

## Results Summary

* Comprehensive EDA: In-depth insights into fraud patterns
* Extensive Feature Engineering: 15+ derived features
* Strong Model Performance: Optimized for imbalanced data
* Explainable Models: Transparent decisions using SHAP
* Business-Oriented Output: Supports ROI and risk analysis

---

## Key Insights

* High-value transactions are strong fraud indicators
* New users show higher likelihood of fraud
* Fraud rates vary significantly by time of day
* Behavioral anomalies suggest fraudulent behavior
* Geographic location is a relevant risk factor

---

## Business Recommendations

1. Implement real-time alerts for high-value transactions
2. Use additional verification for new signups
3. Enhance monitoring during high-risk time windows
4. Analyze user and device patterns for anomalies
5. Use country-based filtering to mitigate risk

---
# fraud_detection
# fraud_detection
