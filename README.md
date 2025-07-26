# Fraud Detection Project
## 10 Academy: Artificial Intelligence Mastery Week 8&9 Challenge

### Project Overview

This project implements a comprehensive fraud detection system for e-commerce and bank transactions using advanced machine learning techniques. The system includes data preprocessing, feature engineering, model building, evaluation, and explainability analysis using SHAP.

### 🎯 Objectives

- Develop accurate fraud detection models for e-commerce and bank transactions
- Handle class imbalance using appropriate sampling techniques
- Implement geolocation analysis and transaction pattern recognition
- Provide model explainability using SHAP analysis
- Balance security and user experience considerations

### 📊 Datasets

1. **Fraud_Data.csv**: E-commerce transaction data
   - Features: user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class
   - Challenge: Highly imbalanced dataset

2. **IpAddress_to_Country.csv**: IP address to country mapping
   - Features: lower_bound_ip_address, upper_bound_ip_address, country

3. **creditcard.csv**: Bank transaction data (PCA-transformed features)
   - Features: Time, V1-V28 (anonymized), Amount, Class
   - Challenge: Extremely imbalanced dataset

### 🏗️ Project Structure

\`\`\`
fraud-detection-project/
├── data/                          # Data files
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_preprocessing_and_eda.py
│   ├── 02_model_building_and_evaluation.py
│   └── 03_model_explainability_shap.py
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_trainer.py
│   ├── explainability.py
│   └── utils.py
├── models/                       # Saved models
├── plots/                        # Generated plots and visualizations
├── results/                      # Analysis results and reports
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── main.py                       # Main execution script
\`\`\`

### 🚀 Getting Started

#### Prerequisites

- Python 3.8 or higher
- pip package manager

#### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/your-username/fraud-detection-project.git
cd fraud-detection-project
\`\`\`

2. Create a virtual environment:
\`\`\`bash
python -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # On Windows: fraud_detection_env\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Run the Flask application:
\`\`\`bash
cd app
python main.py
\`\`\`

5. Open your browser and navigate to:
\`\`\`
http://localhost:5000
\`\`\`

#### Using the Web Interface

The Flask application provides three main pages:

1. **Dashboard** (`/`): Overview of fraud detection metrics and visualizations
2. **Test Model** (`/test`): Interactive form to test the fraud detection model
3. **Analytics** (`/analytics`): Model performance metrics and feature importance

#### Features

- **Simple Dashboard**: Key metrics and fraud pattern visualizations
- **Interactive Testing**: Real-time fraud risk scoring for individual transactions
- **Model Analytics**: Feature importance and performance metrics
- **Responsive Design**: Works on desktop and mobile devices

#### Running the Analysis

1. **Data Preprocessing and EDA**:
\`\`\`bash
python notebooks/01_data_preprocessing_and_eda.py
\`\`\`

2. **Model Building and Evaluation**:
\`\`\`bash
python notebooks/02_model_building_and_evaluation.py
\`\`\`

3. **Model Explainability Analysis**:
\`\`\`bash
python notebooks/03_model_explainability_shap.py
\`\`\`

4. **Complete Pipeline**:
\`\`\`bash
python main.py
\`\`\`

### 📈 Key Features

#### Data Preprocessing
- Missing value handling and data cleaning
- IP address to country mapping
- Time-based feature engineering
- Transaction frequency and velocity features
- Class imbalance analysis

#### Feature Engineering
- **Time Features**: hour_of_day, day_of_week, time_since_signup
- **User Features**: transaction count, total value, average value
- **Device Features**: transaction count, unique users per device
- **Risk Indicators**: high_value_transaction, new_user flags
- **Geolocation**: country mapping from IP addresses

#### Model Building
- **Logistic Regression**: Interpretable baseline model
- **Random Forest**: Powerful ensemble model
- **SMOTE**: Synthetic minority oversampling for class balance
- **Cross-validation**: 5-fold stratified cross-validation

#### Evaluation Metrics
- AUC-ROC (Area Under ROC Curve)
- AUC-PR (Area Under Precision-Recall Curve)
- Precision, Recall, F1-Score
- Confusion Matrix analysis
- Business impact analysis

#### Model Explainability
- **SHAP Analysis**: Feature importance and impact direction
- **Summary Plots**: Global feature importance visualization
- **Waterfall Plots**: Individual prediction explanations
- **Force Plots**: Local feature contributions
- **Business Insights**: Actionable recommendations

### 📊 Results Summary

The project delivers:

1. **Comprehensive EDA**: Deep insights into fraud patterns
2. **Feature Engineering**: 15+ engineered features for better detection
3. **Model Performance**: AUC-PR scores optimized for imbalanced data
4. **Explainability**: SHAP analysis for model transparency
5. **Business Impact**: ROI analysis and cost-benefit evaluation

### 🔍 Key Insights

Based on SHAP analysis, the most important fraud indicators are:

1. **High-value transactions**: Strong predictor of fraud risk
2. **New user status**: Recent signups pose higher risk
3. **Transaction timing**: Certain hours show elevated fraud rates
4. **User behavior patterns**: Frequency and velocity anomalies
5. **Geolocation**: Country-based risk patterns

### 📋 Business Recommendations

1. **Real-time Monitoring**: Implement alerts for high-value transactions
2. **New User Verification**: Additional checks for recent signups
3. **Time-based Rules**: Enhanced monitoring during high-risk hours
4. **Behavioral Analytics**: Track user and device patterns
5. **Geographic Filtering**: Country-based risk assessment

### 🛠️ Technical Implementation

#### Class Imbalance Handling
- **SMOTE**: Synthetic minority oversampling
- **Evaluation Focus**: AUC-PR over accuracy
- **Cost-sensitive Learning**: Balanced class weights

#### Model Selection Criteria
- **Performance**: AUC-PR score prioritized
- **Interpretability**: SHAP explainability
- **Business Impact**: ROI and cost considerations
- **Scalability**: Real-time prediction capability

### 📝 Usage Examples

#### Quick Start Example
\`\`\`python
from src.data_preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.explainability import ModelExplainer

# Initialize components
preprocessor = DataPreprocessor()
trainer = ModelTrainer()
explainer = ModelExplainer()

# Load and preprocess data
fraud_df = preprocessor.load_fraud_data('data/Fraud_Data.csv')
fraud_df_clean = preprocessor.clean_fraud_data(fraud_df)

# Train models
X, y, features = trainer.prepare_features(fraud_df_clean)
X_train, X_test, y_train, y_test = trainer.split_data(X, y)
model = trainer.train_random_forest(X_train, y_train)

# Explain model
tree_explainer = explainer.create_tree_explainer(model, 'random_forest')
shap_values, X_sample = explainer.calculate_shap_values(tree_explainer, X_test, 'random_forest')
