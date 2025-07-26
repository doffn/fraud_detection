"""
Model Training Module
Handles model building, training, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Main class for model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        
    def prepare_features(self, df, target_col='class'):
        """Prepare features for modeling"""
        logger.info("Preparing features for modeling...")
        
        # Define feature columns
        numerical_features = [
            'purchase_value', 'age', 'time_since_signup', 'hour_of_day',
            'day_of_week', 'is_weekend', 'user_transaction_count',
            'user_total_value', 'user_avg_value', 'device_transaction_count',
            'device_total_value', 'device_avg_value', 'device_unique_users',
            'high_value_transaction', 'new_user'
        ]
        
        categorical_features = ['source', 'browser', 'sex', 'country']
        
        # Encode categorical features
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                le = LabelEncoder()
                df[f'{cat_feature}_encoded'] = le.fit_transform(df[cat_feature].astype(str))
                numerical_features.append(f'{cat_feature}_encoded')
                self.label_encoders[cat_feature] = le
        
        # Select available features
        available_features = [col for col in numerical_features if col in df.columns]
        X = df[available_features]
        y = df[target_col]
        
        logger.info(f"Features prepared: {len(available_features)} features")
        return X, y, available_features
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Features scaled")
        return X_train_scaled, X_test_scaled
    
    def train_logistic_regression(self, X_train, y_train, use_smote=True):
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression...")
        
        if use_smote:
            smote = SMOTE(random_state=42)
            model = LogisticRegression(random_state=42, max_iter=1000)
            pipeline = ImbPipeline([('smote', smote), ('classifier', model)])
        else:
            pipeline = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        
        pipeline.fit(X_train, y_train)
        self.models['logistic_regression'] = pipeline
        
        logger.info("Logistic Regression trained")
        return pipeline
    
    def train_random_forest(self, X_train, y_train, use_smote=True):
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        
        if use_smote:
            smote = SMOTE(random_state=42)
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            pipeline = ImbPipeline([('smote', smote), ('classifier', model)])
        else:
            pipeline = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
            )
        
        pipeline.fit(X_train, y_train)
        self.models['random_forest'] = pipeline
        
        logger.info("Random Forest trained")
        return pipeline
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate model performance"""
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        train_auc_roc = roc_auc_score(y_train, y_train_proba)
        test_auc_roc = roc_auc_score(y_test, y_test_proba)
        train_auc_pr = average_precision_score(y_train, y_train_proba)
        test_auc_pr = average_precision_score(y_test, y_test_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        results = {
            'model_name': model_name,
            'train_auc_roc': train_auc_roc,
            'test_auc_roc': test_auc_roc,
            'train_auc_pr': train_auc_pr,
            'test_auc_pr': test_auc_pr,
            'cv_auc_roc_mean': cv_scores.mean(),
            'cv_auc_roc_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_test_pred),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        logger.info(f"{model_name} evaluation completed")
        return results
    
    def save_model(self, model, model_name, filepath):
        """Save trained model"""
        joblib.dump(model, filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
