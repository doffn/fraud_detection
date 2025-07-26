"""
Utility Functions
Common helper functions used across the project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def create_sample_fraud_data(n_samples=10000, fraud_rate=0.05, random_state=42):
    """Create sample fraud detection dataset for testing"""
    np.random.seed(random_state)
    
    # Generate base features
    data = {
        'user_id': range(1, n_samples + 1),
        'signup_time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'purchase_time': pd.date_range('2024-01-02', periods=n_samples, freq='2H'),
        'purchase_value': np.random.exponential(50, n_samples),
        'device_id': np.random.randint(1, n_samples//2, n_samples),
        'source': np.random.choice(['SEO', 'Ads', 'Direct', 'Social'], n_samples),
        'browser': np.random.choice(['Chrome', 'Safari', 'Firefox', 'Edge'], n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'ip_address': [f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}" 
                      for _ in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic fraud labels
    fraud_prob = (
        fraud_rate +
        0.05 * (df['purchase_value'] > df['purchase_value'].quantile(0.95)) +
        0.03 * (df['age'] < 25) +
        0.02 * (df['source'] == 'Ads')
    )
    
    df['class'] = np.random.binomial(1, fraud_prob, n_samples)
    
    return df

def create_sample_ip_country_data():
    """Create sample IP to country mapping data"""
    data = {
        'lower_bound_ip_address': [16777216, 33554432, 50331648, 67108864, 83886080],
        'upper_bound_ip_address': [33554431, 50331647, 67108863, 83886079, 100663295],
        'country': ['USA', 'Canada', 'UK', 'Germany', 'France']
    }
    
    return pd.DataFrame(data)

def plot_class_distribution(y, title="Class Distribution"):
    """Plot class distribution"""
    plt.figure(figsize=(8, 6))
    
    class_counts = pd.Series(y).value_counts()
    colors = ['lightblue', 'red']
    
    plt.pie(class_counts.values, labels=['Non-Fraud', 'Fraud'], 
            autopct='%1.1f%%', colors=colors)
    plt.title(title)
    plt.show()
    
    print(f"Class distribution:")
    print(f"Non-Fraud: {class_counts[0]:,} ({class_counts[0]/len(y)*100:.2f}%)")
    print(f"Fraud: {class_counts[1]:,} ({class_counts[1]/len(y)*100:.2f}%)")

def plot_feature_distributions(df, features, target_col='class', figsize=(15, 10)):
    """Plot feature distributions by class"""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, feature in enumerate(features):
        if feature in df.columns:
            for class_val in df[target_col].unique():
                data = df[df[target_col] == class_val][feature]
                label = 'Fraud' if class_val == 1 else 'Non-Fraud'
                color = 'red' if class_val == 1 else 'blue'
                alpha = 0.7
                
                axes[i].hist(data, bins=30, alpha=alpha, label=label, color=color)
            
            axes[i].set_title(f'{feature} Distribution')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def calculate_fraud_rates(df, categorical_features, target_col='class'):
    """Calculate fraud rates for categorical features"""
    fraud_rates = {}
    
    for feature in categorical_features:
        if feature in df.columns:
            rates = df.groupby(feature)[target_col].agg(['count', 'sum', 'mean'])
            rates.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
            rates = rates.sort_values('fraud_rate', ascending=False)
            fraud_rates[feature] = rates
    
    return fraud_rates

def print_model_comparison(results_list):
    """Print formatted model comparison"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    comparison_data = []
    for result in results_list:
        comparison_data.append({
            'Model': result['model_name'],
            'Train AUC-ROC': f"{result['train_auc_roc']:.4f}",
            'Test AUC-ROC': f"{result['test_auc_roc']:.4f}",
            'Train AUC-PR': f"{result['train_auc_pr']:.4f}",
            'Test AUC-PR': f"{result['test_auc_pr']:.4f}",
            'CV AUC-ROC': f"{result['cv_auc_roc_mean']:.4f} ± {result['cv_auc_roc_std']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Determine best model
    best_idx = np.argmax([result['test_auc_pr'] for result in results_list])
    best_model = results_list[best_idx]['model_name']
    best_score = results_list[best_idx]['test_auc_pr']
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"🎯 BEST AUC-PR SCORE: {best_score:.4f}")
    print("="*80)

def save_results_to_csv(results_dict, filepath):
    """Save results dictionary to CSV"""
    try:
        # Convert results to DataFrame format
        results_df = pd.DataFrame([results_dict])
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results: {e}")

def load_config(config_path):
    """Load configuration from file"""
    # This would typically load from a JSON or YAML file
    # For now, return default configuration
    default_config = {
        'data': {
            'test_size': 0.2,
            'random_state': 42,
            'fraud_data_path': 'data/Fraud_Data.csv',
            'ip_country_path': 'data/IpAddress_to_Country.csv'
        },
        'models': {
            'use_smote': True,
            'logistic_regression': {
                'max_iter': 1000,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'evaluation': {
            'cv_folds': 5,
            'scoring': 'roc_auc'
        },
        'explainability': {
            'max_samples': 500,
            'save_plots': True,
            'plots_dir': 'plots/'
        }
    }
    
    return default_config
