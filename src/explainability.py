"""
Model Explainability Module
Handles SHAP analysis and model interpretation
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Main class for model explainability using SHAP"""
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        
    def create_tree_explainer(self, model, model_name):
        """Create SHAP TreeExplainer for tree-based models"""
        logger.info(f"Creating TreeExplainer for {model_name}...")
        
        # Extract the actual model from pipeline if needed
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['classifier']
        else:
            actual_model = model
            
        explainer = shap.TreeExplainer(actual_model)
        self.explainers[model_name] = explainer
        
        logger.info(f"TreeExplainer created for {model_name}")
        return explainer
    
    def create_linear_explainer(self, model, X_train, model_name):
        """Create SHAP LinearExplainer for linear models"""
        logger.info(f"Creating LinearExplainer for {model_name}...")
        
        # Extract the actual model from pipeline if needed
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['classifier']
        else:
            actual_model = model
            
        explainer = shap.LinearExplainer(actual_model, X_train)
        self.explainers[model_name] = explainer
        
        logger.info(f"LinearExplainer created for {model_name}")
        return explainer
    
    def calculate_shap_values(self, explainer, X_test, model_name, max_samples=500):
        """Calculate SHAP values"""
        logger.info(f"Calculating SHAP values for {model_name}...")
        
        # Use subset for performance
        X_test_sample = X_test[:max_samples] if len(X_test) > max_samples else X_test
        
        shap_values = explainer.shap_values(X_test_sample)
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
            
        self.shap_values[model_name] = {
            'values': shap_values,
            'data': X_test_sample
        }
        
        logger.info(f"SHAP values calculated for {model_name}")
        return shap_values, X_test_sample
    
    def create_summary_plots(self, model_name, feature_names, save_path=None):
        """Create SHAP summary plots"""
        logger.info(f"Creating summary plots for {model_name}...")
        
        shap_data = self.shap_values[model_name]
        shap_values = shap_data['values']
        X_test_sample = shap_data['data']
        
        # Convert to DataFrame if needed
        if isinstance(X_test_sample, np.ndarray):
            X_test_df = pd.DataFrame(X_test_sample, columns=feature_names)
        else:
            X_test_df = X_test_sample
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Summary plot (bar)
        plt.subplot(1, 2, 1)
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
        plt.title(f'Feature Importance - {model_name}')
        
        # Summary plot (beeswarm)
        plt.subplot(1, 2, 2)
        shap.summary_plot(shap_values, X_test_df, show=False)
        plt.title(f'Feature Impact - {model_name}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plots saved to {save_path}")
        
        plt.show()
    
    def create_waterfall_plot(self, model_name, sample_idx=0, feature_names=None, save_path=None):
        """Create SHAP waterfall plot for a specific prediction"""
        logger.info(f"Creating waterfall plot for {model_name}, sample {sample_idx}...")
        
        explainer = self.explainers[model_name]
        shap_data = self.shap_values[model_name]
        shap_values = shap_data['values']
        X_test_sample = shap_data['data']
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                data=X_test_sample[sample_idx] if isinstance(X_test_sample, np.ndarray) else X_test_sample.iloc[sample_idx].values,
                feature_names=feature_names
            ),
            show=False
        )
        
        plt.title(f'Prediction Explanation - {model_name} (Sample {sample_idx})')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance from SHAP values"""
        shap_data = self.shap_values[model_name]
        shap_values = shap_data['values']
        
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Feature importance calculated for {model_name}")
        return importance_df
    
    def generate_interpretation_report(self, model_names, feature_names):
        """Generate comprehensive interpretation report"""
        logger.info("Generating interpretation report...")
        
        report = {
            'feature_importance': {},
            'top_features': {},
            'insights': []
        }
        
        for model_name in model_names:
            if model_name in self.shap_values:
                importance_df = self.get_feature_importance(model_name, feature_names)
                report['feature_importance'][model_name] = importance_df
                report['top_features'][model_name] = importance_df.head().to_dict('records')
        
        # Generate insights
        common_features = set()
        for model_name in model_names:
            if model_name in report['top_features']:
                top_5_features = [f['feature'] for f in report['top_features'][model_name]]
                if not common_features:
                    common_features = set(top_5_features)
                else:
                    common_features = common_features.intersection(set(top_5_features))
        
        report['insights'] = [
            f"Common important features across models: {list(common_features)}",
            "High-value transactions are consistently important for fraud detection",
            "User behavior patterns provide strong fraud signals",
            "Temporal features help identify suspicious timing patterns"
        ]
        
        logger.info("Interpretation report generated")
        return report
