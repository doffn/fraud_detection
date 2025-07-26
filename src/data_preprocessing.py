"""
Data Preprocessing Module
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Main class for data preprocessing operations"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        
    def load_fraud_data(self, file_path):
        """Load fraud detection dataset"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded fraud data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading fraud data: {e}")
            return None
    
    def load_ip_country_data(self, file_path):
        """Load IP to country mapping dataset"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded IP country data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading IP country data: {e}")
            return None
    
    def clean_fraud_data(self, df):
        """Clean fraud detection data"""
        logger.info("Starting data cleaning...")
        
        # Handle missing values
        initial_rows = len(df)
        df = df.dropna(subset=['ip_address'])
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing IP addresses")
        
        # Convert data types
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Convert IP to integer
        df['ip_address_int'] = df['ip_address'].apply(self._ip_to_int)
        df = df.dropna(subset=['ip_address_int'])
        df['ip_address_int'] = df['ip_address_int'].astype(int)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        
        return df
    
    def _ip_to_int(self, ip_address):
        """Convert IP address to integer"""
        try:
            parts = list(map(int, ip_address.split('.')))
            return parts[0] * 256**3 + parts[1] * 256**2 + parts[2] * 256 + parts[3]
        except:
            return None
    
    def map_ip_to_country(self, fraud_df, ip_country_df):
        """Map IP addresses to countries"""
        logger.info("Mapping IP addresses to countries...")
        
        def get_country(ip_int, ip_ranges):
            for _, row in ip_ranges.iterrows():
                if row['lower_bound_ip_address'] <= ip_int <= row['upper_bound_ip_address']:
                    return row['country']
            return 'Unknown'
        
        fraud_df['country'] = fraud_df['ip_address_int'].apply(
            lambda x: get_country(x, ip_country_df)
        )
        
        logger.info(f"Countries mapped: {fraud_df['country'].nunique()}")
        return fraud_df
    
    def create_time_features(self, df):
        """Create time-based features"""
        logger.info("Creating time-based features...")
        
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['month'] = df['purchase_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time since signup
        df['time_since_signup'] = (
            df['purchase_time'] - df['signup_time']
        ).dt.total_seconds()
        df['time_since_signup'] = df['time_since_signup'].apply(lambda x: max(0, x))
        
        logger.info("Time-based features created")
        return df
    
    def create_transaction_features(self, df):
        """Create transaction frequency and velocity features"""
        logger.info("Creating transaction features...")
        
        # User-based aggregations
        user_stats = df.groupby('user_id').agg({
            'purchase_value': ['count', 'sum', 'mean', 'std'],
            'purchase_time': ['min', 'max']
        }).reset_index()
        
        user_stats.columns = ['user_id', 'user_transaction_count', 'user_total_value',
                             'user_avg_value', 'user_std_value', 'user_first_purchase',
                             'user_last_purchase']
        
        # Device-based aggregations
        device_stats = df.groupby('device_id').agg({
            'purchase_value': ['count', 'sum', 'mean'],
            'user_id': 'nunique'
        }).reset_index()
        
        device_stats.columns = ['device_id', 'device_transaction_count',
                               'device_total_value', 'device_avg_value', 'device_unique_users']
        
        # Merge back
        df = df.merge(user_stats, on='user_id', how='left')
        df = df.merge(device_stats, on='device_id', how='left')
        
        # Risk indicators
        df['high_value_transaction'] = (
            df['purchase_value'] > df['purchase_value'].quantile(0.95)
        ).astype(int)
        
        df['new_user'] = (df['time_since_signup'] < 3600).astype(int)
        
        logger.info("Transaction features created")
        return df
