import pandas as pd
import numpy as np
from scripts.utils import ip_to_int

def load_and_clean_fraud_data(file_path):
    """
    Loads Fraud_Data.csv, handles missing values, and corrects data types.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original Fraud_Data.csv shape: {df.shape}")

        # Handle missing values
        # Drop rows where ip_address is missing as it's crucial for geolocation
        initial_rows = df.shape[0]
        df.dropna(subset=['ip_address'], inplace=True)
        print(f"Dropped {initial_rows - df.shape[0]} rows with missing ip_address.")

        # Correct data types
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Convert ip_address to integer
        # Apply ip_to_int function, handling potential errors
        df['ip_address_int'] = df['ip_address'].apply(lambda x: ip_to_int(x) if pd.notna(x) else np.nan)
        df.dropna(subset=['ip_address_int'], inplace=True) # Drop if conversion failed
        df['ip_address_int'] = df['ip_address_int'].astype(int) # Ensure integer type

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        print(f"Removed duplicates. New shape: {df.shape}")

        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred during loading/cleaning Fraud_Data: {e}")
        return None

def load_and_clean_ip_data(file_path):
    """
    Loads IpAddress_to_Country.csv and corrects data types.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original IpAddress_to_Country.csv shape: {df.shape}")

        # Correct data types (ensure bounds are integers)
        df['lower_bound_ip_address'] = df['lower_bound_ip_address'].astype(int)
        df['upper_bound_ip_address'] = df['upper_bound_ip_address'].astype(int)

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        print(f"Removed duplicates. New shape: {df.shape}")

        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred during loading/cleaning IpAddress_to_Country: {e}")
        return None
