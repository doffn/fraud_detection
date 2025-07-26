import pandas as pd
import numpy as np
from scripts.utils import ip_to_int

def map_ip_to_country(fraud_df, ip_country_df):
    """
    Maps IP addresses in fraud_df to countries using ip_country_df.
    Assumes ip_address_int is already created in fraud_df.
    """
    if 'ip_address_int' not in fraud_df.columns:
        print("Error: 'ip_address_int' column not found in fraud_df. Please run data_preprocessing first.")
        return None

    # Sort ip_country_df by lower_bound_ip_address for efficient merging
    ip_country_df_sorted = ip_country_df.sort_values(by='lower_bound_ip_address').reset_index(drop=True)

    # Use pd.merge_asof for range-based lookup
    # This requires both dataframes to be sorted on the key
    # We need to ensure the IP address from fraud_df falls within the range [lower_bound, upper_bound]
    # A direct merge_asof might not perfectly capture the upper bound.
    # A more robust approach for IP ranges is often a custom lookup or a series of merges.

    # For simplicity and common practice with IP ranges, we can perform a merge and then filter.
    # This approach might be memory intensive for very large datasets.
    # A more optimized approach would involve using a custom function with binary search or interval trees.

    # Let's try a more direct approach by iterating or using apply with a lookup function
    # For demonstration, we'll use a less efficient but clear apply method.
    # For production, consider a more optimized solution like `numpy.searchsorted` or `bisect` module.

    def get_country(ip_int, ip_ranges):
        # Find the row where ip_int is within the lower and upper bounds
        # This is inefficient for large dataframes, but demonstrates the logic.
        # A more performant way is to use `searchsorted` on sorted bounds.
        idx = ip_ranges['lower_bound_ip_address'].searchsorted(ip_int, side='right') - 1
        if idx >= 0 and idx < len(ip_ranges):
            row = ip_ranges.iloc[idx]
            if ip_int >= row['lower_bound_ip_address'] and ip_int <= row['upper_bound_ip_address']:
                return row['country']
        return 'Unknown' # Or np.nan

    print("Applying IP to country lookup. This might take a while for large datasets...")
    fraud_df['country'] = fraud_df['ip_address_int'].apply(lambda x: get_country(x, ip_country_df_sorted))
    print(f"Mapped {fraud_df['country'].nunique()} unique countries.")
    print(f"Transactions with 'Unknown' country: {fraud_df[fraud_df['country'] == 'Unknown'].shape[0]}")

    return fraud_df

def create_time_features(df):
    """
    Creates time-based features from signup_time and purchase_time.
    """
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek # Monday=0, Sunday=6
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

    # Handle cases where purchase_time might be before signup_time (data error)
    df['time_since_signup'] = df['time_since_signup'].apply(lambda x: max(0, x))

    return df

def create_transaction_frequency_features(df, time_window_hours=24):
    """
    Calculates transaction frequency and velocity for user_id and device_id
    within a specified time window.
    """
    df_sorted = df.sort_values(by='purchase_time').reset_index(drop=True)

    # Calculate rolling features for user_id
    print(f"Calculating user transaction frequency and velocity for {time_window_hours} hours...")
    df_sorted['user_transactions_24h'] = df_sorted.groupby('user_id').rolling(
        f'{time_window_hours}h', on='purchase_time'
    )['purchase_value'].count().reset_index(level=0, drop=True) - 1 # Exclude current transaction

    df_sorted['user_value_24h'] = df_sorted.groupby('user_id').rolling(
        f'{time_window_hours}h', on='purchase_time'
    )['purchase_value'].sum().reset_index(level=0, drop=True) - df_sorted['purchase_value'] # Exclude current transaction

    # Calculate rolling features for device_id
    print(f"Calculating device transaction frequency and velocity for {time_window_hours} hours...")
    df_sorted['device_transactions_24h'] = df_sorted.groupby('device_id').rolling(
        f'{time_window_hours}h', on='purchase_time'
    )['purchase_value'].count().reset_index(level=0, drop=True) - 1 # Exclude current transaction

    df_sorted['device_value_24h'] = df_sorted.groupby('device_id').rolling(
        f'{time_window_hours}h', on='purchase_time'
    )['purchase_value'].sum().reset_index(level=0, drop=True) - df_sorted['purchase_value'] # Exclude current transaction

    # Fill NaN values that might result from rolling window (e.g., first transaction) with 0
    df_sorted[['user_transactions_24h', 'user_value_24h', 'device_transactions_24h', 'device_value_24h']] = \
        df_sorted[['user_transactions_24h', 'user_value_24h', 'device_transactions_24h', 'device_value_24h']].fillna(0)

    return df_sorted
