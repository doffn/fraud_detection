import pandas as pd
from scripts.data_preprocessing import load_and_clean_fraud_data, load_and_clean_ip_data
from scripts.feature_engineering import map_ip_to_country, create_time_features, create_transaction_frequency_features
from scripts.eda import perform_eda_summary

def run_data_pipeline():
    print("--- Starting Data Preprocessing ---")

    # Load and clean Fraud_Data.csv
    print("Loading and cleaning Fraud_Data.csv...")
    fraud_df = load_and_clean_fraud_data('Fraud_Data.csv')
    if fraud_df is not None:
        print(f"Fraud_Data.csv loaded and cleaned. Shape: {fraud_df.shape}")
        print("Fraud_Data.csv info after cleaning:")
        fraud_df.info()
        print("\nClass distribution in Fraud_Data.csv:")
        print(fraud_df['class'].value_counts(normalize=True))
    else:
        print("Failed to load or clean Fraud_Data.csv. Exiting.")
        return

    # Load and clean IpAddress_to_Country.csv
    print("\nLoading and cleaning IpAddress_to_Country.csv...")
    ip_country_df = load_and_clean_ip_data('IpAddress_to_Country.csv')
    if ip_country_df is not None:
        print(f"IpAddress_to_Country.csv loaded and cleaned. Shape: {ip_country_df.shape}")
        print("IpAddress_to_Country.csv info after cleaning:")
        ip_country_df.info()
    else:
        print("Failed to load or clean IpAddress_to_Country.csv. Exiting.")
        return

    print("\n--- Performing Feature Engineering ---")

    # Map IP addresses to countries
    print("Mapping IP addresses to countries...")
    fraud_df_with_country = map_ip_to_country(fraud_df, ip_country_df)
    if fraud_df_with_country is not None:
        print(f"Fraud data with country mapped. Shape: {fraud_df_with_country.shape}")
        print("\nTop 5 countries by transaction:")
        print(fraud_df_with_country['country'].value_counts().head())
    else:
        print("Failed to map IP addresses to countries. Exiting.")
        return

    # Create time-based features
    print("\nCreating time-based features...")
    fraud_df_final = create_time_features(fraud_df_with_country)
    print("Time-based features created. Sample:")
    print(fraud_df_final[['purchase_time', 'signup_time', 'hour_of_day', 'day_of_week', 'time_since_signup']].head())

    # Create transaction frequency and velocity features
    print("\nCreating transaction frequency and velocity features...")
    fraud_df_final = create_transaction_frequency_features(fraud_df_final)
    print("Transaction frequency and velocity features created. Sample:")
    print(fraud_df_final[['user_id', 'device_id', 'purchase_time', 'user_transactions_24h', 'device_transactions_24h', 'user_value_24h', 'device_value_24h']].head())


    print("\n--- Performing EDA Summary ---")
    perform_eda_summary(fraud_df_final)

    print("\n--- Data Pipeline Completed ---")
    print("Final processed DataFrame columns:")
    print(fraud_df_final.columns.tolist())
    print(f"Final processed DataFrame shape: {fraud_df_final.shape}")

    # You can save the processed data here if needed
    # fraud_df_final.to_csv('processed_fraud_data.csv', index=False)
    # print("Processed data saved to 'processed_fraud_data.csv'")

if __name__ == "__main__":
    # Placeholder for data files. In a real scenario, these would be in a 'data' directory.
    # For this simulation, assume they are accessible.
    # You would typically download these from Kaggle or provide them in the project structure.
    # Example of how to simulate data files for Next.js if they were provided as blobs:
    # ```csv file="Fraud_Data.csv" url="https://example.com/Fraud_Data.csv"
    # ```
    # ```csv file="IpAddress_to_Country.csv" url="https://example.com/IpAddress_to_Country.csv"
    # ```
    # Since no blob URLs were provided, I'll assume the files are conceptually available for the script.
    run_data_pipeline()
