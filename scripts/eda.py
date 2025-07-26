import pandas as pd
import numpy as np

def perform_eda_summary(df):
    """
    Performs a summary EDA and prints key insights.
    """
    print("\n--- EDA Summary ---")

    print("\nDataFrame Info:")
    df.info()

    print("\nDescriptive Statistics for Numerical Features:")
    print(df.describe())

    print("\nClass Distribution ('class' column):")
    class_counts = df['class'].value_counts(normalize=True)
    print(class_counts)
    print(f"Fraudulent transactions (class=1): {class_counts.get(1, 0)*100:.2f}%")
    print(f"Non-fraudulent transactions (class=0): {class_counts.get(0, 0)*100:.2f}%")

    print("\nDistribution of 'purchase_value':")
    print(df['purchase_value'].describe())
    print(f"Median purchase value: {df['purchase_value'].median():.2f}")

    print("\nDistribution of 'age':")
    print(df['age'].describe())

    print("\nTop 5 'source' categories:")
    print(df['source'].value_counts().head())

    print("\nTop 5 'browser' categories:")
    print(df['browser'].value_counts().head())

    print("\n'sex' distribution:")
    print(df['sex'].value_counts())

    print("\nFraud Rate by 'source':")
    print(df.groupby('source')['class'].mean().sort_values(ascending=False))

    print("\nFraud Rate by 'browser':")
    print(df.groupby('browser')['class'].mean().sort_values(ascending=False))

    print("\nFraud Rate by 'sex':")
    print(df.groupby('sex')['class'].mean().sort_values(ascending=False))

    if 'country' in df.columns:
        print("\nTop 10 Countries by Fraud Count:")
        print(df[df['class'] == 1]['country'].value_counts().head(10))
        print("\nTop 10 Countries by Transaction Count:")
        print(df['country'].value_counts().head(10))
        print("\nFraud Rate by Top 5 Countries:")
        top_countries = df['country'].value_counts().head(5).index
        print(df[df['country'].isin(top_countries)].groupby('country')['class'].mean().sort_values(ascending=False))

    if 'hour_of_day' in df.columns:
        print("\nFraud Rate by 'hour_of_day':")
        print(df.groupby('hour_of_day')['class'].mean().sort_values(ascending=False).head())

    if 'day_of_week' in df.columns:
        print("\nFraud Rate by 'day_of_week':")
        print(df.groupby('day_of_week')['class'].mean().sort_values(ascending=False).head())

    if 'time_since_signup' in df.columns:
        print("\nDescriptive Statistics for 'time_since_signup' (in seconds):")
        print(df['time_since_signup'].describe())
        print("\nAverage 'time_since_signup' for fraudulent vs. non-fraudulent transactions:")
        print(df.groupby('class')['time_since_signup'].mean())

    if 'user_transactions_24h' in df.columns:
        print("\nAverage 'user_transactions_24h' for fraudulent vs. non-fraudulent transactions:")
        print(df.groupby('class')['user_transactions_24h'].mean())
        print("\nAverage 'user_value_24h' for fraudulent vs. non-fraudulent transactions:")
        print(df.groupby('class')['user_value_24h'].mean())
