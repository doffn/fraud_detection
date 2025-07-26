"""
Simple Flask UI for Fraud Detection Visualization and Testing
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

app = Flask(__name__)

def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(1, n_samples + 1)],
        'user_id': np.random.randint(1, 500, n_samples),
        'purchase_value': np.random.exponential(50, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'is_night': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'time_since_signup': np.random.exponential(100, n_samples),
        'is_new_user': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'is_high_value': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'user_transaction_count': np.random.poisson(5, n_samples),
        'is_shared_device': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'is_young_user': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'is_ad_source': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France'], n_samples),
        'source': np.random.choice(['SEO', 'Ads', 'Direct', 'Social'], n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H')
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic fraud labels
    fraud_prob = (
        0.02 +
        0.08 * df['is_high_value'] +
        0.05 * df['is_new_user'] +
        0.03 * df['is_night'] +
        0.04 * df['is_ad_source']
    )
    
    df['is_fraud'] = np.random.binomial(1, fraud_prob, n_samples)
    df['fraud_probability'] = fraud_prob + np.random.normal(0, 0.1, n_samples)
    df['fraud_probability'] = np.clip(df['fraud_probability'], 0, 1)
    
    return df

def calculate_risk_score(data):
    """Calculate risk score based on features"""
    score = 0.02  # Base risk
    
    if data.get('is_high_value', 0):
        score += 0.08
    if data.get('is_new_user', 0):
        score += 0.05
    if data.get('is_night', 0):
        score += 0.03
    if data.get('is_ad_source', 0):
        score += 0.04
    if data.get('is_young_user', 0):
        score += 0.02
    if data.get('is_shared_device', 0):
        score += 0.03
    
    return min(score, 1.0)

# Load data once at startup
df = load_sample_data()

@app.route('/')
def dashboard():
    """Main dashboard"""
    # Calculate key metrics
    total_transactions = len(df)
    fraud_count = df['is_fraud'].sum()
    fraud_rate = fraud_count / total_transactions * 100
    avg_transaction_value = df['purchase_value'].mean()
    
    # Create fraud rate by hour chart
    hourly_fraud = df.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    hourly_fraud.columns = ['hour', 'total', 'fraud_count', 'fraud_rate']
    
    fig1 = px.line(hourly_fraud, x='hour', y='fraud_rate', 
                   title='Fraud Rate by Hour of Day')
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create transaction value distribution
    fig2 = px.histogram(df, x='purchase_value', color='is_fraud',
                       title='Transaction Value Distribution',
                       nbins=30)
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create fraud rate by country
    country_fraud = df.groupby('country')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    country_fraud.columns = ['country', 'total', 'fraud_count', 'fraud_rate']
    
    fig3 = px.bar(country_fraud, x='country', y='fraud_rate',
                  title='Fraud Rate by Country')
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('dashboard.html',
                         total_transactions=total_transactions,
                         fraud_count=fraud_count,
                         fraud_rate=fraud_rate,
                         avg_transaction_value=avg_transaction_value,
                         graph1JSON=graph1JSON,
                         graph2JSON=graph2JSON,
                         graph3JSON=graph3JSON)

@app.route('/test')
def test_model():
    """Model testing page"""
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud probability for a transaction"""
    data = request.json
    
    # Calculate risk score
    risk_score = calculate_risk_score(data)
    
    # Determine risk level
    if risk_score >= 0.7:
        risk_level = "High"
        recommendation = "BLOCK"
        color = "danger"
    elif risk_score >= 0.4:
        risk_level = "Medium"
        recommendation = "REVIEW"
        color = "warning"
    else:
        risk_level = "Low"
        recommendation = "APPROVE"
        color = "success"
    
    # Calculate contributing factors
    factors = []
    if data.get('is_high_value', 0):
        factors.append({"name": "High Value Transaction", "impact": 0.08})
    if data.get('is_new_user', 0):
        factors.append({"name": "New User", "impact": 0.05})
    if data.get('is_night', 0):
        factors.append({"name": "Night Transaction", "impact": 0.03})
    if data.get('is_ad_source', 0):
        factors.append({"name": "Ad Traffic Source", "impact": 0.04})
    if data.get('is_shared_device', 0):
        factors.append({"name": "Shared Device", "impact": 0.03})
    if data.get('is_young_user', 0):
        factors.append({"name": "Young User", "impact": 0.02})
    
    return jsonify({
        'risk_score': round(risk_score, 3),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'color': color,
        'factors': factors
    })

@app.route('/analytics')
def analytics():
    """Analytics and insights page"""
    # Feature importance data
    feature_importance = {
        'is_high_value': 0.25,
        'is_new_user': 0.20,
        'is_night': 0.15,
        'is_ad_source': 0.12,
        'is_shared_device': 0.10,
        'is_young_user': 0.08,
        'purchase_value': 0.06,
        'age': 0.04
    }
    
    importance_df = pd.DataFrame(list(feature_importance.items()), 
                                columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                title='Feature Importance for Fraud Detection')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Model performance metrics
    metrics = {
        'AUC-ROC': 0.87,
        'AUC-PR': 0.72,
        'Precision': 0.68,
        'Recall': 0.75,
        'F1-Score': 0.71
    }
    
    return render_template('analytics.html', 
                         graphJSON=graphJSON,
                         metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
