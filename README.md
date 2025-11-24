# VIX Volatility Analysis Dashboard

## Overview
Comprehensive market stress analysis and risk identification dashboard using 35+ years of VIX data.

## Features
- Historical volatility regime analysis
- Crisis period identification and characterization  
- Early warning indicators for market stress
- Machine learning forecasting models
- Interactive visualizations with Plotly

## Performance
- 74% error reduction vs baseline
- Best model MAE: 1.49 VIX points
- 85% crisis detection rate

## Running the Dashboard

### Local Development
```bash
pip install -r requirements.txt
streamlit run vix_dashboard.py
```

### Posit Connect Deployment
1. Fork this repository
2. Deploy to Posit Connect using Git integration
3. Access via your Posit Connect URL

## Model Performance Summary
- XGBoost: MAE 1.49, RMSE 2.68, R² 0.75
- Random Forest: MAE 1.53, RMSE 2.88, R² 0.71  
- H2O DAI: MAE 1.52, Test MAE 1.52
- Prophet: MAE 4.80, RMSE 6.06

## Data Source
Federal Reserve Economic Data (FRED) - VIXCLS
https://fred.stlouisfed.org/series/VIXCLS

## Author
Created for hedge fund investment strategy team
