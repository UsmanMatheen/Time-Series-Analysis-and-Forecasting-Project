# 📈 Bitcoin Price Prediction Using Time Series Analysis

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository presents our final project for MATH 546 - *Introduction to Time Series*, where we developed and compared various models to forecast Bitcoin prices using historical data.

---

## 🧠 Project Objective

Bitcoin is notoriously volatile, making forecasting essential yet challenging. This project aims to:
- Analyze historical Bitcoin price data.
- Build predictive models using Time Series and Machine Learning techniques.
- Compare performance across ARIMA, SARIMA, Random Forest, Gradient Boosting, and LSTM models.
- Provide meaningful insights and accurate forecasts for future price movements.

---

## 📊 Dataset Description

- **Source:** [Kaggle: Bitcoin Historical Data](https://www.kaggle.com/datasets/shiivvvaam/bitcoin-historical-data)
- **Duration:** July 2010 – February 2024
- **Features:**
  - `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Change%`

---

## 🛠️ Data Processing

- **Cleaning:** Handled missing values, converted dates, removed outliers.
- **Standardization:** Scaled numerical features for model consistency.
- **Stationarity:** Applied ADF tests; differencing used where needed.
- **Exploration:** Visualized trends and plotted ACF/PACF to understand temporal dependencies.

---

## 🧪 Models Implemented

### 📉 1. ARIMA (Autoregressive Integrated Moving Average)
- Selected ARIMA(2,1,2) based on AIC criteria.
- Diagnostics revealed good fit but non-normal residuals.

### 🔁 2. SARIMA (Seasonal ARIMA)
- Incorporated seasonal trends found in Bitcoin price behavior.
- Residuals displayed no significant autocorrelation.

### 🌳 3. Random Forest Regressor
- Used Recursive Feature Elimination (RFE) to select features.
- **Metrics:**
  - RMSE: 268.37
  - R²: 0.997

### 🚀 4. Gradient Boosting Regressor
- Adaptive model trained sequentially to reduce errors.
- **Metrics:**
  - RMSE: 499.87
  - R²: 0.998

### 🤖 5. LSTM (Long Short-Term Memory Network)
- Deep learning model with 3 stacked LSTM layers and dropout.
- **Metrics:**
  - RMSE: 1252.03
  - R²: 0.993
- Provided the most accurate sequential learning performance.

---

## 📈 Forecast Visualizations

- All models produced forecast graphs for March 2024 and beyond.
- LSTM and SARIMA models showed upward trends with confidence intervals.
- Seasonal decomposition confirmed cyclic patterns in price behavior.

---

## 📦 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bitcoin-timeseries-forecast.git
   cd bitcoin-timeseries-forecast
2. Install dependencies:
pip install -r requirements.txt

3. Run notebooks

Data cleaning.ipynb
Stationary data acf pacf plots.ipynb
ARIMA Code.ipynb
Random Forest Regressor implementation.ipynb

## Key Highlights
Analyzed 14 years of Bitcoin data.
Applied classical time series and modern machine learning models.
Achieved near-perfect predictions with Random Forest and LSTM.
Identified seasonality and volatility in Bitcoin prices.

## References
Kaggle Dataset: https://www.kaggle.com/datasets/shiivvvaam/bitcoin-historical-data
ADF & Stationarity: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
ARIMA Theory: https://otexts.com/fpp3/arima.html
LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
