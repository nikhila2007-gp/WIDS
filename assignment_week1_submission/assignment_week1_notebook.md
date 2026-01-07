```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import warnings
warnings.filterwarnings("ignore")
TICKER = 'MSFT'
START_DATE = '2023-01-01'
END_DATE = '2025-01-01'
TEST_SIZE = 0.2
print(f"Downloading stock price data for {TICKER}")
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
df = data[['Close']].copy()
df.index = pd.to_datetime(df.index)
print(f"No of data points: {len(df)}")
df.dropna(inplace=True)
def check_stationarity(series):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("series is stationary")
        return True
    else:
        print("series is not stationary")
        return False
print("\nADF test on original series")
is_stationary = check_stationarity(df['Close'])
d = 0
series_to_use = df['Close']
if not is_stationary:
    print("\nApplying first difference")
    df['Close_Diff'] = df['Close'].diff().dropna()
    d = 1
    print("Running ADF test again")
    if not check_stationarity(df['Close_Diff']):
        df['Close_Diff2'] = df['Close_Diff'].diff().dropna()
        d = 2
        series_to_use = df['Close_Diff2']
    else:
        series_to_use = df['Close_Diff']
else:
    series_to_use = df['Close']
print(f"d = {d}")
train_size = int(len(series_to_use) * (1 - TEST_SIZE))
train_data = series_to_use[:train_size]
test_data = series_to_use[train_size:]
start_index_original = len(df['Close']) - len(test_data)
original_test_data = df['Close'].iloc[start_index_original:]
print(f"Training data size: {len(train_data)}")
print(f"Testing data size: {len(test_data)}")
print("\nSearching ARIMA order")
p_values = range(0, 3)
q_values = range(0, 3)
pdq = list(itertools.product(p_values, [d], q_values))
best_aic = float("inf")
best_order = None
best_model = None
for order in pdq:
    try:
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_order = order
            best_model = model_fit
    except:
        continue
print(f"Best ARIMA order: {best_order}")
print("\nForecasting test period")
start = len(train_data)
end = len(train_data) + len(test_data) - 1
forecast_original = best_model.get_prediction(start=start, end=end).predicted_mean
forecast_original.index = original_test_data.index
print("\nEvaluating model performance")
mae = mean_absolute_error(original_test_data, forecast_original)
mse = mean_squared_error(original_test_data, forecast_original)
rmse = np.sqrt(mse)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Actual Price')
plt.plot(original_test_data.index, forecast_original, label='Predicted Price', color='red')
plt.title(f'{TICKER} Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

residuals = original_test_data - forecast_original

plt.figure(figsize=(12, 4))
plt.plot(residuals, color='green')
plt.axhline(0, linestyle='--', color='black')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.grid(True)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
acf_vals = acf(residuals.values.ravel(), nlags=20)
axes[0].bar(range(len(acf_vals)), acf_vals)
axes[0].axhline(0, linestyle='--', color='black')
axes[0].set_title('ACF of Residuals')
pacf_vals = pacf(residuals.values.ravel(), nlags=20)
axes[1].bar(range(len(pacf_vals)), pacf_vals)
axes[1].axhline(0, linestyle='--', color='black')
axes[1].set_title('PACF of Residuals')
plt.show()
