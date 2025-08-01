# Import required libraries
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

# Load the dataset
print("Loading sales dataset...")
df = pd.read_csv('train.csv')
print(f"Dataset loaded! Shape: {df.shape}")

# Display basic information
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset columns:")
print(df.columns.tolist())

print("\nDataset info:")
print(df.info())

# Data preprocessing
print("\nCleaning and preprocessing data...")

# Convert date column to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)

# Sort by date
df = df.sort_values('Order Date')

# Create monthly sales aggregation
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()

print(f"Monthly sales data shape: {monthly_sales.shape}")
print("\nMonthly sales preview:")
print(monthly_sales.head())

# Exploratory Data Analysis
plt.figure()
plt.gcf().set_size_inches(15, 12)

# Plot 1: Sales over time
plt.subplot(3, 2, 1)
plt.plot(monthly_sales['Order Date'], monthly_sales['Sales'])
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)

# Plot 2: Sales distribution
plt.subplot(3, 2, 2)
plt.hist(monthly_sales['Sales'], bins=20, edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('Sales ($)')
plt.ylabel('Frequency')

# Plot 3: Year-over-year comparison
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
yearly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

plt.subplot(3, 2, 3)
for year in yearly_sales['Year'].unique():
    year_data = yearly_sales[yearly_sales['Year'] == year]
    plt.plot(year_data['Month'], year_data['Sales'], marker='o', label=f'Year {year}')
plt.title('Month-wise Sales by Year')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.legend()

# Plot 4: Seasonal patterns
plt.subplot(3, 2, 4)
monthly_avg = df.groupby('Month')['Sales'].mean()
plt.bar(monthly_avg.index, monthly_avg.values)
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Sales ($)')

# Plot 5: Top categories
plt.subplot(3, 2, 5)
top_categories = df.groupby('Category')['Sales'].sum().sort_values(ascending=False).head(5)
plt.bar(top_categories.index, top_categories.values)
plt.title('Top 5 Categories by Sales')
plt.xticks(rotation=45)
plt.ylabel('Total Sales ($)')

# Plot 6: Sales by region
plt.subplot(3, 2, 6)
if 'Region' in df.columns:
    region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    plt.bar(region_sales.index, region_sales.values)
    plt.title('Sales by Region')
    plt.xticks(rotation=45)
    plt.ylabel('Total Sales ($)')

plt.tight_layout()
plt.savefig('sales_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Time series decomposition
print("\nPerforming time series decomposition...")
ts_data = monthly_sales.set_index('Order Date')['Sales']

# Perform seasonal decomposition
decomposition = seasonal_decompose(ts_data, model='additive', period=12)
plt.figure()
plt.gcf().set_size_inches(15, 10)
plt.figure(figsize=(15, 10))
plt.subplot(4, 1, 1)
plt.plot(decomposition.observed)
plt.title('Original Time Series')

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend)
plt.title('Trend Component')

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal)
plt.title('Seasonal Component')

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid)
plt.title('Residual Component')

plt.tight_layout()
plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()

# Prepare data for ARIMA modeling
print("\nPreparing data for ARIMA forecasting...")

# Use 80% data for training
train_size = int(len(ts_data) * 0.8)
train_data = ts_data[:train_size]
test_data = ts_data[train_size:]

print(f"Training data points: {len(train_data)}")
print(f"Testing data points: {len(test_data)}")

# Fit ARIMA model
print("\nFitting ARIMA model...")
# Using ARIMA(1,1,1) - you can experiment with different parameters
model = ARIMA(train_data, order=(1, 1, 1))
fitted_model = model.fit()

print("ARIMA Model Summary:")
print(fitted_model.summary())

# Make predictions
print("\nMaking predictions...")
forecast_steps = len(test_data)
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_index = test_data.index

# Calculate accuracy metrics
mae = mean_absolute_error(test_data, forecast)
rmse = np.sqrt(mean_squared_error(test_data, forecast))
mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

print(f"\nModel Performance:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Square Error (RMSE): ${rmse:,.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
plt.figure()
plt.gcf().set_size_inches(15, 8)
# Plot predictions vs actual
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.plot(train_data.index, train_data.values, label='Training Data', color='blue')
plt.plot(test_data.index, test_data.values, label='Actual', color='green', marker='o')
plt.plot(forecast_index, forecast, label='Forecast', color='red', marker='s')
plt.title('Sales Forecasting Results')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.xticks(rotation=45)

# Plot residuals
plt.subplot(2, 1, 2)
residuals = test_data - forecast
plt.plot(forecast_index, residuals, color='purple', marker='o')
plt.title('Forecast Residuals (Actual - Predicted)')
plt.xlabel('Date')
plt.ylabel('Residual ($)')
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('forecast_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Future forecasting
print("\nGenerating future forecasts...")
future_periods = 6  # Forecast next 6 months
future_forecast = fitted_model.forecast(steps=future_periods)

# Create future dates
last_date = ts_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=future_periods, freq='MS')
plt.figure()
plt.gcf().set_size_inches(15, 6)

# Plot complete forecast
plt.figure(figsize=(15, 6))
plt.plot(ts_data.index, ts_data.values, label='Historical Data', color='blue')
plt.plot(future_dates, future_forecast, label='Future Forecast', color='red', marker='o')
plt.title('Sales Forecast for Next 6 Months')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('future_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary results
results_summary = {
    'Total_Historical_Months': len(ts_data),
    'Training_Months': len(train_data),
    'Testing_Months': len(test_data),
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape,
    'Avg_Monthly_Sales': ts_data.mean(),
    'Total_Sales': ts_data.sum()
}

print(f"\nProject Summary:")
for key, value in results_summary.items():
    if key in ['MAE', 'RMSE', 'Avg_Monthly_Sales', 'Total_Sales']:
        print(f"{key}: ${value:,.2f}")
    else:
        print(f"{key}: {value}")

print("\nFuture 6-Month Forecast:")
for i, (date, sales) in enumerate(zip(future_dates, future_forecast)):
    print(f"{date.strftime('%Y-%m')}: ${sales:,.2f}")

print("\nProject completed successfully!")
print("Generated files:")
print("- sales_analysis.png")
print("- time_series_decomposition.png")
print("- forecast_results.png")
print("- future_forecast.png")
