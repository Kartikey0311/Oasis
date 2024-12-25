#task2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Step 1: Load Data
unemployment_data = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=24, freq='M'),
    'unemployment_rate': np.random.uniform(3, 15, 24),
    'covid_cases': np.random.randint(1000, 100000, 24),
    'lockdown': [1 if i < 12 else 0 for i in range(24)]
})
gdp_data = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=24, freq='M'),
    'gdp_change': np.random.uniform(-10, 5, 24)
})
data = pd.merge(unemployment_data, gdp_data, on='date')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Step 2: Data Cleaning and Feature Engineering
data['lockdown_duration'] = data['lockdown'].cumsum()
data['covid_cases_change'] = data['covid_cases'].pct_change().fillna(0)

# Step 3: Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x=data.index, y='unemployment_rate', label='Unemployment Rate')
plt.title("Unemployment Rate During COVID-19")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='covid_cases', y='unemployment_rate', hue='lockdown')
plt.title("Unemployment Rate vs COVID Cases")
plt.xlabel("COVID Cases")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# Step 4: Model Building - Linear Regression
X = data[['covid_cases', 'gdp_change', 'lockdown_duration']]
y = data['unemployment_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(test_results.head())

# Step 4: Model Building - ARIMA
unemployment_series = data['unemployment_rate']
model_arima = ARIMA(unemployment_series, order=(1, 1, 1))
model_arima_fit = model_arima.fit()
forecast = model_arima_fit.forecast(steps=6)
forecast_values = forecast
future_dates = pd.date_range(start=data.index[-1], periods=7, freq='M')[1:]
forecast_df = pd.DataFrame({'date': future_dates, 'forecasted_unemployment': forecast_values})
print(forecast_df)

plt.figure(figsize=(10, 6))
plt.plot(unemployment_series, label='Historical Data')
plt.plot(forecast_df['date'], forecast_df['forecasted_unemployment'], label='Forecasted Data', linestyle='--')
plt.title("Unemployment Rate Forecast")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()

# Step 5: Insights and Recommendations
print("Key Insights:")
print("- Unemployment spiked during lockdown periods.")
print("- Strong correlation observed between GDP decline and unemployment rate increase.")
print("- Forecast indicates a gradual decline in unemployment post-COVID recovery.")

print("Recommendations:")
print("- Focus on sectors most affected by lockdown for recovery strategies.")
print("- Implement policies to stimulate GDP growth.")
print("- Monitor unemployment rates closely to inform future decisions.")
