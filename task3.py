
#task3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Generate synthetic data
np.random.seed(42)
data = {
    'Brand': np.random.choice(['Toyota', 'Ford', 'BMW', 'Honda'], size=500),
    'Year': np.random.randint(2000, 2023, size=500),
    'Horsepower': np.random.randint(70, 400, size=500),
    'Mileage': np.random.randint(10, 50, size=500),
    'Price': np.random.randint(5000, 50000, size=500)
}

df = pd.DataFrame(data)

# Step 2: Exploratory Data Analysis (EDA)
print("First few rows of the dataset:")
print(df.head())

print("\nCheck for missing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

# Visualize the distribution of the price
plt.figure(figsize=(8, 6))
sns.histplot(df['Price'], kde=True, bins=30)
plt.title('Car Price Distribution')
plt.show()

# Step 3: Preprocessing
# Encode categorical variable
le = LabelEncoder()
df['Brand'] = le.fit_transform(df['Brand'])

# Split data into features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Step 6: Visualize the Results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
