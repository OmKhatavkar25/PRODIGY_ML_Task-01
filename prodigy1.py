# Linear Regression: House Price Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Step 1: Create dataset
# -------------------------------
data = {
    'sqft': [800, 1000, 1200, 1500, 1800, 2000],
    'bedrooms': [1, 2, 2, 3, 3, 4],
    'bathrooms': [1, 1, 2, 2, 3, 3],
    'price': [50000, 65000, 80000, 120000, 150000, 180000]
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Define features and target
# -------------------------------
X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']

# -------------------------------
# Step 3: Split data into train and test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Train Linear Regression model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Make predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 6: Evaluate model
# -------------------------------
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# Step 7: Predict price for a new house
# -------------------------------
new_house = [[1600, 3, 2]]
predicted_price = model.predict(new_house)

print("Predicted Price for new house:", predicted_price[0])
