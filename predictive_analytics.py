import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create historical dataset
data = {
    "Month": pd.date_range(start="2022-01-01", periods=36, freq="MS"),
    "Sales": [
        120, 125, 130, 128, 135, 140,
        145, 150, 148, 155, 160, 165,
        170, 175, 180, 178, 185, 190,
        195, 200, 205, 202, 210, 215,
        220, 225, 230, 228, 235, 240,
        245, 250, 255, 252, 260, 265
    ]
}

df = pd.DataFrame(data)

print("Original Historical Data:")
print(df.head())

# Data cleaning
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df.dropna(inplace=True)

# Create time-based feature
df["Month_Number"] = np.arange(1, len(df) + 1)

# Features and target
X = df[["Month_Number"]]
y = df["Sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# Future prediction
future_months = np.arange(len(df) + 1, len(df) + 7).reshape(-1, 1)
future_predictions = model.predict(future_months)

future_dates = pd.date_range(
    start=df["Month"].max() + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)

future_df = pd.DataFrame({
    "Month": future_dates,
    "Predicted_Sales": future_predictions.round(2)
})

print("\nFuture Sales Predictions:")
print(future_df)

# Save predictions
future_df.to_csv("future_predictions.csv", index=False)

# Visualization
plt.figure(figsize=(10, 5))

plt.plot(
    df["Month"],
    df["Sales"],
    marker="o",
    label="Historical Sales"
)

plt.plot(
    future_df["Month"],
    future_df["Predicted_Sales"],
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

plt.title("Predictive Analytics Using Historical Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.savefig("sales_prediction.png")
plt.show()

print("\nPrediction graph saved as sales_prediction.png")
print("Future predictions saved as future_predictions.csv")