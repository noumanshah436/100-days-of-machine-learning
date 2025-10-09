import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Read the file (assuming space-separated values)
df = pd.read_csv("bodyfat.csv")

# Select predictors and target
X = df.drop(columns=["Density", "BodyFat"])
y = df["BodyFat"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients
coefficients = pd.DataFrame({"Variable": X.columns, "Coefficient": model.coef_})
print(coefficients)

# Model performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# output:
#    Variable  Coefficient
# 0       Age     0.077179
# 1    Weight    -0.053173
# 2    Height    -0.095715
# 3      Neck    -0.251306
# 4     Chest    -0.139476
# 5   Abdomen     0.960621
# 6       Hip    -0.125536
# 7     Thigh     0.145596
# 8      Knee    -0.102809
# 9     Ankle     0.268512
# 10   Biceps     0.279270
# 11  Forearm     0.250926
# 12    Wrist    -1.838267
# R² Score: 0.6135
# Mean Squared Error: 17.9800