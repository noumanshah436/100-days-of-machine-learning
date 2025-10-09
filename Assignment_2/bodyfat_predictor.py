import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("bodyfat.csv")

X = df.drop(columns=["Density", "BodyFat"])
y = df["BodyFat"]


model = LinearRegression()
model.fit(X, y)

# Coefficients
coefficients = pd.DataFrame({"Variable": X.columns, "Coefficient": model.coef_})
print(coefficients)


# Variable  Coefficient
# Age       0.062079
# Weight    -0.088445
# Height    -0.069590
# Neck      -0.470600
# Chest     -0.023864
# Abdomen   0.954773
# Hip       -0.207541
# Thigh     0.236100
# Knee      0.015281
# Ankle     0.173995
# Biceps    0.181602
# Forearm   0.452025
# Wrist     -1.620639