import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([3, 6, 14, 28, 45, 70, 95, 130])

# Train/test split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)

degrees = range(1, 10)
train_errors, val_errors = [], []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    x_train_poly = poly.fit_transform(x_train)
    x_val_poly = poly.transform(x_val)
    
    model = LinearRegression().fit(x_train_poly, y_train)
    
    # Errors
    train_errors.append(mean_squared_error(y_train, model.predict(x_train_poly)))
    val_errors.append(mean_squared_error(y_val, model.predict(x_val_poly)))

plt.plot(degrees, train_errors, label="Training Error", marker='o')
plt.plot(degrees, val_errors, label="Validation Error", marker='o')
plt.xlabel("Degree of Polynomial")
plt.ylabel("MSE")
plt.legend()
plt.show()