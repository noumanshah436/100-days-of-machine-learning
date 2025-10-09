import matplotlib.pyplot as plt
import numpy as np

# Given data
price = np.array([49, 69, 89, 99, 109])
demand = np.array([124, 95, 71, 45, 18])

# Regression parameters (calculated earlier)
m = -1.624
b = 205.39

# Predicted values
predicted_demand = b + m * price

# Plot
plt.figure(figsize=(8,6))
plt.scatter(price, demand, color='blue', label='Actual Data Points')
plt.plot(price, predicted_demand, color='red', linewidth=2, label='Regression Line')

# Labels and title
plt.title("Price vs Demand (Linear Regression)", fontsize=14)
plt.xlabel("Price", fontsize=12)
plt.ylabel("Demand", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
