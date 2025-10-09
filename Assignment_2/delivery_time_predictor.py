import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    "Miles": [89, 66, 78, 111, 44, 77, 80, 66, 109, 76],
    "Deliveries": [4, 1, 3, 6, 1, 3, 3, 2, 5, 3],
    "TravelTime": [7.0, 5.4, 6.6, 7.4, 4.8, 6.4, 7.0, 5.6, 7.3, 6.4],
}

df = pd.DataFrame(data)

X = df.drop(columns=["TravelTime"])
y = df["TravelTime"]

model = LinearRegression()
model.fit(X, y)

print("Intercept (w0):", model.intercept_)
print("Coefficient for miles (w1):", model.coef_[0])
print("Coefficient for deliveries (w2):", model.coef_[1])

prediction = model.predict([[100, 10]])
print("Predicted travel time:", prediction[0])

# Intercept (w0): 3.73215813168261
# Coefficient for miles (w1): 0.02622256612267869
# Coefficient for deliveries (w2): 0.18404051772650504

# Predicted travel time: 8.19481992121553