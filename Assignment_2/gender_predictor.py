import pandas as pd
from sklearn import linear_model

df = pd.read_csv("heights_weights.csv")

X = df.drop(columns=["Gender"])
y = df["Gender"]

# Fit (train) the Logistic Regression classifier
clf = linear_model.LogisticRegression(C=1e40, solver='newton-cg')
fitted_model = clf.fit(X, y)

# Print intercept and coefficients separately
print("Intercept (w0):", clf.intercept_[0])
print("Coefficient for Height (w1):", clf.coef_[0][0])
print("Coefficient for Weight (w2):", clf.coef_[0][1])

# Predict for height=70, weight=180
prediction = clf.predict([[70, 180]])
print("Predicted Gender:", prediction[0])


