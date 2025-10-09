import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron, LogisticRegression


"""
here's a side-by-side Python comparison of Perceptron vs Logistic Regression,
 including how they learn and their decision boundaries.
"""
# -----------------------------
# Generate Simple 2D Dataset
# -----------------------------
np.random.seed(0)

# Class 0: centered at (1, 1)
X0 = np.random.randn(20, 2) + np.array([1, 1])
# Class 1: centered at (3, 3)
X1 = np.random.randn(20, 2) + np.array([3, 3])

X = np.vstack((X0, X1))
y = np.hstack((np.zeros(20), np.ones(20)))  # Labels: 0 and 1

# -----------------------------
# Train Models
# -----------------------------
perceptron = Perceptron(max_iter=1000, tol=1e-3)
log_reg = LogisticRegression()

perceptron.fit(X, y)
log_reg.fit(X, y)

# -----------------------------
# Decision Boundary Function
# -----------------------------
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Model prediction
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# -----------------------------
# Plot Both Models
# -----------------------------
plot_decision_boundary(perceptron, X, y, "Perceptron Decision Boundary")
plot_decision_boundary(log_reg, X, y, "Logistic Regression Decision Boundary")


# If you'd like to see how the sigmoid turns raw scores into probabilities:
z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))
plt.plot(z, sigmoid)
plt.title("Sigmoid Function in Logistic Regression")
plt.xlabel("z = w·x + b")
plt.ylabel("Probability (P(y=1|x))")
plt.grid(True)
plt.show()


"""

🧠 What You'll See

Perceptron plot:
A sharp, hard boundary — it just splits the plane into two halves.
It doesn't care about probabilities, just whether a point is on one side or the other.

Logistic Regression plot:
A smoother boundary — behind the scenes, it calculates probabilities using the sigmoid function.
Even points near the edge get a probability (e.g., 0.4 or 0.6).

"""