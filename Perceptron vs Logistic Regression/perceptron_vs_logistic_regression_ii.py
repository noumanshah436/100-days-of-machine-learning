import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron, LogisticRegression


"""
this next part will visualize how the Perceptron and Logistic Regression update their decision boundaries 
during training, so you can see the difference between the “jumping” and “smoothing” behavior.

Below is a complete Python example using matplotlib.animation — 
"""
# -----------------------------
# Generate linearly separable data
# -----------------------------
np.random.seed(1)
X, y = make_blobs(n_samples=40, centers=2, random_state=2, cluster_std=1.2)

# -----------------------------
# Utility: plot decision boundary
# -----------------------------
def plot_boundary(ax, w, b, color, label):
    x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
    y_vals = -(w[0]*x_vals + b) / w[1]
    ax.plot(x_vals, y_vals, color=color, label=label, lw=2)

# -----------------------------
# Custom Perceptron training loop (manual updates)
# -----------------------------
def train_perceptron(X, y, lr=0.1, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0
    history = []

    for _ in range(epochs):
        for xi, target in zip(X, y):
            pred = 1 if np.dot(w, xi) + b > 0 else 0
            update = lr * (target - pred)
            w += update * xi
            b += update
        history.append((w.copy(), b))
    return history

# -----------------------------
# Custom Logistic Regression (manual gradient descent)
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic(X, y, lr=0.1, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0
    history = []

    for _ in range(epochs):
        linear = np.dot(X, w) + b
        pred = sigmoid(linear)
        error = y - pred
        w += lr * np.dot(X.T, error)
        b += lr * np.sum(error)
        history.append((w.copy(), b))
    return history

# -----------------------------
# Train both models
# -----------------------------
perc_hist = train_perceptron(X, y, lr=0.1, epochs=20)
log_hist = train_logistic(X, y, lr=0.1, epochs=20)

# -----------------------------
# Setup the plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolor='k', s=60)
ax.set_xlim(X[:,0].min()-1, X[:,0].max()+1)
ax.set_ylim(X[:,1].min()-1, X[:,1].max()+1)
ax.set_title("Perceptron vs Logistic Regression — Training Animation")

perc_line, = ax.plot([], [], 'b-', lw=2, label="Perceptron")
log_line,  = ax.plot([], [], 'r--', lw=2, label="Logistic Regression")
ax.legend()

# -----------------------------
# Animation function
# -----------------------------
def update(frame):
    # Remove only previous decision lines, not the scatter plot base
    for artist in ax.lines[:]:
        artist.remove()

    w_p, b_p = perc_hist[frame]
    w_l, b_l = log_hist[frame]

    plot_boundary(ax, w_p, b_p, 'blue', 'Perceptron')
    plot_boundary(ax, w_l, b_l, 'red', 'Logistic Regression')

    ax.set_title(f"Training Iteration {frame + 1}")
    return ax.lines


anim = FuncAnimation(fig, update, frames=len(perc_hist), interval=800, repeat=False)
plt.show()



"""
🧠 What You'll Observe

Perceptron:	The decision line jumps abruptly whenever it finds a misclassified point. Updates happen in steps
 and only when a mistake is made.

Logistic Regression:  The line slides smoothly toward the optimal boundary, adjusting gradually at every iteration — even for correctly classified points.


🎯 Key Takeaway

The Perceptron is more reactive — it corrects only mistakes.

Logistic Regression is more statistical — it continuously adjusts to minimize total loss.

That's why logistic regression is more stable and robust, especially on noisy data.
"""