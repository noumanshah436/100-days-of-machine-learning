import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Training Data
# -----------------------------
# Format: [x1, x2, label]
data = np.array([
    [0, 1, -1],  # a
    [2, 0, -1],  # b
    [1, 0, +1]   # c
])

# Learning rate
eta = 1.0

# Initial weights: w0 (bias), w1, w2
w = np.array([-1.5, 0.0, 2.0])  # [w0, w1, w2]

def predict(x, w):
    """Predict class (+1 or -1) for input x=[x1, x2]."""
    z = w[0] + w[1]*x[0] + w[2]*x[1]
    return 1 if z > 0 else -1

def plot_decision_boundary(w, title, ax):
    """Plot points and decision boundary."""
    ax.set_title(title)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Plot data points
    for x1, x2, label in data:
        ax.scatter(x1, x2, c='red' if label == -1 else 'blue', s=80, label=f'class {label}')
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # Decision boundary: w0 + w1*x1 + w2*x2 = 0  =>  x2 = (-w0 - w1*x1)/w2
    x_vals = np.linspace(-1, 3, 100)
    if w[2] != 0:
        y_vals = (-w[0] - w[1]*x_vals) / w[2]
        ax.plot(x_vals, y_vals, 'k--', label="Decision Boundary")

# -----------------------------
# Plot before training
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(w, "Before Training", axes[0])

# -----------------------------
# Perceptron Learning Algorithm
# -----------------------------
epochs = 0
while True:
    error_count = 0
    for x1, x2, label in data:
        x = np.array([x1, x2])
        y_hat = predict(x, w)

        if y_hat != label:
            # Update weights: w = w + eta * y * [1, x1, x2]
            w = w + eta * label * np.array([1, x1, x2])
            error_count += 1
            print(f"Update {epochs+1}: for point ({x1}, {x2}), label={label} → new weights: {w}")
    epochs += 1
    if error_count == 0 or epochs > 50:
        break

# -----------------------------
# Plot after training
# -----------------------------
plot_decision_boundary(w, "After Training", axes[1])
plt.show()

print(f"\nFinal weights after {epochs} epochs: {w}")
