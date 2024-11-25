import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# MLP Class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def _activate(self, Z):
        if self.activation_fn == 'tanh':
            return np.tanh(Z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, Z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._activate(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))
        return self.A2

    def backward(self, X, y):
        m = y.shape[0]
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (1 - np.tanh(self.Z1)**2)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

# Generate data
def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int).reshape(-1, 1)
    return X, y

# Visualization function
def update(frame, mlp, X, y, ax):
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    ax.clear()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid_points).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=["blue", "red"])
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolor="k")
    ax.set_title(f"Step {frame * 10}")

# Main visualization loop
def visualize(activation, lr, steps):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    fig, ax = plt.subplots(figsize=(8, 6))
    ani = FuncAnimation(fig, update, frames=steps // 10, fargs=(mlp, X, y, ax), repeat=False)
    ani.save(os.path.join(RESULTS_DIR, "mlp_visualization.gif"), writer="pillow", fps=10)
    plt.close()

if __name__ == "__main__":
    visualize(activation="tanh", lr=0.1, steps=1000)
