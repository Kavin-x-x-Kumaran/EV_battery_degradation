"""
Minimal from-scratch NumPy MLP.

Kept dependency-free and tiny on purpose: the whole point of the federated
layer is that we need to average WEIGHTS across clients, and a plain NumPy
model makes that trivial (no framework-specific state dicts / optimizer
buffers to reconcile). This also doubles as the "edge-deployable" model --
its full parameter set is a few hundred floats, easy to quantize and run on
constrained BMS hardware.

Architecture: y = x @ Ws + tanh(x @ W1 + b1) @ W2 + b2
The linear skip path (Ws) matters for degradation: SoH keeps falling below
anything seen in training, and a pure tanh network saturates there, while
the skip path lets the model extrapolate the trend. Small L2 weight decay
keeps the tiny dataset from being memorised.
"""
import numpy as np


class SoHRegressor:
    def __init__(self, n_features, hidden=16, seed=0, weight_decay=1e-3):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.hidden = hidden
        self.W1 = rng.normal(0, 0.3, (n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.3, (hidden, 1))
        self.b2 = np.zeros(1)
        self.Ws = np.zeros((n_features, 1))
        self.weight_decay = weight_decay

    # ---- forward ----
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2 + X @ self.Ws
        return z2.ravel(), a1, z1

    def predict(self, X):
        y, _, _ = self.forward(X)
        return y

    # ---- training (plain gradient descent, MSE loss) ----
    def train_step(self, X, y, lr=0.01):
        n = X.shape[0]
        y_pred, a1, z1 = self.forward(X)
        err = (y_pred - y)  # dL/dy_pred for MSE (up to constant factor)
        loss = np.mean(err ** 2)

        dW2 = a1.T @ (err.reshape(-1, 1)) / n
        db2 = np.mean(err)
        d_a1 = (err.reshape(-1, 1) @ self.W2.T) * (1 - np.tanh(z1) ** 2)
        dW1 = X.T @ d_a1 / n
        db1 = np.mean(d_a1, axis=0)
        dWs = X.T @ (err.reshape(-1, 1)) / n

        wd = self.weight_decay
        dW1 += wd * self.W1
        dW2 += wd * self.W2
        dWs += wd * self.Ws

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.Ws -= lr * dWs
        return loss

    def fit(self, X, y, epochs=200, lr=0.05):
        losses = []
        for _ in range(epochs):
            losses.append(self.train_step(X, y, lr=lr))
        return losses

    # ---- weight get/set for FedAvg ----
    def get_weights(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), self.Ws.copy()]

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, self.Ws = [w.copy() for w in weights]

    def n_params(self):
        return sum(w.size for w in self.get_weights())


def fed_avg(weight_list, sample_counts):
    """Weighted average of client weight lists (McMahan et al. FedAvg)."""
    total = sum(sample_counts)
    avg = None
    for weights, n in zip(weight_list, sample_counts):
        scale = n / total
        scaled = [w * scale for w in weights]
        if avg is None:
            avg = scaled
        else:
            avg = [a + s for a, s in zip(avg, scaled)]
    return avg
