# ===============================
# Logistic Regression From Scratch
# ===============================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


# -------------------------------
# 1. Load and preprocess dataset
# -------------------------------

data = load_iris()
X = data.data
y = data.target

# Binary classification: Setosa vs Others
y = (y == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# -------------------------------
# 2. Logistic Regression Class
# -------------------------------

class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_hat):
        return -np.mean(
            y * np.log(y_hat + 1e-9) +
            (1 - y) * np.log(1 - y_hat + 1e-9)
        )

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros((n, 1))
        self.b = 0
        self.losses = []

        for _ in range(self.epochs):
            z = np.dot(X, self.W) + self.b
            y_hat = self.sigmoid(z)

            self.losses.append(self.loss(y, y_hat))

            dw = (1 / m) * np.dot(X.T, (y_hat - y))
            db = (1 / m) * np.sum(y_hat - y)

            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        return (self.sigmoid(z) >= 0.5).astype(int)


# -------------------------------
# 3. Train Scratch Model
# -------------------------------

model = LogisticRegressionScratch(learning_rate=0.01, epochs=2000)
model.fit(X_train, y_train)

y_pred_scratch = model.predict(X_test)


# -------------------------------
# 4. Evaluation (Scratch Model)
# -------------------------------

print("Accuracy (From Scratch):", accuracy_score(y_test, y_pred_scratch))
print("Precision:", precision_score(y_test, y_pred_scratch))
print("Recall:", recall_score(y_test, y_pred_scratch))
print("F1 Score:", f1_score(y_test, y_pred_scratch))


# -------------------------------
# 5. Loss Convergence Plot
# -------------------------------

plt.plot(model.losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Convergence (From Scratch)")
plt.show()


# -------------------------------
# 6. Sklearn Comparison
# -------------------------------

sk_model = LogisticRegression(max_iter=2000)
sk_model.fit(X_train, y_train.ravel())

y_pred_sk = sk_model.predict(X_test)

print("Accuracy (Sklearn):", accuracy_score(y_test, y_pred_sk))