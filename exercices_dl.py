import numpy as np

# Données AND
X = np.array([[0,0], [0,1], [1,0], [1,1]])
# y = np.array([0, 0, 0, 1])  # AND
y = np.array([1, 0, 0, 0]) # NOR 

np.random.seed(42)
w = np.random.randn(2)
b = np.random.randn()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X, w, b):
    return sigmoid(np.dot(X, w) + b)

def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

learning_rate = 0.1
epochs = 2000

for i in range(epochs):
    y_pred = predict(X, w, b)
    # Gradient de la MSE + sigmoïde
    grad_w = -2 * np.dot((y - y_pred) * y_pred * (1 - y_pred), X) / len(X)
    grad_b = -2 * np.mean((y - y_pred) * y_pred * (1 - y_pred))
    # MISE À JOUR CORRECTE : w = w - lr * gradient
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    if i % 200 == 0:
        print(f"Epoch {i}, Loss: {loss(y, y_pred):.4f}")

y_pred = predict(X, w, b)
print("\nPrédictions finales :", y_pred.round(4))
print("Classes prédites :", (y_pred > 0.5).astype(int))