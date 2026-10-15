import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import networkx as nx


tf.keras.utils.set_random_seed(42)

iris = load_iris()
X = iris.data   # 4 features
y = iris.target # classes 0, 1, 2
print("Dimensions X :", X.shape)
print("Dimensions y :", y.shape)

print("Exemple de  X :\n", X[:5])
print("Exemple de  y :\n", y[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))
print("Sortie encode :\n", y_train[:5])

##########################################
##      Normalisation   (standardisation)
# a faire apres separation pour evité les fuite
##########################################
#StandardScaler : centre les données autour de 0 
# avec un écart-type de 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Features normalisées :\n", X_train[:5])

model = Sequential([
    Input(shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()

# Couche 1 (Dense 8)  : 4 entrées × 8 neurones + 8 biais = 32 + 8 = 40
# Couche 2 (Dense 3)  : 8 entrées × 3 neurones + 3 biais = 24 + 3 = 27
# Total : 67 paramètres


##Étape 7 : Compiler le modèle
layers = [4, 8, 3]

G = nx.DiGraph()
pos = {}
layer_positions = []

for l, n_neurons in enumerate(layers):
    layer_pos = []
    for i in range(n_neurons):
        node_id = f"L{l}N{i}"
        G.add_node(node_id, layer=l)
        pos[node_id] = (l, i)  # Positionner les neurones en colonnes
        layer_pos.append(node_id)
    layer_positions.append(layer_pos)

for l in range(len(layers) - 1):
    for src in layer_positions[l]:
        for dst in layer_positions[l + 1]:
            G.add_edge(src, dst)

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=False,
        node_size=500, node_color="lightgreen", edge_color="gray")
plt.title("Réseau de neurones pour Iris (4-8-3)", fontsize=14)
plt.show()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#Étape 8 : Entraîner le modèle
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=5,
    validation_split=0.1
)

history = model.fit(X_train, y_train, epochs=100, batch_size=5, validation_split=0.1)

# Étape 9 : Évaluer sur le test
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")


#Étape 10 : Prédire sur de nouveaux exemples
predictions = model.predict(X_test)
print("Predictions :\n", predictions[:5].round(3))
print("Classes predites :", np.argmax(predictions, axis=1)[:20])
print("Vraies classes :", np.argmax(y_test, axis=1)[:20])

## Etape 11 : Visualiser la prédication pour quelques exemples
predictions = model.predict(X_test)
print("Predictions :\n", predictions[:5].round(3))
print("Classes predites :", np.argmax(predictions, axis=1)[:20])
print("Vraies classes :", np.argmax(y_test, axis=1)[:20])


