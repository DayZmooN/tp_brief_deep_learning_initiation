# Deep Learning - mes exos

Voilà mes exercices de Deep Learning. J'ai commencé par le perceptron tout simple à la main, puis j'ai enchaîné sur les MLP, les CNN et enfin les RNN. Tout est dans l'ordre du cours.

## Ce qu'il y a dedans

- **Introduction au Deep Learning.html** → le support de cours interactif qu'on nous a donné. Il y a des labos où on peut jouer avec le neurone, les activations, les optimizers. Faut juste l'ouvrir dans un navigateur.
- **exercices_dl.py** → mon premier perceptron avec la MSE, pour les portes AND et NOR.
- **exercice_1_perceptron.ipynb** → le même mais avec la sigmoïde et l'entropie croisée. J'ai compris à ce moment-là pourquoi la BCE descend vachement plus vite que la MSE.
- **exercice_2_iris_mlp.ipynb** → MLP sur Iris. Petit réseau 4-8-3, ça marche bien (~96%).
- **exercice_2_mlp_irs.py** → la version script du même exo, avec un dessin du réseau fait avec networkx parce que je voulais voir à quoi ça ressemblait.
- **exercice_2_titanic.ipynb** → classification binaire sur Titanic. Le plus galère niveau préparation des données (valeurs manquantes, one-hot, scaler...).
- **exercice_3_cnn_mnist.ipynb** → CNN sur MNIST, 98.8% au final.
- **exercice_4_cnn_fashion_mnist.ipynb** → pareil mais sur Fashion-MNIST, 91%.
- **exercice_RNN.ipynb** → génération de texte avec les poèmes de Victor Hugo. J'ai comparé SimpleRNN et LSTM.
- **victor_hugo-texts.txt** → le corpus que j'ai utilisé pour le RNN.
- **requirements.txt** → les dépendances.

## Résultats vite fait

- Perceptron AND → sorties ≈ [0, 0, 0, 0.97] après 2000 époques
- Iris → 96% sur le test
- Titanic → 79% (F1 de 0.79)
- MNIST → 98.8%
- Fashion-MNIST → 91%
- Victor Hugo → SimpleRNN loss 1.55 vs LSTM loss 0.43

Le LSTM écrase clairement le SimpleRNN sur le texte, c'est assez flagrant sur les courbes.

## Installer

```bash
python -m venv venv
source venv/bin/activate     # Linux/mac
# venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Lancer

Pour les notebooks :

```bash
jupyter notebook
```

Pour les scripts :

```bash
python exercices_dl.py
python exercice_2_mlp_irs.py
```

Pour le support de cours, j'ouvre juste le HTML dans Firefox, ça marche direct.

## Trucs que j'ai retenus

- Un neurone seul, il fait AND/OR/NOR mais pas XOR. C'est pour ça qu'on empile des couches.
- **Toujours séparer train/test AVANT le scaler**, sinon on triche. J'ai failli me faire avoir sur Iris et Titanic. Le `fit` du `StandardScaler` doit se faire sur le train seulement, sinon la moyenne et l'écart-type du test contaminent l'entraînement.
- Adam c'est le plus simple pour démarrer, SGD c'est bien pour comprendre mais faut vraiment régler le learning rate au poil.
- Sur les images, faut penser à normaliser (`/255`) et à ajouter la dimension du canal avec `expand_dims`, sinon le Conv2D râle.
- Pour la génération de texte, la température c'est rigolo : à 0.5 le modèle radote, à 1.2 il part dans tous les sens.

## Structure

```
.
├── Introduction au Deep Learning.html
├── exercices_dl.py
├── exercice_1_perceptron.ipynb
├── exercice_2_iris_mlp.ipynb
├── exercice_2_mlp_irs.py
├── exercice_2_titanic.ipynb
├── exercice_3_cnn_mnist.ipynb
├── exercice_4_cnn_fashion_mnist.ipynb
├── exercice_RNN.ipynb
├── victor_hugo-texts.txt
├── requirements.txt
└── README.md
```

## Environnement

Python 3.14 avec un venv, TensorFlow/Keras, NumPy, pandas, scikit-learn, seaborn, matplotlib et networkx. Rien d'exotique.

---

Fait dans le cadre du cours d'Intro au Deep Learning, 2025-2026.
