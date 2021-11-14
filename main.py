
#________________________________ MODULES A IMPORTER ______________________________________#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#__________________________________________________________________________________________
# Expérience de prédiction de cancer de la poitrine

# Chargement des données
cancer = load_breast_cancer()

X = cancer['data']
y = cancer['target']

# Répartition des échantillons de test et d'apprentissage

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Normalisation des données

scaler = StandardScaler(copy = True, with_mean = True, with_std = True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Premier apprentissage

mlp = MLPClassifier(activation='relu',
                    alpha=0.0001,
                    batch_size='auto',
                    beta_1=0.9,
                    beta_2=0.999,
                    early_stopping=False,
                    epsilon=(10**(-8)),
                    hidden_layer_sizes=(30,30,30),
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    max_iter=200,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    power_t=0.5,
                    random_state=None,
                    shuffle=True,
                    solver='adam',
                    tol=0.0001,
                    validation_fraction=0.1,
                    verbose=False,
                    warm_start=False)

mlp.fit(X_train, y_train)

# Evaluation de la performance sur l'ensemble de test

predictions = mlp.predict(X_test)
print(classification_report(y_test, predictions))
