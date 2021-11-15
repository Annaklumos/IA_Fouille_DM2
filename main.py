
#________________________________ MODULES A IMPORTER ______________________________________#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import cross_val_score
import statistics as st

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
report = classification_report(y_test, predictions, output_dict=True)
print(report)

# Variation du nombre de couches et du nb de neurones dans les couches cachées

# _______ 1 couche _______
L_precision_1 = []; L_recall_1 = []; L_cross_vall_1 = []
for i in range (1,31):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(i),
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_1.append(moy_precision)
    L_recall_1.append(moy_recall)
    L_cross_vall_1.append(moy_cross_val)


# _______ 2 couches _______
L_precision_2 = []; L_recall_2 = []; L_cross_vall_2 = []
for i in range (1,31):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(i, i),
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n = n + 1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_2.append(moy_precision)
    L_recall_2.append(moy_recall)
    L_cross_vall_2.append(moy_cross_val)


# _______ 3 couches _______
L_precision_3 = []; L_recall_3 = []; L_cross_vall_3 = []
for i in range (1,31):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(i, i, i),
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n = n + 1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_3.append(moy_precision)
    L_recall_3.append(moy_recall)
    L_cross_vall_3.append(moy_cross_val)


# _______ 4 couches _______
L_precision_4 = []; L_recall_4 = []; L_cross_vall_4 = []
for i in range (1,31):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(i, i, i, i),
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n = n + 1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_4.append(moy_precision)
    L_recall_4.append(moy_recall)
    L_cross_vall_4.append(moy_cross_val)


# _______ 5 couches _______
L_precision_5 = []; L_recall_5 = []; L_cross_vall_5 = []
for i in range (1,31):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(i, i, i, i, i),
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n = n + 1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_5.append(moy_precision)
    L_recall_5.append(moy_recall)
    L_cross_vall_5.append(moy_cross_val)

nb_neurones = range(1,31)

#Tracé de la précision du MLP en fonction du nb de couches cachées

plt.plot(nb_neurones, L_precision_1, label = '1 couche')
plt.plot(nb_neurones, L_precision_2, label = '2 couche')
plt.plot(nb_neurones, L_precision_3, label = '3 couche')
plt.plot(nb_neurones, L_precision_4, label = '4 couche')
plt.plot(nb_neurones, L_precision_5, label = '5 couche')
plt.title('Précision')
plt.xlabel('Nombre de neurones par couches')
plt.ylabel('Moyenne de la précision sur 3 apprentissages')
#plt.ylim(0.95, 1)
plt.legend()
plt.show()

#Tracé du recall du MLP en fonction du nb de couches cachées

plt.plot(nb_neurones, L_recall_1, label = '1 couche')
plt.plot(nb_neurones, L_recall_2, label = '2 couche')
plt.plot(nb_neurones, L_recall_3, label = '3 couche')
plt.plot(nb_neurones, L_recall_4, label = '4 couche')
plt.plot(nb_neurones, L_recall_5, label = '5 couche')
plt.xlabel('Nombre de neurones par couches')
plt.ylabel('Moyenne du rappel sur 3 apprentissages')
#plt.ylim(0.95, 1)
plt.title('Recall')
plt.legend()
plt.show()

#Tracé de l'erreur par validation croisé à 10 plis en fonction du nb de couches cachées

plt.plot(nb_neurones, L_cross_vall_1, label = '1 couche')
plt.plot(nb_neurones, L_cross_vall_2, label = '2 couche')
plt.plot(nb_neurones, L_cross_vall_3, label = '3 couche')
plt.plot(nb_neurones, L_cross_vall_4, label = '4 couche')
plt.plot(nb_neurones, L_cross_vall_5, label = '5 couche')
plt.xlabel('Nombre de neurones par couches')
plt.ylabel("Moyenne du taux d'erreur par validation croisée à 10 plis sur 3 apprentissages")
#plt.ylim(0.95, 1)
plt.title('Cross validation')
plt.legend()
plt.show()

#Multiplot pour chaque MLP à k couches

plt.figure()
plt.subplot(321)
plt.plot(nb_neurones, L_precision_1, marker='o', label = 'Précision')
plt.plot(nb_neurones, L_recall_1, label = 'Rappel')
plt.plot(nb_neurones, L_cross_vall_1, label = "Taux d'erreur")
plt.xlabel('Nombre de neurones par couches')
plt.title('1 couche')
plt.legend()
plt.subplot(322)
plt.plot(nb_neurones, L_precision_2, marker='o', label = 'Précision')
plt.plot(nb_neurones, L_recall_2, label = 'Rappel')
plt.plot(nb_neurones, L_cross_vall_2, label = "Taux d'erreur")
plt.xlabel('Nombre de neurones par couches')
plt.title('2 couches')
plt.legend()
plt.subplot(323)
plt.plot(nb_neurones, L_precision_3, marker='o', label = 'Précision')
plt.plot(nb_neurones, L_recall_3, label = 'Rappel')
plt.plot(nb_neurones, L_cross_vall_3, label = "Taux d'erreur")
plt.xlabel('Nombre de neurones par couches')
plt.title('3 couches')
plt.legend()
plt.subplot(324)
plt.plot(nb_neurones, L_precision_4, marker='o', label = 'Précision')
plt.plot(nb_neurones, L_recall_4, label = 'Rappel')
plt.plot(nb_neurones, L_cross_vall_4, label = "Taux d'erreur")
plt.xlabel('Nombre de neurones par couches')
plt.title('4 couches')
plt.legend()
plt.subplot(325)
plt.plot(nb_neurones, L_precision_5, marker='o', label = 'Précision')
plt.plot(nb_neurones, L_recall_5, label = 'Rappel')
plt.plot(nb_neurones, L_cross_vall_5, label = "Taux d'erreur")
plt.xlabel('Nombre de neurones par couches')
plt.title('5 couches')
plt.legend()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.35)
plt.show()

# Variation du nombre de couches de neurones et du nombre d'itération effectuées lors de l'apprentissage

# _______ 1 couche _______
L_precision_11 = []; L_recall_11 = []; L_cross_vall_11 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(10),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_11.append(moy_precision)
    L_recall_11.append(moy_recall)
    L_cross_vall_11.append(moy_cross_val)

# _______ 2 couche _______
L_precision_12 = []; L_recall_12 = []; L_cross_vall_12 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(10, 10),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_12.append(moy_precision)
    L_recall_12.append(moy_recall)
    L_cross_vall_12.append(moy_cross_val)

# _______ 3 couche _______
L_precision_13 = []; L_recall_13 = []; L_cross_vall_13 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(10, 10, 10),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_13.append(moy_precision)
    L_recall_13.append(moy_recall)
    L_cross_vall_13.append(moy_cross_val)

# _______ 4 couche _______
L_precision_14 = []; L_recall_14 = []; L_cross_vall_14 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(10, 10, 10, 10),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_14.append(moy_precision)
    L_recall_14.append(moy_recall)
    L_cross_vall_14.append(moy_cross_val)

# _______ 5 couche _______
L_precision_15 = []; L_recall_15 = []; L_cross_vall_15 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(10, 10, 10, 10, 10),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_15.append(moy_precision)
    L_recall_15.append(moy_recall)
    L_cross_vall_15.append(moy_cross_val)

nb_iterations = range(1,1001, 25)

#Multiplot pour chaque MLP à k couches

plt.figure()
plt.subplot(321)
plt.plot(nb_iterations, L_precision_11, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_11, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_11, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 1 couche de 10 neurones')
plt.legend()
plt.subplot(322)
plt.plot(nb_iterations, L_precision_12, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_12, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_12, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 2 couches de 10 neurones')
plt.legend()
plt.subplot(323)
plt.plot(nb_iterations, L_precision_13, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_13, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_13, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 3 couches de 10 neurones')
plt.legend()
plt.subplot(324)
plt.plot(nb_iterations, L_precision_14, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_14, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_14, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 4 couches de 10 neurones')
plt.legend()
plt.subplot(325)
plt.plot(nb_iterations, L_precision_15, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_15, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_15, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 5 couches de 10 neurones')
plt.legend()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.35)
plt.show()

# Variation du nombre de neurones et du nombre d'itération effectuées lors de l'apprentissage

# _______ 1 couche _______
L_precision_111 = []; L_recall_111 = []; L_cross_vall_111 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(1, 1, 1),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_111.append(moy_precision)
    L_recall_111.append(moy_recall)
    L_cross_vall_111.append(moy_cross_val)

# _______ 2 couche _______
L_precision_112 = []; L_recall_112 = []; L_cross_vall_112 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(10, 10, 10),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_112.append(moy_precision)
    L_recall_112.append(moy_recall)
    L_cross_vall_112.append(moy_cross_val)

# _______ 3 couche _______
L_precision_113 = []; L_recall_113 = []; L_cross_vall_113 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(20, 20, 20),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_113.append(moy_precision)
    L_recall_113.append(moy_recall)
    L_cross_vall_113.append(moy_cross_val)

# _______ 4 couche _______
L_precision_114 = []; L_recall_114 = []; L_cross_vall_114 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(50, 50, 50),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_114.append(moy_precision)
    L_recall_114.append(moy_recall)
    L_cross_vall_114.append(moy_cross_val)

# _______ 5 couche _______
L_precision_115 = []; L_recall_115 = []; L_cross_vall_115 = []
for i in range (1,1001, 25):
    n=0; L_moy_precision = []; L_moy_recall = []; L_moy_cross_val = []
    while n<3:
        mlp = MLPClassifier(activation='relu',
                            alpha=0.0001,
                            batch_size='auto',
                            beta_1=0.9,
                            beta_2=0.999,
                            early_stopping=False,
                            epsilon=(10 ** (-8)),
                            hidden_layer_sizes=(100, 100, 100),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=i,
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
        predictions = mlp.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        L_moy_precision.append(report['weighted avg']['precision'])
        L_moy_recall.append(report['weighted avg']['recall'])
        L_moy_cross_val.append(st.mean(cross_val_score(mlp, X_train, y_train, cv=10, scoring='accuracy')))
        n=n+1
    moy_precision = st.mean(L_moy_precision)
    moy_recall = st.mean(L_moy_recall)
    moy_cross_val = st.mean(L_moy_cross_val)
    L_precision_115.append(moy_precision)
    L_recall_115.append(moy_recall)
    L_cross_vall_115.append(moy_cross_val)

nb_iterations = range(1,1001, 25)

#Multiplot pour chaque MLP à k couches

plt.figure()
plt.subplot(321)
plt.plot(nb_iterations, L_precision_111, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_111, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_111, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 3 couche de 1 neurone')
plt.legend()
plt.subplot(322)
plt.plot(nb_iterations, L_precision_112, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_112, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_112, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 3 couches de 10 neurones')
plt.legend()
plt.subplot(323)
plt.plot(nb_iterations, L_precision_113, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_113, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_113, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 3 couches de 20 neurones')
plt.legend()
plt.subplot(324)
plt.plot(nb_iterations, L_precision_114, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_114, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_114, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 3 couches de 50 neurones')
plt.legend()
plt.subplot(325)
plt.plot(nb_iterations, L_precision_115, marker='o', label = 'Précision')
plt.plot(nb_iterations, L_recall_115, label = 'Rappel')
plt.plot(nb_iterations, L_cross_vall_115, label = "Taux d'erreur")
plt.xlabel("Nombre d'itération maximum")
plt.title('Evaluation des performances pour 3 couches de 100 neurones')
plt.legend()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.35)
plt.show()

