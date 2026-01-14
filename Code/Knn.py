# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:16:27 2025

@author: Gabriele
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

# --- CARICAMENTO DATI ---
# Uso i nomi delle colonne come nel dataset originale
colonne = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]
colonne_da_trasformare = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "breast", "breast-quad", "irradiat"]

# Provo a caricare il file (gestisco entrambi i percorsi per sicurezza)
try:
    df = pd.read_csv("../Dataset/breast-cancer.csv", names=colonne)
except:
    df = pd.read_csv("Dataset/breast-cancer.csv", names=colonne)

# Trasformo le variabili categoriche in numeri (dummy)
df_nuovo = pd.get_dummies(df, columns=colonne_da_trasformare)

# Separo le feature (X) dal target (y)
X = df_nuovo.drop(["class"], axis=1)
y = pd.get_dummies(df["class"])["recurrence-events"]

# Divido in Training e Test (uso 75/25 come split)
# Tengo il random_state fisso così i risultati non cambiano ogni volta che premo RUN
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=13)


# =============================================================================
# STUDIO DEL VALORE K (Grafico 1)
# =============================================================================
lista_errori = []

for i in range(1, 20):
    test_knn = KNeighborsClassifier(n_neighbors=i)
    test_knn.fit(X_train, y_train)
    pred_i = test_knn.predict(X_test)
    lista_errori.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), lista_errori, color='red', linestyle='--', marker='o', markerfacecolor='blue')
plt.title('Ricerca del valore K ottimale')
plt.xlabel('Valore K')
plt.ylabel('Errore Medio')
plt.show()


# =============================================================================
# MODELLO BASE (Sbilanciato)
# =============================================================================
modello_standard = KNeighborsClassifier(n_neighbors=2)
modello_standard.fit(X_train, y_train)
pred_standard = modello_standard.predict(X_test)
prob_standard = modello_standard.predict_proba(X_test)[:, 1]

print("\n--- RISULTATI MODELLO STANDARD ---")
print(classification_report(y_test, pred_standard))

# Matrice di Confusione (Grafico 2)
plt.figure()
matrice_1 = confusion_matrix(y_test, pred_standard)
sns.heatmap(matrice_1, annot=True, fmt='g', cmap='Blues')
plt.title("Matrice di Confusione (Dati Originali)")
plt.show()

# Curva ROC (Grafico 3)
fpr, tpr, _ = roc_curve(y_test, prob_standard)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='AUC: ' + str(round(roc_auc_score(y_test, prob_standard), 2)))
plt.title("Curva ROC (Standard)")
plt.legend()
plt.show()

# Curva Precision-Recall (Grafico 4)
p, r, _ = precision_recall_curve(y_test, pred_standard)
plt.figure()
plt.step(r, p, where='post', color='blue')
plt.fill_between(r, p, alpha=0.2, color='blue')
plt.title("Precision-Recall (Standard)")
plt.show()


# =============================================================================
# BILANCIAMENTO CON SMOTE
# =============================================================================
# Uso SMOTE per creare esempi sintetici della classe minoritaria
sm = SMOTE(random_state=0)
X_resampled, y_resampled = sm.fit_resample(X, y.ravel())

# Rifaccio lo split sui dati nuovi
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resampled, y_resampled, train_size=0.75, random_state=13)

modello_smote = KNeighborsClassifier(n_neighbors=2)
modello_smote.fit(X_train_res, y_train_res)
pred_res = modello_smote.predict(X_test_res)
prob_res = modello_smote.predict_proba(X_test_res)[:, 1]

print("\n--- RISULTATI DOPO SMOTE ---")
print(classification_report(y_test_res, pred_res))

# Matrice di Confusione SMOTE (Grafico 5)
plt.figure()
matrice_2 = confusion_matrix(y_test_res, pred_res)
sns.heatmap(matrice_2, annot=True, fmt='g', cmap='Greens')
plt.title("Matrice di Confusione (SMOTE)")
plt.show()

# Curva ROC SMOTE (Grafico 6)
fpr2, tpr2, _ = roc_curve(y_test_res, prob_res)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr2, tpr2, label='AUC: ' + str(round(roc_auc_score(y_test_res, prob_res), 2)))
plt.title("Curva ROC (SMOTE)")
plt.legend()
plt.show()

# Curva Precision-Recall SMOTE (Grafico 7)
p2, r2, _ = precision_recall_curve(y_test_res, pred_res)
plt.figure()
plt.step(r2, p2, where='post', color='green')
plt.fill_between(r2, p2, alpha=0.2, color='green')
plt.title("Precision-Recall (SMOTE)")
plt.show()


# =============================================================================
# ANALISI STABILITÀ (Grafico 8)
# =============================================================================
# Uso la Cross Validation per vedere quanto è stabile il modello bilanciato
punteggi_cv = cross_val_score(modello_smote, X_resampled, y_resampled, cv=5)

plt.figure(figsize=(6, 4))
nomi_metriche = ['Varianza', 'Deviazione Standard']
valori_metriche = [np.var(punteggi_cv), np.std(punteggi_cv)]
plt.bar(nomi_metriche, valori_metriche, color=['orange', 'purple'])
plt.title("Metriche di Stabilità (CV=5)")
plt.show()