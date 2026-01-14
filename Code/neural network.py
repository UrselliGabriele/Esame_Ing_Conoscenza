# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:23:16 2025
@author: Gabriele
"""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Input

# --- 1. FUNZIONE PER CREARE LA RETE ---
def create_model():
    network = Sequential()
    # shape=41: numero di feature dopo il get_dummies
    network.add(Input(shape=(41,))) 
    network.add(Dense(17, activation='relu'))
    network.add(Dense(1, activation='sigmoid'))
    network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return network

np.random.seed(7)

# --- 2. CARICAMENTO E PRE-PROCESSING ---
try:
    dataset = pd.read_csv("../Dataset/breast-cancer.csv", sep=",", 
                         names=["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"], 
                         dtype={'deg-malig':np.int32})
except FileNotFoundError:
    dataset = pd.read_csv("Dataset/breast-cancer.csv", sep=",", names=["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"])

# Trasformazione variabili categoriche
network_data = pd.get_dummies(dataset, columns=["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "breast", "breast-quad", "irradiat"])
network_data = network_data.drop(["class"], axis=1)

X = network_data.astype('float32')
y = pd.get_dummies(dataset["class"])["recurrence-events"].astype('float32')

# Bilanciamento Dataset (SMOTE)
sm = SMOTE(random_state=0)
X_res, y_res = sm.fit_resample(X, y.ravel())

# Split per l'addestramento principale
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size=0.75, random_state=13)

# --- 3. ADDESTRAMENTO MODELLO SINGOLO ---
model = create_model()
print("\n--- Inizio addestramento Rete Neurale ---")
model.fit(X_train, y_train, epochs=130, batch_size=10, verbose=1) 

# Predizioni
predictions = model.predict(X_test)
rounded = [round(x[0]) for x in predictions]

print('\nClassification report:\n', classification_report(y_test, rounded))

# --- 4. GENERAZIONE GRAFICI CLASSICI ---

# Grafico 1: Matrice di Confusione
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, rounded)
sn.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.title('Matrice di Confusione (Neural Network)')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.show()

# Grafico 2: Curva ROC
fpr, tpr, _ = roc_curve(y_test, predictions)
plt.figure(figsize=(7, 5))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='AUC: %.3f' % roc_auc_score(y_test, predictions))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend()
plt.show()

# Grafico 3: Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, rounded)
plt.figure(figsize=(7, 5))
plt.step(recall, precision, where='post', color='b')
plt.fill_between(recall, precision, alpha=0.2, color='b')
plt.title('Curva Precision-Recall (AP=%.2f)' % average_precision_score(y_test, rounded))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# --- 5. CALCOLO VARIANZA---
# Usiamo un ciclo for per evitare il bug di scikeras/sklearn
print("\n--- Calcolo Cross-Validation (5 fold) ---")

kf = KFold(n_splits=5, shuffle=True, random_state=7)
cv_scores = []

# Trasformiamo X_res e y_res in DataFrame/Series per usare .iloc se necessario
X_res_df = pd.DataFrame(X_res)
y_res_s = pd.Series(y_res)

fold = 1
for train_idx, val_idx in kf.split(X_res_df):
    X_t_cv, X_v_cv = X_res_df.iloc[train_idx], X_res_df.iloc[val_idx]
    y_t_cv, y_v_cv = y_res_s.iloc[train_idx], y_res_s.iloc[val_idx]
    
    # Creiamo un modello "fresco" per ogni fold
    m_cv = create_model()
    m_cv.fit(X_t_cv, y_t_cv, epochs=30, batch_size=10, verbose=0)
    
    # Valutazione
    _, accuracy = m_cv.evaluate(X_v_cv, y_v_cv, verbose=0)
    cv_scores.append(accuracy)
    print(f"Fold {fold} completato. Accuracy: {accuracy:.4f}")
    fold += 1

# Calcolo metriche di stabilità
varianza = np.var(cv_scores)
dev_std = np.std(cv_scores)

print('\nMedia Accuracy CV: {:.4f}'.format(np.mean(cv_scores)))
print('Varianza: {:.6f}'.format(varianza))
print('Deviazione Standard: {:.4f}'.format(dev_std))

# Grafico 4: Istogramma della Varianza (Stabilità)
plt.figure(figsize=(6, 4))
plt.bar(['Varianza', 'Dev Standard'], [varianza, dev_std], color=['orange', 'purple'])
plt.title('Grafico Stabilità: Varianza e Dev. Standard')
plt.ylabel('Valore Metrica')
plt.show()
