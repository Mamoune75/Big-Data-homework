import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


np.random.seed(42)
data_size = 5000

df = pd.DataFrame({
    "Revenu": np.random.randint(15000, 100000, data_size),  # Revenu annuel
    "Âge": np.random.randint(18, 70, data_size),  # Âge du client
    "Montant_Prêt": np.random.randint(1000, 50000, data_size),  # Montant du prêt
    "Durée_Prêt": np.random.randint(6, 60, data_size),  # Durée en mois
    "Historique_Paiement": np.random.choice([0, 1, 2], data_size, p=[0.7, 0.2, 0.1]),  # 0 = bon, 1 = moyen, 2 = mauvais
    "Risque_Crédit": np.random.choice([0, 1], data_size, p=[0.85, 0.15])  # 0 = remboursement, 1 = défaut de paiement
})

# Afficher les premières lignes du dataset
df.head()


X = df.drop(columns=["Risque_Crédit"])  # Variables explicatives
y = df["Risque_Crédit"]  # Variable cible

# Séparer les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Afficher la taille des ensembles
X_train.shape, X_test.shape


model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Précision du modèle : {accuracy:.2f}")


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Bon client", "Mauvais client"], yticklabels=["Bon client", "Mauvais client"])
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.show()


feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importances.plot(kind="bar", title="Importance des variables")
plt.show()
