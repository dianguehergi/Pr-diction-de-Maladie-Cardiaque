import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import joblib

# Charger les données
df = pd.read_csv('HeartDiseaseUCI1.csv')
X = df.drop(columns='num')  # Caractéristiques (âge, cholestérol, etc.)
y = df['num']  # Variable cible (1 = malade, 0 = non malade)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paramètres pour l'optimisation de chaque modèle
param_grids = {
    "RandomForest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # Distance de Minkowski (1 = distance de Manhattan, 2 = distance Euclidienne)
    },
    "LogisticRegression": {
        'C': [0.01, 0.1, 1, 10],  # Régularisation
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [100, 200, 500]
    }
}

# Initialiser les modèles
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(random_state=42)
}

# Variables pour stocker le meilleur modèle
best_model = None
best_model_name = ""
best_net_benefit = float('-inf')  # Initialiser à une valeur très basse

# Évaluer chaque modèle
results = []

for name, model in models.items():
    # Optimiser les hyperparamètres avec GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='recall', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Meilleur modèle après optimisation
    best_model_optimized = grid_search.best_estimator_
    
    # Prédictions sur l'ensemble de test
    y_pred = best_model_optimized.predict(X_test)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calcul des valeurs pour les bénéfices nets
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    net_benefit = (TP * 30000) - (FN * 50000) - (FP * 10000)
    
    # Sauvegarder les résultats pour chaque modèle
    results.append({
        "Model": name,
        "Best Params": grid_search.best_params_,
        "Accuracy": accuracy,
        "Recall": recall,
        "F1 Score": f1,
        "Net Benefit (€)": net_benefit
    })
    
    # Sélectionner le modèle avec le bénéfice net le plus élevé
    if net_benefit > best_net_benefit:
        best_net_benefit = net_benefit
        best_model = best_model_optimized
        best_model_name = name

# Afficher les résultats des modèles
results_df = pd.DataFrame(results)
print("Évaluation des modèles après optimisation :")
print(results_df)

print(f"\nMeilleur modèle : {best_model_name} avec un bénéfice net de {best_net_benefit}€")

# Sauvegarder le meilleur modèle pour le déploiement
joblib.dump(best_model, 'meilleur_model.pkl')
print(f"Modèle {best_model_name} sauvegardé sous 'meilleur_model.pkl'")
