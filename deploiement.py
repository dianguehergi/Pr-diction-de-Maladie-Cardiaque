import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Configurer le style pour les graphiques
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Charger le meilleur modèle
model = joblib.load('meilleur_model.pkl')
print("Modèle chargé pour le déploiement.")

# Charger les données pour obtenir des informations sur les colonnes et l'ensemble de test
df = pd.read_csv('HeartDiseaseUCI1.csv')
X = df.drop(columns='num')
y = df['num']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction principale de l'application
def main():
    st.title("Tableau de Bord de Prédiction de Maladie Cardiaque")
    
    # Section 1 : Prédiction pour un patient en fonction des caractéristiques saisies
    st.sidebar.header("Entrée des Caractéristiques du Patient")
    age = st.sidebar.number_input('Âge', min_value=0, max_value=120, value=50)
    sex = st.sidebar.selectbox('Sexe', ['Femme', 'Homme'])
    cp = st.sidebar.selectbox('Type de douleur thoracique', [
        'Type 0 : Asymptomatique', 
        'Type 1 : Angine typique', 
        'Type 2 : Angine atypique', 
        'Type 3 : Douleur non angineuse'
    ])
    trestbps = st.sidebar.number_input('Pression artérielle au repos (mm Hg)', min_value=0, value=120)
    chol = st.sidebar.number_input('Cholestérol sérique (mg/dl)', min_value=0, value=200)
    fbs = st.sidebar.selectbox('Glycémie à jeun > 120 mg/dl', ['Non', 'Oui'])
    restecg = st.sidebar.selectbox('Résultats de l\'ECG au repos', [
        '0 : Normal', 
        '1 : Anomalie de l\'onde ST-T', 
        '2 : Hypertrophie ventriculaire gauche'
    ])
    thalach = st.sidebar.number_input('Fréquence cardiaque maximale atteinte', min_value=0, value=150)
    exang = st.sidebar.selectbox('Angine induite par l\'exercice', ['Non', 'Oui'])
    oldpeak = st.sidebar.number_input('Dépression ST induite par l\'exercice', min_value=0.0, value=1.0, step=0.1)
    slope = st.sidebar.selectbox('Pente du segment ST', [
        '0 : Montant', 
        '1 : Plat', 
        '2 : Descendant'
    ])
    ca = st.sidebar.number_input('Nombre de vaisseaux colorés par fluoroscopie', min_value=0, max_value=4, value=0)
    thal = st.sidebar.selectbox('Thalassémie', [
        '0 : Normal', 
        '1 : Défaut fixe', 
        '2 : Défaut réversible', 
        '3 : Défaut inconnu'
    ])

    # Conversion des choix en valeurs numériques pour le modèle
    sex = 1 if sex == 'Homme' else 0
    cp = int(cp.split(':')[0].strip().split(' ')[1])
    fbs = 1 if fbs == 'Oui' else 0
    restecg = int(restecg.split(':')[0].strip())
    exang = 1 if exang == 'Oui' else 0
    slope = int(slope.split(':')[0].strip())
    thal = int(thal.split(':')[0].strip())

    # Préparer les données d'entrée pour la prédiction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal],
    })

    # Prédiction de la maladie pour les caractéristiques saisies
    st.subheader("Résultat de la Prédiction")
    if st.sidebar.button('Prédire'):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Affichage du résultat de la prédiction
        if prediction[0] == 1:
            st.write("Le patient est **malade**.")
        else:
            st.write("Le patient est **non malade**.")
        
        st.write(f"Probabilité de maladie : {prediction_proba[0][1] * 100:.2f}%")

        # Visualisation des caractéristiques du patient par rapport aux distributions des patients malades et non malades
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        sns.histplot(df[df['num'] == 0]['age'], color='blue', label='Non Malade', kde=True, ax=axes[0, 0])
        sns.histplot(df[df['num'] == 1]['age'], color='red', label='Malade', kde=True, ax=axes[0, 0])
        axes[0, 0].axvline(age, color='green', linestyle='--', label='Patient Actuel')
        axes[0, 0].set_title('Âge')
        axes[0, 0].legend()

        sns.histplot(df[df['num'] == 0]['chol'], color='blue', label='Non Malade', kde=True, ax=axes[0, 1])
        sns.histplot(df[df['num'] == 1]['chol'], color='red', label='Malade', kde=True, ax=axes[0, 1])
        axes[0, 1].axvline(chol, color='green', linestyle='--', label='Patient Actuel')
        axes[0, 1].set_title('Cholestérol')

        sns.histplot(df[df['num'] == 0]['trestbps'], color='blue', label='Non Malade', kde=True, ax=axes[1, 0])
        sns.histplot(df[df['num'] == 1]['trestbps'], color='red', label='Malade', kde=True, ax=axes[1, 0])
        axes[1, 0].axvline(trestbps, color='green', linestyle='--', label='Patient Actuel')
        axes[1, 0].set_title('Pression Artérielle au Repos')

        sns.histplot(df[df['num'] == 0]['thalach'], color='blue', label='Non Malade', kde=True, ax=axes[1, 1])
        sns.histplot(df[df['num'] == 1]['thalach'], color='red', label='Malade', kde=True, ax=axes[1, 1])
        axes[1, 1].axvline(thalach, color='green', linestyle='--', label='Patient Actuel')
        axes[1, 1].set_title('Fréquence Cardiaque Max')
        
        fig.tight_layout()
        st.pyplot(fig)

    # Section 2 : Évaluation globale du modèle
    st.header('Évaluation du Modèle sur l’Ensemble de Test')
    
    # Calcul du bénéfice net
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    TN, FP, FN, TP = cm.ravel()
    net_benefit = (TP * 30000) - (FN * 50000) - (FP * 10000)
    
    st.subheader("Bénéfice Net")
    st.write(f"Bénéfice Net estimé : {net_benefit} €")

    # Affichage de la matrice de confusion
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Malade', 'Malade'], yticklabels=['Non Malade', 'Malade'])
    ax.set_ylabel('Vrai label')
    ax.set_xlabel('Prédiction')
    st.pyplot(fig)

    # Rapport de classification
    st.subheader("Rapport de Classification")
    report = classification_report(y_test, y_pred_test, target_names=['Non Malade', 'Malade'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)

    # Visualisation des coûts et bénéfices en fonction des catégories
    labels = ['Faux Positifs (FP)', 'Faux Négatifs (FN)', 'Vrais Positifs (TP)', 'Vrais Négatifs (TN)']
    costs_benefits = [-FP * 10000, -FN * 50000, TP * 30000, 0]
    colors = ['orange', 'red', 'green', 'blue']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=labels, y=costs_benefits, palette=colors)
    ax.set_title("Bénéfice Net par Catégorie de Prédiction")
    ax.set_ylabel("Montant (€)")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
