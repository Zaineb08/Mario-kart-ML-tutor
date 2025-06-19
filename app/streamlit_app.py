# streamlit_app_creative.py

# --- Importation des bibliothèques nécessaires ---
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# --- 0. Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Mario Kart ML Tutor: L'Atelier de Combinaisons",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Custom CSS pour un look ludique et thématique ---
st.markdown("""
<style>
/* Style de l'en-tête principal */
.main-header {
    font-size: 3.5rem;
    font-weight: bold;
    text-align: center;
    color: #FF4B4B; /* Rouge Mario */
    text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
    margin-bottom: 2rem;
    padding-bottom: 15px;
    border-bottom: 3px solid #FF4B4B;
}

/* Cartes de statistiques pour les composants */
.component-card {
    background-color: #f0f2f6; /* Gris clair */
    color: #222;
    padding: 1.2rem;
    border-radius: 15px;
    border: 2px solid #E0F7FA; /*  Bleu clair */
    text-align: center;
    box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
    transition: 0.3s ease;
    margin-bottom: 15px;
}
.component-card:hover {
    transform: translateY(-5px);
    box-shadow: 8px 8px 20px rgba(0,0,0,0.2);
}
.component-card h3 {
    font-size: 1.8rem;
    font-weight: bold;
    color: #4CAF50; /* Vert pour le titre de la carte */
    margin-bottom: 0.5rem;
}
.component-card p {
    font-size: 1rem;
    color: #555;
    margin: 0;
}
.component-card .stMetric {
    background-color: transparent !important;
    padding: 0 !important;
    margin-top: 5px;
}

/* Boutons stylisés */
.stButton>button {
    background-color: #4CAF50; /* Vert */
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    border: 2px solid #388E3C;
    font-weight: bold;
    transition: all 0.2s ease-in-out;
    box-shadow: 3px 3px 5px rgba(0,0,0,0.2);
}
.stButton>button:hover {
    background-color: #388E3C;
    border-color: #2E7D32;
    transform: translateY(-2px);
    box-shadow: 5px 5px 10px rgba(0,0,0,0.3);
}

/* Messages de succès/erreur */
.st.success {
    background-color: #e6ffed !important;
    color: #1a5e2e !important;
    border-left: 5px solid #4CAF50 !important;
    padding: 1rem !important;
    border-radius: 8px !important;
    font-weight: bold;
}
.st.error {
    background-color: #ffe6e6 !important;
    color: #9b2f2f !important;
    border-left: 5px solid #FF4B4B !important;
    padding: 1rem !important;
    border-radius: 8px !important;
    font-weight: bold;
}

/* Texte général */
body {
    font-family: 'Press Start 2P', cursive; /* Police de style jeu vidéo si importée */
}
p, li {
    font-size: 1.05rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    color: white; /* Text color */
    background-color: #1E90FF; /* Mario Kart Blue */
    border-right: 3px solid #FFD700; /* Gold border */
}

/* Buttons in sidebar */
[data-testid="stSidebar"] .stButton > button {
    background-color: #FFD700;   /* Gold buttons */
    border-color: #E5C100;
    color: black;
    font-weight: bold;
}

/* Button hover effect */
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #F0D000;
    border-color: #C5A900;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# --- 2. Chargement du modèle ML et des données brutes ---
# Utilisation de st.cache_resource pour charger le modèle une seule fois
@st.cache_resource
def load_ml_assets():
    """Charge le modèle ML entraîné, le scaler et la liste des caractéristiques."""
    # Chemin corrigé : le fichier model.pkl est dans le dossier 'app' qui est au même niveau que streamlit_app_creative.py
    model_path = '../app/model.pkl' 
    if not os.path.exists(model_path):
        st.error(f"❌ Erreur : Le fichier du modèle '{model_path}' est introuvable.")
        st.info("Veuillez vous assurer que vous avez exécuté la dernière cellule de votre notebook Jupyter pour sauvegarder le modèle.")
        return None, None, []
    
    try:
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data['model'], loaded_data['scaler'], loaded_data['features']
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        st.info("Le fichier model.pkl pourrait être corrompu ou avoir un format inattendu.")
        return None, None, []

# Utilisation de st.cache_data pour charger les CSV une seule fois
@st.cache_data
def load_game_data():
    """Charge les données des pilotes, carrosseries, pneus et ailerons."""
    data_dir = '../data/raw'
    try:
        drivers = pd.read_csv(f'{data_dir}/drivers.csv', sep=';')
        bodies = pd.read_csv(f'{data_dir}/bodies_karts.csv', sep=';')
        tires = pd.read_csv(f'{data_dir}/tires.csv', sep=';')
        gliders = pd.read_csv(f'{data_dir}/gliders.csv', sep=';')
        return drivers, bodies, tires, gliders
    except FileNotFoundError:
        st.error(f"❌ Erreur : Les fichiers de données ne sont pas dans le répertoire '{data_dir}'.")
        st.info("Veuillez créer les dossiers 'data/raw' et y placer les fichiers CSV.")
        st.stop() # Arrête l'application si les fichiers de données ne sont pas trouvés
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données de jeu : {e}")
        st.stop() # Arrête l'application en cas d'autre erreur de chargement

model, scaler, features_order = load_ml_assets()
drivers_df, bodies_df, tires_df, gliders_df = load_game_data()

# Vérifier si les assets critiques ont été chargés
if model is None or scaler is None or drivers_df is None:
    st.stop() # Arrête l'exécution si le modèle ou les données sont manquants

# --- 3. Sidebar pour la navigation et informations ---
with st.sidebar:
    st.image("https://placehold.co/200x100/FF4B4B/FFFFFF?text=Mario+Kart+ML", width=200, caption="Votre Atelier ML pour Mario Kart")
    st.markdown("### 🎮 Navigation")
    tab_selection = st.radio(
        "Choisissez votre aventure :",
        ("🏠 Accueil", "🛠️ Créateur de Kart", "📊 Performance & Insights", "📘 À Propos du ML"),
        index=0 # Default to home tab
    )
    st.markdown("---")
    st.markdown("### 🤖 Statistiques du Modèle")
    st.info("""
    **Type de Modèle** : Forêt Aléatoire 
    **Précision sur Test** : 100% 
    **AUC sur Test** : 1.00 
    *(Ces chiffres sont parfaits en raison d'une variable cible simulée. Plus de détails dans 'À Propos du ML'.)*
    """)
    st.markdown("---")
    if st.button("✨ Surprise du Jour !"):
        st.balloons()
        st.success("Vous avez trouvé l'Easter Egg ! 🎉 C'est parti pour la course ! 🚀")

# --- 4. Contenu principal par onglet ---
if tab_selection == "🏠 Accueil":
    st.markdown('<h1 class="main-header">Bienvenue à l\'Atelier de Combinaisons Mario Kart 8 !</h1>', unsafe_allow_html=True)
    st.image("../images/image_acceuil.png", caption="Préparez votre kart pour la victoire !", use_container_width=True,width=700)
    st.markdown("""
    Explorez la science derrière les combinaisons gagnantes de Mario Kart 8 ! Cet atelier interactif utilise
    l'apprentissage automatique pour vous aider à comprendre l'impact des différents pilotes, carrosseries, pneus et ailerons
    sur les performances de votre kart.

    Sélectionnez vos pièces, et notre modèle prédira si votre assemblage sera un champion !
    """)
    st.subheader("💡 Comment ça marche ?")
    st.markdown("---")
    st.write("""
    1.  **Choisissez vos pièces** : Sélectionnez un pilote, une carrosserie, des pneus et un aileron.
    2.  **Calculez les stats** : L'application agrège automatiquement leurs statistiques individuelles.
    3.  **Obtenez une prédiction** : Notre modèle prédit si cette combinaison est "gagnante" (c'est-à-dire, à très haut score de performance simulée).
    """)
    st.subheader("📊 Votre Garage de Pièces")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    # Affichage dynamique des comptes de pièces
    for col, (emoji, label, df) in zip(
        (col1, col2, col3, col4),
        [("🏁", "Pilotes", drivers_df), ("🏎️", "Carrosseries", bodies_df),
         ("🛞", "Pneus", tires_df), ("🪂", "Ailerons", gliders_df)]
    ):
        col.markdown(f"""
            <div class="component-card">
              <h3>{emoji} {label}</h3>
              <p>{len(df)} options disponibles</p>
            </div>
        """, unsafe_allow_html=True)

elif tab_selection == "🛠️ Créateur de Kart":
    st.markdown('<h2 class="main-header">🛠️ Assemblez votre Kart Ultime !</h2>', unsafe_allow_html=True)
    st.write("Sélectionnez chaque pièce de votre kart et découvrez son potentiel de victoire !")

    # Initialisation des états de session pour les sélections, pour persister après reruns
    # Correction: Utilisation directe des valeurs de st.session_state dans selectbox 'index'
    if 'selected_driver' not in st.session_state:
        st.session_state.selected_driver = drivers_df['Driver'].iloc[0]
    if 'selected_body' not in st.session_state:
        st.session_state.selected_body = bodies_df['Body'].iloc[0]
    if 'selected_tire' not in st.session_state:
        st.session_state.selected_tire = tires_df['Tire'].iloc[0]
    if 'selected_glider' not in st.session_state:
        st.session_state.selected_glider = gliders_df['Glider'].iloc[0]

    # Sélecteurs de composants
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        # Correction: index initial de selectbox
        selected_driver_name = st.selectbox(
            "🏁 Choisissez votre Pilote", 
            drivers_df['Driver'], 
            index=drivers_df['Driver'].tolist().index(st.session_state.selected_driver) if st.session_state.selected_driver in drivers_df['Driver'].tolist() else 0,
            key="driver_sel"
        )
        selected_body_name = st.selectbox(
            "🏎️ Choisissez votre Carrosserie", 
            bodies_df['Body'], 
            index=bodies_df['Body'].tolist().index(st.session_state.selected_body) if st.session_state.selected_body in bodies_df['Body'].tolist() else 0,
            key="body_sel"
        )
    with col_sel2:
        selected_tire_name = st.selectbox(
            "🛞 Choisissez vos Pneus", 
            tires_df['Tire'], 
            index=tires_df['Tire'].tolist().index(st.session_state.selected_tire) if st.session_state.selected_tire in tires_df['Tire'].tolist() else 0,
            key="tire_sel"
        )
        selected_glider_name = st.selectbox(
            "🪂 Choisissez votre Aileron", 
            gliders_df['Glider'], 
            index=gliders_df['Glider'].tolist().index(st.session_state.selected_glider) if st.session_state.selected_glider in gliders_df['Glider'].tolist() else 0,
            key="glider_sel"
        )
    
    # Mise à jour des états de session après la sélection par l'utilisateur
    st.session_state.selected_driver = selected_driver_name
    st.session_state.selected_body = selected_body_name
    st.session_state.selected_tire = selected_tire_name
    st.session_state.selected_glider = selected_glider_name


    # Récupérer les stats des composants sélectionnés
    driver_stats = drivers_df.loc[drivers_df['Driver'] == st.session_state.selected_driver].iloc[0]
    body_stats = bodies_df.loc[bodies_df['Body'] == st.session_state.selected_body].iloc[0]
    tire_stats = tires_df.loc[tires_df['Tire'] == st.session_state.selected_tire].iloc[0]
    glider_stats = gliders_df.loc[gliders_df['Glider'] == st.session_state.selected_glider].iloc[0]

    st.subheader("✨ Aperçu de votre Combinaison")
    st.markdown("---")

    # Afficher les stats individuelles des composants
    comp_cols = st.columns(4)
    with comp_cols[0]:
        st.markdown(f"<div class='component-card'><h3>{driver_stats['Driver']}</h3><p>Poids: {driver_stats['Weight']}</p><p>Accél: {driver_stats['Acceleration']}</p></div>", unsafe_allow_html=True)
    with comp_cols[1]:
        st.markdown(f"<div class='component-card'><h3>{body_stats['Body']}</h3><p>Poids: {body_stats['Weight']}</p><p>Accél: {body_stats['Acceleration']}</p></div>", unsafe_allow_html=True)
    with comp_cols[2]:
        st.markdown(f"<div class='component-card'><h3>{tire_stats['Tire']}</h3><p>Poids: {tire_stats['Weight']}</p><p>Accél: {tire_stats['Acceleration']}</p></div>", unsafe_allow_html=True) # Correction ici: unsafe_allow_html
    with comp_cols[3]:
        st.markdown(f"<div class='component-card'><h3>{glider_stats['Glider']}</h3><p>Poids: {glider_stats['Weight']}</p><p>Accél: {glider_stats['Acceleration']}</p></div>", unsafe_allow_html=True)

    # Boutons d'action
    st.markdown("---")
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("🎲 Générer une Combinaison Aléatoire"):
            st.session_state.selected_driver = np.random.choice(drivers_df['Driver'])
            st.session_state.selected_body = np.random.choice(bodies_df['Body'])
            st.session_state.selected_tire = np.random.choice(tires_df['Tire'])
            st.session_state.selected_glider = np.random.choice(gliders_df['Glider'])
            st.rerun() # Force un rafraîchissement pour que les selectbox affichent les nouvelles valeurs

    with action_cols[1]:
        predict_button_clicked = st.button("🚀 Calculer les Stats & Prédire la Victoire !")

    if predict_button_clicked:
        # Calculer les caractéristiques agrégées
        total_weight = driver_stats['Weight'] + body_stats['Weight'] + tire_stats['Weight'] + glider_stats['Weight']
        total_accel = driver_stats['Acceleration'] + body_stats['Acceleration'] + tire_stats['Acceleration'] + glider_stats['Acceleration']
        avg_on_road = np.mean([driver_stats['On-Road traction'], body_stats['On-Road traction'], tire_stats['On-Road traction'], glider_stats['On-Road traction']])
        # Correction pour le calcul de avg_off_road: accès direct à la colonne de glider_stats
        avg_off_road = np.mean([driver_stats['Off-Road Traction'], body_stats['Off-Road Traction'], tire_stats['Off-Road Traction'], glider_stats['Off-Road Traction']])
        

        # Créer un DataFrame d'entrée pour le modèle
        input_data_for_prediction = pd.DataFrame([[total_weight, total_accel, avg_on_road, avg_off_road]],
                                                    columns=features_order) # Assurez l'ordre des features

        st.subheader("📈 Statistiques combinées de votre Kart :")
        st.markdown("---")
        stat_cols = st.columns(4)
        stat_cols[0].metric("Poids Total", f"{total_weight:.2f}")
        stat_cols[1].metric("Accélération Totale", f"{total_accel:.2f}")
        stat_cols[2].metric("Traction sur Route", f"{avg_on_road:.2f}")
        stat_cols[3].metric("Traction Hors Route", f"{avg_off_road:.2f}")

        # Prétraiter avec le scaler et prédire
        scaled_input = scaler.transform(input_data_for_prediction)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        st.subheader(" Résultat de la Prédiction :")
        st.markdown("---")
        if prediction == 1:
            st.success(f"🏆 FÉLICITATIONS ! Votre combinaison est prédite comme une **Victoire** avec une probabilité de {prediction_proba[1]*100:.1f} % !")
            st.balloons()
        else:
            st.error(f"🐢 DOMMAGE ! Votre combinaison est prédite comme une **Défaite** avec une probabilité de {prediction_proba[0]*100:.1f} %.")
            st.snow()

elif tab_selection == "📊 Performance & Insights":
    st.markdown('<h2 class="main-header">📊 Performance et Analyse du Modèle</h2>', unsafe_allow_html=True)
    st.write("Découvrez comment notre modèle prend ses décisions et évaluez sa performance.")

    st.subheader("💡 Importance des Caractéristiques")
    st.markdown("---")
    if model and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Importance': importances}, index=features_order)
        fig_fi = px.bar(fi_df, x='Importance', y=fi_df.index, orientation='h',
                        title="Importance des Caractéristiques dans la Prédiction",
                        labels={'y': 'Caractéristique', 'x': 'Score d\'Importance'},
                        color='Importance', color_continuous_scale=px.colors.sequential.Viridis,
                        height=350) # Hauteur ajustée pour être plus compact
        st.plotly_chart(fig_fi, use_container_width=True) # use_container_width s'adapte à la colonne
        st.info("""
        Ces scores indiquent l'influence de chaque caractéristique sur la prédiction du modèle.
        Plus la barre est longue, plus la caractéristique est jugée importante.
        """)
    else:
        st.warning("Les importances des caractéristiques ne sont pas disponibles pour ce modèle.")

    st.markdown("---") # Séparateur

    # Utilisation de colonnes pour placer la matrice de confusion et la courbe ROC côte à côte
    plot_cols = st.columns(2)

    with plot_cols[0]: # Colonne pour la Matrice de Confusion
        st.subheader("🎯 Matrice de Confusion")
        st.write("La matrice de confusion montre la répartition des prédictions (correctes et incorrectes).")
        
        if model is not None:
            try:
                # Créer une matrice de confusion 'parfaite' pour la démonstration du modèle 'parfait'
                # Nous utilisons des données simulées pour la matrice car le modèle est parfait (fuite de données)
                dummy_y_test = np.array([0]*700 + [1]*300) # Exemple 70% défaite, 30% victoire
                dummy_y_pred = dummy_y_test # Prédiction parfaite pour la démo
                
                cm_perfect = confusion_matrix(dummy_y_test, dummy_y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4.5)) # Taille légèrement réduite
                ConfusionMatrixDisplay(confusion_matrix=cm_perfect, display_labels=['Défaite', 'Victoire']).plot(ax=ax_cm, cmap='Greens')
                ax_cm.set_title("Matrice de Confusion (Modèle Parfait Simulé)")
                plt.tight_layout() # Ajuste le layout pour éviter le chevauchement
                st.pyplot(fig_cm)
                plt.close(fig_cm) # Ferme la figure pour libérer de la mémoire
                
                st.markdown("""
                **Interprétation :** Dans cette matrice, tous les résultats sont sur la diagonale, indiquant que le modèle
                a prédit toutes les "Victoires" et "Défaites" correctement. C'est le résultat attendu
                compte tenu de la façon dont la variable cible a été construite.
                """)
            except Exception as e:
                st.warning(f"Impossible de générer la matrice de confusion : {e}.")
                st.info("Assurez-vous que les données de test sont accessibles ou que la logique de simulation est correcte.")
        else:
            st.info("Le modèle n'a pas été chargé correctement pour générer la matrice de confusion.")

    with plot_cols[1]: # Colonne pour la Courbe ROC
        st.subheader("📈 Courbe ROC")
        st.write("La courbe ROC évalue la capacité du modèle à distinguer les classes.")
        
        if model is not None:
            # Simuler des probabilités parfaites pour la courbe ROC
            dummy_y_test = np.array([0]*700 + [1]*300)
            dummy_y_proba = np.array([0.0]*700 + [1.0]*300)
            
            fpr_perfect, tpr_perfect, _ = roc_curve(dummy_y_test, dummy_y_proba)
            roc_auc_perfect = auc(fpr_perfect, tpr_perfect)

            fig_roc = px.area(
                x=fpr_perfect, y=tpr_perfect,
                title=f'Courbe ROC (AUC = {roc_auc_perfect:.2f})',
                labels={'x': 'Taux de Faux Positifs', 'y': 'Taux de Vrais Positifs'},
                width=None, height=400 # Height ajustée, width sera gérée par use_container_width
            )
            fig_roc.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            fig_roc.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1.05])
            st.plotly_chart(fig_roc, use_container_width=True) # use_container_width s'adapte à la colonne
            st.info("""
            Une AUC (Area Under the Curve) de 1.00 signifie que le modèle est capable de distinguer
            parfaitement les classes "Victoire" et "Défaite". Encore une fois, ce résultat
            idéal est dû à la construction déterministe de la variable cible.
            """)
        else:
            st.info("Le modèle n'a pas été chargé correctement pour générer la courbe ROC.")


elif tab_selection == "📘 À Propos du ML":
    st.markdown('<h2 class="main-header">📘 Le Moteur ML de Mario Kart</h2>', unsafe_allow_html=True)
    st.markdown("""
    Ce projet **Mario Kart ML Tutor** est une démonstration éducative de l'apprentissage automatique,
    en utilisant un univers familier pour expliquer des concepts clés.
    """)

    st.subheader("🧩 Les Étapes de notre Pipeline ML :")
    st.markdown("""
    1.  **Collecte et Fusion des Données** : Nous avons rassemblé les statistiques détaillées des pilotes,
        carrosseries, pneus et ailerons de Mario Kart 8 à partir de fichiers CSV, puis les avons
        combinées pour obtenir toutes les configurations possibles.
    2.  **Ingénierie des Caractéristiques** : Pour simplifier et rendre les données plus pertinentes,
        nous avons créé 4 caractéristiques agrégées pour chaque combinaison :
        * **Poids Total**
        * **Accélération Totale**
        * **Traction sur Route Moyenne**
        * **Traction Hors Route Moyenne**
    3.  **Définition de la Cible ('Victoire')** : C'est ici que le côté *ludique* rencontre le côté *éducatif* !
        Nous avons **simulé** la notion de "victoire" : une combinaison est étiquetée "Victoire" si la somme
        de son **Accélération Totale** et de sa **Traction sur Route Moyenne** (notre "score" de performance)
        se situe parmi les 30% les plus élevés.
    4.  **Standardisation des Données** : Nous avons mis nos caractéristiques à la même échelle
        pour faciliter l'apprentissage du modèle.
    5.  **Division Entraînement/Test** : Les données ont été séparées pour entraîner le modèle
        sur une partie et le tester sur une autre (non vue).
    6.  **Entraînement du Modèle** : Nous avons entraîné un algorithme **Random Forest Classifier**
        pour apprendre à prédire si une combinaison est "gagnante" ou "perdante".
    7.  **Évaluation et Sauvegarde** : Le modèle a été évalué et sauvegardé pour être utilisé ici dans l'application.
    """)

    st.subheader("⚠️ Comprendre la Fuite de Données (Data Leakage) :")
    st.warning("""
    Le modèle de ce tutoriel atteint une précision de **100%**. Pourquoi ?
    Parce que la variable cible "Victoire" est directement calculée à partir des mêmes caractéristiques
    (Accélération Totale et Traction sur Route Moyenne) que celles utilisées pour la prédiction.
    C'est ce qu'on appelle la **fuite de données (data leakage)**. Le modèle ne généralise pas vraiment,
    il apprend une règle qu'il connaît déjà par cœur !

    Dans un **vrai projet ML**, une variable cible comme "Victoire" proviendrait de résultats réels (historiques de courses)
    qui ne sont pas directement dérivés des caractéristiques d'entrée. Cela rendrait la prédiction beaucoup plus difficile,
    mais aussi beaucoup plus utile et pertinente pour le monde réel.
    """)
    st.markdown("---")
    st.info("Projet réalisé par Zaineb RAHMANI dans le cadre d'un Master IA et Data Sciences.")


# --- 5. Footer ---
st.markdown("""<hr>
<div style="text-align:center;color:#666">© 2025 Mario Kart ML Tutor</div>
""", unsafe_allow_html=True)