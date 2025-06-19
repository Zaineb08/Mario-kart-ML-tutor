# streamlit_app_bilingual.py

# --- Import necessary libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# --- Language Configuration ---
LANGUAGES = {
    'fr': {
        'flag': '🇫🇷',
        'name': 'Français',
        'page_title': "Mario Kart ML Tutor: L'Atelier de Combinaisons",
        'main_header': "Bienvenue à l'Atelier de Combinaisons Mario Kart 8 !",
        'navigation': "🎮 Navigation",
        'nav_options': ["🏠 Accueil", "🛠️ Créateur de Kart", "📊 Performance & Insights", "📘 À Propos du ML"],
        'model_stats': "🤖 Statistiques du Modèle",
        'model_info': """
        **Type de Modèle** : Forêt Aléatoire 
        **Précision sur Test** : 100% 
        **AUC sur Test** : 1.00 
        *(Ces chiffres sont parfaits en raison d'une variable cible simulée. Plus de détails dans 'À Propos du ML'.)*
        """,
        'surprise_button': "✨ Surprise du Jour !",
        'surprise_message': "Vous avez trouvé l'Easter Egg ! 🎉 C'est parti pour la course ! 🚀",
        'welcome_text': """
        Explorez la science derrière les combinaisons gagnantes de Mario Kart 8 ! Cet atelier interactif utilise
        l'apprentissage automatique pour vous aider à comprendre l'impact des différents pilotes, carrosseries, pneus et ailerons
        sur les performances de votre kart.

        Sélectionnez vos pièces, et notre modèle prédira si votre assemblage sera un champion !
        """,
        'how_it_works': "💡 Comment ça marche ?",
        'steps': [
            "**Choisissez vos pièces** : Sélectionnez un pilote, une carrosserie, des pneus et un aileron.",
            "**Calculez les stats** : L'application agrège automatiquement leurs statistiques individuelles.",
            "**Obtenez une prédiction** : Notre modèle prédit si cette combinaison est \"gagnante\"."
        ],
        'parts_garage': "📊 Votre Garage de Pièces",
        'parts_labels': ["Pilotes", "Carrosseries", "Pneus", "Ailerons"],
        'options_available': "options disponibles",
        'kart_builder_title': "🛠️ Assemblez votre Kart Ultime !",
        'kart_builder_subtitle': "Sélectionnez chaque pièce de votre kart et découvrez son potentiel de victoire !",
        'select_driver': "🏁 Choisissez votre Pilote",
        'select_body': "🏎️ Choisissez votre Carrosserie",
        'select_tire': "🛞 Choisissez vos Pneus",
        'select_glider': "🪂 Choisissez votre Aileron",
        'combination_preview': "✨ Aperçu de votre Combinaison",
        'random_button': "🎲 Générer une Combinaison Aléatoire",
        'predict_button': "🚀 Calculer les Stats & Prédire la Victoire !",
        'combined_stats': "📈 Statistiques combinées de votre Kart :",
        'total_weight': "Poids Total",
        'total_accel': "Accélération Totale",
        'on_road_traction': "Traction sur Route",
        'off_road_traction': "Traction Hors Route",
        'prediction_result': "Résultat de la Prédiction :",
        'victory_message': "🏆 FÉLICITATIONS ! Votre combinaison est prédite comme une **Victoire** avec une probabilité de {:.1f} % !",
        'defeat_message': "🐢 DOMMAGE ! Votre combinaison est prédite comme une **Défaite** avec une probabilité de {:.1f} %.",
        'performance_title': "📊 Performance et Analyse du Modèle",
        'performance_subtitle': "Découvrez comment notre modèle prend ses décisions et évaluez sa performance.",
        'feature_importance': "💡 Importance des Caractéristiques",
        'confusion_matrix': "🎯 Matrice de Confusion",
        'roc_curve': "📈 Courbe ROC",
        'about_ml_title': "📘 Le Moteur ML de Mario Kart",
        'weight': "Poids",
        'acceleration': "Accél",
        'defeat': "Défaite",
        'victory': "Victoire"
    },
    'en': {
        'flag': '🇺🇸',
        'name': 'English',
        'page_title': "Mario Kart ML Tutor: The Combination Workshop",
        'main_header': "Welcome to the Mario Kart 8 Combination Workshop!",
        'navigation': "🎮 Navigation",
        'nav_options': ["🏠 Home", "🛠️ Kart Builder", "📊 Performance & Insights", "📘 About ML"],
        'model_stats': "🤖 Model Statistics",
        'model_info': """
        **Model Type**: Random Forest 
        **Test Accuracy**: 100% 
        **Test AUC**: 1.00 
        *(These perfect scores are due to a simulated target variable. More details in 'About ML'.)*
        """,
        'surprise_button': "✨ Daily Surprise!",
        'surprise_message': "You found the Easter Egg! 🎉 Let's race! 🚀",
        'welcome_text': """
        Explore the science behind winning Mario Kart 8 combinations! This interactive workshop uses
        machine learning to help you understand the impact of different drivers, bodies, tires, and gliders
        on your kart's performance.

        Select your parts, and our model will predict if your build will be a champion!
        """,
        'how_it_works': "💡 How it works?",
        'steps': [
            "**Choose your parts**: Select a driver, body, tires, and glider.",
            "**Calculate stats**: The app automatically aggregates their individual statistics.",
            "**Get a prediction**: Our model predicts if this combination is \"winning\"."
        ],
        'parts_garage': "📊 Your Parts Garage",
        'parts_labels': ["Drivers", "Bodies", "Tires", "Gliders"],
        'options_available': "options available",
        'kart_builder_title': "🛠️ Build your Ultimate Kart!",
        'kart_builder_subtitle': "Select each part of your kart and discover its victory potential!",
        'select_driver': "🏁 Choose your Driver",
        'select_body': "🏎️ Choose your Body",
        'select_tire': "🛞 Choose your Tires",
        'select_glider': "🪂 Choose your Glider",
        'combination_preview': "✨ Your Combination Preview",
        'random_button': "🎲 Generate Random Combination",
        'predict_button': "🚀 Calculate Stats & Predict Victory!",
        'combined_stats': "📈 Your Kart's Combined Statistics:",
        'total_weight': "Total Weight",
        'total_accel': "Total Acceleration",
        'on_road_traction': "On-Road Traction",
        'off_road_traction': "Off-Road Traction",
        'prediction_result': "Prediction Result:",
        'victory_message': "🏆 CONGRATULATIONS! Your combination is predicted as a **Victory** with {:.1f}% probability!",
        'defeat_message': "🐢 TOO BAD! Your combination is predicted as a **Defeat** with {:.1f}% probability.",
        'performance_title': "📊 Model Performance and Analysis",
        'performance_subtitle': "Discover how our model makes decisions and evaluate its performance.",
        'feature_importance': "💡 Feature Importance",
        'confusion_matrix': "🎯 Confusion Matrix",
        'roc_curve': "📈 ROC Curve",
        'about_ml_title': "📘 The Mario Kart ML Engine",
        'weight': "Weight",
        'acceleration': "Accel",
        'defeat': "Defeat",
        'victory': "Victory"
    }
}

# --- 0. Streamlit page configuration ---
st.set_page_config(
    page_title="Mario Kart ML Tutor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize language in session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # Default to English

# Language selector in the top right
col1, col2, col3 = st.columns([6, 1, 1])
with col2:
    if st.button(f"{LANGUAGES['fr']['flag']} FR", key="fr_btn", 
                 type="primary" if st.session_state.language == 'fr' else "secondary"):
        st.session_state.language = 'fr'
        st.rerun()
with col3:
    if st.button(f"{LANGUAGES['en']['flag']} EN", key="en_btn",
                 type="primary" if st.session_state.language == 'en' else "secondary"):
        st.session_state.language = 'en'
        st.rerun()

# Get current language texts
lang = LANGUAGES[st.session_state.language]

# --- 1. Custom CSS for playful and thematic look ---
st.markdown("""
<style>
/* Main header style */
.main-header {
    font-size: 3.5rem;
    font-weight: bold;
    text-align: center;
    color: #FF4B4B; /* Mario Red */
    text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
    margin-bottom: 2rem;
    padding-bottom: 15px;
    border-bottom: 3px solid #FF4B4B;
}

/* Component cards for statistics */
.component-card {
    background-color: #f0f2f6; /* Light gray */
    color: #222;
    padding: 1.2rem;
    border-radius: 15px;
    border: 2px solid #E0F7FA; /* Light blue */
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
    color: #4CAF50; /* Green for card title */
    margin-bottom: 0.5rem;
}
.component-card p {
    font-size: 1rem;
    color: #555;
    margin: 0;
}

/* Styled buttons */
.stButton>button {
    background-color: #4CAF50; /* Green */
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

/* Language buttons */
button[key="fr_btn"], button[key="en_btn"] {
    width: 60px !important;
    height: 40px !important;
    font-size: 12px !important;
    padding: 5px !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    color: white;
    background-color: #1E90FF; /* Mario Kart Blue */
    border-right: 3px solid #FFD700; /* Gold border */
}

[data-testid="stSidebar"] .stButton > button {
    background-color: #FFD700;   /* Gold buttons */
    border-color: #E5C100;
    color: black;
    font-weight: bold;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #F0D000;
    border-color: #C5A900;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# --- 2. Loading ML model and raw data ---
@st.cache_resource
def load_ml_assets():
    """Load the trained ML model, scaler and feature list."""
    # Try multiple possible paths for the model
    possible_model_paths = [
        'model.pkl', 
        'app/model.pkl',
        '../app/model.pkl',
        './model.pkl'
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        if st.session_state.language == 'fr':
            st.error(f"❌ Erreur : Aucun fichier de modèle trouvé dans les emplacements suivants: {possible_model_paths}")
            st.info("Veuillez vous assurer que vous avez exécuté la dernière cellule de votre notebook Jupyter pour sauvegarder le modèle dans le bon répertoire.")
        else:
            st.error(f"❌ Error: No model file found in the following locations: {possible_model_paths}")
            st.info("Please make sure you have run the last cell of your Jupyter notebook to save the model in the correct directory.")
        return None, None, []
    
    try:
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data['model'], loaded_data['scaler'], loaded_data['features']
    except Exception as e:
        if st.session_state.language == 'fr':
            st.error(f"❌ Erreur lors du chargement du modèle : {e}")
            st.info("Le fichier model.pkl pourrait être corrompu ou avoir un format inattendu.")
        else:
            st.error(f"❌ Error loading model: {e}")
            st.info("The model.pkl file might be corrupted or have an unexpected format.")
        return None, None, []

@st.cache_data
def load_game_data():
    """Load drivers, bodies, tires and gliders data."""
    # Try multiple possible paths for the data directory
    possible_data_dirs = [
        'data/raw',
        '../data/raw',
        './data/raw',
        'raw'
    ]
    
    data_dir = None
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
    
    if data_dir is None:
        if st.session_state.language == 'fr':
            st.error(f"❌ Erreur : Aucun répertoire de données trouvé dans les emplacements suivants: {possible_data_dirs}")
            st.info("Veuillez créer le dossier 'data/raw' et y placer les fichiers CSV.")
        else:
            st.error(f"❌ Error: No data directory found in the following locations: {possible_data_dirs}")
            st.info("Please create the 'data/raw' folder and place the CSV files there.")
        st.stop()
    
    try:
        drivers = pd.read_csv(f'{data_dir}/drivers.csv', sep=';')
        bodies = pd.read_csv(f'{data_dir}/bodies_karts.csv', sep=';')
        tires = pd.read_csv(f'{data_dir}/tires.csv', sep=';')
        gliders = pd.read_csv(f'{data_dir}/gliders.csv', sep=';')
        return drivers, bodies, tires, gliders
    except FileNotFoundError as e:
        if st.session_state.language == 'fr':
            st.error(f"❌ Erreur : Fichier CSV manquant dans le répertoire '{data_dir}': {e}")
            st.info("Veuillez vous assurer que tous les fichiers CSV requis sont présents : drivers.csv, bodies_karts.csv, tires.csv, gliders.csv")
        else:
            st.error(f"❌ Error: Missing CSV file in directory '{data_dir}': {e}")
            st.info("Please ensure all required CSV files are present: drivers.csv, bodies_karts.csv, tires.csv, gliders.csv")
        st.stop()
    except Exception as e:
        if st.session_state.language == 'fr':
            st.error(f"❌ Erreur lors du chargement des données de jeu : {e}")
        else:
            st.error(f"❌ Error loading game data: {e}")
        st.stop()

model, scaler, features_order = load_ml_assets()
drivers_df, bodies_df, tires_df, gliders_df = load_game_data()

# Check if critical assets have been loaded
if model is None or scaler is None or drivers_df is None:
    st.stop()

# --- 3. Sidebar for navigation and information ---
with st.sidebar:
    st.image("https://placehold.co/200x100/FF4B4B/FFFFFF?text=Mario+Kart+ML", width=200, caption="Your ML Workshop for Mario Kart")
    st.markdown(f"### {lang['navigation']}")
    tab_selection = st.radio(
        "Choose your adventure:" if st.session_state.language == 'en' else "Choisissez votre aventure :",
        lang['nav_options'],
        index=0
    )
    st.markdown("---")
    st.markdown(f"### {lang['model_stats']}")
    st.info(lang['model_info'])
    st.markdown("---")
    if st.button(lang['surprise_button']):
        st.balloons()
        st.success(lang['surprise_message'])

# --- 4. Main content by tab ---
if tab_selection == lang['nav_options'][0]:  # Home
    st.markdown(f'<h1 class="main-header">{lang["main_header"]}</h1>', unsafe_allow_html=True)
    
    # Try to load the home image, fallback to placeholder if not found
    image_paths = ["images/image_acceuil.png", "../images/image_acceuil.png", "./image_acceuil.png"]
    image_loaded = False
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            st.image(img_path, caption="Prepare your kart for victory!" if st.session_state.language == 'en' else "Préparez votre kart pour la victoire !", use_container_width=True, width=700)
            image_loaded = True
            break
    
    if not image_loaded:
        st.image("https://placehold.co/700x400/4CAF50/FFFFFF?text=Mario+Kart+8+Workshop", 
                caption="Prepare your kart for victory!" if st.session_state.language == 'en' else "Préparez votre kart pour la victoire !", 
                use_container_width=True)
    
    st.markdown(lang['welcome_text'])
    
    st.subheader(lang['how_it_works'])
    st.markdown("---")
    for i, step in enumerate(lang['steps'], 1):
        st.write(f"{i}. {step}")
    
    st.subheader(lang['parts_garage'])
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    for col, (emoji, label, df) in zip(
        (col1, col2, col3, col4),
        [("🏁", lang['parts_labels'][0], drivers_df), ("🏎️", lang['parts_labels'][1], bodies_df),
         ("🛞", lang['parts_labels'][2], tires_df), ("🪂", lang['parts_labels'][3], gliders_df)]
    ):
        col.markdown(f"""
            <div class="component-card">
              <h3>{emoji} {label}</h3>
              <p>{len(df)} {lang['options_available']}</p>
            </div>
        """, unsafe_allow_html=True)

elif tab_selection == lang['nav_options'][1]:  # Kart Builder
    st.markdown(f'<h2 class="main-header">{lang["kart_builder_title"]}</h2>', unsafe_allow_html=True)
    st.write(lang['kart_builder_subtitle'])

    # Initialize session states for selections
    if 'selected_driver' not in st.session_state:
        st.session_state.selected_driver = drivers_df['Driver'].iloc[0]
    if 'selected_body' not in st.session_state:
        st.session_state.selected_body = bodies_df['Body'].iloc[0]
    if 'selected_tire' not in st.session_state:
        st.session_state.selected_tire = tires_df['Tire'].iloc[0]
    if 'selected_glider' not in st.session_state:
        st.session_state.selected_glider = gliders_df['Glider'].iloc[0]

    # Component selectors
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_driver_name = st.selectbox(
            lang['select_driver'], 
            drivers_df['Driver'], 
            index=drivers_df['Driver'].tolist().index(st.session_state.selected_driver) if st.session_state.selected_driver in drivers_df['Driver'].tolist() else 0,
            key="driver_sel"
        )
        selected_body_name = st.selectbox(
            lang['select_body'], 
            bodies_df['Body'], 
            index=bodies_df['Body'].tolist().index(st.session_state.selected_body) if st.session_state.selected_body in bodies_df['Body'].tolist() else 0,
            key="body_sel"
        )
    with col_sel2:
        selected_tire_name = st.selectbox(
            lang['select_tire'], 
            tires_df['Tire'], 
            index=tires_df['Tire'].tolist().index(st.session_state.selected_tire) if st.session_state.selected_tire in tires_df['Tire'].tolist() else 0,
            key="tire_sel"
        )
        selected_glider_name = st.selectbox(
            lang['select_glider'], 
            gliders_df['Glider'], 
            index=gliders_df['Glider'].tolist().index(st.session_state.selected_glider) if st.session_state.selected_glider in gliders_df['Glider'].tolist() else 0,
            key="glider_sel"
        )
    
    # Update session states after user selection
    st.session_state.selected_driver = selected_driver_name
    st.session_state.selected_body = selected_body_name
    st.session_state.selected_tire = selected_tire_name
    st.session_state.selected_glider = selected_glider_name

    # Get stats for selected components
    driver_stats = drivers_df.loc[drivers_df['Driver'] == st.session_state.selected_driver].iloc[0]
    body_stats = bodies_df.loc[bodies_df['Body'] == st.session_state.selected_body].iloc[0]
    tire_stats = tires_df.loc[tires_df['Tire'] == st.session_state.selected_tire].iloc[0]
    glider_stats = gliders_df.loc[gliders_df['Glider'] == st.session_state.selected_glider].iloc[0]

    st.subheader(lang['combination_preview'])
    st.markdown("---")

    # Display individual component stats
    comp_cols = st.columns(4)
    with comp_cols[0]:
        st.markdown(f"<div class='component-card'><h3>{driver_stats['Driver']}</h3><p>{lang['weight']}: {driver_stats['Weight']}</p><p>{lang['acceleration']}: {driver_stats['Acceleration']}</p></div>", unsafe_allow_html=True)
    with comp_cols[1]:
        st.markdown(f"<div class='component-card'><h3>{body_stats['Body']}</h3><p>{lang['weight']}: {body_stats['Weight']}</p><p>{lang['acceleration']}: {body_stats['Acceleration']}</p></div>", unsafe_allow_html=True)
    with comp_cols[2]:
        st.markdown(f"<div class='component-card'><h3>{tire_stats['Tire']}</h3><p>{lang['weight']}: {tire_stats['Weight']}</p><p>{lang['acceleration']}: {tire_stats['Acceleration']}</p></div>", unsafe_allow_html=True)
    with comp_cols[3]:
        st.markdown(f"<div class='component-card'><h3>{glider_stats['Glider']}</h3><p>{lang['weight']}: {glider_stats['Weight']}</p><p>{lang['acceleration']}: {glider_stats['Acceleration']}</p></div>", unsafe_allow_html=True)

    # Action buttons
    st.markdown("---")
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button(lang['random_button']):
            st.session_state.selected_driver = np.random.choice(drivers_df['Driver'])
            st.session_state.selected_body = np.random.choice(bodies_df['Body'])
            st.session_state.selected_tire = np.random.choice(tires_df['Tire'])
            st.session_state.selected_glider = np.random.choice(gliders_df['Glider'])
            st.rerun()

    with action_cols[1]:
        predict_button_clicked = st.button(lang['predict_button'])

    if predict_button_clicked:
        # Calculate aggregated features
        total_weight = driver_stats['Weight'] + body_stats['Weight'] + tire_stats['Weight'] + glider_stats['Weight']
        total_accel = driver_stats['Acceleration'] + body_stats['Acceleration'] + tire_stats['Acceleration'] + glider_stats['Acceleration']
        avg_on_road = np.mean([driver_stats['On-Road traction'], body_stats['On-Road traction'], tire_stats['On-Road traction'], glider_stats['On-Road traction']])
        avg_off_road = np.mean([driver_stats['Off-Road Traction'], body_stats['Off-Road Traction'], tire_stats['Off-Road Traction'], glider_stats['Off-Road Traction']])
        

        # Create input DataFrame for the model
        input_data_for_prediction = pd.DataFrame([[total_weight, total_accel, avg_on_road, avg_off_road]],
                                                    columns=features_order)

        st.subheader(lang['combined_stats'])
        st.markdown("---")
        stat_cols = st.columns(4)
        stat_cols[0].metric(lang['total_weight'], f"{total_weight:.2f}")
        stat_cols[1].metric(lang['total_accel'], f"{total_accel:.2f}")
        stat_cols[2].metric(lang['on_road_traction'], f"{avg_on_road:.2f}")
        stat_cols[3].metric(lang['off_road_traction'], f"{avg_off_road:.2f}")

        # Preprocess with scaler and predict
        scaled_input = scaler.transform(input_data_for_prediction)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        st.subheader(lang['prediction_result'])
        st.markdown("---")
        if prediction == 1:
            st.success(lang['victory_message'].format(prediction_proba[1]*100))
            st.balloons()
        else:
            st.error(lang['defeat_message'].format(prediction_proba[0]*100))
            st.snow()

elif tab_selection == lang['nav_options'][2]:  # Performance & Insights
    st.markdown(f'<h2 class="main-header">{lang["performance_title"]}</h2>', unsafe_allow_html=True)
    st.write(lang['performance_subtitle'])

    st.subheader(lang['feature_importance'])
    st.markdown("---")
    if model and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Importance': importances}, index=features_order)
        
        title_text = "Feature Importance in Prediction" if st.session_state.language == 'en' else "Importance des Caractéristiques dans la Prédiction"
        x_label = "Importance Score" if st.session_state.language == 'en' else "Score d'Importance"
        y_label = "Feature" if st.session_state.language == 'en' else "Caractéristique"
        
        fig_fi = px.bar(fi_df, x='Importance', y=fi_df.index, orientation='h',
                        title=title_text,
                        labels={'y': y_label, 'x': x_label},
                        color='Importance', color_continuous_scale=px.colors.sequential.Viridis,
                        height=350)
        st.plotly_chart(fig_fi, use_container_width=True)
        
        info_text = ("These scores indicate the relative importance of each feature in making predictions. Higher scores mean the feature has more influence on whether a combination is predicted as winning or losing." 
                    if st.session_state.language == 'en' else 
                    "Ces scores indiquent l'importance relative de chaque caractéristique dans les prédictions. Des scores plus élevés signifient que la caractéristique a plus d'influence sur le fait qu'une combinaison soit prédite comme gagnante ou perdante.")
        st.info(info_text)
    else:
        st.warning("Feature importance not available for this model type." if st.session_state.language == 'en' else "L'importance des caractéristiques n'est pas disponible pour ce type de modèle.")

    # Load test data for confusion matrix and ROC curve if available
    try:
        # Try to load test predictions
        test_data_paths = ['test_predictions.pkl', 'app/test_predictions.pkl', '../app/test_predictions.pkl']
        test_data = None
        
        for path in test_data_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    test_data = pickle.load(f)
                break
        
        if test_data and 'y_test' in test_data and 'y_pred' in test_data and 'y_proba' in test_data:
            y_test = test_data['y_test']
            y_pred = test_data['y_pred']
            y_proba = test_data['y_proba']
            
            # Confusion Matrix
            st.subheader(lang['confusion_matrix'])
            st.markdown("---")
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Create confusion matrix plot
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                        display_labels=[lang['defeat'], lang['victory']])
            disp.plot(ax=ax_cm, cmap='Blues', values_format='d')
            
            title_text = "Confusion Matrix - Test Set" if st.session_state.language == 'en' else "Matrice de Confusion - Ensemble de Test"
            ax_cm.set_title(title_text, fontsize=16, fontweight='bold')
            st.pyplot(fig_cm)
            
            cm_info = ("The confusion matrix shows how well our model performed on the test data. Perfect diagonal values indicate 100% accuracy." 
                      if st.session_state.language == 'en' else 
                      "La matrice de confusion montre les performances de notre modèle sur les données de test. Des valeurs diagonales parfaites indiquent une précision de 100%.")
            st.info(cm_info)
            
            # ROC Curve
            st.subheader(lang['roc_curve'])
            st.markdown("---")
            
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                       label='Random classifier')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            
            xlabel_text = 'False Positive Rate' if st.session_state.language == 'en' else 'Taux de Faux Positifs'
            ylabel_text = 'True Positive Rate' if st.session_state.language == 'en' else 'Taux de Vrais Positifs'
            title_text = 'ROC Curve - Test Set' if st.session_state.language == 'en' else 'Courbe ROC - Ensemble de Test'
            
            ax_roc.set_xlabel(xlabel_text, fontsize=12)
            ax_roc.set_ylabel(ylabel_text, fontsize=12)
            ax_roc.set_title(title_text, fontsize=16, fontweight='bold')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, alpha=0.3)
            st.pyplot(fig_roc)
            
            roc_info = ("The ROC curve shows the trade-off between sensitivity and specificity. An AUC of 1.0 indicates perfect classification." 
                       if st.session_state.language == 'en' else 
                       "La courbe ROC montre le compromis entre sensibilité et spécificité. Un AUC de 1.0 indique une classification parfaite.")
            st.info(roc_info)
            
        else:
            no_test_data = ("Test performance data not available. The model statistics shown are based on training data." 
                           if st.session_state.language == 'en' else 
                           "Données de performance de test non disponibles. Les statistiques du modèle affichées sont basées sur les données d'entraînement.")
            st.warning(no_test_data)
    
    except Exception as e:
        error_msg = (f"Could not load test performance data: {e}" 
                    if st.session_state.language == 'en' else 
                    f"Impossible de charger les données de performance de test : {e}")
        st.warning(error_msg)

elif tab_selection == lang['nav_options'][3]:  # About ML
    st.markdown(f'<h2 class="main-header">{lang["about_ml_title"]}</h2>', unsafe_allow_html=True)
    
    about_ml_content = """
    ### 🎯 What is Machine Learning?
    
    Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario.
    
    ### 🔬 How This Mario Kart Model Works
    
    Our Mario Kart predictor uses a **Random Forest** algorithm, which:
    
    1. **Collects Data**: We gather statistics for all drivers, bodies, tires, and gliders
    2. **Creates Features**: We combine individual component stats into total metrics
    3. **Trains the Model**: The algorithm learns patterns from thousands of combinations
    4. **Makes Predictions**: Given new combinations, it predicts win/loss probability
    
    ### 📊 The Features We Use
    
    - **Total Weight**: Sum of all component weights
    - **Total Acceleration**: Sum of all component acceleration values  
    - **Average On-Road Traction**: Mean traction on regular surfaces
    - **Average Off-Road Traction**: Mean traction on difficult terrain
    
    ### ⚠️ Important Note About This Demo
    
    The "perfect" accuracy scores (100%) you see are because this is an educational simulation. In real ML projects:
    
    - Accuracy is rarely perfect
    - We use techniques like cross-validation
    - We split data into training/validation/test sets
    - We watch for overfitting
    
    ### 🎮 Real-World Applications
    
    Similar ML techniques are used in:
    - **Game Development**: Balancing characters and items
    - **Sports Analytics**: Predicting team performance
    - **E-commerce**: Recommending products
    - **Healthcare**: Diagnosing conditions
    - **Finance**: Detecting fraud
    
    ### 🚀 Next Steps in ML Learning
    
    To dive deeper into machine learning:
    1. Learn Python and pandas for data manipulation
    2. Study statistics and probability
    3. Practice with real datasets
    4. Experiment with different algorithms
    5. Build your own projects!
    """ if st.session_state.language == 'en' else """
    ### 🎯 Qu'est-ce que l'Apprentissage Automatique ?
    
    L'Apprentissage Automatique (ML) est un sous-ensemble de l'intelligence artificielle qui permet aux ordinateurs d'apprendre et de prendre des décisions à partir de données sans être explicitement programmés pour chaque scénario.
    
    ### 🔬 Comment Fonctionne ce Modèle Mario Kart
    
    Notre prédicteur Mario Kart utilise un algorithme de **Forêt Aléatoire**, qui :
    
    1. **Collecte les Données** : Nous rassemblons les statistiques de tous les pilotes, carrosseries, pneus et ailerons
    2. **Crée des Caractéristiques** : Nous combinons les stats individuelles en métriques totales
    3. **Entraîne le Modèle** : L'algorithme apprend les patterns à partir de milliers de combinaisons
    4. **Fait des Prédictions** : Avec de nouvelles combinaisons, il prédit la probabilité de victoire/défaite
    
    ### 📊 Les Caractéristiques que Nous Utilisons
    
    - **Poids Total** : Somme de tous les poids des composants
    - **Accélération Totale** : Somme de toutes les valeurs d'accélération des composants
    - **Traction Moyenne sur Route** : Traction moyenne sur surfaces régulières
    - **Traction Moyenne Hors Route** : Traction moyenne sur terrain difficile
    
    ### ⚠️ Note Importante sur cette Démo
    
    Les scores de précision "parfaits" (100%) que vous voyez sont dus au fait que c'est une simulation éducative. Dans de vrais projets ML :
    
    - La précision est rarement parfaite
    - Nous utilisons des techniques comme la validation croisée
    - Nous divisons les données en ensembles d'entraînement/validation/test
    - Nous surveillons le surapprentissage
    
    ### 🎮 Applications dans le Monde Réel
    
    Des techniques ML similaires sont utilisées dans :
    - **Développement de Jeux** : Équilibrage des personnages et objets
    - **Analyse Sportive** : Prédiction des performances d'équipe
    - **E-commerce** : Recommandation de produits
    - **Santé** : Diagnostic de conditions
    - **Finance** : Détection de fraude
    
    ### 🚀 Prochaines Étapes dans l'Apprentissage ML
    
    Pour approfondir l'apprentissage automatique :
    1. Apprendre Python et pandas pour la manipulation de données
    2. Étudier les statistiques et probabilités
    3. S'exercer avec de vrais jeux de données
    4. Expérimenter avec différents algorithmes
    5. Construire vos propres projets !
    """
    
    st.markdown(about_ml_content)
    
    # Add some interactive elements
    st.markdown("---")
    
    expander_title = "🔍 Explore Random Forest Algorithm" if st.session_state.language == 'en' else "🔍 Explorer l'Algorithme Forêt Aléatoire"
    with st.expander(expander_title):
        rf_explanation = """
        **Random Forest** works by:
        
        1. Creating many decision trees (a "forest")
        2. Each tree votes on the prediction
        3. The majority vote becomes the final prediction
        4. This reduces overfitting and improves accuracy
        
        **Advantages:**
        - Handles both numerical and categorical data
        - Reduces overfitting compared to single decision trees
        - Provides feature importance scores
        - Works well with default parameters
        
        **In Mario Kart Context:**
        Each tree might focus on different aspects:
        - Tree 1: "Heavy karts with high acceleration win"
        - Tree 2: "Good traction is most important"
        - Tree 3: "Balanced stats perform best"
        
        The forest combines all these "opinions" for a robust prediction!
        """ if st.session_state.language == 'en' else """
        **La Forêt Aléatoire** fonctionne en :
        
        1. Créant de nombreux arbres de décision (une "forêt")
        2. Chaque arbre vote sur la prédiction
        3. Le vote majoritaire devient la prédiction finale
        4. Cela réduit le surapprentissage et améliore la précision
        
        **Avantages :**
        - Gère les données numériques et catégorielles
        - Réduit le surapprentissage par rapport aux arbres de décision simples
        - Fournit des scores d'importance des caractéristiques
        - Fonctionne bien avec les paramètres par défaut
        
        **Dans le Contexte Mario Kart :**
        Chaque arbre peut se concentrer sur différents aspects :
        - Arbre 1 : "Les karts lourds avec haute accélération gagnent"
        - Arbre 2 : "La bonne traction est plus importante"
        - Arbre 3 : "Les stats équilibrées performent mieux"
        
        La forêt combine toutes ces "opinions" pour une prédiction robuste !
        """
        
        st.markdown(rf_explanation)
    
    # Fun fact section
    fun_fact_title = "🎲 Fun ML Fact" if st.session_state.language == 'en' else "🎲 Fait Amusant ML"
    st.markdown(f"### {fun_fact_title}")
    
    fun_facts = [
        "Random Forest was invented by Leo Breiman in 2001!",
        "Netflix uses ML algorithms similar to this to recommend movies!",
        "The Random Forest algorithm is used in genomics research!",
        "Google's search algorithm uses hundreds of ML models!",
        "ML models help detect credit card fraud in real-time!"
    ] if st.session_state.language == 'en' else [
        "La Forêt Aléatoire a été inventée par Leo Breiman en 2001 !",
        "Netflix utilise des algorithmes ML similaires pour recommander des films !",
        "L'algorithme Forêt Aléatoire est utilisé dans la recherche génomique !",
        "L'algorithme de recherche de Google utilise des centaines de modèles ML !",
        "Les modèles ML aident à détecter la fraude de carte de crédit en temps réel !"
    ]
    
    if st.button("🎯 Show Random Fact!" if st.session_state.language == 'en' else "🎯 Montrer un Fait Aléatoire !"):
        st.info(np.random.choice(fun_facts))

# --- Footer ---
st.markdown("---")
footer_text = """
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏎️ <strong>Mario Kart ML Tutor</strong> - Educational Demo</p>
    <p>Built with ❤️ using Streamlit, scikit-learn</p>
    <p>© 2025 Zainebr</p>
</div>
""" if st.session_state.language == 'en' else """
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏎️ <strong>Mario Kart ML Tutor</strong> - Démo Éducative</p>
    <p>Construit en utilisant Streamlit, scikit-learn</p>
    <p>© 2025 Zainebr</p>
</div>
"""

st.markdown(footer_text, unsafe_allow_html=True)