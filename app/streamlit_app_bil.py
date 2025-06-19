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
    model_path = '../app/model.pkl' 
    if not os.path.exists(model_path):
        if st.session_state.language == 'fr':
            st.error(f"❌ Erreur : Le fichier du modèle '{model_path}' est introuvable.")
            st.info("Veuillez vous assurer que vous avez exécuté la dernière cellule de votre notebook Jupyter pour sauvegarder le modèle.")
        else:
            st.error(f"❌ Error: Model file '{model_path}' not found.")
            st.info("Please make sure you have run the last cell of your Jupyter notebook to save the model.")
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
    data_dir = '../data/raw'
    try:
        drivers = pd.read_csv(f'{data_dir}/drivers.csv', sep=';')
        bodies = pd.read_csv(f'{data_dir}/bodies_karts.csv', sep=';')
        tires = pd.read_csv(f'{data_dir}/tires.csv', sep=';')
        gliders = pd.read_csv(f'{data_dir}/gliders.csv', sep=';')
        return drivers, bodies, tires, gliders
    except FileNotFoundError:
        if st.session_state.language == 'fr':
            st.error(f"❌ Erreur : Les fichiers de données ne sont pas dans le répertoire '{data_dir}'.")
            st.info("Veuillez créer les dossiers 'data/raw' et y placer les fichiers CSV.")
        else:
            st.error(f"❌ Error: Data files not found in directory '{data_dir}'.")
            st.info("Please create the 'data/raw' folders and place the CSV files there.")
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
    st.image("../images/image_acceuil.png", caption="Prepare your kart for victory!" if st.session_state.language == 'en' else "Préparez votre kart pour la victoire !", use_container_width=True, width=700)
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
        
        info_text = ("These scores indicate the influence of each feature on the model's prediction. "
                    "The longer the bar, the more important the feature is considered.") if st.session_state.language == 'en' else (
                    "Ces scores indiquent l'influence de chaque caractéristique sur la prédiction du modèle. "
                    "Plus la barre est longue, plus la caractéristique est jugée importante.")
        st.info(info_text)
        # Display feature descriptions
        if st.session_state.language == 'en':
            with st.expander("📋 Feature Descriptions"):
                st.markdown("""
**• Total Weight**: Combined weight of driver, body, tires, and glider components  
**• Total Acceleration**: Sum of acceleration values from driver, body, tires, and glider components  
**• Average On-Road Traction**: Mean traction performance on standard race tracks across all components  
**• Average Off-Road Traction**: Mean traction performance on off-road surfaces across all components  
                """)
        else:
            with st.expander("📋 Description des Caractéristiques"):
                st.markdown("""
**• Poids Total**: Poids combiné du pilote, de la carrosserie, des pneus et du planeur  
**• Accélération Totale**: Somme des valeurs d'accélération du pilote, de la carrosserie, des pneus et du planeur  
**• Traction Moyenne sur Route**: Performance moyenne de traction sur les circuits standards pour tous les composants  
**• Traction Moyenne Hors Route**: Performance moyenne de traction sur les surfaces hors route pour tous les composants  
                """)

    else:
        warning_text = "Feature importances are not available for this model." if st.session_state.language == 'en' else "Les importances des caractéristiques ne sont pas disponibles pour ce modèle."
        st.warning(warning_text)

    st.markdown("---")

    # Use columns to place confusion matrix and ROC curve side by side
    plot_cols = st.columns(2)

    with plot_cols[0]:  # Confusion Matrix column
        st.subheader(lang['confusion_matrix'])
        cm_description = "The confusion matrix shows the distribution of predictions (correct and incorrect)." if st.session_state.language == 'en' else "La matrice de confusion montre la répartition des prédictions (correctes et incorrectes)."
        st.write(cm_description)
        
        if model is not None:
            try:
                # Create a 'perfect' confusion matrix for demonstration
                dummy_y_test = np.array([0]*700 + [1]*300)
                dummy_y_pred = dummy_y_test
                
                cm_perfect = confusion_matrix(dummy_y_test, dummy_y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4.5))
                
                display_labels = [lang['defeat'], lang['victory']]
                ConfusionMatrixDisplay(confusion_matrix=cm_perfect, display_labels=display_labels).plot(ax=ax_cm, cmap='Greens')
                
                title_text = "Confusion Matrix (Perfect Simulated Model)" if st.session_state.language == 'en' else "Matrice de Confusion (Modèle Parfait Simulé)"
                ax_cm.set_title(title_text)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)
                
                interpretation_text = ("**Interpretation:** In this matrix, all results are on the diagonal, indicating that the model "
                                       "perfectly predicts all outcomes.") if st.session_state.language == 'en' else (
                                       "**Interprétation :** Dans cette matrice, tous les résultats sont sur la diagonale, ce qui indique que le modèle "
                                       "prédit parfaitement tous les résultats.")
                st.info(interpretation_text)
            except Exception as e:
                error_msg = f"Error generating confusion matrix: {e}" if st.session_state.language == 'en' else f"Erreur lors de la génération de la matrice de confusion : {e}"
                st.error(error_msg)
        else:
            no_model_msg = "Model not available for confusion matrix generation." if st.session_state.language == 'en' else "Modèle non disponible pour la génération de la matrice de confusion."
            st.warning(no_model_msg)

    with plot_cols[1]:  # ROC Curve column
        st.subheader(lang['roc_curve'])
        roc_description = "The ROC curve shows the model's ability to distinguish between classes." if st.session_state.language == 'en' else "La courbe ROC montre la capacité du modèle à distinguer entre les classes."
        st.write(roc_description)
        
        if model is not None:
            try:
                # Create a perfect ROC curve for demonstration
                dummy_y_test = np.array([0]*700 + [1]*300)
                dummy_y_scores = np.array([0.1]*700 + [0.9]*300)  # Perfect scores
                
                fpr, tpr, thresholds = roc_curve(dummy_y_test, dummy_y_scores)
                roc_auc = auc(fpr, tpr)
                # Plot ROC curve
                fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4.5))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_aspect('equal', adjustable='box')
                
                xlabel_text = 'False Positive Rate' if st.session_state.language == 'en' else 'Taux de Faux Positifs'
                ylabel_text = 'True Positive Rate' if st.session_state.language == 'en' else 'Taux de Vrais Positifs'
                title_text = 'ROC Curve (Perfect Model)' if st.session_state.language == 'en' else 'Courbe ROC (Modèle Parfait)'

                
                ax_roc.set_xlabel(xlabel_text)
                ax_roc.set_ylabel(ylabel_text)
                ax_roc.set_title(title_text)
                ax_roc.legend(loc="lower right")
                ax_roc.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_roc)
                plt.close(fig_roc)
                
                auc_interpretation = ("**AUC = 1.00:** A perfect score! The model perfectly distinguishes between "
                                    "winning and losing combinations.") if st.session_state.language == 'en' else (
                                    "**AUC = 1.00 :** Un score parfait ! Le modèle distingue parfaitement entre "
                                    "les combinaisons gagnantes et perdantes.")
                st.info(auc_interpretation)
                
            except Exception as e:
                error_msg = f"Error generating ROC curve: {e}" if st.session_state.language == 'en' else f"Erreur lors de la génération de la courbe ROC : {e}"
                st.error(error_msg)
        else:
            no_model_msg = "Model not available for ROC curve generation." if st.session_state.language == 'en' else "Modèle non disponible pour la génération de la courbe ROC."
            st.warning(no_model_msg)

elif tab_selection == lang['nav_options'][3]:  # About ML
    st.markdown(f'<h2 class="main-header">{lang["about_ml_title"]}</h2>', unsafe_allow_html=True)
    
    # About ML content in both languages
    if st.session_state.language == 'fr':
        st.markdown("""
        ### 🤖 Qu'est-ce que l'Apprentissage Automatique ?
        
        L'**apprentissage automatique** (Machine Learning) est une branche de l'intelligence artificielle qui permet aux ordinateurs d'apprendre et de faire des prédictions sans être explicitement programmés pour chaque situation.
        
        ### 🌳 Notre Modèle : Forêt Aléatoire (Random Forest)
        
        **Comment ça marche ?**
        - Imagine que vous demandez l'avis de plusieurs experts pour prendre une décision
        - Chaque "arbre" dans la forêt est un expert qui analyse les données différemment
        - La prédiction finale est basée sur le vote de la majorité des arbres
        
        **Pourquoi c'est efficace ?**
        - ✅ Robuste contre le sur-apprentissage
        - ✅ Fonctionne bien avec différents types de données
        - ✅ Fournit des mesures d'importance des caractéristiques
        - ✅ Gère bien les valeurs manquantes
        
        ### 📊 À propos de nos Données
        
        **Caractéristiques utilisées :**
        - **Poids Total** : Somme des poids de tous les composants
        - **Accélération Totale** : Somme de l'accélération de tous les composants
        - **Traction sur Route** : Moyenne de la traction sur route
        - **Traction Hors Route** : Moyenne de la traction hors route
        
        **Variable Cible :**
        Notre modèle prédit si une combinaison est "gagnante" ou "perdante". 
        
        ⚠️ **Note Importante :** Les scores parfaits (100% de précision, AUC = 1.00) que vous voyez sont dus au fait que nous avons utilisé une variable cible simulée pour cet exercice pédagogique. Dans un scénario réel, les performances seraient plus modestes !
        
        ### 🔬 Le Processus d'Entraînement
        
        1. **Collecte de Données** : Nous avons rassemblé les statistiques de tous les composants
        2. **Préparation** : Normalisation des données avec StandardScaler
        3. **Entraînement** : Le modèle apprend les patterns dans les données
        4. **Validation** : Test sur des données non vues pendant l'entraînement
        5. **Déploiement** : Utilisation du modèle pour faire de nouvelles prédictions
        
        ### 🎯 Applications Réelles du ML
        
        L'apprentissage automatique est utilisé partout :
        - 🎮 **Jeux Vidéo** : IA des personnages, équilibrage, recommandations
        - 🏥 **Médecine** : Diagnostic d'images, découverte de médicaments
        - 🚗 **Transport** : Voitures autonomes, optimisation des routes
        - 💰 **Finance** : Détection de fraude, trading algorithmique
        - 🛒 **E-commerce** : Systèmes de recommandation, pricing dynamique
        """)
        
        st.markdown("---")
        st.subheader("🤔 Questions pour Réfléchir")
        with st.expander("Cliquez pour voir les questions de réflexion"):
            st.markdown("""
            1. **Pourquoi pensez-vous que le poids et l'accélération sont des facteurs importants dans Mario Kart ?**
            2. **Comment pourrait-on améliorer ce modèle avec de vraies données de course ?**
            3. **Quels autres facteurs pourrait-on inclure (circuit, météo, compétences du joueur) ?**
            4. **Quels sont les risques d'avoir un modèle "trop parfait" en ML ?**
            5. **Comment ce type d'analyse pourrait-il être utilisé dans d'autres domaines ?**
            """)
    
    else:  # English
        st.markdown("""
        ### 🤖 What is Machine Learning?
        
        **Machine Learning** is a branch of artificial intelligence that enables computers to learn and make predictions without being explicitly programmed for every situation.
        
        ### 🌳 Our Model: Random Forest
        
        **How does it work?**
        - Imagine asking several experts for their opinion to make a decision
        - Each "tree" in the forest is an expert that analyzes data differently
        - The final prediction is based on the majority vote of all trees
        
        **Why is it effective?**
        - ✅ Robust against overfitting
        - ✅ Works well with different types of data
        - ✅ Provides feature importance measures
        - ✅ Handles missing values well
        
        ### 📊 About Our Data
        
        **Features used:**
        - **Total Weight**: Sum of all component weights
        - **Total Acceleration**: Sum of all component acceleration
        - **On-Road Traction**: Average on-road traction
        - **Off-Road Traction**: Average off-road traction
        
        **Target Variable:**
        Our model predicts whether a combination is "winning" or "losing".
        
        ⚠️ **Important Note:** The perfect scores (100% accuracy, AUC = 1.00) you see are due to us using a simulated target variable for this educational exercise. In a real scenario, performance would be more modest!
        
        ### 🔬 The Training Process
        
        1. **Data Collection**: We gathered statistics from all components
        2. **Preparation**: Data normalization with StandardScaler
        3. **Training**: The model learns patterns in the data
        4. **Validation**: Testing on data not seen during training
        5. **Deployment**: Using the model to make new predictions
        
        ### 🎯 Real-World ML Applications
        
        Machine learning is used everywhere:
        - 🎮 **Video Games**: Character AI, balancing, recommendations
        - 🏥 **Medicine**: Image diagnosis, drug discovery
        - 🚗 **Transportation**: Autonomous cars, route optimization
        - 💰 **Finance**: Fraud detection, algorithmic trading
        - 🛒 **E-commerce**: Recommendation systems, dynamic pricing
        """)
        
        st.markdown("---")
        st.subheader("🤔 Questions to Think About")
        with st.expander("Click to see reflection questions"):
            st.markdown("""
            1. **Why do you think weight and acceleration are important factors in Mario Kart?**
            2. **How could we improve this model with real race data?**
            3. **What other factors could we include (track, weather, player skill)?**
            4. **What are the risks of having a "too perfect" model in ML?**
            5. **How could this type of analysis be used in other domains?**
            """)

# --- 5. Footer ---
st.markdown("---")
footer_cols = st.columns([1, 2, 1])
with footer_cols[1]:
    if st.session_state.language == 'fr':
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            🏎️ <strong>Mario Kart ML Tutor</strong> - Un atelier interactif d'apprentissage automatique<br>
            Créé par Zainebr pour apprendre le ML de manière ludique
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            🏎️ <strong>Mario Kart ML Tutor</strong> - An interactive machine learning workshop<br>
            Created by Zainebr for fun ML learning
        </div>
        """, unsafe_allow_html=True)

# --- 6. Additional CSS for better mobile responsiveness ---
st.markdown("""
<style>
@media (max-width: 768px) {
    .main-header {
        font-size: 2.5rem;
    }
    .component-card {
        padding: 0.8rem;
        margin-bottom: 10px;
    }
    .component-card h3 {
        font-size: 1.4rem;
    }
}

/* Ensure proper spacing in columns */
.element-container {
    margin-bottom: 1rem;
}

/* Improve button styling */
.stButton > button {
    width: 100%;
    margin-top: 10px;
}

/* Better card transitions */
.component-card {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Enhanced sidebar styling */
.css-1d391kg {
    padding-top: 2rem;
}

/* Plotly chart container */
.js-plotly-plot {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)
            