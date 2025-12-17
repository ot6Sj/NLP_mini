import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
import os
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px

try:
    stop_en = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_en = set(stopwords.words('english'))

# Page config
st.set_page_config(
    page_title="Medical AI Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour l interface
st.markdown("""
    <style>
    /* Global background */
    .stApp {
        background: radial-gradient(circle at top left, #8f94fb 0, #4e54c8 35%, #1b1036 100%);
    }

    /* Center content and constrain width */
    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }

    /* Header */
    .header-container {
        text-align: center;
        padding: 2rem 1rem;
        border-radius: 24px;
        margin-bottom: 2.2rem;
        background: linear-gradient(135deg, rgba(102,126,234,0.95), rgba(118,75,162,0.98));
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(14px);
    }
    .main-title {
        color: #ffffff;
        font-size: 2.7rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.04em;
    }
    .sub-title {
        color: rgba(255,255,255,0.9);
        font-size: 1.05rem;
        margin-top: 0.6rem;
    }

    /* Main white cards */
    .glass-card {
        background: rgba(255,255,255,0.96);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 16px 35px rgba(15,23,42,0.25);
        border: 1px solid rgba(148,163,184,0.25);
        margin-bottom: 1.5rem;
    }

    /* Prediction card */
    .prediction-card {
        background: radial-gradient(circle at top, #667eea 0, #764ba2 50%, #3b2667 100%);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0 1.8rem 0;
        box-shadow: 0 18px 40px rgba(15,23,42,0.55);
        border: 1px solid rgba(255,255,255,0.16);
        animation: slideIn 0.5s ease-out;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Text area & input */
    .stTextArea textarea, .stTextInput input {
        border-radius: 14px !important;
        border: 1px solid #d0d7ff !important;
        padding: 0.85rem 0.9rem !important;
        font-size: 0.95rem !important;
        box-shadow: 0 10px 25px rgba(15,23,42,0.08) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 1px rgba(102,126,234,0.4) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 999px;
        box-shadow: 0 14px 28px rgba(88, 113, 221, 0.55);
        transition: all 0.18s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 18px 34px rgba(88, 113, 221, 0.7);
        background: linear-gradient(135deg, #7a8cfb, #8c5ad9);
    }

    /* Sidebar glass effect */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141226 0%, #2b2250 60%, #3a275d 100%) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(15,23,42,0.15);
        backdrop-filter: blur(16px);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5 {
        color: #f9fafb !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] li {
        color: #e5e7eb !important;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stMetric {
        background: rgba(15,23,42,0.65);
        border-radius: 14px;
        padding: 0.5rem 0.85rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.35);
        border: 1px solid rgba(148,163,184,0.35);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22c55e, #eab308);
    }

    /* Pills / badges */
    .class-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        background: rgba(79,70,229,0.06);
        border-radius: 999px;
        margin: 0.2rem;
        font-size: 0.84rem;
        border: 1px solid rgba(129,140,248,0.7);
        color: #white293b;
    }

    /* Expander header */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #4c51bf, #805ad5);
        color: #f9fafb !important;
        border-radius: 12px !important;
        font-weight: 600;
    }

    /* Footer text contrast */
    .footer-text {
        color: #white !important;
    }
    </style>
            """, unsafe_allow_html=True)

# pretraitement et chargement du model
def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def get_top_words(classifier, tfidf, class_label, k=10):
    feature_names = np.array(tfidf.get_feature_names_out())
    class_index = list(classifier.classes_).index(class_label)
    coef = classifier.coef_[class_index]
    top_indices = np.argsort(coef)[-k:]
    return list(reversed(feature_names[top_indices]))

@st.cache_resource
def load_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "model.pkl"), "rb") as f:
            clf = pickle.load(f)
        with open(os.path.join(script_dir, "tfidf.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        return clf, tfidf, None
    except FileNotFoundError:
        return None, None, "Model files not found. Put model.pkl and tfidf.pkl next to app.py"
    except Exception as e:
        return None, None, f"Error loading model: {e}"

@st.cache_resource
def load_dataset():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "all_medicine databased.csv")
        df = pd.read_csv(data_path, low_memory=False)
        return df, None
    except FileNotFoundError:
        return None, "Dataset file not found. Put all_medicine databased.csv next to app.py"
    except Exception as e:
        return None, f"Error loading dataset: {e}"

clf, tfidf, error = load_model()
df_meds, data_error = load_dataset()


THERAPEUTIC_COL = "Therapeutic Class"
HABIT_FORMING_COL = "Habit Forming"
MED_NAME_COL = "name"  

# traduct
if "language" not in st.session_state:
    st.session_state.language = "en"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

english_examples = {
    "diabetes": "Treatment for type 2 diabetes mellitus. Helps control blood glucose levels. May cause hypoglycemia, low blood sugar, nausea, and dizziness.",
    "antibiotic": "Treatment for bacterial infections including respiratory tract infections. Common side effects include nausea, diarrhea, and allergic reactions.",
    "cardiac": "Treatment for hypertension and high blood pressure. Helps reduce cholesterol levels and prevents heart attacks. May cause dizziness and fatigue.",
    "neurological": "Treatment for epilepsy and seizures. Helps control neuropathic pain and anxiety. May cause drowsiness and weight gain.",
    "respiratory": "Treatment for asthma and allergic conditions. Relieves cough and nasal congestion. Common cold symptoms relief.",
}

translations = {
    "en": {
        "title": "🔬 Medical AI Classifier",
        "subtitle": "Advanced Therapeutic Class Prediction System",
        "model_status": "📊 Model Status",
        "model_active": "✅ Model Active",
        "available_classes": "Available Classes",
        "model_accuracy": "Model Accuracy",
        "about": "📖 About",
        "about_text": (
            "This AI-powered system predicts therapeutic classifications "
            "for medications using advanced machine learning algorithms.\n\n"
            "**Features:**\n"
            "- 22 Therapeutic Classes\n"
            "- 99.5% Accuracy\n"
            "- Real-time Predictions\n"
            "- Confidence Scoring"
        ),
        "how_to_use": "🎯 How to Use",
        "how_to_steps": "1. 📝 Enter medication description\n2. 🔍 Click predict button\n3. 📊 View results & insights\n",
        "project_info": "👨‍💻 Project Info",
        "project_details": (
            "**Academic Project by\n"
            "Marouan LAAZIBI and Adnane RAHAOUI**  \n"
            "Natural Language Processing  \n"
            "2025"
        ),
        "input_description": "📝 Input Medical Description",
        "placeholder": (
            "Enter medication details here...\n\nExample: Treatment for type 2 diabetes mellitus. "
            "Helps control blood glucose levels. May cause hypoglycemia, nausea, and dizziness..."
        ),
        "analyze_button": "🔍 Analyze & Predict",
        "analyzing": "🔄 Analyzing medication data...",
        "predicted_class": "🎯 Predicted Class",
        "confidence": "Confidence",
        "confidence_score": "Confidence Score",
        "top_3": "📊 Top 3 Predictions",
        "key_indicators": "🔑 Key Indicators",
        "warning_empty": "⚠️ Please enter a medical description first!",
        "quick_examples": "💡 Quick Examples",
        "click_to_try": "*Click to try:*",
        "available_classes_title": "📚 Available Classes",
        "view_all_classes": "View All 22 Classes",
        "model_not_found": "❌ Model Not Found",
        "setup_instructions": "📝 Setup Instructions",
        "file_structure": "📁 File Structure",
        "run_command": "🚀 Run Command",
        "footer": "🔬 Medical AI Classifier | Academic Project 2025",
        "powered_by": "Powered by Machine Learning & Natural Language Processing",
        "diabetes": "💊 Diabetes Medication",
        "antibiotic": "💉 Antibiotic",
        "cardiac": "❤️ Cardiac Medicine",
        "neurological": "🧠 Neurological",
        "respiratory": "🌬 Respiratory",
        "diabetes_text": english_examples["diabetes"],
        "antibiotic_text": english_examples["antibiotic"],
        "cardiac_text": english_examples["cardiac"],
        "neurological_text": english_examples["neurological"],
        "respiratory_text": english_examples["respiratory"],
        "risk_profile_title": "📉 Side Effect Risk Profile",
        "risk_profile_intro": "Based on our internal database, here are the most frequently reported side effects for this therapeutic class.",
        "no_data_for_class": "No statistical data available in the dataset for this therapeutic class.",
        "vigilance_title": "⚠️ Vigilance Indicator",
        "vigilance_text": "Percentage of medicines in this class that are reported as habit-forming in the dataset.",
        "vigilance_no_data": "No Habit Forming information available for this class in the dataset.",
        "substitute_section_title": "💊 Medic brands title",
        "substitute_input_label": "Type the exact medicine name to look for cheaper or alternative substitutes (brand or generic).",
        "substitute_input_placeholder": "Example: augmentin 625 duo tablet",
        "substitute_no_data": "Dataset not loaded. Please make sure all_medicine databased.csv is in the same folder as app.py.",
        "substitute_no_match": "No medicine found with this name in the dataset.",
        "substitute_results_title": "Substitutes found in the database",
        "substitute_for": "Substitutes for",
    },
    "fr": {
        "title": "🔬 Classificateur IA Médical",
        "subtitle": "Système Avancé de Prédiction de Classe Thérapeutique",
        "model_status": "📊 État du Modèle",
        "model_active": "✅ Modèle Actif",
        "available_classes": "Classes Disponibles",
        "model_accuracy": "Précision du Modèle",
        "about": "📖 À Propos",
        "about_text": (
            "Ce système alimenté par IA prédit les classifications thérapeutiques "
            "des médicaments en utilisant des algorithmes d'apprentissage automatique avancés.\n\n"
            "**Caractéristiques :**\n"
            "- 22 Classes Thérapeutiques\n"
            "- 99.5% de Précision\n"
            "- Prédictions en Temps Réel\n"
            "- Score de Confiance"
        ),
        "how_to_use": "🎯 Comment Utiliser",
        "how_to_steps": "1. 📝 Entrez la description du médicament\n2. 🔍 Cliquez sur le bouton prédire\n3. 📊 Consultez les résultats\n",
        "project_info": "👨‍💻 Info Projet",
        "project_details": (
            "**Projet Académique par\n"
            "Marouan LAAZIBI et Adnane RAHAOUI**  \n"
            "Traitement du Langage Naturel  \n"
            "2025"
        ),
        "input_description": "📝 Entrez la Description Médicale",
        "placeholder": (
            "Entrez les détails du médicament ici...\n\nExemple: Traitement du diabète de type 2. "
            "Aide à contrôler les niveaux de glucose sanguin. Peut causer hypoglycémie, nausées et vertiges..."
        ),
        "analyze_button": "🔍 Analyser & Prédire",
        "analyzing": "🔄 Analyse des données médicales...",
        "predicted_class": "🎯 Classe Prédite",
        "confidence": "Confiance",
        "confidence_score": "Score de Confiance",
        "top_3": "📊 Top 3 Prédictions",
        "key_indicators": "🔑 Indicateurs Clés",
        "warning_empty": "⚠️ Veuillez entrer une description médicale d'abord !",
        "quick_examples": "💡 Exemples Rapides",
        "click_to_try": "*Cliquez pour essayer :*",
        "available_classes_title": "📚 Classes Disponibles",
        "view_all_classes": "Voir Toutes les 22 Classes",
        "model_not_found": "❌ Modèle Non Trouvé",
        "setup_instructions": "📝 Instructions d'Installation",
        "file_structure": "📁 Structure des Fichiers",
        "run_command": "🚀 Commande d'Exécution",
        "footer": "🔬 Classificateur IA Médical | Projet Académique 2025",
        "powered_by": "Propulsé par l'Apprentissage Automatique et le Traitement du Langage Naturel",
        "diabetes": "💊 Médicament Diabète",
        "antibiotic": "💉 Antibiotique",
        "cardiac": "❤️ Médicament Cardiaque",
        "neurological": "🧠 Neurologique",
        "respiratory": "🌬 Respiratoire",
        "diabetes_text": "Traitement du diabète de type 2...",
        "antibiotic_text": "Traitement des infections bactériennes...",
        "cardiac_text": "Traitement de l'hypertension...",
        "neurological_text": "Traitement de l'épilepsie...",
        "respiratory_text": "Traitement de l'asthme...",
        "risk_profile_title": "📉 Profil de Risque (Effets Secondaires)",
        "risk_profile_intro": "Selon notre base de données interne, voici les effets secondaires les plus fréquemment signalés pour cette classe thérapeutique.",
        "no_data_for_class": "Aucune donnée statistique disponible dans le dataset pour cette classe thérapeutique.",
        "vigilance_title": "⚠️ Indicateur de Vigilance",
        "vigilance_text": "Pourcentage de médicaments de cette classe signalés comme créant une accoutumance dans la base de données.",
        "vigilance_no_data": "Aucune information sur la dépendance (Habit Forming) n’est disponible pour cette classe dans le dataset.",
        "substitute_section_title": "💊 Marques de médicaments",
        "substitute_input_label": "Tapez le nom exact du médicament pour chercher des substituts (marque ou générique).",
        "substitute_input_placeholder": "Exemple : augmentin 625 duo tablet",
        "substitute_no_data": "Dataset non chargé. Assurez-vous que le fichier all_medicine databased.csv est dans le même dossier que app.py.",
        "substitute_no_match": "Aucun médicament trouvé avec ce nom dans la base de données.",
        "substitute_results_title": "Substituts trouvés dans la base de données",
        "substitute_for": "Substituts pour",
    },
}

def t(key: str) -> str:
    return translations[st.session_state.language][key]

# header et traduction du fr a en
st.markdown(
    f"""
    <div class="header-container">
        <h1 class="main-title">{t('title')}</h1>
        <p class="sub-title">{t('subtitle')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_lang1, col_lang2 = st.columns([5, 1])
with col_lang2:
    lang_option = st.selectbox(
        "🌐",
        options=["English", "Français"],
        index=0 if st.session_state.language == "en" else 1,
        label_visibility="collapsed",
    )
    if lang_option == "English" and st.session_state.language != "en":
        st.session_state.language = "en"
        st.rerun()
    elif lang_option == "Français" and st.session_state.language != "fr":
        st.session_state.language = "fr"
        st.rerun()

# sidebar 
with st.sidebar:
    st.markdown(f"### {t('model_status')}")
    if error:
        st.error(f"❌ {error}")
    elif clf is not None and tfidf is not None:
        st.success(t("model_active"))
        st.metric(t("available_classes"), len(clf.classes_))
        st.metric(t("model_accuracy"), "99.5%")

    st.markdown("---")
    st.markdown("### Dataset")
    if data_error:
        st.error(f"❌ {data_error}")
    elif df_meds is not None:
        st.success("✅ Dataset loaded")
        st.metric("Rows", len(df_meds))
    else:
        st.info("Dataset not available")

    st.markdown("---")
    st.markdown(f"### {t('about')}")
    st.info(t("about_text"))

    st.markdown("---")
    st.markdown(f"### {t('how_to_use')}")
    st.markdown(t("how_to_steps"))

    st.markdown("---")
    st.markdown(f"### {t('project_info')}")
    st.markdown(t("project_details"))

# main 
if clf is not None and tfidf is not None:
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown(f"### {t('input_description')}")
        user_input = st.text_area(
            "",
            height=200,
            placeholder=t("placeholder"),
            value=st.session_state.input_text,
            label_visibility="collapsed",
        )
        if user_input != st.session_state.input_text:
            st.session_state.input_text = user_input

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button(t("analyze_button"), use_container_width=True)

        if predict_button:
            if user_input.strip():
                with st.spinner(t("analyzing")):
                    text_for_model = user_input
                    cleaned_input = clean_text(text_for_model)
                    input_vec = tfidf.transform([cleaned_input])
                    prediction = clf.predict(input_vec)[0]
                    probabilities = clf.predict_proba(input_vec)[0]
                    class_index = list(clf.classes_).index(prediction)
                    confidence = probabilities[class_index] * 100

                st.markdown("---")
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <h2 style="margin: 0;">{t('predicted_class')}</h2>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0;">{prediction}</h1>
                        <h3 style="margin: 0;">{t('confidence')}: {confidence:.2f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": t("confidence_score"), "font": {"size": 24}},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": "#667eea"},
                        },
                    )
                )
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"### {t('top_3')}")
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                for i, idx in enumerate(top_3_indices):
                    class_name = clf.classes_[idx]
                    prob = probabilities[idx] * 100
                    col_rank, col_name, col_prob = st.columns([0.5, 2, 1])
                    with col_rank:
                        st.markdown("### 🥇" if i == 0 else "### 🥈" if i == 1 else "### 🥉")
                    with col_name:
                        st.markdown(f"**{class_name}**")
                    with col_prob:
                        st.markdown(f"**{prob:.1f}%**")
                    st.progress(prob / 100)
                    st.markdown("")

                st.markdown(f"### {t('key_indicators')}")
                top_words = get_top_words(clf, tfidf, prediction, k=15)
                words_html = " ".join(
                    [f'<span class="class-badge">#{word}</span>' for word in top_words]
                )
                st.markdown(f'<div style="line-height: 2.5;">{words_html}</div>', unsafe_allow_html=True)

                # effect risk profile 
                st.markdown(f"### {t('risk_profile_title')}")
                st.markdown(t("risk_profile_intro"))

                if df_meds is None or data_error:
                    st.info(t("substitute_no_data"))
                else:
                    side_effect_cols = [c for c in df_meds.columns if c.startswith("sideEffect")]
                    if THERAPEUTIC_COL in df_meds.columns and side_effect_cols:
                        df_class = df_meds[df_meds[THERAPEUTIC_COL] == prediction]
                        if df_class.empty:
                            st.info(t("no_data_for_class"))
                        else:
                            se_values = df_class[side_effect_cols].values.ravel()
                            effects = []
                            for val in se_values:
                                if pd.notna(val):
                                    txt = str(val).lower()
                                    parts = [
                                        p.strip()
                                        for p in re.split(r",|;", txt)
                                        if p.strip() and p.lower() != "nan"
                                    ]
                                    effects.extend(parts)
                            if not effects:
                                st.info(t("no_data_for_class"))
                            else:
                                counter = Counter(effects)
                                total_mentions = sum(counter.values())
                                top_5 = counter.most_common(5)
                                risk_df = pd.DataFrame(
                                    [
                                        {
                                            "Side effect": eff,
                                            "Count": cnt,
                                            "Percentage": round(cnt / total_mentions * 100, 1),
                                        }
                                        for eff, cnt in top_5
                                    ]
                                )
                                st.dataframe(risk_df, use_container_width=True)
                                fig_se = px.bar(
                                    risk_df,
                                    x="Side effect",
                                    y="Percentage",
                                    text="Percentage",
                                )
                                fig_se.update_traces(texttemplate="%{text}%", textposition="outside")
                                fig_se.update_layout(
                                    yaxis_title="%",
                                    xaxis_title="",
                                    height=350,
                                    margin=dict(l=20, r=20, t=40, b=40),
                                )
                                st.plotly_chart(fig_se, use_container_width=True)
                    else:
                        st.info("Side effect columns not found in dataset.")

                # Vigilance indicator
                st.markdown(f"### {t('vigilance_title')}")
                st.markdown(t("vigilance_text"))

                if df_meds is None or data_error:
                    st.info(t("substitute_no_data"))
                else:
                    if THERAPEUTIC_COL in df_meds.columns and HABIT_FORMING_COL in df_meds.columns:
                        df_class = df_meds[df_meds[THERAPEUTIC_COL] == prediction]
                        hf_series = (
                            df_class[HABIT_FORMING_COL].dropna().astype(str).str.lower()
                        )
                        if len(hf_series) == 0:
                            st.info(t("vigilance_no_data"))
                        else:
                            habit_true = hf_series.isin(["yes", "true", "1", "oui"])
                            habit_rate = habit_true.mean() * 100
                            col_v1, col_v2 = st.columns(2)
                            with col_v1:
                                st.metric("Habit forming", f"{habit_rate:.1f}%")
                            with col_v2:
                                st.markdown(f"**{len(df_class)}** medicines in this class in the dataset.")
                    else:
                        st.info("Habit Forming column not found in dataset.")
            else:
                st.warning(t("warning_empty"))

    with col2:
        st.markdown(f"### {t('quick_examples')}")
        st.markdown(t("click_to_try"))
        examples = {
            t("diabetes"): ("diabetes", t("diabetes_text")),
            t("antibiotic"): ("antibiotic", t("antibiotic_text")),
            t("cardiac"): ("cardiac", t("cardiac_text")),
            t("neurological"): ("neurological", t("neurological_text")),
            t("respiratory"): ("respiratory", t("respiratory_text")),
        }
        for name, (ex_key, text) in examples.items():
            if st.button(name, key=f"ex_{ex_key}", use_container_width=True):
                st.session_state.input_text = text
                st.session_state.selected_example = ex_key
                st.rerun()

        st.markdown("---")
        st.markdown(f"### {t('available_classes_title')}")
        if clf is not None:
            with st.expander(t("view_all_classes"), expanded=False):
                classes = sorted(clf.classes_)
                for i in range(0, len(classes), 2):
                    if i + 1 < len(classes):
                        st.markdown(
                            f"""
                            <div style="display: flex; gap: 0.5rem;">
                                <span class="class-badge" style="flex: 1;">✓ {classes[i]}</span>
                                <span class="class-badge" style="flex: 1;">✓ {classes[i+1]}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<span class="class-badge">✓ {classes[i]}</span>',
                            unsafe_allow_html=True,
                        )

    # substitutes search 
    st.markdown("---")
    st.markdown(f"## {t('substitute_section_title')}")
    if df_meds is None or data_error:
        st.info(t("substitute_no_data"))
    else:
        if MED_NAME_COL not in df_meds.columns:
            st.info("Medicine name column not found in dataset.")
        else:
            drug_query = st.text_input(
                t("substitute_input_label"),
                placeholder=t("substitute_input_placeholder"),
            )
            if drug_query:
                mask = df_meds[MED_NAME_COL].astype(str).str.contains(
                    drug_query, case=False, na=False
                )
                results = df_meds[mask]
                if results.empty:
                    st.warning(t("substitute_no_match"))
                else:
                    st.markdown(f"### {t('substitute_results_title')}")
                    sub_cols = [
                        c for c in df_meds.columns if c.lower().startswith("substitute")
                    ]
                    for _, row in results.head(5).iterrows():
                        med_name = row[MED_NAME_COL]
                        subs = [
                            str(row[c])
                            for c in sub_cols
                            if pd.notna(row[c]) and str(row[c]).strip()
                        ]
                        with st.expander(f"{t('substitute_for')} **{med_name}**"):
                            if subs:
                                st.markdown("\n".join([f"- {s}" for s in subs]))
                            else:
                                st.markdown("No substitutes listed for this medicine in the dataset.")

else:
    # modele n existe pas
    st.error(f"### {t('model_not_found')}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {t('setup_instructions')}")
        st.markdown(
            "error"
        )

# footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: black; padding: 2rem 0;">
        <p>{t('footer')}</p>
        <p style="font-size: 0.9rem;">{t('powered_by')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
