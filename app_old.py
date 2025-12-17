import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
import os
import plotly.graph_objects as go
import plotly.express as px

# Download stopwords if not already there (first time only)
try:
    stop_en = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_en = set(stopwords.words('english'))

# Set page configuration
st.set_page_config(
    page_title="Medical AI Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the design wkda
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #3C2653 100%);
    }
    
    /* Custom container styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .sub-title {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: border 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .sidebar-content, [data-testid="stSidebar"] > div:first-child {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem;
    }
    
    /* Sidebar text colors */
    [data-testid="stSidebar"] .element-container {
        color: #1a1a1a;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Sidebar metrics */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-weight: bold;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
    }
    
    /* Example button styling */
    .example-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        margin: 0.3rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Class badge */
    .class-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Clean up messy text (remove URLs, special chars, etc.)
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Get the most important words for each drug class
def get_top_words(classifier, tfidf, class_label, k=10):
    feature_names = np.array(tfidf.get_feature_names_out())
    class_index = list(classifier.classes_).index(class_label)
    coef = classifier.coef_[class_index]
    top_indices = np.argsort(coef)[-k:]
    return list(reversed(feature_names[top_indices]))

# Load model and vectorizer (cached so we don't reload every time - saves time!)
@st.cache_resource
def load_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(script_dir, 'model.pkl')
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        
        tfidf_path = os.path.join(script_dir, 'tfidf.pkl')
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        
        return clf, tfidf, None
    except FileNotFoundError as e:
        return None, None, f"Model files not found. Please ensure 'model.pkl' and 'tfidf.pkl' are in the same directory as app.py"
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# Load the model (yarbi ykon tma)
clf, tfidf, error = load_model()

# Language selector
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Keep track of what the user types
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Remember which example button was clicked
if 'input_language' not in st.session_state:
    st.session_state.input_language = 'en'

# English versions of examples (so the model doesn't get confused with French!)
english_examples = {
    'diabetes': 'Treatment for type 2 diabetes mellitus. Helps control blood glucose levels. May cause hypoglycemia, low blood sugar, nausea, and dizziness.',
    'antibiotic': 'Treatment for bacterial infections including respiratory tract infections. Common side effects include nausea, diarrhea, and allergic reactions.',
    'cardiac': 'Treatment for hypertension and high blood pressure. Helps reduce cholesterol levels and prevents heart attacks. May cause dizziness and fatigue.',
    'neurological': 'Treatment for epilepsy and seizures. Helps control neuropathic pain and anxiety. May cause drowsiness and weight gain.',
    'respiratory': 'Treatment for asthma and allergic conditions. Relieves cough and nasal congestion. Common cold symptoms relief.'
}

# Translation dictionary
translations = {
    'en': {
        'title': '🔬 Medical AI Classifier',
        'subtitle': 'Advanced Therapeutic Class Prediction System',
        'model_status': '📊 Model Status',
        'model_active': '✅ Model Active',
        'available_classes': 'Available Classes',
        'model_accuracy': 'Model Accuracy',
        'about': '📖 About',
        'about_text': '''This AI-powered system predicts therapeutic classifications 
        for medications using advanced machine learning algorithms.
        
        **Features:**
        - 22 Therapeutic Classes
        - 99.5% Accuracy
        - Real-time Predictions
        - Confidence Scoring''',
        'how_to_use': '🎯 How to Use',
        'how_to_steps': '''1. 📝 Enter medication description\n2. 🔍 Click predict button\n3. 📊 View results & insights\n''',
        'project_info': '👨‍💻 Project Info',
        'project_details': '''**Academic Project by\nMarouan LAAZIBI and Adnane RAHAOUI**  
    Natural Language Processing  
    2025''',
        'input_description': '📝 Input Medical Description',
        'placeholder': 'Enter medication details here...\n\nExample: Treatment for type 2 diabetes mellitus. Helps control blood glucose levels. May cause hypoglycemia, nausea, and dizziness...',
        'analyze_button': '🔍 Analyze & Predict',
        'analyzing': '🔄 Analyzing medication data...',
        'predicted_class': '🎯 Predicted Class',
        'confidence': 'Confidence',
        'confidence_score': 'Confidence Score',
        'top_3': '📊 Top 3 Predictions',
        'key_indicators': '🔑 Key Indicators',
        'warning_empty': '⚠️ Please enter a medical description first!',
        'quick_examples': '💡 Quick Examples',
        'click_to_try': '*Click to try:*',
        'selected_example': '📋 Selected Example',
        'use_example': '📝 Use This Example',
        'available_classes_title': '📚 Available Classes',
        'view_all_classes': 'View All 22 Classes',
        'model_not_found': '❌ Model Not Found',
        'setup_instructions': '📝 Setup Instructions',
        'file_structure': '📁 File Structure',
        'run_command': '🚀 Run Command',
        'footer': '🔬 Medical AI Classifier | Academic Project 2025',
        'powered_by': 'Powered by Machine Learning & Natural Language Processing',
        'diabetes': '💊 Diabetes Medication',
        'antibiotic': '💉 Antibiotic',
        'cardiac': '❤️ Cardiac Medicine',
        'neurological': '🧠 Neurological',
        'respiratory': '🌬 Respiratory',
        'diabetes_text': 'Treatment for type 2 diabetes mellitus. Helps control blood glucose levels. May cause hypoglycemia, low blood sugar, nausea, and dizziness.',
        'antibiotic_text': 'Treatment for bacterial infections including respiratory tract infections. Common side effects include nausea, diarrhea, and allergic reactions.',
        'cardiac_text': 'Treatment for hypertension and high blood pressure. Helps reduce cholesterol levels and prevents heart attacks. May cause dizziness and fatigue.',
        'neurological_text': 'Treatment for epilepsy and seizures. Helps control neuropathic pain and anxiety. May cause drowsiness and weight gain.',
        'respiratory_text': 'Treatment for asthma and allergic conditions. Relieves cough and nasal congestion. Common cold symptoms relief.',
    },
    'fr': {
        'title': '🔬 Classificateur IA Médical',
        'subtitle': 'Système Avancé de Prédiction de Classe Thérapeutique',
        'model_status': '📊 État du Modèle',
        'model_active': '✅ Modèle Actif',
        'available_classes': 'Classes Disponibles',
        'model_accuracy': 'Précision du Modèle',
        'about': '📖 À Propos',
        'about_text': '''Ce système alimenté par IA prédit les classifications thérapeutiques 
        des médicaments en utilisant des algorithmes d\'apprentissage automatique avancés.
        
        **Caractéristiques:**
        - 22 Classes Thérapeutiques
        - 99.5% de Précision
        - Prédictions en Temps Réel
        - Score de Confiance''',
        'how_to_use': '🎯 Comment Utiliser',
        'how_to_steps': '''1. 📝 Entrez la description du médicament\n2. 🔍 Cliquez sur le bouton prédire\n3. 📊 Consultez les résultats\n''',
        'project_info': '👨‍💻 Info Projet',
        'project_details': '''**Projet Académique par\nMarouan LAAZIBI et Adnane RAHAOUI**  
    Traitement du Langage Naturel  
    2025''',
        'input_description': '📝 Entrez la Description Médicale',
        'placeholder': 'Entrez les détails du médicament ici...\n\nExemple: Traitement du diabète de type 2. Aide à contrôler les niveaux de glucose sanguin. Peut causer hypoglycémie, nausées et vertiges...',
        'analyze_button': '🔍 Analyser & Prédire',
        'analyzing': '🔄 Analyse des données médicales...',
        'predicted_class': '🎯 Classe Prédite',
        'confidence': 'Confiance',
        'confidence_score': 'Score de Confiance',
        'top_3': '📊 Top 3 Prédictions',
        'key_indicators': '🔑 Indicateurs Clés',
        'warning_empty': '⚠️ Veuillez entrer une description médicale d\'abord!',
        'quick_examples': '💡 Exemples Rapides',
        'click_to_try': '*Cliquez pour essayer:*',
        'selected_example': '📋 Exemple Sélectionné',
        'use_example': '📝 Utiliser cet Exemple',
        'available_classes_title': '📚 Classes Disponibles',
        'view_all_classes': 'Voir Toutes les 22 Classes',
        'model_not_found': '❌ Modèle Non Trouvé',
        'setup_instructions': '📝 Instructions d\'Installation',
        'file_structure': '📁 Structure des Fichiers',
        'run_command': '🚀 Commande d\'Exécution',
        'footer': '🔬 Classificateur IA Médical | Projet Académique 2025',
        'powered_by': 'Propulsé par l\'Apprentissage Automatique et le Traitement du Langage Naturel',
        'diabetes': '💊 Médicament Diabète',
        'antibiotic': '💉 Antibiotique',
        'cardiac': '❤️ Médicament Cardiaque',
        'neurological': '🧠 Neurologique',
        'respiratory': '🌬 Respiratoire',
        'diabetes_text': 'Traitement du diabète de type 2. Aide à contrôler les niveaux de glucose sanguin. Peut causer hypoglycémie, faible taux de sucre, nausées et vertiges.',
        'antibiotic_text': 'Traitement des infections bactériennes incluant les infections respiratoires. Effets secondaires communs: nausées, diarrhée et réactions allergiques.',
        'cardiac_text': 'Traitement de l\'hypertension et de la haute pression artérielle. Aide à réduire les niveaux de cholestérol et prévient les crises cardiaques. Peut causer vertiges et fatigue.',
        'neurological_text': 'Traitement de l\'épilepsie et des crises. Aide à contrôler la douleur neuropathique et l\'anxiété. Peut causer somnolence et gain de poids.',
        'respiratory_text': 'Traitement de l\'asthme et des conditions allergiques. Soulage la toux et la congestion nasale. Soulagement des symptômes du rhume commun.',
    }
}
#ot6_j
def t(key):
    """Quick translation helper"""
    return translations[st.session_state.language][key]

# Header
st.markdown(f"""
    <div class="header-container">
        <h1 class="main-title">{t('title')}</h1>
        <p class="sub-title">{t('subtitle')}</p>
    </div>
""", unsafe_allow_html=True)

# Language switcher in top right
col_lang1, col_lang2 = st.columns([5, 1])
with col_lang2:
    lang_option = st.selectbox(
        "🌐",
        options=['English', 'Français'],
        index=0 if st.session_state.language == 'en' else 1,
        label_visibility="collapsed"
    )
    # Switch language if needed
    if lang_option == 'English' and st.session_state.language != 'en':
        st.session_state.language = 'en'
        st.rerun()
    elif lang_option == 'Français' and st.session_state.language != 'fr':
        st.session_state.language = 'fr'
        st.rerun()

# Sidebar - where all the cool info lives
with st.sidebar:
    st.markdown(f"### {t('model_status')}")
    
    if error:
        st.error(f"❌ {error}")
    elif clf is not None and tfidf is not None:
        st.success(t('model_active'))
        st.metric(t('available_classes'), len(clf.classes_))
        st.metric(t('model_accuracy'), "99.5%")
    
    st.markdown("---")
    
    st.markdown(f"### {t('about')}")
    st.info(t('about_text'))
    
    st.markdown("---")
    
    st.markdown(f"### {t('how_to_use')}")
    st.markdown(t('how_to_steps'))
    
    st.markdown("---")
    
    st.markdown(f"### {t('project_info')}")
    st.markdown(t('project_details'))

# Main content area
if clf is not None and tfidf is not None:
    
    # Two columns - left for input, right for examples
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown(f"### {t('input_description')}")
        
        # Text input area
        user_input = st.text_area(
            "",
            height=200,
            placeholder=t('placeholder'),
            value=st.session_state.input_text,
            label_visibility="collapsed"
        )
        
        # Update state only if user actually changed something
        if user_input != st.session_state.input_text:
            st.session_state.input_text = user_input
        
        # Big shiny predict button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button(t('analyze_button'), type="primary", use_container_width=True)
        
        # Show results when button is clicked
        if predict_button:
            if user_input.strip():
                with st.spinner(t('analyzing')):
                    # Smart trick: use English version for examples, user text otherwise
                    if 'selected_example' in st.session_state and st.session_state.input_text == user_input:
                        # This is an example - use English for better accuracy
                        text_for_model = english_examples[st.session_state.selected_example]
                    else:
                        # User typed custom text - use it as is
                        text_for_model = user_input
                        # Clear example flag since user modified text
                        if 'selected_example' in st.session_state:
                            del st.session_state.selected_example
                    
                    # Prepare text and get prediction
                    cleaned_input = clean_text(text_for_model)
                    input_vec = tfidf.transform([cleaned_input])
                    
                    prediction = clf.predict(input_vec)[0]
                    probabilities = clf.predict_proba(input_vec)[0]
                    
                    class_index = list(clf.classes_).index(prediction)
                    confidence = probabilities[class_index] * 100
                
                st.markdown("---")
                
                # Main prediction card
                st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style="margin: 0;">{t('predicted_class')}</h2>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0;">{prediction}</h1>
                        <h3 style="margin: 0;">{t('confidence')}: {confidence:.2f}%</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Fancy gauge chart (because numbers are boring)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': t('confidence_score'), 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#667eea"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#ffcccb'},
                            {'range': [50, 75], 'color': '#ffffcc'},
                            {'range': [75, 100], 'color': '#90EE90'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90}}))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#344386", 'family': "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top 3 predictions (medals for everyone!)
                st.markdown(f"### {t('top_3')}")
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                
                for i, idx in enumerate(top_3_indices):
                    class_name = clf.classes_[idx]
                    prob = probabilities[idx] * 100
                    
                    col_rank, col_name, col_prob = st.columns([0.5, 2, 1])
                    
                    with col_rank:
                        if i == 0:
                            st.markdown("### 🥇")
                        elif i == 1:
                            st.markdown("### 🥈")
                        else:
                            st.markdown("### 🥉")
                    
                    with col_name:
                        st.markdown(f"**{class_name}**")
                    
                    with col_prob:
                        st.markdown(f"**{prob:.1f}%**")
                    
                    st.progress(prob / 100)
                    st.markdown("")
                
                # Show the words that made the model decide
                st.markdown(f"### {t('key_indicators')}")
                top_words = get_top_words(clf, tfidf, prediction, k=15)
                
                # Display as pretty badges
                words_html = " ".join([f'<span class="class-badge">#{word}</span>' for word in top_words])
                st.markdown(f'<div style="line-height: 2.5;">{words_html}</div>', unsafe_allow_html=True)
                
            else:
                st.warning(t('warning_empty'))
    
    with col2:
        st.markdown(f"### {t('quick_examples')}")
        st.markdown(t('click_to_try'))
        
        # Example buttons - click to auto-fill!
        examples = {
            t('diabetes'): ('diabetes', t('diabetes_text')),
            t('antibiotic'): ('antibiotic', t('antibiotic_text')),
            t('cardiac'): ('cardiac', t('cardiac_text')),
            t('neurological'): ('neurological', t('neurological_text')),
            t('respiratory'): ('respiratory', t('respiratory_text'))
        }
        
        for name, (example_key, text) in examples.items():
            if st.button(name, key=f"example_{name}", use_container_width=True):
                # Show text in current language but remember which example for model
                st.session_state.input_text = text
                st.session_state.selected_example = example_key
                st.rerun()
        
        st.markdown("---")
        
        st.markdown(f"### {t('available_classes_title')}")
        
        with st.expander(t('view_all_classes'), expanded=False):
            classes = sorted(clf.classes_)
            
            # Show classes in two columns (looks cleaner!)
            for i in range(0, len(classes), 2):
                if i + 1 < len(classes):
                    st.markdown(f"""
                        <div style="display: flex; gap: 0.5rem;">
                            <span class="class-badge" style="flex: 1;">✓ {classes[i]}</span>
                            <span class="class-badge" style="flex: 1;">✓ {classes[i+1]}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="class-badge">✓ {classes[i]}</span>', unsafe_allow_html=True)

else:
    # Uh oh, model not found - show setup instructions
    st.error(f"### {t('model_not_found')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {t('setup_instructions')}")
        st.markdown("""
        **Step 1: Export Model from Notebook**
        
        ```python
        import pickle
        
        # Save model
        with open('model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        
        # Save vectorizer
        with open('tfidf.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        ```
        """)
    
    with col2:
        st.markdown(f"### {t('file_structure')}")
        st.code("""
        NLP_prjct/
        ├── app.py
        ├── model.pkl
        └── tfidf.pkl
        """)
        
        st.markdown(f"### {t('run_command')}")
        st.code("python -m streamlit run app.py")

# Footer
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: black; padding: 2rem 0;">
        <p>{t('footer')}</p>
        <p style="font-size: 0.9rem;">{t('powered_by')}</p>
    </div>
""", unsafe_allow_html=True)