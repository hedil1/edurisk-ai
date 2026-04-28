import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import shap
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import tensorflow as tf
from tensorflow.keras.models import load_model

# ================= CONFIG =================
st.set_page_config(
    page_title="EduRisk AI — Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stSelectbox > label {
    color: #8b949e !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem !important;
    font-weight: 700;
    color: #58a6ff !important;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

hr { border-color: #21262d !important; }

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(31,111,235,0.4) !important;
}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 7px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important;
}

/* ---- CARDS ---- */
.risk-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.risk-badge-urgent {
    background: rgba(248,81,73,0.15); border: 1px solid rgba(248,81,73,0.4);
    color: #f85149; padding: 0.3rem 0.8rem; border-radius: 100px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; display: inline-block;
}
.risk-badge-high {
    background: rgba(210,153,34,0.15); border: 1px solid rgba(210,153,34,0.4);
    color: #d2991e; padding: 0.3rem 0.8rem; border-radius: 100px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; display: inline-block;
}
.risk-badge-low {
    background: rgba(63,185,80,0.15); border: 1px solid rgba(63,185,80,0.4);
    color: #3fb950; padding: 0.3rem 0.8rem; border-radius: 100px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; display: inline-block;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem;
    font-weight: 700; color: #e6edf3; margin-bottom: 0.25rem;
}
.section-sub { font-size: 0.8rem; color: #8b949e; margin-bottom: 1.25rem; }

.detail-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0; border-bottom: 1px solid #21262d; font-size: 0.87rem;
}
.detail-row:last-child { border-bottom: none; }
.detail-key { color: #8b949e; }
.detail-val { color: #e6edf3; font-weight: 500; }

.top-risk-row {
    display: flex; align-items: center; gap: 0.75rem;
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 0.75rem 1rem; margin-bottom: 0.5rem; transition: border-color 0.2s;
}
.top-risk-row:hover { border-color: #388bfd; }
.rank-num {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem;
    font-weight: 700; color: #58a6ff; width: 28px; text-align: center;
}
.student-name { font-weight: 600; flex: 1; font-size: 0.9rem; }
.prob-bar-wrap { width: 120px; background: #21262d; border-radius: 100px; height: 6px; }
.prob-bar-fill { height: 6px; border-radius: 100px; }
.prob-val { font-size: 0.82rem; color: #8b949e; min-width: 42px; text-align: right; }

/* ---- RECO CARDS ---- */
.reco-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 12px;
    padding: 1rem 1.25rem; margin-bottom: 0.6rem; display: flex;
    align-items: flex-start; gap: 0.85rem;
}
.reco-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 0.1rem; }
.reco-title { font-weight: 600; font-size: 0.9rem; color: #e6edf3; margin-bottom: 0.2rem; }
.reco-desc { font-size: 0.8rem; color: #8b949e; line-height: 1.5; }
.reco-priority-urgent {
    background: rgba(248,81,73,0.1); border-left: 3px solid #f85149;
}
.reco-priority-high {
    background: rgba(210,153,34,0.1); border-left: 3px solid #d2991e;
}
.reco-priority-low {
    background: rgba(63,185,80,0.1); border-left: 3px solid #3fb950;
}

/* ---- ACTION LOG ---- */
.action-entry {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 0.85rem 1.1rem; margin-bottom: 0.5rem; font-size: 0.85rem;
}
.action-meta { color: #8b949e; font-size: 0.75rem; margin-top: 0.3rem; }
.action-impact-positif { color: #3fb950; font-weight: 600; }
.action-impact-neutre { color: #8b949e; font-weight: 600; }
.action-impact-négatif { color: #f85149; font-weight: 600; }

/* ---- CONFIDENCE ---- */
.conf-bar {
    height: 10px; border-radius: 100px;
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
}
</style>
""", unsafe_allow_html=True)

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
HISTORY_FILE = os.path.join(BASE_DIR, "..", "action_history.json")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_excel(os.path.join(BASE_DIR, "..", "dataset_finale.xlsx"))
    df.columns = df.columns.str.strip()

    # 🔥 Conversion automatique des colonnes numériques
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    df = df.fillna(0)
    return df


df = load_data()

# 🔥 Nettoyage renforcé colonnes sensibles
for col in df.columns:
    if any(k in col.lower() for k in ["temps", "duree", "time", "connexion", "progression", "score", "interaction"]):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


# ================= LOAD MODELS =================
@st.cache_resource
def load_all_models():
    columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    rf      = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    xgb_m   = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    # Try .h5 first (TF 2.16 compatible), fallback to .keras
    _h5_path = os.path.join(MODEL_DIR, "ann_model.h5")
    _keras_path = os.path.join(MODEL_DIR, "ann_model.keras")
    _model_path = _h5_path if os.path.exists(_h5_path) else _keras_path
    ann = load_model(_model_path, compile=False)
    return columns, scaler, rf, xgb_m, ann

try:
    columns, scaler, rf, xgb_m, ann = load_all_models()
    models_loaded = True
except Exception as e:
    st.sidebar.error(f"⚠️ Models not found: {e}")
    models_loaded = False

# ================= PREPARE FEATURES =================
if models_loaded:
    X_df = df.reindex(columns=columns, fill_value=0)
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0)
    X_scaled = scaler.transform(X_df)

# ================= ACTION HISTORY =================
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_history(hist):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

# ================= HELPERS =================
MODEL_OPTIONS = {
    "🌲 Random Forest": "rf",
    "⚡ XGBoost": "xgb",
    "🧠 Neural Network (ANN)": "ann",
    "🔀 Ensemble (moyenne)": "ensemble",
}

def predict_proba_single(i, model_key="ensemble"):
    x = X_scaled[i].reshape(1, -1)
    if model_key == "rf":
        return rf.predict_proba(x)[0][1]
    elif model_key == "xgb":
        return xgb_m.predict_proba(x)[0][1]
    elif model_key == "ann":
        return float(ann.predict(x, verbose=0)[0][0])
    else:
        p1 = float(ann.predict(x, verbose=0)[0][0])
        p2 = rf.predict_proba(x)[0][1]
        p3 = xgb_m.predict_proba(x)[0][1]
        return (p1 + p2 + p3) / 3

def predict_confidence(i, model_key="ensemble"):
    """Returns (mean_proba, std_proba) as confidence measure across 3 models."""
    x = X_scaled[i].reshape(1, -1)
    p1 = float(ann.predict(x, verbose=0)[0][0])
    p2 = rf.predict_proba(x)[0][1]
    p3 = xgb_m.predict_proba(x)[0][1]
    probas = [p1, p2, p3]
    mean_p = np.mean(probas)
    std_p  = np.std(probas)
    # confidence = 1 - normalized std (how much models agree)
    confidence = max(0.0, 1.0 - (std_p * 3))
    return mean_p, std_p, confidence, probas

def risk_status(score_eng, proba):
    if score_eng < 0.4:
        return "URGENT", "🚨 URGENT"
    if proba > 0.6:
        return "HIGH", "🔥 HIGH RISK"
    return "LOW", "🟢 LOW RISK"

def badge_html(level):
    cls = {"URGENT": "urgent", "HIGH": "high", "LOW": "low"}.get(level, "low")
    labels = {"URGENT": "🚨 URGENT", "HIGH": "🔥 HIGH RISK", "LOW": "🟢 LOW RISK"}
    return f'<span class="risk-badge-{cls}">{labels[level]}</span>'

def bar_color(p):
    if p > 0.7: return "#f85149"
    if p > 0.5: return "#d2991e"
    return "#3fb950"

# ================= SHAP =================
@st.cache_data(show_spinner=False)
def compute_shap_rf(idx):
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_scaled[idx].reshape(1, -1))

    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]  # classe positive
    else:
        sv = shap_vals[0]

    sv = np.array(sv).flatten()  # 🔥 FIX CRITIQUE

    return sv, columns


def compute_shap_xgb(idx):
    explainer = shap.TreeExplainer(xgb_m)
    shap_vals = explainer.shap_values(X_scaled[idx].reshape(1, -1))

    if isinstance(shap_vals, list):
        sv = shap_vals[0]
    else:
        sv = shap_vals[0]

    sv = np.array(sv).flatten()  # 🔥 FIX CRITIQUE

    return sv, columns

# ================= RECOMMENDATIONS =================
def generate_recommendations(student, proba, level, shap_vals, col_names):
    recos = []
    score_eng   = float(student.get("score engagement (kpo)", 1))
    nb_co       = float(student.get("nb connexion", student.get("connexions", 10)))
    progression = float(student.get("progression", student.get("taux progression", 50)))
    interactions = float(student.get("nb interactions", student.get("interactions", 5)))

    # Identify top negative SHAP features (pushing toward dropout)
    shap_neg = sorted(zip(shap_vals, col_names), key=lambda x: x[0])
    top_bad   = [c for v, c in shap_neg[:3] if v < 0]

    if level == "URGENT":
        recos.append({
            "icon": "🚨", "priority": "urgent",
            "title": "Appel téléphonique immédiat",
            "desc": "Contacter l'élève dans les 24h. Score d'engagement critique (<0.4). Identifier les obstacles à la connexion et proposer un plan de rattrapage personnalisé."
        })
        recos.append({
            "icon": "👨‍🏫", "priority": "urgent",
            "title": "Assignation d'un tuteur dédié",
            "desc": "Assigner un tuteur pédagogique pour un suivi hebdomadaire. Prévoir 2 sessions de 30 min la première semaine."
        })

    if nb_co < 5:
        recos.append({
            "icon": "📲", "priority": "high",
            "title": "Campagne de ré-engagement",
            "desc": f"L'élève ne s'est connecté que {int(nb_co)} fois. Envoyer une séquence d'emails de relance + notification push avec contenu exclusif pour recréer l'habitude de connexion."
        })

    if progression < 30:
        recos.append({
            "icon": "🎯", "priority": "high",
            "title": "Module de démarrage simplifié",
            "desc": f"Progression à {progression:.0f}%. Proposer un parcours accéléré avec les 3 modules essentiels. Débloquer des badges de progression pour stimuler la motivation."
        })

    if interactions < 3:
        recos.append({
            "icon": "💬", "priority": "high",
            "title": "Invitation session live Q&A",
            "desc": "Très peu d'interactions détectées. Inviter l'élève à la prochaine session live gratuite. L'interaction sociale est un facteur clé de rétention."
        })

    if proba > 0.5 and level != "URGENT":
        recos.append({
            "icon": "🎁", "priority": "high",
            "title": "Offre de module bonus",
            "desc": "Débloquer un module bonus gratuit comme incentive. Peut inclure un certificat intermédiaire ou un accès à du contenu premium pour 30 jours."
        })

    if any("connexion" in c.lower() or "login" in c.lower() for c in top_bad):
        recos.append({
            "icon": "⏰", "priority": "high",
            "title": "Rappel de planning personnalisé",
            "desc": "La fréquence de connexion est la cause principale identifiée. Envoyer un planning hebdomadaire adapté aux horaires connus de l'élève."
        })

    if level == "LOW":
        recos.append({
            "icon": "⭐", "priority": "low",
            "title": "Programme ambassadeur",
            "desc": "Profil stable et engagé. Inviter l'élève à rejoindre le programme ambassadeur : parrainage d'autres étudiants en échange de réduction sur le prochain module."
        })
        recos.append({
            "icon": "📈", "priority": "low",
            "title": "Suivi standard — rapport mensuel",
            "desc": "Continuer le suivi standard. Envoyer un rapport de progression mensuel automatisé pour maintenir l'engagement."
        })

    return recos[:6]  # max 6 recommendations

# ================= KPI RADAR DATA =================
# ================= KPI RADAR DATA =================
def get_radar_kpis(student, df):
    """Extract 6 key KPIs normalized 0-100 for radar chart."""

    def pct_rank(val, series):
        series = pd.to_numeric(series, errors='coerce')  # 🔥 FIX
        series = series.dropna()

        if len(series) == 0:
            return 0.0

        return float((series <= val).mean() * 100)

    def safe_val(x):
        try:
            return float(pd.to_numeric(x, errors='coerce'))
        except:
            return 0.0

    kpis = {}

    eng_col = "score engagement (kpo)"
    co_col  = next((c for c in df.columns if "connexion" in c.lower()), None)
    pro_col = next((c for c in df.columns if "progression" in c.lower() or "taux" in c.lower()), None)
    int_col = next((c for c in df.columns if "interaction" in c.lower()), None)
    quiz_col = next((c for c in df.columns if ("quiz" in c.lower() or "score" in c.lower()) and c != eng_col), None)
    time_col = next((c for c in df.columns if "temps" in c.lower() or "duree" in c.lower() or "time" in c.lower()), None)

    if eng_col in df.columns:
        kpis["Engagement"] = pct_rank(safe_val(student.get(eng_col, 0)), df[eng_col])

    if co_col:
        kpis["Connexions"] = pct_rank(safe_val(student.get(co_col, 0)), df[co_col])

    if pro_col:
        kpis["Progression"] = pct_rank(safe_val(student.get(pro_col, 0)), df[pro_col])

    if int_col:
        kpis["Interactions"] = pct_rank(safe_val(student.get(int_col, 0)), df[int_col])

    if quiz_col:
        kpis["Score Quiz"] = pct_rank(safe_val(student.get(quiz_col, 0)), df[quiz_col])

    if time_col:
        kpis["Temps Étude"] = pct_rank(safe_val(student.get(time_col, 0)), df[time_col])

    # fallback si peu de KPI
    if len(kpis) < 4:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            if c not in kpis and len(kpis) < 6:
                kpis[c[:15]] = pct_rank(safe_val(student.get(c, 0)), df[c])

    return kpis

# ================= TIMELINE =================
def build_timeline(student, df, idx):
    """
    Build synthetic engagement timeline from available numeric cols.
    If real time-series cols exist (e.g. mois1, mois2...) use them,
    otherwise simulate a plausible trend from aggregate stats.
    """
    # Try to detect month-based columns
    month_cols = sorted([c for c in df.columns if any(f"mois{i}" in c.lower() or f"month{i}" in c.lower() or f"m{i}_" in c.lower() for i in range(1, 13))])

    if month_cols:
        vals = [float(student.get(c, 0)) for c in month_cols]
        labels = [f"M{i+1}" for i in range(len(vals))]
        return labels, vals

    # Synthetic: simulate 6-month trend based on engagement score
    eng = float(student.get("score engagement (kpo)", 0.5))
    co  = float(student.get(next((c for c in df.columns if "connexion" in c.lower()), df.columns[1]), 5))
    pro = float(student.get(next((c for c in df.columns if "progression" in c.lower()), df.columns[2]), 30))

    # Percentile of this student globally
    pct = float(np.mean([
        (df.select_dtypes(include=[np.number]).iloc[:, 0] <= df.select_dtypes(include=[np.number]).iloc[idx, 0]).mean()
    ]))

    # Simulate 6 months: starts mid, trends toward current engagement
    trend = []
    base  = min(0.8, eng + 0.3)
    for m in range(6):
        noise  = np.random.normal(0, 0.04)
        decay  = (eng - base) * (m / 5)
        trend.append(round(max(0, min(1, base + decay + noise)), 3))

    return [f"M{i+1}" for i in range(6)], trend


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem 0'>
        <div style='font-family: Space Grotesk, sans-serif; font-size: 1.4rem; font-weight: 700; color: #e6edf3; line-height: 1.2'>
            EduRisk <span style='color:#58a6ff'>AI</span>
        </div>
        <div style='font-size: 0.75rem; color: #8b949e; margin-top: 0.2rem'>
            Dropout Early-Warning System v2
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**ÉLÈVE**")
    name_list = df["candidat"].astype(str).tolist() if "candidat" in df.columns else [str(i) for i in df.index]
    student_name = st.selectbox("Sélectionner un élève", name_list, label_visibility="collapsed")

    if "candidat" in df.columns:
        idx = df[df["candidat"].astype(str) == student_name].index[0]
    else:
        idx = int(student_name)

    iloc_idx = df.index.get_loc(idx)

    st.markdown("---")

    st.markdown("**MODÈLE**")
    model_label = st.radio("Choisir le modèle", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")
    selected_model = MODEL_OPTIONS[model_label]

    st.markdown("---")
    run_btn = st.button("🚀 Analyser cet élève", use_container_width=True)

    st.markdown("---")
    st.markdown("**📝 NOUVELLE ACTION**")
    action_type = st.selectbox("Type d'intervention", [
        "📞 Appel téléphonique", "📧 Email de relance", "💬 Session live",
        "🎁 Module bonus envoyé", "👨‍🏫 Session tutorat", "📋 Autre"
    ], label_visibility="collapsed")
    action_note = st.text_area("Note / Résultat", placeholder="Ex: Élève a répondu positivement, prévu session vendredi...", height=80, label_visibility="collapsed")
    action_impact = st.selectbox("Impact observé", ["— Non évalué —", "Positif ✅", "Neutre ➖", "Négatif ❌"], label_visibility="collapsed")

    if st.button("💾 Enregistrer l'action", use_container_width=True):
        if action_note.strip():
            hist = load_history()
            key  = str(student_name)
            if key not in hist:
                hist[key] = []
            hist[key].append({
                "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "type": action_type,
                "note": action_note.strip(),
                "impact": action_impact
            })
            save_history(hist)
            st.success("✅ Action enregistrée !")
        else:
            st.warning("Ajoutez une note avant d'enregistrer.")

# ================= MAIN =================
student = df.iloc[iloc_idx]

# ---- Header ----
st.markdown(f"""
<div style='margin-bottom: 1.5rem'>
    <div style='font-family: Space Grotesk, sans-serif; font-size: 1.9rem; font-weight: 700; color: #e6edf3'>
        🎓 Tableau de bord
    </div>
    <div style='font-size: 0.85rem; color: #8b949e; margin-top: 0.2rem'>
        Plateforme de prédiction d'abandon — Tunisie EdTech
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Global metrics ----
if models_loaded:
    with st.spinner("Calcul des métriques globales..."):
        all_probas = np.array([predict_proba_single(i, selected_model) for i in range(len(df))])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👥 Total Élèves", len(df))
    m2.metric("🚨 Risque Urgent", int((df.get("score engagement (kpo)", pd.Series([1]*len(df))) < 0.4).sum()))
    m3.metric("🔥 Risque Élevé", int((all_probas > 0.6).sum()))
    m4.metric("📊 Risque Moyen", f"{all_probas.mean()*100:.1f}%")
else:
    st.info("⚠️ Entraînez d'abord les modèles avec train.py")

st.markdown("---")

# ================================================================
# MAIN TWO-COLUMN LAYOUT
# ================================================================
col_left, col_right = st.columns([1.05, 1], gap="large")

# ==================== LEFT COLUMN ====================
with col_left:
    st.markdown('<div class="section-title">👤 Profil de l\'élève</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Informations détaillées — {student_name}</div>', unsafe_allow_html=True)

    score_eng = float(student.get("score engagement (kpo)", 0))
    k1, k2, k3 = st.columns(3)
    k1.metric("Score Engagement", f"{score_eng:.2f}")
    k2.metric("Connexions", int(student.get("nb connexion", student.get("connexions", 0))))
    k3.metric("Progression", f"{float(student.get('progression', student.get('taux progression', 0))):.0f}%")

    # Tabs: Details | Radar | Timeline
    tab_details, tab_radar, tab_timeline = st.tabs(["📋 Détails", "🕸️ Radar KPIs", "📈 Timeline"])

    with tab_details:
        st.markdown('<div class="risk-card">', unsafe_allow_html=True)
        skip_cols = {"candidat"}
        rows_html = ""
        for col_name in df.columns:
            if col_name in skip_cols:
                continue
            val = student[col_name]
            if isinstance(val, float):
                val_str = f"{val:.3f}" if val != int(val) else str(int(val))
            else:
                val_str = str(val)
            rows_html += f"""
            <div class="detail-row">
                <span class="detail-key">{col_name}</span>
                <span class="detail-val">{val_str}</span>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_radar:
        kpis = get_radar_kpis(student, df)
        if kpis:
            labels = list(kpis.keys())
            values = list(kpis.values())
            labels_closed = labels + [labels[0]]
            values_closed = values + [values[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill='toself',
                fillcolor='rgba(31,111,235,0.15)',
                line=dict(color='#388bfd', width=2),
                name=student_name,
                hovertemplate='%{theta}: %{r:.1f}e percentile<extra></extra>'
            ))
            # Average line
            avg_vals = [50] * len(labels)
            avg_closed = avg_vals + [avg_vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_closed,
                theta=labels_closed,
                line=dict(color='#484f58', width=1.5, dash='dot'),
                name='Moyenne cohorte',
                hoverinfo='skip'
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='#161b22',
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color='#8b949e', size=9), gridcolor='#21262d', linecolor='#21262d'),
                    angularaxis=dict(tickfont=dict(color='#c9d1d9', size=10), gridcolor='#21262d', linecolor='#21262d'),
                ),
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                showlegend=True,
                legend=dict(font=dict(color='#8b949e', size=10), bgcolor='#0d1117'),
                margin=dict(l=40, r=40, t=30, b=30),
                height=340,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Percentile de l'élève par rapport à la cohorte (100 = meilleur)")
        else:
            st.info("Données insuffisantes pour le radar.")

    with tab_timeline:
        t_labels, t_vals = build_timeline(student, df, iloc_idx)

        colors = ['#f85149' if v < 0.4 else '#d2991e' if v < 0.65 else '#3fb950' for v in t_vals]
        fig_tl = go.Figure()
        fig_tl.add_shape(type="rect", x0=-0.5, x1=len(t_labels)-0.5, y0=0, y1=0.4,
                         fillcolor="rgba(248,81,73,0.07)", line_width=0)
        fig_tl.add_shape(type="rect", x0=-0.5, x1=len(t_labels)-0.5, y0=0.4, y1=0.65,
                         fillcolor="rgba(210,153,34,0.07)", line_width=0)
        fig_tl.add_trace(go.Scatter(
            x=t_labels, y=t_vals,
            mode='lines+markers',
            line=dict(color='#388bfd', width=2.5),
            marker=dict(color=colors, size=9, line=dict(width=2, color='#0d1117')),
            fill='tozeroy',
            fillcolor='rgba(31,111,235,0.08)',
            hovertemplate='%{x}: %{y:.3f}<extra></extra>'
        ))
        fig_tl.add_hline(y=0.4, line_dash="dash", line_color="#f85149", line_width=1,
                         annotation_text="Seuil urgent", annotation_font_color="#f85149", annotation_font_size=10)
        fig_tl.add_hline(y=0.65, line_dash="dash", line_color="#d2991e", line_width=1,
                         annotation_text="Seuil risque", annotation_font_color="#d2991e", annotation_font_size=10)
        fig_tl.update_layout(
            paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            xaxis=dict(gridcolor='#21262d', tickfont=dict(color='#8b949e')),
            yaxis=dict(gridcolor='#21262d', tickfont=dict(color='#8b949e'), range=[0, 1.05], title="Score engagement"),
            margin=dict(l=10, r=10, t=20, b=20),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_tl, use_container_width=True)
        st.caption("Évolution du score d'engagement dans le temps (zones : rouge=urgent, orange=risque)")


# ==================== RIGHT COLUMN ====================
with col_right:

    # ---- Prediction result ----
    st.markdown('<div class="section-title">🤖 Résultat de prédiction</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Modèle actif : {model_label}</div>', unsafe_allow_html=True)

    if models_loaded and run_btn:
        # Prediction + confidence
        mean_p, std_p, confidence, model_probas = predict_confidence(iloc_idx, selected_model)
        proba = predict_proba_single(iloc_idx, selected_model)
        level, label = risk_status(score_eng, proba)
        pct = int(proba * 100)
        bar_clr = bar_color(proba)

        # Store for use below
        st.session_state["last_proba"]  = proba
        st.session_state["last_level"]  = level
        st.session_state["last_conf"]   = confidence
        st.session_state["last_std"]    = std_p
        st.session_state["last_m_probas"] = model_probas

        # Risk card
        st.markdown(f"""
        <div class="risk-card" style="border-color: {bar_clr}33">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem">
                <div>
                    <div style="font-size:0.78rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.06em; font-weight:600; margin-bottom:0.3rem">Résultat</div>
                    {badge_html(level)}
                </div>
                <div style="font-family: Space Grotesk, sans-serif; font-size: 3rem; font-weight:700; color:{bar_clr}; line-height:1">
                    {pct}<span style="font-size:1.4rem">%</span>
                </div>
            </div>
            <div style="background:#21262d; border-radius:100px; height:8px; margin-bottom:1rem">
                <div style="width:{pct}%; background:{bar_clr}; height:8px; border-radius:100px"></div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; font-size:0.83rem">
                <div>
                    <span style="color:#8b949e">Score engagement</span><br>
                    <span style="font-weight:600; color:#e6edf3">{score_eng:.3f}</span>
                </div>
                <div>
                    <span style="color:#8b949e">Probabilité dropout</span><br>
                    <span style="font-weight:600; color:{bar_clr}">{proba:.3f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- SCORE DE CONFIANCE ----
        conf_pct = int(confidence * 100)
        conf_clr = "#3fb950" if confidence > 0.7 else "#d2991e" if confidence > 0.4 else "#f85149"
        conf_label = "Élevée" if confidence > 0.7 else "Modérée" if confidence > 0.4 else "Faible"

        st.markdown(f"""
        <div class="risk-card" style="padding:1rem 1.25rem; margin-top:-0.5rem">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem">
                <div style="font-size:0.78rem; color:#8b949e; font-weight:600; text-transform:uppercase; letter-spacing:0.06em">
                    🎯 Score de confiance
                </div>
                <div style="font-size:0.88rem; font-weight:700; color:{conf_clr}">{conf_label} — {conf_pct}%</div>
            </div>
            <div style="background:#21262d; border-radius:100px; height:6px; margin-bottom:0.75rem">
                <div class="conf-bar" style="width:{conf_pct}%; background:{conf_clr}; height:6px; border-radius:100px"></div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:0.5rem; font-size:0.78rem; text-align:center">
                <div style="background:#21262d; border-radius:8px; padding:0.4rem">
                    <div style="color:#8b949e">🌲 RF</div>
                    <div style="font-weight:700; color:#e6edf3">{model_probas[1]:.2f}</div>
                </div>
                <div style="background:#21262d; border-radius:8px; padding:0.4rem">
                    <div style="color:#8b949e">⚡ XGB</div>
                    <div style="font-weight:700; color:#e6edf3">{model_probas[2]:.2f}</div>
                </div>
                <div style="background:#21262d; border-radius:8px; padding:0.4rem">
                    <div style="color:#8b949e">🧠 ANN</div>
                    <div style="font-weight:700; color:#e6edf3">{model_probas[0]:.2f}</div>
                </div>
            </div>
            <div style="font-size:0.72rem; color:#484f58; margin-top:0.5rem; text-align:center">
                Écart-type entre modèles : {std_p:.3f} — {"✅ Consensus fort" if std_p < 0.1 else "⚠️ Divergence modérée" if std_p < 0.2 else "❗ Forte divergence"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if level == "URGENT":
            st.error("⚠️ Action immédiate requise — contacter le tuteur pédagogique.")
        elif level == "HIGH":
            st.warning("📬 Relance recommandée — planifier un entretien de suivi.")
        else:
            st.success("✅ Profil stable — continuer le suivi standard.")

    elif not run_btn:
        # Show last result if available
        if "last_proba" in st.session_state:
            proba  = st.session_state["last_proba"]
            level  = st.session_state["last_level"]
        else:
            st.markdown("""
            <div class="risk-card" style="text-align:center; padding: 2.5rem 1.5rem">
                <div style="font-size:2.5rem; margin-bottom:0.75rem">🚀</div>
                <div style="color:#8b949e; font-size:0.9rem">Cliquez sur <strong style="color:#58a6ff">Analyser cet élève</strong><br>dans la barre latérale pour lancer la prédiction.</div>
            </div>
            """, unsafe_allow_html=True)
            proba, level = None, None

    # ---- Tabs: SHAP | Recommendations | Top10 | History ----
    st.markdown("---")
    tab_shap, tab_reco, tab_top10, tab_hist = st.tabs([
        "🔍 SHAP Explication", "💡 Recommandations", "🔥 Top 10 Risque", "📋 Historique"
    ])

    # ---- SHAP ----
    with tab_shap:
        if models_loaded and (run_btn or "last_proba" in st.session_state):
            with st.spinner("Calcul SHAP..."):
                shap_key = selected_model if selected_model in ("rf", "xgb") else "rf"
                if shap_key == "rf":
                    sv, feat_names = compute_shap_rf(iloc_idx)
                else:
                    sv, feat_names = compute_shap_xgb(iloc_idx)

            # Top 10 features by absolute SHAP
            pairs = sorted(
    [(float(v), n) for v, n in zip(sv, feat_names)],
    key=lambda x: abs(x[0]),
    reverse=True
)[:12]
            shap_vals_plot = [v for v, _ in pairs]
            feat_labels    = [n[:20] for _, n in pairs]
            colors_shap    = ["#f85149" if v > 0 else "#3fb950" for v in shap_vals_plot]

            fig_shap = go.Figure(go.Bar(
                x=shap_vals_plot,
                y=feat_labels,
                orientation='h',
                marker_color=colors_shap,
                hovertemplate='%{y}: %{x:.4f}<extra></extra>'
            ))
            fig_shap.update_layout(
                paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
                xaxis=dict(gridcolor='#21262d', tickfont=dict(color='#8b949e'), title="Contribution SHAP"),
                yaxis=dict(tickfont=dict(color='#c9d1d9'), autorange='reversed'),
                margin=dict(l=10, r=10, t=30, b=10),
                height=360,
                title=dict(text=f"Waterfall SHAP — {shap_key.upper()}", font=dict(color='#8b949e', size=11), x=0)
            )
            fig_shap.add_vline(x=0, line_color="#30363d", line_width=1.5)
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("🔴 Rouge = pousse vers le dropout · 🟢 Vert = facteur protecteur")

            # Top 3 explanation text
            top_risk_feats   = [(n, v) for v, n in pairs if v > 0][:3]
            top_protect_feats = [(n, v) for v, n in pairs if v < 0][:2]
            if top_risk_feats:
                st.markdown("**Facteurs de risque principaux :**")
                for name, val in top_risk_feats:
                    st.markdown(f"- `{name}` contribue **+{val:.4f}** vers le dropout")
            if top_protect_feats:
                st.markdown("**Facteurs protecteurs :**")
                for name, val in top_protect_feats:
                    st.markdown(f"- `{name}` contribue **{val:.4f}** contre le dropout")
        else:
            st.info("Lancez l'analyse pour voir l'explication SHAP.")

    # ---- Recommendations ----
    with tab_reco:
        if models_loaded and (run_btn or "last_proba" in st.session_state):
            cur_proba = st.session_state.get("last_proba", 0.5)
            cur_level = st.session_state.get("last_level", "LOW")

            # Need SHAP for recommendations
            try:
                sv_reco, feat_names_reco = compute_shap_rf(iloc_idx)
            except:
                sv_reco = np.zeros(len(columns))
                feat_names_reco = columns

            recos = generate_recommendations(student, cur_proba, cur_level, sv_reco, feat_names_reco)

            st.markdown(f"<div class='section-sub'>{len(recos)} actions recommandées pour {student_name}</div>", unsafe_allow_html=True)

            for r in recos:
                priority_class = f"reco-priority-{r['priority']}"
                st.markdown(f"""
                <div class="reco-card {priority_class}">
                    <div class="reco-icon">{r['icon']}</div>
                    <div>
                        <div class="reco-title">{r['title']}</div>
                        <div class="reco-desc">{r['desc']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Lancez l'analyse pour voir les recommandations.")

    # ---- Top 10 ----
    with tab_top10:
        st.markdown(f'<div class="section-sub">Classés par probabilité d\'abandon — {model_label}</div>', unsafe_allow_html=True)
        if models_loaded:
            top_indices = np.argsort(all_probas)[::-1][:10]
            for rank, i in enumerate(top_indices, 1):
                name = df.iloc[i].get("candidat", f"Élève {i}") if "candidat" in df.columns else f"Élève {i}"
                p = all_probas[i]
                pct_w = int(p * 100)
                b_clr = bar_color(p)
                lv, _ = risk_status(float(df.iloc[i].get("score engagement (kpo)", 1)), p)
                bdg = badge_html(lv)

                st.markdown(f"""
                <div class="top-risk-row">
                    <div class="rank-num">#{rank}</div>
                    <div class="student-name">{name}</div>
                    {bdg}
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill" style="width:{pct_w}%; background:{b_clr}"></div>
                    </div>
                    <div class="prob-val">{p:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Modèles non chargés.")

    # ---- Historique ----
    with tab_hist:
        hist = load_history()
        key  = str(student_name)
        entries = hist.get(key, [])

        if entries:
            st.markdown(f"<div class='section-sub'>{len(entries)} intervention(s) enregistrée(s)</div>", unsafe_allow_html=True)

            # Impact summary
            impacts = [e["impact"] for e in entries]
            pos = sum(1 for i in impacts if "Positif" in i)
            neu = sum(1 for i in impacts if "Neutre" in i)
            neg = sum(1 for i in impacts if "Négatif" in i)

            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("✅ Positif", pos)
            hc2.metric("➖ Neutre", neu)
            hc3.metric("❌ Négatif", neg)

            st.markdown("---")

            for entry in reversed(entries):
                impact_class = "neutre"
                if "Positif" in entry["impact"]:
                    impact_class = "positif"
                elif "Négatif" in entry["impact"]:
                    impact_class = "négatif"

                st.markdown(f"""
                <div class="action-entry">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start">
                        <div style="font-weight:600; font-size:0.88rem">{entry['type']}</div>
                        <div class="action-impact-{impact_class}" style="font-size:0.78rem">{entry['impact']}</div>
                    </div>
                    <div style="color:#c9d1d9; margin-top:0.35rem; font-size:0.85rem; line-height:1.5">{entry['note']}</div>
                    <div class="action-meta">🕐 {entry['date']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Clear history
            if st.button("🗑️ Effacer l'historique de cet élève"):
                hist.pop(key, None)
                save_history(hist)
                st.rerun()
        else:
            st.markdown("""
            <div style="text-align:center; padding:2rem; color:#484f58">
                <div style="font-size:2rem; margin-bottom:0.5rem">📋</div>
                Aucune intervention enregistrée pour cet élève.<br>
                <span style="font-size:0.82rem">Utilisez la barre latérale pour ajouter une action.</span>
            </div>
            """, unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#484f58; font-size:0.75rem; padding: 0.5rem 0 1rem 0'>
    EduRisk AI v2 · SHAP · Confiance · Timeline · Recommandations · Historique · CRISP-DM · Tunisie EdTech
</div>
""", unsafe_allow_html=True)