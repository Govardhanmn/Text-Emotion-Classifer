"""
app.py  --  EmotiSense: Emotion Detection UI
Streamlit app with ML (Linear SVM + TF-IDF) and DL (Bidirectional LSTM) models.
Launch: streamlit run app.py
"""

import os, re, json, pickle, warnings, base64
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -- Page config (must be first) ----------------------------------------------
st.set_page_config(
    page_title="EmotiSense AI - Intelligent Emotion Analysis",
    page_icon=":performing_arts:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Paths --------------------------------------------------------------------
BASE   = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE, "models", "ml")
DL_DIR = os.path.join(BASE, "models", "dl")

# -- Emotion config (all non-ASCII kept inside HTML strings only) -------------
EMOTION = {
    "anger"   : {"emoji": "\U0001F620", "color": "#FF4757", "bg": "rgba(255,71,87,0.12)",   "label": "Anger"},
    "fear"    : {"emoji": "\U0001F628", "color": "#9B59B6", "bg": "rgba(155,89,182,0.12)",  "label": "Fear"},
    "joy"     : {"emoji": "\U0001F604", "color": "#FFA502", "bg": "rgba(255,165,2,0.12)",   "label": "Joy"},
    "love"    : {"emoji": "\u2764\uFE0F", "color": "#FF6B9D", "bg": "rgba(255,107,157,0.12)", "label": "Love"},
    "sadness" : {"emoji": "\U0001F622", "color": "#4A90D9", "bg": "rgba(74,144,217,0.12)",  "label": "Sadness"},
    "surprise": {"emoji": "\U0001F632", "color": "#2ED573", "bg": "rgba(46,213,115,0.12)",  "label": "Surprise"},
}

EXAMPLES = {
    "Joy"     : "I just got promoted and I feel on top of the world right now!",
    "Sadness" : "I miss my grandmother so much, she was my whole world.",
    "Anger"   : "I cannot believe they lied to me after everything I did for them!",
    "Love"    : "Every moment spent with you makes my heart feel so full and warm.",
    "Fear"    : "I heard strange footsteps outside my door late at night and froze.",
    "Surprise": "I had absolutely no idea they were planning a party just for me!",
}

# -- CSS ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: #080c18 !important;
    color: #e2e8f8 !important;
}

/* Hide default Streamlit chrome completely */
#MainMenu, footer, header, [data-testid="stHeader"] { 
    display: none !important; 
}
[data-testid="stToolbar"] { display: none !important; }
[data-testid="block-container"] { padding-top: 0rem !important; margin-top: 0 !important; }

/* Hide empty markdown containers that hold styles */
div:has(> div[data-testid="stMarkdownContainer"] > style) {
    display: none !important;
}

/* Hero */
.hero {
    margin-top: -4.5rem !important;
    background: linear-gradient(135deg, #1a1f3a 0%, #0d1229 50%, #121830 100%);
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 16px;
    padding: 18px 24px 16px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(118,75,162,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 1.8rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #a78bfa, #667eea, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.1;
}
.hero-sub {
    color: #94a3b8; font-size: 0.9rem; margin-top: 6px;
    font-weight: 400; max-width: 560px;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1a1f3a, #131829);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 20px 24px;
    text-align: center; margin-bottom: 16px;
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-emoji { font-size: 3.5rem; line-height: 1; margin-bottom: 8px; }
.result-label { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.3px; margin-bottom: 4px; }
.result-conf  { font-size: 0.9rem; color: #94a3b8; font-weight: 500; }

/* Text area */
textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e2e8f8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
textarea:focus {
    border-color: rgba(102,126,234,0.6) !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
}

/* Buttons */
.stButton > button {
    width: 100%; padding: 12px 20px !important;
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important; font-weight: 700 !important;
    font-size: 0.9rem !important; border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(102,126,234,0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    color: #8899bb !important;
    box-shadow: none !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1121 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }

/* Divider */
hr { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 20px 0; }

/* Section headings */
h4 { color: #e2e8f8 !important; font-weight: 600 !important; margin-bottom: 12px !important; font-size: 1.05rem !important;}
</style>
""", unsafe_allow_html=True)


# -- NLTK setup (cached) ------------------------------------------------------
@st.cache_resource
def _setup_nltk():
    for pkg in ["stopwords", "wordnet", "punkt", "omw-1.4", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), PorterStemmer()

STOP_WORDS, STEMMER = _setup_nltk()


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(STEMMER.stem(w) for w in text.split() if w not in STOP_WORDS)


# -- Model loading (cached) ---------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_all():
    if not (os.path.exists(os.path.join(ML_DIR, "svm_model.pkl")) and
            os.path.exists(os.path.join(DL_DIR, "bilstm_model.h5"))):
        return None

    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    with open(os.path.join(ML_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(ML_DIR, "svm_model.pkl"), "rb") as f:
        svm = pickle.load(f)
    with open(os.path.join(DL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    dl_model = tf.keras.models.load_model(os.path.join(DL_DIR, "bilstm_model.h5"))

    with open(os.path.join(BASE, "models", "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(BASE, "models", "config.json")) as f:
        cfg = json.load(f)

    return dict(tfidf=tfidf, svm=svm, tokenizer=tokenizer,
                dl_model=dl_model, le=le, cfg=cfg, pad=pad_sequences)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_ml(text, models):
    vec    = models["tfidf"].transform([preprocess(text)])
    label  = models["svm"].predict(vec)[0]
    probs  = _softmax(models["svm"].decision_function(vec)[0])
    return models["le"].classes_[label], probs, models["le"].classes_


def predict_dl(text, models):
    seq   = models["tokenizer"].texts_to_sequences([preprocess(text)])
    pad   = models["pad"](seq, maxlen=models["cfg"]["max_len"],
                          padding="post", truncating="post")
    probs = models["dl_model"].predict(pad, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return models["le"].classes_[idx], probs, models["le"].classes_


# -- Chart --------------------------------------------------------------------
def prob_chart(probs, classes):
    order  = np.argsort(probs)
    colors = [EMOTION[c]["color"] for c in classes]
    labels = [f"{EMOTION[c]['emoji']}  {EMOTION[c]['label']}" for c in classes]

    fig = go.Figure(go.Bar(
        x=[float(probs[i]) for i in order],
        y=[labels[i]       for i in order],
        orientation="h",
        marker=dict(color=[colors[i] for i in order], opacity=0.85),
        text=[f"{probs[i]*100:.1f}%" for i in order],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=13),
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(255,255,255,0.02)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=0, r=60, t=8, b=8),
        height=240,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   tickformat=".0%", range=[0, 1.18],
                   showline=False, zeroline=False, color="#64748b"),
        yaxis=dict(showgrid=False, showline=False, color="#94a3b8", tickfont_size=12),
        bargap=0.35,
    )
    return fig


# =============================================================================
# APP STATE
# =============================================================================
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "ML"
if "res_data" not in st.session_state:
    st.session_state.res_data = None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 22px;">
      <div style="font-size:1.5rem;font-weight:800;
           background:linear-gradient(90deg,#a78bfa,#667eea);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        EmotiSense
      </div>
      <div style="font-size:0.8rem;color:#475569;margin-top:4px;">
        Emotion Detection Engine
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Select Model**")
    
    if st.button("⚡ ML Model (Fast)",
                 use_container_width=True,
                 type="primary" if st.session_state.model_choice == "ML" else "secondary"):
        st.session_state.model_choice = "ML"
        st.session_state.res_data = None # Clear result on switch
    if st.button("🧠 DL Model (Accurate)",
                 use_container_width=True,
                 type="primary" if st.session_state.model_choice == "DL" else "secondary"):
        st.session_state.model_choice = "DL"
        st.session_state.res_data = None # Clear result on switch

    st.markdown("---")
    st.markdown("**Detectable emotions**")
    for em, info in EMOTION.items():
        st.markdown(
            f'<span style="color:{info["color"]}">{info["emoji"]}</span>'
            f'&nbsp;<span style="color:#94a3b8;">{info["label"]}</span>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem;color:#334155;">'
        'Dataset: <a href="https://www.kaggle.com/datasets/praveengovi/'
        'emotions-dataset-for-nlp" style="color:#667eea;">Kaggle &rarr;</a>'
        '</div>',
        unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================
models = load_all()

# Models not ready
if models is None:
    st.error("Models not found. Run the training script first.")
    st.stop()

cfg = models["cfg"]
is_ml     = st.session_state.model_choice == "ML"
acc_val   = cfg.get("ml_accuracy", 0) if is_ml else cfg.get("dl_accuracy", 0)
f1_val    = cfg.get("ml_f1",       0) if is_ml else cfg.get("dl_f1",       0)
mdl_name  = "Linear SVM + TF-IDF" if is_ml else "Bidirectional LSTM"
mdl_icon  = "⚡" if is_ml else "🧠"

# Check for custom background image
bg_img_path = os.path.join(BASE, "header_bg.png")
if os.path.exists(bg_img_path):
    with open(bg_img_path, "rb") as f:
        bg_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .hero {{
        background: url('data:image/png;base64,{bg_b64}') !important;
        background-size: cover !important;
        background-position: right center !important;
        background-repeat: no-repeat !important;
        padding: 60px 32px 50px !important;
        min-height: 180px !important;
    }}
    .hero::before {{ display: none; /* Hide default gradient circle if image is used */ }}
    </style>
    """, unsafe_allow_html=True)

# Hero
st.markdown(f"""
<div class="hero">
  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
      <div>
          <div class="hero-title">Emotion Intelligence Dashboard</div>
          <div class="hero-sub">
            Decode human emotions from text instantly..
          </div>
      </div>
      <div style="text-align:right;">
          <div style="font-weight:600;color:#e2e8f8;font-size:0.9rem;">{mdl_icon} Active: {mdl_name}</div>
          <div style="font-size:0.8rem;color:#64748b;margin-top:2px;">
            Test accuracy: <b style="color:#a78bfa;">{acc_val:.1%}</b> &middot;
            F1-Macro: <b style="color:#a78bfa;">{f1_val:.4f}</b>
          </div>
      </div>
  </div>
</div>
""", unsafe_allow_html=True)


# -- Input & Top Results section ----------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("#### Enter text")

    # Example buttons
    ex_cols = st.columns(3)
    for i, (label, sentence) in enumerate(EXAMPLES.items()):
        em = label.lower()
        icon = EMOTION[em]["emoji"] if em in EMOTION else ""
        with ex_cols[i % 3]:
            if st.button(f"{icon} {label}", key=f"ex_{i}", use_container_width=True):
                st.session_state["user_text"] = sentence
                st.session_state.res_data = None # Clear old result

    user_text = st.text_area(
        label="Input",
        label_visibility="collapsed",
        placeholder="Type or paste any sentence here...",
        height=180,
        value=st.session_state.get("user_text", ""),
        key="text_area",
    )
    st.session_state["user_text"] = user_text

    analyse_btn = st.button("Detect Emotion", use_container_width=True)

# Process detection
if analyse_btn and user_text.strip():
    with st.spinner("Analysing..."):
        if is_ml:
            emotion, probs, classes = predict_ml(user_text, models)
        else:
            emotion, probs, classes = predict_dl(user_text, models)
        st.session_state.res_data = {"emotion": emotion, "probs": probs, "classes": classes}
elif analyse_btn and not user_text.strip():
    st.warning("Please enter some text before clicking Detect Emotion.")


with right:
    st.markdown("#### Detection Result")
    if st.session_state.res_data:
        data     = st.session_state.res_data
        emotion  = data["emotion"]
        probs    = data["probs"]
        classes  = data["classes"]
        
        conf     = float(probs[list(classes).index(emotion)])
        em_info  = EMOTION[emotion]

        # Top Result Card (Smaller)
        st.markdown(f"""
        <div class="result-card" style="border-color:{em_info['color']}44;
             background:linear-gradient(135deg,{em_info['bg']},rgba(13,18,41,0.8));">
          <div class="result-emoji" style="font-size: 3rem;">{em_info['emoji']}</div>
          <div class="result-label" style="font-size: 1.5rem; color:{em_info['color']};">{em_info['label']}</div>
          <div class="result-conf">
            Confidence: <b style="color:{em_info['color']};">{conf*100:.1f}%</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Plotly chart below it
        st.plotly_chart(prob_chart(probs, classes), use_container_width=True, config={"displayModeBar": False})
        
    else:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.1);
             border-radius: 12px; height: 260px; display: flex; align-items: center; justify-content: center;
             color: #64748b; font-size: 0.9rem; text-align: center; padding: 20px;">
          Type some text on the left and click "Detect Emotion" <br>to see the results here.
        </div>
        """, unsafe_allow_html=True)


# -- Down part remaining ------------------------------------------------------
if st.session_state.res_data:
    st.markdown("---")
    st.markdown("#### All emotion scores")
    
    data = st.session_state.res_data
    probs = data["probs"]
    classes = data["classes"]
    emotion = data["emotion"]

    em_cols = st.columns(len(classes))
    sorted_idx = np.argsort(probs)[::-1]

    for i, idx in enumerate(sorted_idx):
        em   = classes[idx]
        prob = probs[idx]
        info = EMOTION[em]
        is_top = (em == emotion)
        border = f"2px solid {info['color']}" if is_top else "1px solid rgba(255,255,255,0.07)"
        glow   = f"0 0 14px {info['color']}44" if is_top else "none"
        with em_cols[i]:
            st.markdown(f"""
            <div style="background:{info['bg']};border:{border};border-radius:14px;
                 padding:14px 8px;text-align:center;box-shadow:{glow};">
              <div style="font-size:1.6rem;">{info['emoji']}</div>
              <div style="font-size:0.7rem;color:{info['color']};font-weight:600;
                   text-transform:uppercase;letter-spacing:0.6px;margin:6px 0 4px;">
                {info['label']}
              </div>
              <div style="font-size:1.1rem;font-weight:800;color:{info['color']};">
                {prob*100:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)
