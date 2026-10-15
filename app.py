from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from sklearn.datasets import load_wine

# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "wine_classifier.pth"
SCALER_PATH = BASE_DIR / "wine_scaler.joblib"

INPUT_FEATURES = 13
HIDDEN_UNITS = 64
NUM_CLASSES = 3

CLASS_NAMES = ["Barbera", "Grignolino", "Barolo"]
SAMPLES_PER_CLASS = {0: 4, 1: 3, 2: 3}

torch.set_num_threads(1)

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="VinoAI — Enterprise Wine Intelligence",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Premium High-Legibility Typography & Styling
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #F8F6F2 0%, #EFECE6 100%);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #12080C !important;
        border-right: 1px solid #28141B;
    }

    [data-testid="stSidebar"] * {
        color: #EFE6E9 !important;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #46121D 0%, #25060D 100%);
        padding: 36px 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 30px -8px rgba(70, 18, 29, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
        color: #FFFFFF;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #ECC7CE;
        margin-top: 8px;
        font-weight: 400;
        max-width: 650px;
        line-height: 1.5;
    }

    /* High Visibility Form Labels & Input Headers */
    .stForm label, .stSelectbox label {
        font-weight: 600 !important;
        color: #1F1216 !important;
        font-size: 0.95rem !important;
    }

    /* Prediction Result Badge */
    .prediction-badge {
        background: linear-gradient(135deg, #571624 0%, #360D16 100%);
        color: white;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(87, 22, 36, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .prediction-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #ECC7CE;
        font-weight: 700;
    }

    .prediction-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin-top: 8px;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }

    /* Primary Button Styling */
    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #571624 0%, #46121D 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 16px rgba(87, 22, 36, 0.3) !important;
    }

    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(87, 22, 36, 0.45) !important;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(46, 204, 113, 0.15);
        border: 1px solid rgba(46, 204, 113, 0.3);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #2ECC71;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PyTorch Model
# =========================================================

class WineClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_FEATURES, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, NUM_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


# =========================================================
# Artifact Loaders
# =========================================================

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        raise FileNotFoundError("Model or scaler file missing in directory.")

    scaler = joblib.load(SCALER_PATH)
    model = WineClassifier()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler


@st.cache_data
def get_dataset():
    dataset = load_wine()
    X, y = dataset.data, dataset.target
    feature_names = list(dataset.feature_names)
    target_names = list(dataset.target_names)

    selected_indices = []
    for class_index, sample_count in SAMPLES_PER_CLASS.items():
        class_indices = np.flatnonzero(y == class_index)
        selected_indices.extend(class_indices[:sample_count].tolist())

    return X, y, feature_names, target_names, selected_indices


try:
    model, scaler = load_artifacts()
    X, y, feature_names, original_class_names, selected_indices = get_dataset()
except Exception as exc:
    st.error(f"Initialization error: {exc}")
    st.stop()


# =========================================================
# Sidebar with Logo & System Status Only
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 14px; padding: 12px 0 20px 0;">
            <svg width="42" height="42" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="40" height="40" rx="12" fill="#571624"/>
                <path d="M20 10V26M20 26C15.5817 26 12 22.4183 12 18V12H28V18C28 22.4183 24.4183 26 20 26ZM15 30H25" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
            <div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #FFF; line-height: 1.1;">VinoAI</div>
                <div style="font-size: 0.75rem; color: #ECC7CE; letter-spacing: 0.05em;">ANALYTICS ENGINE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="status-pill">
            <span style="width: 8px; height: 8px; background-color: #2ECC71; border-radius: 50%; display: inline-block;"></span>
            System Online & Ready
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Hero Banner
# =========================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">Wine Cultivar Analytics Suite</div>
        <div class="hero-subtitle">Perform high-precision machine learning classification based on advanced chemical measurements and spectrometer attributes.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Sample Selector Panel
# =========================================================

def sample_label(idx):
    return f"Sample #{idx + 1:02d}"

selected_sample = st.selectbox(
    "Select Reference Sample Profile",
    options=list(range(len(selected_indices))),
    format_func=sample_label,
    index=0,
)

dataset_index = selected_indices[selected_sample]
sample_values = X[dataset_index]
reference_class_index = int(y[dataset_index])
reference_class_name = CLASS_NAMES[reference_class_index]

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# Feature Input Form with Categorized Tabs
# =========================================================

with st.form("wine_prediction_form"):
    st.markdown("### Chemical Composition Parameters")

    tab1, tab2, tab3 = st.tabs([
        "Alcohol & Phenolic Compounds",
        "Color Metrics & Spectrometry",
        "Minerals & Acidity Profile",
    ])

    input_values = [0.0] * INPUT_FEATURES

    with tab1:
        c1, c2, c3 = st.columns(3)
        input_values[0] = c1.number_input("Alcohol", value=float(sample_values[0]), format="%.4f")
        input_values[5] = c2.number_input("Total Phenols", value=float(sample_values[5]), format="%.4f")
        input_values[6] = c3.number_input("Flavanoids", value=float(sample_values[6]), format="%.4f")

        c4, c5 = st.columns(2)
        input_values[7] = c4.number_input("Nonflavanoid Phenols", value=float(sample_values[7]), format="%.4f")
        input_values[8] = c5.number_input("Proanthocyanins", value=float(sample_values[8]), format="%.4f")

    with tab2:
        c1, c2, c3 = st.columns(3)
        input_values[9] = c1.number_input("Color Intensity", value=float(sample_values[9]), format="%.4f")
        input_values[10] = c2.number_input("Hue", value=float(sample_values[10]), format="%.4f")
        input_values[11] = c3.number_input("OD280/OD315 Diluted Wines", value=float(sample_values[11]), format="%.4f")

    with tab3:
        c1, c2, c3 = st.columns(3)
        input_values[1] = c1.number_input("Malic Acid", value=float(sample_values[1]), format="%.4f")
        input_values[2] = c2.number_input("Ash", value=float(sample_values[2]), format="%.4f")
        input_values[3] = c3.number_input("Alcalinity of Ash", value=float(sample_values[3]), format="%.4f")

        c4, c5 = st.columns(2)
        input_values[4] = c4.number_input("Magnesium", value=float(sample_values[4]), format="%.4f")
        input_values[12] = c5.number_input("Proline", value=float(sample_values[12]), format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "Execute Cultivar Classification Analysis",
        use_container_width=True,
    )


# =========================================================
# Prediction & Visual Analytics Dashboard
# =========================================================

if submitted:
    try:
        input_array = np.asarray(input_values, dtype=np.float64).reshape(1, INPUT_FEATURES)
        
        if not np.isfinite(input_array).all():
            st.error("Please enter valid numerical parameters.")
            st.stop()

        scaled_input = scaler.transform(input_array)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index].item())

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Analysis Results & Probability Distribution")

        res_col1, res_col2 = st.columns([1, 1.6])

        with res_col1:
            st.markdown(
                f"""
                <div class="prediction-badge">
                    <div class="prediction-label">Predicted Cultivar Class</div>
                    <div class="prediction-value">{predicted_class}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Model Confidence", f"{confidence:.1%}")
            m2.metric("True Sample Target", reference_class_name)

        with res_col2:
            prob_df = pd.DataFrame({
                "Wine Class": CLASS_NAMES,
                "Probability": [float(p) for p in probabilities],
            })

            fig = px.bar(
                prob_df,
                x="Probability",
                y="Wine Class",
                orientation="h",
                text=prob_df["Probability"].map(lambda v: f"{v:.1%}"),
                color="Wine Class",
                color_discrete_sequence=["#571624", "#853142", "#B36E7B"],
            )

            fig.update_layout(
                template="plotly_white",
                height=240,
                showlegend=False,
                margin=dict(l=10, r=25, t=10, b=10),
                xaxis=dict(showgrid=False, range=[0, 1.18], visible=False),
                yaxis=dict(title=None, tickfont=dict(size=13, color="#1F1216", family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            fig.update_traces(textposition="outside", textfont=dict(size=12, family="Inter", color="#1F1216", weight=600))
            st.plotly_chart(fig, use_container_width=True)

        if predicted_index == reference_class_index:
            st.success("Model prediction matches the reference sample dataset target.")
        else:
            st.warning("Model prediction differs from the reference sample dataset target.")

    except Exception as exc:
        st.error(f"Prediction Execution Error: {exc}")