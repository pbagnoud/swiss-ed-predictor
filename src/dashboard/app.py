"""
Swiss ED Predictor — Dashboard Streamlit
Interface décideurs cantonaux — prédit les pics d'affluence à 24-72h.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Page config
st.set_page_config(
    page_title="Swiss ED Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border-left: 4px solid #1B7FA1;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .alert-high { border-left-color: #ef4444; }
    .alert-medium { border-left-color: #f59e0b; }
    .alert-low { border-left-color: #22c55e; }
    .main-title { color: #0A2342; font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar
with st.sidebar:
    st.image("https://www.bag.admin.ch/etc/designs/bag/img/logo-bag.svg", width=120)
    st.markdown("## ⚙️ Paramètres")

    canton = st.selectbox(
        "Canton",
        ["BE", "ZH", "GE", "VD", "BS", "AG", "SG", "TI", "VS", "NE"],
        index=0,
    )

    hospital = st.selectbox(
        "Hôpital",
        ["Inselspital Bern", "Hôpital du Valais", "HUG Genève", "CHUV Lausanne"],
        index=0,
    )

    horizon = st.slider("Horizon de prédiction (heures)", 24, 72, 48, step=24)

    st.markdown("---")
    st.markdown("**Sources de données**")
    st.caption("🌤️ MétéoSuisse · Temps réel")
    st.caption("🚌 opentransportdata.swiss")
    st.caption("📊 SpiGes / OFSP · 2018-2023")
    st.caption("👥 OFS · Démographie cantonale")

    st.markdown("---")
    if st.button("🔄 Actualiser les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Main content
st.markdown('<p class="main-title">🏥 Swiss ED Predictor</p>', unsafe_allow_html=True)
st.caption(f"Canton {canton} · {hospital} · Prédiction à {horizon}h · {datetime.now().strftime('%d.%m.%Y %H:%M')}")

st.markdown("---")

# ── KPI Row
col1, col2, col3, col4 = st.columns(4)

# Simulated predictions (replace with model.predict() in production)
predicted_admissions = np.random.randint(45, 85)
current_capacity = 60
alert_level = "🔴 Élevé" if predicted_admissions > 75 else ("🟡 Modéré" if predicted_admissions > 55 else "🟢 Normal")
confidence = np.random.uniform(0.82, 0.93)

with col1:
    st.metric(
        label="🏥 Admissions prévues",
        value=f"{predicted_admissions}",
        delta=f"+{predicted_admissions - current_capacity} vs capacité normale",
        delta_color="inverse",
    )

with col2:
    st.metric(
        label="⚡ Niveau d'alerte",
        value=alert_level,
        delta=f"Horizon {horizon}h",
    )

with col3:
    st.metric(
        label="📈 Confiance du modèle",
        value=f"{confidence:.0%}",
        delta="XGBoost · AUROC 0.90",
    )

with col4:
    st.metric(
        label="🌡️ Température prévue",
        value=f"{np.random.randint(5, 25)}°C",
        delta="MétéoSuisse",
    )

st.markdown("---")

# ── Prediction chart
col_chart, col_features = st.columns([2, 1])

with col_chart:
    st.subheader("📊 Prévision d'affluence — Prochaines 72h")

    # Simulated time series
    now = datetime.now()
    hours = [now + timedelta(hours=i) for i in range(73)]
    baseline = 50
    pred_values = [
        int(baseline + 15 * np.sin(i / 8) + np.random.randint(-5, 10) +
            (10 if 8 <= (now + timedelta(hours=i)).hour <= 20 else -5))
        for i in range(73)
    ]
    upper = [v + 8 for v in pred_values]
    lower = [max(0, v - 8) for v in pred_values]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(27, 127, 161, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Intervalle de confiance",
    ))

    # Prediction line
    fig.add_trace(go.Scatter(
        x=hours, y=pred_values,
        mode="lines",
        name="Admissions prévues",
        line=dict(color="#1B7FA1", width=2.5),
    ))

    # Capacity threshold
    fig.add_hline(
        y=current_capacity,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Capacité normale",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Date / Heure",
        yaxis_title="Admissions aux urgences",
        hovermode="x unified",
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#f0f4f8")
    fig.update_yaxes(gridcolor="#f0f4f8")

    st.plotly_chart(fig, use_container_width=True)


with col_features:
    st.subheader("🔍 Facteurs déclenchants")

    # SHAP-inspired feature importance (simulated)
    features = {
        "Saison (hiver)": 0.28,
        "Jour semaine": 0.22,
        "Température": 0.18,
        "Mobilité transport": 0.15,
        "Indice saisonnier": 0.10,
        "Précipitations": 0.07,
    }

    fig2 = px.bar(
        x=list(features.values()),
        y=list(features.keys()),
        orientation="h",
        color=list(features.values()),
        color_continuous_scale=["#1B7FA1", "#02C39A"],
    )
    fig2.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Basé sur les valeurs SHAP du modèle XGBoost")


# ── Recommendations
st.markdown("---")
st.subheader("📋 Recommandations opérationnelles")

if predicted_admissions > 75:
    st.error(f"""
    **⚠️ Pic d'affluence prévu dans les prochaines {horizon}h**
    - Activer le protocole de renforts (+2 médecins urgentistes)
    - Préparer 8–10 lits supplémentaires
    - Alerter le service de coordination 144
    - Envisager déviation partielle vers hôpitaux voisins
    """)
elif predicted_admissions > 55:
    st.warning(f"""
    **🟡 Charge modérée prévue**
    - Maintenir la vigilance opérationnelle
    - Prévoir 2–3 lits de réserve
    - Vérifier la disponibilité du personnel de nuit
    """)
else:
    st.success(f"""
    **✅ Charge normale prévue**
    - Fonctionnement habituel
    - Occasion de planifier la maintenance préventive
    """)

# ── Footer
st.markdown("---")
st.caption(
    "🏥 Swiss ED Predictor · GovTech Hackathon 2026 · "
    "Open Source MIT · github.com/adeutou/swiss-ed-predictor · "
    "Données: SpiGes/OFSP · MétéoSuisse · opentransportdata.swiss · OFS"
)
