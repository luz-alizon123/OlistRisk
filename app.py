"""
app.py — Olist Risk Engine MVP
================================
Punto de entrada de la aplicación multipágina.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Olist Risk Engine — MVP",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.metric-card {
    background: #0f1117; border: 1px solid #1e2130;
    border-radius: 12px; padding: 20px 24px; text-align: center;
}
.metric-value { font-family: 'DM Mono', monospace; font-size: 1.9rem;
    font-weight: 500; color: #00d4aa; line-height: 1; margin-bottom: 4px; }
.metric-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.08em; }
.metric-delta { font-size: 0.78rem; color: #9ca3af; margin-top: 4px; }
.section-title { font-size: 0.70rem; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 14px;
    padding-bottom: 7px; border-bottom: 1px solid #1e2130; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📦 Olist Risk Engine")
st.markdown("**Motor de Inteligencia Logística**")
st.markdown("*Detecta la firma de riesgo de cada pedido al momento de la compra*")
st.markdown("---")

with st.sidebar:
    st.markdown("## 📦 Olist Risk")
    st.markdown("**MVP v4.0.0**")
    st.markdown("---")
    st.markdown("""
    **Navegación:**
    Usá el menú de páginas arriba a la izquierda ⬆️

    - 📊 Dashboard Operativo
    - 🎯 Simulador de Pedido
    - 🔍 Monitor del Modelo
    - 🎤 Storytelling Demo Day
    - ⚖️ Gobernanza y Ética
    """)
    st.markdown("---")

    model_path = Path("models/champion_model_v3.pkl")
    if model_path.exists():
        st.success("✅ Modelo champion cargado")
        st.caption("champion_model_v3.pkl — LogReg Tuneado")
    else:
        st.error("⚠️ No se encontró champion_model_v3.pkl")
        st.caption("Verificá la carpeta /models")

    st.markdown("---")
    st.caption("⚠️ Este MVP usa el modelo ya entrenado. "
               "Ningún botón de esta aplicación reentrena el modelo.")

st.markdown("""
### Bienvenido al panel interno de Olist Risk

Este sistema **no está diseñado para mostrarse al comprador final**.
Es una herramienta interna para los equipos de operaciones, logística
y atención al vendedor.

Usá el menú lateral para navegar entre las distintas vistas del sistema:

- **Dashboard Operativo** — KPIs del mes actual, mapa de riesgo por estado, rankings
- **Simulador de Pedido** — evaluá el riesgo de un pedido hipotético en tiempo real
- **Monitor del Modelo** — métricas históricas y detección de drift (PSI)
- **Storytelling Demo Day** — la historia completa del proyecto para presentar a directivos
- **Gobernanza y Ética** — sesgos conocidos, limitaciones y políticas de uso responsable
""")

st.markdown("---")
st.caption("Sistema de Inteligencia Logística — Olist E-commerce · Sprint 4 MVP")
