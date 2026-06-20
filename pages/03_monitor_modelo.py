"""
pages/03_monitor_modelo.py
============================
Implementa la regla de reentrenamiento exacta solicitada:
  SI AUC < 0.65 O PSI > 0.25 en MÁS DE 3 variables críticas ⇒ Reentrenar
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.predict import cargar_champion
from utils.metrics import evaluar_split
from utils.monitoring import (
    cargar_splits_monitoreo, calcular_psi_monitoreo,
    diagnostico_global, estilizar_tabla_psi,
    evaluar_regla_reentrenamiento, UMBRAL_AUC_MINIMO,
    UMBRAL_PSI_CRITICO, MAX_VARIABLES_INESTABLES_PERMITIDAS,
)
from utils.plots import grafico_evolucion_metricas, grafico_barras_drift, TEMA, GRID

st.set_page_config(page_title="Monitor del Modelo", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Monitor del Modelo")
st.markdown("**Salud del modelo, métricas históricas y regla de reentrenamiento**")
st.markdown("---")

DATA_DIR = Path("data")
TARGET   = "is_late_delivery"

art      = cargar_champion()
modelo   = art["modelo"]
features = art["features_finales"]
umbral   = art.get("umbral_optimo", 0.50)

# ══════════════════════════════════════════════════════════════════════════
# MÉTRICAS POR SPLIT
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Métricas Train / Val / Backtest / Live</div>',
            unsafe_allow_html=True)

@st.cache_data
def calcular_metricas_todos_splits(_umbral):
    splits = cargar_splits_monitoreo()
    resultados = []
    for nombre, df in splits.items():
        X = df[features].fillna(0)
        y = df[TARGET]
        proba = modelo.predict_proba(X)[:, 1]
        r = evaluar_split(y, proba, umbral=_umbral)
        r["split"] = nombre.capitalize()
        resultados.append(r)
    return pd.DataFrame(resultados)

df_bt = calcular_metricas_todos_splits(0.50)
cols_tabla = ["split","N","Tasa_real_%","AUC-ROC","Gini","PR-AUC","KS",
              "F1","Recall","Precision","Accuracy"]
st.dataframe(df_bt[cols_tabla], hide_index=True, use_container_width=True)
st.plotly_chart(grafico_evolucion_metricas(df_bt), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# REGLA DE REENTRENAMIENTO — el panel principal de esta página
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🚦 Regla de decisión: ¿reentrenar el modelo?</div>',
            unsafe_allow_html=True)

st.markdown(f"""
<div style="background:#0f1117; border:1px solid #1e2130; border-radius:12px;
     padding:16px 20px; margin-bottom:16px; font-family:'DM Mono',monospace; font-size:0.88rem;">
SI &nbsp;<b style="color:#D85A30">AUC < {UMBRAL_AUC_MINIMO}</b>&nbsp; O &nbsp;
<b style="color:#D85A30">PSI > {UMBRAL_PSI_CRITICO}</b> en
<b style="color:#D85A30">más de {MAX_VARIABLES_INESTABLES_PERMITIDAS} variables críticas</b>
&nbsp;⇒&nbsp; <b style="color:#E24B4A">REENTRENAR</b><br>
EN CUALQUIER OTRO CASO &nbsp;⇒&nbsp; <b style="color:#1D9E75">MODELO ESTABLE</b>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def calcular_drift():
    splits = cargar_splits_monitoreo()
    return calcular_psi_monitoreo(splits)

df_psi = calcular_drift()

# AUC de referencia: el más reciente y representativo de "producción" = Live
auc_live = df_bt[df_bt["split"] == "Live"]["AUC-ROC"].values
auc_live = float(auc_live[0]) if len(auc_live) > 0 else 0.0

decision = evaluar_regla_reentrenamiento(df_psi, auc_live, split_referencia="Live")

color_decision = decision["color"]
st.markdown(f"""
<div style="background:{color_decision}15; border:2px solid {color_decision}; border-radius:14px;
     padding:22px 26px; margin-bottom:18px;">
    <span style="color:{color_decision}; font-weight:700; font-size:1.4rem;">
        ● {decision['decision']}
    </span>
    <div style="margin-top:10px; font-size:0.9rem; color:#d1d5db;">
        <b>Evaluado sobre:</b> Live (Julio 2018 — período más reciente, proxy de producción)<br>
        <b>AUC actual:</b> {decision['auc_actual']}<br>
        <b>Variables con PSI > {UMBRAL_PSI_CRITICO}:</b> {decision['n_vars_inestables']}
        {f"({', '.join(decision['vars_inestables'])})" if decision['vars_inestables'] else ""}
    </div>
    <div style="margin-top:12px; font-size:0.85rem; color:#9ca3af;">
        <b>Razones:</b>
        <ul style="margin:4px 0 0 18px;">
        {''.join(f'<li>{r}</li>' for r in decision['razones'])}
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

if decision["decision"] == "REENTRENAR":
    st.markdown("""
    <div style="background:#0f1117; border-left:3px solid #f59e0b; border-radius:0 8px 8px 0;
         padding:11px 15px; font-size:0.85rem; color:#d1d5db;">
    📌 <b>Interpretación:</b> este resultado refleja drift real detectado —
    no es un error del pipeline. La variable <code>estimated_delivery_days</code> cambió
    su distribución entre Train y Live (la promesa media de entrega bajó de ~25 a ~20 días
    en julio 2018, posiblemente por una política comercial distinta). El sistema de
    monitoreo está funcionando como debe: detectando que el contexto de negocio cambió y
    que el modelo necesita actualizarse con datos más recientes antes de seguir en producción.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DETALLE PSI POR VARIABLE
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Detalle de PSI por variable y período</div>',
            unsafe_allow_html=True)
st.caption("PSI < 0.15: estable | 0.15–0.25: moderado | > 0.25: inestable (cuenta para la regla)")

if not df_psi.empty:
    st.plotly_chart(grafico_barras_drift(df_psi.rename(columns={"_color":"Color"})),
                    use_container_width=True)
    st.dataframe(estilizar_tabla_psi(df_psi), use_container_width=True, hide_index=True)
    st.markdown("""
    <div style="margin-top:10px; font-size:0.82rem; color:#9ca3af;">
    🟢 Estable &nbsp;&nbsp; 🟡 Moderado &nbsp;&nbsp; 🔴 Inestable
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("No se pudieron calcular los splits para monitoreo de drift.")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">SHAP — Explicabilidad del modelo</div>',
            unsafe_allow_html=True)

shap_path = Path("reports/shap_importancia.png")

@st.cache_resource
def generar_shap_si_no_existe():
    if shap_path.exists():
        return True, None
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        train = pd.read_parquet(DATA_DIR / "split_train.parquet")
        X_sample = train[features].fillna(0).sample(min(2000, len(train)), random_state=42)
        explainer = shap.LinearExplainer(modelo, X_sample)
        shap_values = explainer.shap_values(X_sample)
        Path("reports").mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=features,
                          plot_type="bar", show=False, max_display=20)
        plt.title("SHAP — Importancia Global")
        plt.tight_layout()
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        return True, None
    except Exception as e:
        return False, str(e)

ok, error = generar_shap_si_no_existe()
if ok and shap_path.exists():
    st.image(str(shap_path), use_container_width=True)
else:
    st.warning(f"No se pudo generar SHAP automáticamente. {error or ''}")
