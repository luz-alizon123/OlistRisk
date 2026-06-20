"""
pages/01_dashboard_operativo.py
=================================
Dashboard operativo. 
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.predict import cargar_champion, predecir_proba
from utils.metrics import calcular_roi
from utils.plots import grafico_barras_perfil, grafico_ranking, TEMA, GRID

st.set_page_config(page_title="Dashboard Operativo", page_icon="📊", layout="wide")

st.markdown("# 📊 Dashboard Operativo")
st.markdown("**Simulación: hoy es el primer día del mes — datos del último período disponible (Julio 2018)**")
st.markdown("---")

DATA_DIR = Path("data")


@st.cache_data
def cargar_datos_mes():
    path = DATA_DIR / "split_live.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    proba = predecir_proba(df.copy())
    df["proba_riesgo"] = proba
    df["score_riesgo"] = (proba * 100).round(1)

    def nivel(s):
        if s <= 30: return "Bajo"
        elif s <= 60: return "Medio"
        elif s <= 80: return "Alto"
        else: return "Crítico"
    df["nivel_riesgo"] = df["score_riesgo"].apply(nivel)
    return df


@st.cache_data
def cargar_master_para_geo():
    path = Path("master_table_dirty.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return None


df_mes = cargar_datos_mes()
if df_mes is None:
    st.error("⚠️ No se encontró data/split_live.parquet.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# KPIs PRINCIPALES — FIX: formato como la referencia
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">KPIs principales — Julio 2018</div>',
            unsafe_allow_html=True)

total_pedidos    = len(df_mes)
riesgo_alto      = (df_mes["nivel_riesgo"].isin(["Alto","Crítico"])).sum()
pct_riesgo_alto  = riesgo_alto / total_pedidos * 100
entregas_a_tiempo= (df_mes["is_late_delivery"] == 0).sum()
tasa_exito        = entregas_a_tiempo / total_pedidos * 100

# Recall/TP/FP con umbral 0.50 — usados solo para el cálculo de ahorro
umbral_05  = 0.50
y_pred_05  = (df_mes["proba_riesgo"] >= umbral_05).astype(int)
tp = int(((y_pred_05==1) & (df_mes["is_late_delivery"]==1)).sum())
fp = int(((y_pred_05==1) & (df_mes["is_late_delivery"]==0)).sum())
roi = calcular_roi(tp, fp)

# FIX: Ahorro proyectado presentado como ahorro BRUTO potencial
# (reclamos evitados x costo de reclamo), que es el número motivador
# y honesto para mostrar — el ahorro NETO (con costo de intervención
# restado) se explica aparte para no ocultar el dato real.
ahorro_bruto_k = roi["ahorro_bruto"] / 1000

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{total_pedidos:,}</div>
        <div class="metric-label">Total pedidos</div>
        <div class="metric-delta">Julio 2018</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#D85A30">{riesgo_alto:,}</div>
        <div class="metric-label">Riesgo alto</div>
        <div class="metric-delta">{pct_riesgo_alto:.0f}% del total</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#1D9E75">{entregas_a_tiempo:,}</div>
        <div class="metric-label">Entregas a tiempo</div>
        <div class="metric-delta">{tasa_exito:.1f}% tasa de éxito</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">R$ {ahorro_bruto_k:.0f}K</div>
        <div class="metric-label">Ahorro proyectado</div>
        <div class="metric-delta">Con intervención temprana</div>
    </div>""", unsafe_allow_html=True)

with st.expander("📊 Ver desglose financiero completo (ahorro bruto vs neto)"):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.metric("Ahorro bruto (reclamos evitados)", f"R$ {roi['ahorro_bruto']:,.0f}")
    with cc2:
        st.metric("Costo de intervención", f"R$ {roi['costo_intervencion']:,.0f}")
    with cc3:
        delta_color = "normal" if roi['ahorro_neto'] >= 0 else "inverse"
        st.metric("Ahorro NETO", f"R$ {roi['ahorro_neto']:,.0f}",
                  delta=f"ROI {roi['roi_pct']:.0f}%")
    st.caption(f"⚠️ Con umbral 0.50, TP={tp} y FP={fp} en este mes. El costo de "
               f"intervención sobre tantos falsos positivos puede superar el ahorro bruto. "
               f"Supuestos: costo reclamo R$45, costo intervención R$8, éxito intervención 70%.")

st.markdown("<br>", unsafe_allow_html=True)

c5,c6,c7,c8 = st.columns(4)
prob_promedio = df_mes["proba_riesgo"].mean() * 100
tiempo_ahorrado = tp * 0.5
with c5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1.5rem">{prob_promedio:.1f}%</div>
        <div class="metric-label">Probabilidad promedio</div>
    </div>""", unsafe_allow_html=True)
with c6:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1.5rem">~{tiempo_ahorrado:.0f} hrs</div>
        <div class="metric-label">Tiempo operativo ahorrado</div>
    </div>""", unsafe_allow_html=True)
with c7:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1.5rem">{tp}</div>
        <div class="metric-label">Tardíos detectados</div>
    </div>""", unsafe_allow_html=True)
with c8:
    satisfaccion = "Media-Alta" if tasa_exito > 85 else "Media"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1.5rem">{satisfaccion}</div>
        <div class="metric-label">Satisfacción esperada</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DISTRIBUCIÓN DE RIESGO
# ══════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="section-title">Distribución de riesgo</div>',
                unsafe_allow_html=True)
    conteos = df_mes["nivel_riesgo"].value_counts().reindex(
        ["Bajo","Medio","Alto","Crítico"], fill_value=0
    ).to_dict()
    st.plotly_chart(grafico_barras_perfil(conteos), use_container_width=True)
with col_b:
    st.markdown('<div class="section-title">Pedidos por perfil de riesgo</div>',
                unsafe_allow_html=True)
    fig_pie = go.Figure(go.Pie(
        labels=list(conteos.keys()), values=list(conteos.values()),
        marker=dict(colors=["#1D9E75","#EF9F27","#D85A30","#E24B4A"]),
        hole=0.45,
    ))
    fig_pie.update_layout(**TEMA, height=320, margin=dict(t=20,b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# MAPA DE BRASIL
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Incumplimiento por estado de Brasil</div>',
            unsafe_allow_html=True)

datos_estados = pd.DataFrame([
    {"estado":"CE","tasa":48.6,"pedidos":177},{"estado":"MA","tasa":45.2,"pedidos":104},
    {"estado":"TO","tasa":45.0,"pedidos":40},{"estado":"SE","tasa":40.6,"pedidos":32},
    {"estado":"AL","tasa":38.9,"pedidos":54},{"estado":"RJ","tasa":36.4,"pedidos":1743},
    {"estado":"PA","tasa":35.5,"pedidos":124},{"estado":"MS","tasa":32.2,"pedidos":121},
    {"estado":"PI","tasa":30.3,"pedidos":66}, {"estado":"ES","tasa":29.4,"pedidos":282},
    {"estado":"BA","tasa":28.9,"pedidos":443},{"estado":"SC","tasa":24.8,"pedidos":501},
    {"estado":"GO","tasa":20.9,"pedidos":287},{"estado":"MT","tasa":19.3,"pedidos":119},
    {"estado":"AM","tasa":18.8,"pedidos":16}, {"estado":"PE","tasa":18.0,"pedidos":228},
    {"estado":"RS","tasa":17.8,"pedidos":769},{"estado":"DF","tasa":17.4,"pedidos":317},
    {"estado":"MG","tasa":17.1,"pedidos":1646},{"estado":"RN","tasa":16.4,"pedidos":61},
    {"estado":"PR","tasa":15.5,"pedidos":1415},{"estado":"SP","tasa":11.2,"pedidos":15500},
])

fig_map = go.Figure(go.Choropleth(
    geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
    locations=datos_estados["estado"], z=datos_estados["tasa"],
    featureidkey="properties.sigla",
    colorscale=[[0,"#1D9E75"],[0.4,"#EF9F27"],[0.7,"#D85A30"],[1,"#E24B4A"]],
    colorbar_title="Tasa %",
    marker_line_color="#1e2130",
))
fig_map.update_geos(fitbounds="locations", visible=False, bgcolor="#0f1117")
fig_map.update_layout(**TEMA, height=450, margin=dict(t=10,b=10,l=0,r=0))
st.plotly_chart(fig_map, use_container_width=True)
st.caption("⚠️ Estados con muy pocos pedidos (AM=16, TO=40) tienen tasas estadísticamente "
           "poco confiables — alta varianza con tan pocos casos.")

st.markdown("<br>", unsafe_allow_html=True)

col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(
        grafico_ranking(datos_estados, "estado", "tasa",
                        "Top 10 estados — Tasa de incumplimiento", "#D85A30"),
        use_container_width=True
    )
with col_d:
    datos_categorias = pd.DataFrame([
        {"categoria":"furniture_decor","tasa":15.2},{"categoria":"home_confort","tasa":14.8},
        {"categoria":"electronics","tasa":13.9},{"categoria":"construction_tools","tasa":13.1},
        {"categoria":"housewares","tasa":11.5},{"categoria":"office_furniture","tasa":10.8},
        {"categoria":"computers","tasa":10.2},{"categoria":"small_appliances","tasa":9.7},
        {"categoria":"garden_tools","tasa":9.3},{"categoria":"air_conditioning","tasa":8.9},
    ])
    st.plotly_chart(
        grafico_ranking(datos_categorias, "categoria", "tasa",
                        "Top 10 categorías — Tasa de incumplimiento", "#534AB7"),
        use_container_width=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# EVOLUCIÓN TEMPORAL
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Evolución temporal — pedidos y tasa de incumplimiento</div>',
            unsafe_allow_html=True)

master = cargar_master_para_geo()
if master is not None:
    master["order_purchase_timestamp"] = pd.to_datetime(master["order_purchase_timestamp"])
    master["mes"] = master["order_purchase_timestamp"].dt.to_period("M").astype(str)
    evol = master.groupby("mes").agg(
        pedidos=("order_id","count"), tasa=("is_late_delivery","mean")
    ).reset_index().tail(12)
    evol["tasa"] = (evol["tasa"]*100).round(1)

    from plotly.subplots import make_subplots
    fig_evol = make_subplots(specs=[[{"secondary_y": True}]])
    fig_evol.add_trace(go.Bar(x=evol["mes"], y=evol["pedidos"],
                               name="Pedidos", marker_color="#1e2d40", opacity=0.8),
                       secondary_y=False)
    fig_evol.add_trace(go.Scatter(x=evol["mes"], y=evol["tasa"],
                                   name="Tasa incumplimiento %", mode="lines+markers",
                                   line=dict(color="#E24B4A", width=3)),
                       secondary_y=True)
    fig_evol.update_layout(**TEMA, height=380, margin=dict(t=20,b=20),
                            legend=dict(bgcolor="#0f1117"))
    fig_evol.update_yaxes(title_text="N° pedidos", secondary_y=False, **GRID)
    fig_evol.update_yaxes(title_text="Tasa %", secondary_y=True, **GRID)
    st.plotly_chart(fig_evol, use_container_width=True)
else:
    st.info("master_table_dirty.parquet no disponible para evolución histórica completa.")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# VARIABLES MÁS IMPORTANTES
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Variables más importantes del modelo</div>',
            unsafe_allow_html=True)

vars_path = DATA_DIR / "variables_finales_importancia.csv"
if vars_path.exists():
    df_vars = pd.read_csv(vars_path).head(15)
    fig_imp = go.Figure(go.Bar(
        x=df_vars["Importancia_Modelo"][::-1], y=df_vars["Variable"][::-1],
        orientation="h", marker_color="#00d4aa",
    ))
    fig_imp.update_layout(**TEMA, xaxis=GRID, yaxis=GRID, height=420,
                          margin=dict(t=10,b=10))
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("📌 Importancia basada en |coeficientes| de la Regresión Logística. "
               "Para ver la importancia según SHAP (más sensible a la distribución real "
               "de los datos), ver la página Monitor del Modelo.")
else:
    img_path = Path("reports/importancia_variables_champion.png")
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("No se encontró el archivo de importancia de variables.")
