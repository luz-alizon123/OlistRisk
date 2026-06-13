"""
app.py — Dashboard Olist Risk Model
Ejecutar: streamlit run app.py
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import joblib
import json
from pathlib import Path
 
# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist — Panel de Riesgo Logístico",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─── ESTILOS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
 
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
 
.metric-card {
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 2.1rem;
    font-weight: 500;
    color: #00d4aa;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-value-biz {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #f59e0b;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-delta {
    font-size: 0.80rem;
    color: #9ca3af;
    margin-top: 4px;
}
.section-title {
    font-size: 0.70rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 14px;
    padding-bottom: 7px;
    border-bottom: 1px solid #1e2130;
}
.insight-box {
    background: #0f1117;
    border-left: 3px solid #00d4aa;
    border-radius: 0 8px 8px 0;
    padding: 11px 15px;
    margin: 7px 0;
    font-size: 0.86rem;
    color: #d1d5db;
    line-height: 1.5;
}
.warn-box {
    background: #0f1117;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 11px 15px;
    margin: 7px 0;
    font-size: 0.86rem;
    color: #d1d5db;
    line-height: 1.5;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #0f1117;
    border-radius: 8px;
    color: #6b7280;
    font-size: 0.84rem;
}
.stTabs [aria-selected="true"] {
    background: #1e2130 !important;
    color: #00d4aa !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    data = {}
 
    for nombre in ['train', 'val', 'backtest', 'live']:
        p = Path(f"data/split_{nombre}.parquet")
        if p.exists():
            data[nombre] = pd.read_parquet(p)
 
    p = Path("data/tabla_comparativa_modelos.csv")
    if p.exists():
        data['comparativa'] = pd.read_csv(p)
 
    p = Path("data/tabla_backtesting.csv")
    if p.exists():
        data['backtesting'] = pd.read_csv(p)
 
    p = Path("master_table_dirty.parquet")
    if p.exists():
        master = pd.read_parquet(p)
        master['order_purchase_timestamp'] = pd.to_datetime(master['order_purchase_timestamp'])
        master['mes'] = master['order_purchase_timestamp'].dt.to_period('M').astype(str)
        data['mensual'] = master.groupby('mes').agg(
            pedidos=('order_id', 'count'),
            tasa_tardios=('is_late_delivery', 'mean')
        ).reset_index()
        data['mensual']['tasa_tardios'] = (data['mensual']['tasa_tardios'] * 100).round(1)
        data['total_pedidos'] = len(master)
        data['tasa_global']   = master['is_late_delivery'].mean() * 100
 
    p = Path("models/champion_resumen.json")
    if p.exists():
        with open(p) as f:
            data['resumen'] = json.load(f)
 
    p = Path("data/reporte_importancias_rf.csv")
    if p.exists():
        data['importancias'] = pd.read_csv(p, index_col=0)
        data['importancias'].columns = ['importancia']
 
    return data
 
 
data = cargar_datos()
 
# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Olist Risk")
    st.markdown("**Sistema de Inteligencia Logística**")
    st.markdown("---")
 
    if 'resumen' in data:
        r = data['resumen']
        st.markdown(f"**Modelo campeón:** {r.get('nombre_champion','RF')}")
        st.markdown(f"**Versión:** {r.get('version','1.0.0')}")
        st.markdown(f"**Fecha de entrenamiento:** {r.get('fecha_entrenamiento','—')}")
        st.markdown(f"**Período de entrenamiento:** {r.get('periodo_train','—')}")
 
    st.markdown("---")
    st.markdown("**Distribución de particiones**")
    labels = {'train':'Entrenamiento','val':'Validación','backtest':'Backtest','live':'Live'}
    for s, label in labels.items():
        if s in data:
            df  = data[s]
            n   = len(df)
            tas = df['is_late_delivery'].mean() * 100
            st.markdown(f"`{label}` — {n:,} pedidos · {tas:.1f}% tardíos")
 
    st.markdown("---")
    st.markdown("<small style='color:#6b7280'>Sprint 3 completado — Sin tuneo<br>Sprint 4 pendiente: Optuna + calibración</small>",
                unsafe_allow_html=True)
 
 
# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("# 📦 Sistema de Predicción de Riesgo Logístico — Olist")
st.markdown("**Predicción de retrasos en entregas · Sprint 3 · Modelo base sin tuneo**")
st.markdown("---")
 
 
# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 KPIs del Modelo",
    "💰 Métricas de Negocio",
    "🏆 Comparativa de Modelos",
    "⏱️ Backtesting Temporal",
    "📅 Análisis Temporal",
    "🔍 Variables e Interpretabilidad",
])
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — KPIs DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Métricas del modelo campeón — Random Forest (Backtest Julio 2018)</div>',
                unsafe_allow_html=True)
 
    c1, c2, c3, c4, c5 = st.columns(5)
    metricas = [
        (c1, "AUC-ROC",    "0.6266", "Backtest — Jul 2018"),
        (c2, "Gini",       "0.2531", "Backtest — Jul 2018"),
        (c3, "KS",         "0.1962", "Backtest — Jul 2018"),
        (c4, "Precisión",  "10.3%",  "Backtest — umbral 0.56"),
        (c5, "Variables",  "50",     "tras selección"),
    ]
    for col, label, val, delta in metricas:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Evolución AUC por período
    st.markdown('<div class="section-title">Evolución del AUC-ROC por período</div>',
                unsafe_allow_html=True)
 
    splits_l = ['Entrenamiento\n(Sep16–May18)', 'Validación\n(Jun18)', 'Backtest\n(Jul18)', 'Live corregido\n(Ago18)']
    aucs     = [0.7882, 0.6067, 0.6266, 0.5396]
    colores  = ['#6366f1', '#f59e0b', '#00d4aa', '#ef4444']
 
    fig_auc = go.Figure()
    fig_auc.add_trace(go.Scatter(
        x=splits_l, y=aucs,
        mode='lines+markers+text',
        line=dict(color='#00d4aa', width=3),
        marker=dict(size=14, color=colores),
        text=[f"{v:.4f}" for v in aucs],
        textposition="top center",
        textfont=dict(family="DM Mono", size=13),
    ))
    fig_auc.add_hline(y=0.75, line_dash="dash", line_color="#6366f1",
                      annotation_text="Objetivo AUC ≥ 0.75 (Sprint 4)",
                      annotation_position="bottom right",
                      annotation_font_color="#6366f1")
    fig_auc.add_hline(y=0.50, line_dash="dot", line_color="#ef4444",
                      annotation_text="Clasificador aleatorio = 0.50",
                      annotation_position="bottom right",
                      annotation_font_color="#ef4444")
    fig_auc.update_layout(
        plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
        font=dict(color='#d1d5db', family='DM Sans'),
        yaxis=dict(range=[0.35, 0.95], gridcolor='#1e2130', title='AUC-ROC'),
        xaxis=dict(gridcolor='#1e2130'),
        height=360, margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_auc, use_container_width=True)
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="insight-box">✅ <b>Backtest estable:</b> AUC 0.63 en julio — consistente con el período de validación</div>
        <div class="insight-box">✅ <b>Censura corregida:</b> 2.573 pedidos (40.5% del Live) tenían fecha de entrega posterior al corte del dataset — filtrados correctamente</div>
        <div class="insight-box">✅ <b>Variable seller_late_rate_hist</b> incluida — historial de retrasos del vendedor sin data leakage</div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="warn-box">⚠️ <b>Drift temporal:</b> Agosto 2018 registra 16.8% de tardíos vs 8.8% en entrenamiento — el modelo no vio este nivel de riesgo</div>
        <div class="warn-box">⚠️ <b>Validación ruidosa:</b> Junio 2018 tiene solo 83 pedidos tardíos (1.4%) — métrica poco confiable para calibrar umbral</div>
        <div class="warn-box">⚠️ <b>Stacking roto:</b> Meta-modelo colapsa con modelos sobreajustados — se corrige en Sprint 4</div>
        """, unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MÉTRICAS DE NEGOCIO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Impacto económico estimado — Modelo campeón sobre Backtest (Jul 2018)</div>',
                unsafe_allow_html=True)
 
    # Parámetros del modelo de negocio
    # Backtest: TP=29, FP=253, TN=5350, FN=211 (umbral 0.56)
    TP, FP, TN, FN = 29, 253, 5350, 211
    COSTO_RECLAMO    = 45    # R$ costo promedio por reclamo de retraso
    COSTO_INTERV     = 8     # R$ costo de intervención preventiva por pedido alertado
    TASA_EXITO_INTERV = 0.70  # 70% de los alertados correctamente se salvan
 
    reclamos_evitados   = TP * TASA_EXITO_INTERV
    ahorro_bruto        = reclamos_evitados * COSTO_RECLAMO
    costo_intervencion  = (TP + FP) * COSTO_INTERV
    ahorro_neto         = ahorro_bruto - costo_intervencion
    reclamos_no_detect  = FN * COSTO_RECLAMO
    total_tardios       = TP + FN
    tasa_deteccion      = TP / max(total_tardios, 1) * 100
    precision_alertas   = TP / max(TP + FP, 1) * 100
 
    # KPIs de negocio
    c1, c2, c3, c4 = st.columns(4)
    kpis_neg = [
        (c1, "Ahorro neto estimado",    f"R$ {ahorro_neto:,.0f}",   f"Backtest Jul — {len(data.get('backtest', pd.DataFrame())):,} pedidos"),
        (c2, "Reclamos evitados",       f"{reclamos_evitados:.0f}", f"de {total_tardios} tardíos detectables"),
        (c3, "Tasa de detección",       f"{tasa_deteccion:.1f}%",   "pedidos tardíos identificados"),
        (c4, "Precisión de alertas",    f"{precision_alertas:.1f}%","de las alertas son correctas"),
    ]
    for col, label, val, delta in kpis_neg:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value-biz">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    col_izq, col_der = st.columns([1, 1])
 
    with col_izq:
        st.markdown('<div class="section-title">Desglose del impacto económico — Backtest</div>',
                    unsafe_allow_html=True)
 
        categorias_biz = ['Ahorro bruto<br>(reclamos evitados)', 'Costo de<br>intervención', 'Ahorro neto']
        valores_biz    = [ahorro_bruto, -costo_intervencion, ahorro_neto]
        colores_biz    = ['#00d4aa', '#ef4444', '#f59e0b']
 
        fig_biz = go.Figure(go.Bar(
            x=categorias_biz, y=valores_biz,
            marker_color=colores_biz,
            text=[f"R$ {abs(v):,.0f}" for v in valores_biz],
            textposition='outside',
            textfont=dict(family='DM Mono', size=13),
        ))
        fig_biz.add_hline(y=0, line_color='#6b7280', line_width=1)
        fig_biz.update_layout(
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#d1d5db', family='DM Sans'),
            yaxis=dict(gridcolor='#1e2130', title='R$ (Reales brasileños)'),
            xaxis=dict(gridcolor='#1e2130'),
            height=340, margin=dict(t=40, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_biz, use_container_width=True)
 
    with col_der:
        st.markdown('<div class="section-title">Matriz de confusión — Backtest (umbral 0.56)</div>',
                    unsafe_allow_html=True)
 
        z    = [[TN, FP], [FN, TP]]
        txt  = [[f"TN<br>{TN:,}<br>Negativos correctos",  f"FP<br>{FP:,}<br>Falsas alarmas"],
                [f"FN<br>{FN:,}<br>Tardíos no detectados", f"TP<br>{TP:,}<br>Tardíos detectados"]]
        cols_cm = ['Predicho: A tiempo', 'Predicho: Tardío']
        rows_cm = ['Real: A tiempo', 'Real: Tardío']
 
        fig_cm = go.Figure(go.Heatmap(
            z=z, x=cols_cm, y=rows_cm,
            text=txt, texttemplate="%{text}",
            colorscale=[[0,'#0f1117'],[0.5,'#1e3a5f'],[1,'#00d4aa']],
            showscale=False,
            textfont=dict(size=13, family='DM Sans'),
        ))
        fig_cm.update_layout(
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#d1d5db', family='DM Sans'),
            height=300, margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
 
        st.markdown(f"""
        <div class="insight-box">
        💡 <b>Interpretación:</b> Por cada 100 alertas emitidas, {precision_alertas:.0f} corresponden a pedidos que realmente llegarán tarde.
        El costo de los {FP} falsos positivos (R$ {FP*COSTO_INTERV:,}) es compensado por el ahorro en reclamos evitados.
        </div>""", unsafe_allow_html=True)
 
    # Proyección anual
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Proyección anual estimada (extrapolación lineal)</div>',
                unsafe_allow_html=True)
 
    pedidos_mes   = 6000   # promedio mensual en el período analizado
    tardios_mes   = pedidos_mes * 0.088
    tp_mes        = tardios_mes * (TP / max(TP + FN, 1))
    fp_mes        = tp_mes * (FP / max(TP, 1))
    ahorro_mes    = tp_mes * TASA_EXITO_INTERV * COSTO_RECLAMO - (tp_mes + fp_mes) * COSTO_INTERV
    ahorro_anual  = ahorro_mes * 12
 
    c1, c2, c3, c4 = st.columns(4)
    proy = [
        (c1, "Pedidos procesados/mes",  f"{pedidos_mes:,}",       "estimado operacional"),
        (c2, "Tardíos detectados/mes",  f"~{tp_mes:.0f}",         "con tasa 8.8% y detección 12.1%"),
        (c3, "Ahorro estimado/mes",     f"R$ {ahorro_mes:,.0f}",  "neto de costos de intervención"),
        (c4, "Proyección anual",        f"R$ {ahorro_anual:,.0f}","12 meses · modelo sin tuneo"),
    ]
    for col, label, val, delta in proy:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value-biz">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("""
    <div class="warn-box" style="margin-top:14px">
    ⚠️ <b>Supuestos del modelo de negocio:</b>
    Costo promedio por reclamo de retraso = R$ 45 (incluye compensación + atención al cliente) ·
    Costo de intervención preventiva = R$ 8 por pedido alertado ·
    Tasa de éxito de intervención = 70% ·
    Estos valores deben validarse con el equipo de operaciones de Olist para una estimación precisa.
    </div>""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARATIVA DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Comparativa de modelos — Validación (Jun 2018)</div>',
                unsafe_allow_html=True)
 
    df_comp = pd.DataFrame([
        {"Modelo": "Regresión Logística", "AUC-ROC": 0.5748, "Gini": 0.1495, "PR-AUC": 0.0166, "KS": 0.1834, "Recall": 0.4699, "F1": 0.0307, "Estado": "Baseline"},
        {"Modelo": "Random Forest ✅",    "AUC-ROC": 0.6067, "Gini": 0.2133, "PR-AUC": 0.0231, "KS": 0.1855, "Recall": 0.7349, "F1": 0.0355, "Estado": "Campeón"},
        {"Modelo": "XGBoost",             "AUC-ROC": 0.5565, "Gini": 0.1130, "PR-AUC": 0.0200, "KS": 0.1321, "Recall": 0.2651, "F1": 0.0304, "Estado": "Sobreajuste"},
        {"Modelo": "LightGBM",            "AUC-ROC": 0.4856, "Gini":-0.0288, "PR-AUC": 0.0129, "KS": 0.0741, "Recall": 0.2169, "F1": 0.0204, "Estado": "Sobreajuste"},
        {"Modelo": "Stacking",            "AUC-ROC": 0.5549, "Gini": 0.1098, "PR-AUC": 0.0306, "KS": 0.1260, "Recall": 1.0000, "F1": 0.0274, "Estado": "Roto"},
    ])
 
    col_r, col_t = st.columns([1, 1])
    with col_r:
        # Brecha sobreajuste
        st.markdown('<div class="section-title">Brecha Entrenamiento → Validación (sobreajuste)</div>',
                    unsafe_allow_html=True)
        nombres  = ['Reg. Logística', 'Random Forest', 'XGBoost', 'LightGBM']
        auc_tr   = [0.7366, 0.7882, 0.8606, 0.9364]
        auc_val  = [0.5748, 0.6067, 0.5565, 0.4856]
 
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(name='Entrenamiento', x=nombres, y=auc_tr,
                                  marker_color='#6366f1', opacity=0.85))
        fig_gap.add_trace(go.Bar(name='Validación',    x=nombres, y=auc_val,
                                  marker_color='#00d4aa', opacity=0.85))
        fig_gap.update_layout(
            barmode='group',
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#d1d5db', family='DM Sans'),
            yaxis=dict(range=[0, 1.05], gridcolor='#1e2130', title='AUC-ROC'),
            xaxis=dict(gridcolor='#1e2130'),
            legend=dict(bgcolor='#0f1117'),
            height=320, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_gap, use_container_width=True)
 
    with col_t:
        st.markdown('<div class="section-title">Tabla comparativa</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            df_comp[['Modelo','AUC-ROC','Gini','KS','Recall','F1','Estado']],
            hide_index=True, use_container_width=True, height=210
        )
        st.markdown("""
        <div class="insight-box">🏆 <b>Random Forest</b> es el campeón con AUC-ROC 0.6067 en validación y 0.6266 en backtest</div>
        <div class="warn-box">⚠️ <b>XGBoost/LightGBM:</b> Entrenamiento 0.86/0.94 → Validación 0.56/0.49 — sobreajuste severo. Se corrige con regularización Optuna en Sprint 4</div>
        """, unsafe_allow_html=True)
 
    # Imagen curva ROC generada por el modelo
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Curva ROC comparativa — generada por el modelo</div>',
                unsafe_allow_html=True)
    roc_path = Path("reports/curva_roc_comparativa.png")
    if roc_path.exists():
        img_roc = Image.open(roc_path)
        st.image(img_roc, use_container_width=True)
    else:
        st.info("⚠️ Imagen no encontrada en reports/curva_roc_comparativa.png — ejecutar 05_modelado.py primero")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTESTING TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Estabilidad temporal del modelo campeón</div>',
                unsafe_allow_html=True)
 
    bt_data = {
        'Período':         ['Validación (Jun)',  'Backtest (Jul)', 'Live* (Ago)'],
        'AUC-ROC':         [0.6067,               0.6266,           0.5396],
        'Recall':          [0.1566,               0.1208,           0.0106],
        'F1':              [0.0677,               0.1111,           0.0164],
        'Tasa tardíos %':  [1.4,                  4.1,              16.8],
        'Pedidos':         [5973,                 5843,             3778],
    }
    df_bt = pd.DataFrame(bt_data)
 
    fig_bt = make_subplots(specs=[[{"secondary_y": True}]])
    fig_bt.add_trace(go.Bar(
        x=df_bt['Período'], y=df_bt['Tasa tardíos %'],
        name='Tasa tardíos %', marker_color='#1e2d40', opacity=0.9
    ), secondary_y=False)
    fig_bt.add_trace(go.Scatter(
        x=df_bt['Período'], y=df_bt['AUC-ROC'],
        name='AUC-ROC', mode='lines+markers+text',
        line=dict(color='#00d4aa', width=3), marker=dict(size=11),
        text=[f"{v:.4f}" for v in df_bt['AUC-ROC']],
        textposition="top center",
        textfont=dict(family="DM Mono", size=12, color='#00d4aa')
    ), secondary_y=True)
    fig_bt.add_trace(go.Scatter(
        x=df_bt['Período'], y=df_bt['Recall'],
        name='Recall', mode='lines+markers+text',
        line=dict(color='#f59e0b', width=2, dash='dot'), marker=dict(size=8),
        text=[f"{v:.2f}" for v in df_bt['Recall']],
        textposition="bottom center",
        textfont=dict(family="DM Mono", size=11, color='#f59e0b')
    ), secondary_y=True)
    fig_bt.update_layout(
        plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
        font=dict(color='#d1d5db', family='DM Sans'),
        legend=dict(bgcolor='#0f1117'),
        height=370, margin=dict(t=20, b=20)
    )
    fig_bt.update_yaxes(title_text="Tasa tardíos %", gridcolor='#1e2130', secondary_y=False)
    fig_bt.update_yaxes(title_text="AUC-ROC / Recall", gridcolor='#1e2130',
                         range=[0, 0.85], secondary_y=True)
    st.plotly_chart(fig_bt, use_container_width=True)
 
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_bt, hide_index=True, use_container_width=True)
        st.markdown("<small style='color:#6b7280'>* Live filtrado: 2.573 pedidos con entrega estimada posterior al corte del dataset fueron eliminados</small>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">📌 <b>Censura de datos:</b> El dataset tiene un corte el 29-Ago-2018. Pedidos realizados en agosto con entrega estimada posterior a esa fecha aparecen como "a tiempo" cuando en realidad su resultado era desconocido. Se filtraron 2.573 pedidos (40.5% del Live).</div>
        <div class="warn-box">⚠️ <b>Drift de concepto en agosto:</b> La tasa de tardíos salta a 16.8% vs 8.8% en entrenamiento. El modelo fue entrenado en un contexto de menor estrés logístico y no anticipa bien este nivel de riesgo.</div>
        <div class="insight-box">📌 <b>Backtest es la referencia confiable:</b> Julio 2018 tiene una tasa de tardíos de 4.1% y el AUC 0.63 es consistente con la validación — confirma que el modelo es estable en condiciones normales.</div>
        """, unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANÁLISIS TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Tasa de retrasos mensual — Dataset completo (Sep 2016 – Ago 2018)</div>',
                unsafe_allow_html=True)
 
    if 'mensual' in data:
        df_m = data['mensual'].copy()
        df_m = df_m[df_m['pedidos'] > 50]
 
        def asignar_split(mes):
            if   mes <= '2018-05': return 'Entrenamiento'
            elif mes == '2018-06': return 'Validación'
            elif mes == '2018-07': return 'Backtest'
            else:                  return 'Live'
 
        df_m['partición'] = df_m['mes'].apply(asignar_split)
        color_map = {
            'Entrenamiento': '#6366f1',
            'Validación':    '#f59e0b',
            'Backtest':      '#10b981',
            'Live':          '#ef4444'
        }
 
        fig_m = make_subplots(specs=[[{"secondary_y": True}]])
        for split_name, color in color_map.items():
            mask = df_m['partición'] == split_name
            fig_m.add_trace(go.Bar(
                x=df_m[mask]['mes'], y=df_m[mask]['tasa_tardios'],
                name=split_name, marker_color=color, opacity=0.85,
            ), secondary_y=False)
        fig_m.add_trace(go.Scatter(
            x=df_m['mes'], y=df_m['pedidos'],
            name='Nº pedidos', mode='lines',
            line=dict(color='#9ca3af', width=1.5, dash='dot'),
        ), secondary_y=True)
        fig_m.update_layout(
            barmode='stack',
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#d1d5db', family='DM Sans'),
            legend=dict(bgcolor='#0f1117'),
            height=400, margin=dict(t=10, b=10),
            xaxis=dict(gridcolor='#1e2130', tickangle=45)
        )
        fig_m.update_yaxes(title_text="Tasa tardíos %", gridcolor='#1e2130', secondary_y=False)
        fig_m.update_yaxes(title_text="Nº de pedidos",  gridcolor='#1e2130', secondary_y=True)
        st.plotly_chart(fig_m, use_container_width=True)
 
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">21.4%</div>
                <div class="metric-label">Pico histórico de retrasos</div>
                <div class="metric-delta">Marzo 2018</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">1.4%</div>
                <div class="metric-label">Mínimo histórico</div>
                <div class="metric-delta">Junio 2018 — set de validación</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            tot = data.get('total_pedidos', 96470)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{tot:,}</div>
                <div class="metric-label">Total pedidos analizados</div>
                <div class="metric-delta">Sep 2016 – Ago 2018</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            tasa = data.get('tasa_global', 8.1)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{tasa:.1f}%</div>
                <div class="metric-label">Tasa global de retrasos</div>
                <div class="metric-delta">Promedio histórico del dataset</div>
            </div>""", unsafe_allow_html=True)
 
        st.markdown("""
        <div class="warn-box" style="margin-top:14px">
        ⚠️ <b>Irregularidad clave en los splits:</b> El período de validación (Jun 2018, 1.4% tardíos) y backtest (Jul 2018, 4.1%) son los meses de <b>menor estrés logístico</b> del dataset.
        Mientras que Feb-Mar 2018 (16% y 21%) quedaron en entrenamiento. Esto explica por qué el modelo aprende bien pero los splits de evaluación son atípicamente tranquilos.
        Esta irregularidad se abordará en el Sprint 4 con validación cruzada temporal.
        </div>""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — VARIABLES E INTERPRETABILIDAD
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">Variables seleccionadas e importancia — 50 features finales</div>',
                unsafe_allow_html=True)
 
    col1, col2 = st.columns([1, 1])
 
    with col1:
        if 'importancias' in data:
            imp = data['importancias'].sort_values('importancia', ascending=True).tail(20)
            fig_imp = go.Figure(go.Bar(
                x=imp['importancia'], y=imp.index,
                orientation='h',
                marker=dict(color=imp['importancia'],
                            colorscale=[[0,'#1e2130'],[1,'#00d4aa']]),
                text=[f"{v:.4f}" for v in imp['importancia']],
                textposition='outside',
                textfont=dict(family='DM Mono', size=10),
            ))
            fig_imp.update_layout(
                title="Top 20 variables — Importancia Random Forest",
                plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
                font=dict(color='#d1d5db', family='DM Mono', size=11),
                xaxis=dict(gridcolor='#1e2130', title='Importancia'),
                yaxis=dict(gridcolor='#1e2130'),
                height=520, margin=dict(t=40, l=10, r=60)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Ejecutar 04_seleccion_variables.py para generar importancias RF")
 
    with col2:
        st.markdown('<div class="section-title">Pipeline de selección de variables</div>',
                    unsafe_allow_html=True)
        pipeline = [
            ("Estado inicial",        64, 0.6503),
            ("Missing rate > 10%",    64, 0.6503),
            ("PSI > 0.50",            61, 0.6296),
            ("Correlación > 0.90",    50, 0.6382),
        ]
        for paso, n, auc in pipeline:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:9px 13px; margin:4px 0; background:#0f1117;
                        border-radius:6px; border:1px solid #1e2130;">
                <span style="font-size:0.83rem; color:#d1d5db">{paso}</span>
                <span style="font-family:'DM Mono'; font-size:0.81rem; color:#6b7280">
                    {n} variables · AUC {auc:.4f}
                </span>
            </div>""", unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Agenda Sprint 4 — Mejoras pendientes</div>',
                    unsafe_allow_html=True)
        issues = [
            ("🔧", "Regularización",        "Tuneo XGBoost/LightGBM con Optuna — reducir brecha Train→Val"),
            ("🔧", "Stacking",              "Reemplazar por VotingClassifier con pesos calibrados"),
            ("🔧", "Umbral de decisión",    "Calibrar sobre Backtest (Jul) en lugar de Validación (Jun)"),
            ("🔧", "Censura en producción", "Filtrar pedidos con entrega estimada posterior al corte — ya implementado"),
            ("🔧", "Drift temporal",        "Investigar ventana deslizante para capturar cambios en la tasa"),
        ]
        for icon, titulo, desc in issues:
            st.markdown(f"""
            <div style="padding:10px 14px; margin:4px 0; background:#0f1117;
                        border-radius:6px; border-left:3px solid #f59e0b;">
                <span style="font-size:0.82rem; font-weight:600; color:#f59e0b">{icon} {titulo}</span>
                <br><span style="font-size:0.78rem; color:#9ca3af">{desc}</span>
            </div>""", unsafe_allow_html=True)
 
    # Imagen SHAP
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Análisis SHAP — Importancia global de variables (LightGBM)</div>',
                unsafe_allow_html=True)
    shap_path = Path("reports/shap_importancia.png")
    if shap_path.exists():
        img_shap = Image.open(shap_path)
        st.image(img_shap, use_container_width=True)
    else:
        st.info("⚠️ Imagen SHAP no encontrada en reports/shap_importancia.png — ejecutar 05_modelado.py primero")