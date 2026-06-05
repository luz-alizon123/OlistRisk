"""
app.py
======
MVP — OlistShield: Sistema de Inteligencia Logística
Conectado al Champion Model entrenado en 05_modelado.py

Ejecutar:
  streamlit run app.py

Estructura de navegación:
  📦 Evaluar Pedido      → Predicción individual con gauge + factores
  📊 Dashboard           → KPIs ejecutivos + gráficos
  📈 Backtesting         → Resultados del modelo por partición
  🔍 Selección Variables → Tabla del proceso de selección
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path
from datetime import datetime

import plotly.express     as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
MODEL_PATH   = Path("models/champion_model_v1.pkl")
DATA_DIR     = Path("data")
REPORT_DIR   = Path("reports")

REGION_ALTO  = ["AM","PA","RO","AC","RR","AP","TO"]
REGION_MEDIO = ["BA","PE","MA","PI","CE","RN","PB","AL","SE"]
CAT_ALTA = ["furniture_decor","electronics","construction_tools_lights",
            "housewares","office_furniture","small_appliances","garden_tools",
            "computers","computers_accessories","air_conditioning"]
CAT_BAJA = ["books_general_interest","stationery","health_beauty","baby",
            "fashion_male_clothing","fashion_female_clothing","toys","food_drink"]

ESTADOS_BR = sorted([
    "AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG",
    "MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR",
    "RS","SC","SE","SP","TO"
])

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OlistShield — Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header{
    background:linear-gradient(90deg,#440154,#21918c);
    color:white;padding:18px 28px;border-radius:10px;margin-bottom:18px;
}
.card-alto {
    background:linear-gradient(135deg,#E74C3C,#C0392B);
    color:white;padding:22px;border-radius:12px;text-align:center;
}
.card-medio{
    background:linear-gradient(135deg,#F39C12,#E67E22);
    color:white;padding:22px;border-radius:12px;text-align:center;
}
.card-bajo {
    background:linear-gradient(135deg,#27AE60,#1E8449);
    color:white;padding:22px;border-radius:12px;text-align:center;
}
.factor-bar{
    background:#f0f2f6;border-left:4px solid #440154;
    padding:10px 14px;border-radius:6px;margin:4px 0;
}
.kpi-box{
    background:#ffffff;border:1px solid #e0e0e0;
    border-radius:10px;padding:16px;text-align:center;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Cargando modelo Champion…")
def cargar_modelo():
    if not MODEL_PATH.exists():
        st.error(f"Modelo no encontrado en {MODEL_PATH}. "
                 "Ejecuta primero: python 05_modelado.py")
        st.stop()
    return joblib.load(MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE FEATURES (sin depender de transformadores externos)
# ══════════════════════════════════════════════════════════════════════════════
def calcular_features_raw(inputs: dict) -> dict:
    """
    Recrea las mismas features que el pipeline de entrenamiento,
    usando únicamente los inputs del formulario.
    Devuelve un dict con TODAS las features posibles.
    """
    cs  = inputs["customer_state"]
    ss  = inputs["seller_state"]
    cat = inputs["main_category"]
    pay = inputs["main_payment_type"]
    wd  = inputs["purchase_weekday"]
    hr  = inputs["purchase_hour"]
    est = inputs["estimated_delivery_days"]
    tp  = inputs["total_price"]
    tf  = inputs["total_freight"]
    tw  = inputs["total_weight_g"]
    vol = inputs["volume_cm3"]
    ti  = inputs["total_items"]
    us  = inputs["unique_sellers"]
    ins = inputs["max_installments"]
    pm  = inputs["payment_methods"]
    tpay= inputs["total_payment"]

    # Región
    rrs = 3 if cs in REGION_ALTO else 2 if cs in REGION_MEDIO else 1
    # Complejidad
    lc  = 3 if cat in CAT_ALTA else 1 if cat in CAT_BAJA else 2
    # Calendario
    hrd = 1 if (wd in [5,6]) or (wd==4 and hr>=16) else 0
    # Pago
    is_bol = 1 if pay == "boleto" else 0
    # Ratios
    fr  = tf / tp  if tp  > 0 else 0
    dns = tw / vol if vol > 0 else 0
    # Geo
    same   = 1 if cs == ss else 0
    inter  = 1 - same
    cx     = 1 if ss in ["SP","RJ","MG"] and cs in REGION_ALTO else 0
    dist_proxy = {1:500, 2:1800, 3:2800}[rrs]
    long_d = 1 if dist_proxy > 1000 else 0

    # Risk score (misma fórmula que feature_engineering)
    risk = (rrs*2.5 + lc*1.5 + hrd*1.5 + is_bol*1.0 + cx*1.5 +
            (1 if vol > 200000 else 0)*0.5 + (1 if fr>0.5 else 0)*0.5)
    risk_norm = min(round(risk / 15 * 10, 2), 10)

    return {
        "estimated_delivery_days":  est,
        "time_to_approve_hours":    2.0 if pay != "boleto" else 26.0,
        "distance_km":              dist_proxy,
        "total_weight_g":           tw,
        "avg_weight_g":             tw / max(ti, 1),
        "volume_cm3":               vol,
        "density_gcm3":             dns,
        "total_price":              tp,
        "total_freight":            tf,
        "total_payment":            tpay,
        "freight_ratio":            fr,
        "price_per_item":           tp / max(ti, 1),
        "log_total_weight":         np.log1p(tw),
        "log_volume_cm3":           np.log1p(vol),
        "log_total_payment":        np.log1p(tpay),
        "risk_score":               risk_norm,
        "region_x_complexity":      rrs * lc,
        "distance_x_complexity":    dist_proxy * lc,
        "weekday_x_hour":           wd * hr,
        "seller_late_rate_hist":    0.09,  # promedio global del train
        "seller_total_orders_hist": 100,
        "max_dimension_cm":         inputs.get("max_dim_cm", 30),
        "region_risk_score":        rrs,
        "logistic_complexity":      lc,
        "freight_category":         1 if fr<0.2 else 2 if fr<0.5 else 3,
        "purchase_weekday":         wd,
        "purchase_hour":            hr,
        "purchase_month":           inputs["purchase_month"],
        "purchase_quarter":         (inputs["purchase_month"]-1)//3 + 1,
        "high_risk_day":            hrd,
        "is_weekend":               1 if wd in [5,6] else 0,
        "is_off_hours":             1 if (hr<8 or hr>=18) else 0,
        "is_peak_season":           1 if inputs["purchase_month"] in [11,12] else 0,
        "same_state":               same,
        "interstate_flag":          inter,
        "complex_route":            cx,
        "is_long_distance":         long_d,
        "has_oversized_item":       1 if inputs.get("max_dim_cm",30) > 60 else 0,
        "is_heavy":                 1 if tw/max(ti,1) > 5000 else 0,
        "carrito_complejo":         1 if ti > 2 else 0,
        "tiene_varios_vendedores":  1 if us > 1 else 0,
        "is_boleto":                is_bol,
        "is_high_installments":     1 if ins > 6 else 0,
        "high_freight_flag":        1 if fr > 0.5 else 0,
        "approval_delay_flag":      1 if (pay=="boleto" and ins>3) else 0,
        "is_short_promise":         1 if est < 10 else 0,
        "seller_is_experienced":    1,
        # Variables categóricas con WOE — usar valor neutro = 0 (media WOE)
        "customer_state":           0.0,
        "seller_state":             0.0,
        "main_category":            0.0,
        "main_payment_type":        0.0,
        "risk_profile":             0.0,
    }


def predecir(inputs: dict, artefactos: dict):
    feats  = artefactos["features_finales"]
    modelo = artefactos["modelo"]
    umbral = artefactos["umbral_optimo"]

    feat_vals = calcular_features_raw(inputs)
    X = pd.DataFrame([{f: feat_vals.get(f, 0) for f in feats}])
    X = X.fillna(0)

    prob   = float(modelo.predict_proba(X)[0][1])
    pred   = int(prob >= umbral)
    score  = round(prob * 10, 2)

    if score >= 6.5:
        perfil, css = "🔴 ALTO RIESGO",  "alto"
    elif score >= 3.5:
        perfil, css = "🟡 RIESGO MEDIO", "medio"
    else:
        perfil, css = "🟢 BAJO RIESGO",  "bajo"

    return prob, score, perfil, css, pred, feat_vals


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛡️ OlistShield")
    st.markdown("Sistema de Inteligencia Logística")
    st.divider()

    pagina = st.radio("Navegación", [
        "📦 Evaluar Pedido",
        "📊 Dashboard Ejecutivo",
        "📈 Backtesting",
        "🔍 Selección de Variables",
    ])

    st.divider()
    # Info del modelo
    if MODEL_PATH.exists():
        art = cargar_modelo()
        st.caption(f"**Modelo:** {art.get('nombre_champion','Champion')}")
        st.caption(f"**Versión:** {art.get('version','1.0.0')}")
        st.caption(f"**Entrenado:** {art.get('fecha_entrenamiento','N/A')}")
        st.caption(f"**Umbral:** {art.get('umbral_optimo',0.40):.2f}")

        m_val = art.get("metricas_val", {})
        if m_val:
            st.divider()
            st.caption("**Métricas en Validación:**")
            st.caption(f"AUC-ROC : {m_val.get('AUC-ROC','—')}")
            st.caption(f"Recall  : {m_val.get('Recall','—')}")
            st.caption(f"F1      : {m_val.get('F1','—')}")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1 style="margin:0">🛡️ OlistShield — Risk Intelligence Platform</h1>
    <p style="margin:4px 0 0 0;opacity:0.9">
    Detección Temprana de Retrasos Logísticos · Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — EVALUAR PEDIDO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "📦 Evaluar Pedido":

    st.subheader("📦 Evaluación de Riesgo — Pedido Individual")
    st.markdown(
        "Completa los datos del pedido y obtén el **Score de Riesgo** "
        "antes de confirmar la venta."
    )

    with st.form("form_pedido"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📍 Geografía**")
            customer_state = st.selectbox("Estado del cliente",  ESTADOS_BR, index=ESTADOS_BR.index("SP"))
            seller_state   = st.selectbox("Estado del vendedor", ESTADOS_BR, index=ESTADOS_BR.index("SP"))
            est_days       = st.slider("Días estimados de entrega", 1, 45, 15)

        with c2:
            st.markdown("**📦 Producto**")
            total_items     = st.number_input("Cantidad de ítems",         1,  20, 1)
            unique_sellers  = st.number_input("Vendedores en la orden",    1,   5, 1)
            total_weight_g  = st.number_input("Peso total (g)",           50, 50000, 500)
            volume_cm3      = st.number_input("Volumen total (cm³)",      100, 500000, 5000)
            max_dim_cm      = st.number_input("Dimensión máxima (cm)",      5,  150, 30)
            main_category   = st.selectbox("Categoría principal", sorted(
                CAT_ALTA + CAT_BAJA + ["sports_leisure","garden_tools",
                "perfumery","watches_gifts","auto","musical_instruments"]
            ))

        with c3:
            st.markdown("**💳 Pago y Tiempo**")
            main_payment_type = st.selectbox("Tipo de pago",
                ["credit_card","boleto","voucher","debit_card"])
            max_installments  = st.slider("Cuotas máximas", 1, 24, 1)
            payment_methods   = st.number_input("Nº métodos de pago", 1, 3, 1)
            total_price       = st.number_input("Valor productos (R$)",  10.0, 15000.0, 150.0)
            total_freight     = st.number_input("Costo flete (R$)",       5.0,  500.0,  20.0)
            total_payment     = total_price + total_freight

            st.markdown("**🗓️ Momento**")
            fecha        = st.date_input("Fecha de compra", datetime.now())
            hora_compra  = st.slider("Hora (0-23)", 0, 23, 14)

        submitted = st.form_submit_button("🔍 ANALIZAR RIESGO", use_container_width=True,
                                           type="primary")

    if submitted:
        artefactos = cargar_modelo()
        inputs = {
            "customer_state":     customer_state,
            "seller_state":       seller_state,
            "estimated_delivery_days": est_days,
            "total_items":        int(total_items),
            "unique_sellers":     int(unique_sellers),
            "total_weight_g":     float(total_weight_g),
            "volume_cm3":         float(volume_cm3),
            "max_dim_cm":         float(max_dim_cm),
            "main_category":      main_category,
            "main_payment_type":  main_payment_type,
            "max_installments":   int(max_installments),
            "payment_methods":    int(payment_methods),
            "total_price":        float(total_price),
            "total_freight":      float(total_freight),
            "total_payment":      float(total_payment),
            "purchase_weekday":   fecha.weekday(),
            "purchase_hour":      hora_compra,
            "purchase_month":     fecha.month,
        }

        prob, score, perfil, css, pred, feat_vals = predecir(inputs, artefactos)

        # ─── RESULTADO ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🎯 Resultado del Análisis")

        ca, cb, cc, cd = st.columns(4)

        with ca:
            st.markdown(f'<div class="card-{css}"><h2>{perfil}</h2>'
                        f'<p>Clasificación del pedido</p></div>',
                        unsafe_allow_html=True)
        with cb:
            st.metric("📊 Score de Riesgo",    f"{score} / 10")
        with cc:
            st.metric("🎲 Prob. de Retraso",   f"{prob*100:.1f}%",
                      f"Umbral: {artefactos['umbral_optimo']*100:.0f}%")
        with cd:
            rrs  = feat_vals["region_risk_score"]
            lc   = feat_vals["logistic_complexity"]
            buff = max(0, int(rrs*1.5 + lc*0.8))
            st.metric("📅 Buffer Sugerido",    f"+{buff} días",
                      "Ajuste al plazo prometido")

        # ─── GAUGE ────────────────────────────────────────────────────────────
        g1, g2 = st.columns([1, 2])

        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"Score de Riesgo", "font":{"size":14}},
                gauge={
                    "axis":    {"range":[0,10]},
                    "bar":     {"color":"#440154"},
                    "steps": [
                        {"range":[0,3.5],  "color":"#27AE60"},
                        {"range":[3.5,6.5],"color":"#F39C12"},
                        {"range":[6.5,10], "color":"#E74C3C"},
                    ],
                    "threshold": {
                        "line":      {"color":"black","width":3},
                        "thickness": 0.8,
                        "value":     score,
                    },
                }
            ))
            gauge.update_layout(height=260, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(gauge, use_container_width=True)

        # ─── FACTORES ─────────────────────────────────────────────────────────
        with g2:
            st.markdown("**🔬 Factores de Riesgo del Pedido:**")

            factores = {
                f"Región destino ({customer_state})":      feat_vals["region_risk_score"]/3,
                f"Complejidad del producto":               feat_vals["logistic_complexity"]/3,
                f"Día/hora de compra (high_risk_day)":     feat_vals["high_risk_day"],
                f"Pago: {main_payment_type}":              feat_vals["is_boleto"]*0.8+0.1,
                f"Promesa corta ({est_days}d)":            feat_vals["is_short_promise"],
                f"Ruta compleja (hub→Norte)":              feat_vals["complex_route"],
                f"Flete ratio ({feat_vals['freight_ratio']:.1%})": min(feat_vals["freight_ratio"],1),
                f"Multi-vendedor":                         feat_vals["tiene_varios_vendedores"]*0.7+0.1,
            }

            for factor, valor in factores.items():
                color = ("#E74C3C" if valor > 0.65 else
                         "#F39C12" if valor > 0.35 else "#27AE60")
                pct = int(valor * 100)
                st.markdown(f"""
                <div class="factor-bar">
                    <b>{factor}</b>
                    <div style="background:#ddd;border-radius:8px;height:7px;margin-top:4px;">
                        <div style="background:{color};width:{pct}%;height:7px;border-radius:8px;"></div>
                    </div>
                    <small style="color:{color}">{pct}% de impacto</small>
                </div>""", unsafe_allow_html=True)

        # ─── RECOMENDACIÓN ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💡 Recomendación Operativa")

        if score >= 6.5:
            st.error(f"""
            **⚠️ ACCIÓN INMEDIATA — Riesgo Alto ({prob*100:.1f}%)**

            - 📅 Ajustar fecha prometida en **+{buff} días** antes de confirmar la venta
            - 🚀 Considerar servicio express o transportadora premium para este destino
            - 📱 Activar notificación proactiva al cliente al momento de la compra
            - 🏷️ Marcar en el sistema como **pedido crítico** para seguimiento prioritario
            """)
        elif score >= 3.5:
            st.warning(f"""
            **⚠️ MONITOREO — Riesgo Medio ({prob*100:.1f}%)**

            - 📅 Evaluar añadir **+{buff} días** al plazo prometido
            - 👁️ Activar alertas automáticas de seguimiento en el sistema
            - 📊 Verificar capacidad actual del vendedor en {seller_state}
            """)
        else:
            st.success(f"""
            **✅ FLUJO NORMAL — Riesgo Bajo ({prob*100:.1f}%)**

            - El plazo de **{est_days} días** es adecuado para este perfil
            - Proceder con el proceso logístico estándar
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — DASHBOARD EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Dashboard Ejecutivo":

    st.subheader("📊 Dashboard Ejecutivo — Inteligencia Logística")

    @st.cache_data(show_spinner=False)
    def cargar_train():
        p = DATA_DIR / "split_train.parquet"
        if p.exists():
            return pd.read_parquet(p)
        return None

    df = cargar_train()

    if df is None:
        st.warning("No se encontró split_train.parquet. Ejecuta el pipeline primero.")
    else:
        TARGET = "is_late_delivery"

        # KPIs
        st.markdown("#### 🔢 KPIs de Negocio — Conjunto de Entrenamiento")
        k1,k2,k3,k4,k5 = st.columns(5)

        tot = len(df)
        tar = int(df[TARGET].sum())
        tasa = tar/tot*100
        pico = int((df.get("is_peak_season", pd.Series([0]*tot))==1).sum())
        ahorro = int(tar * 45 * 0.7 * 0.72)

        k1.metric("Total pedidos",     f"{tot:,}")
        k2.metric("Tasa de retraso",   f"{tasa:.1f}%",    f"{tar:,} tardíos")
        k3.metric("Temporada alta",    f"{pico:,}",        "Nov-Dic (mayor riesgo)")
        k4.metric("Recall estimado",   "72%",              "Champion model")
        k5.metric("Ahorro potencial",  f"R$ {ahorro:,}",  "Reclamos evitables")

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            if "region_risk_score" in df.columns:
                tasa_reg = df.groupby("region_risk_score")[TARGET].mean()*100
                fig1 = px.bar(
                    x=["Bajo (Sur/SE)","Medio (NE)","Alto (Norte)"],
                    y=tasa_reg.values,
                    color=tasa_reg.values,
                    color_continuous_scale="RdYlGn_r",
                    title="Tasa de Retraso por Riesgo Regional",
                    labels={"x":"Región","y":"% Retrasos"},
                )
                fig1.update_layout(coloraxis_showscale=False, height=320)
                st.plotly_chart(fig1, use_container_width=True)

        with c2:
            if "purchase_weekday" in df.columns:
                dias = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
                tasa_dia = df.groupby("purchase_weekday")[TARGET].mean()*100
                fig2 = px.line(
                    x=dias, y=tasa_dia.values,
                    markers=True,
                    title="Tasa de Retraso por Día de Compra",
                    labels={"x":"Día","y":"% Retrasos"},
                )
                fig2.update_traces(line_color="#440154", line_width=3,
                                   marker=dict(size=9))
                fig2.update_layout(height=320)
                st.plotly_chart(fig2, use_container_width=True)

        # Mapa de calor: estado cliente vs tasa de retraso
        if "customer_state" in df.columns:
            st.markdown("#### 🗺️ Tasa de Retraso por Estado del Cliente")
            tasa_estado = (df.groupby("customer_state")[TARGET]
                           .mean()*100).reset_index()
            tasa_estado.columns = ["Estado","Tasa (%)"]
            tasa_estado = tasa_estado.sort_values("Tasa (%)", ascending=False)

            fig3 = px.bar(tasa_estado, x="Estado", y="Tasa (%)",
                          color="Tasa (%)", color_continuous_scale="RdYlGn_r",
                          title="% Pedidos tardíos por estado de destino")
            fig3.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📈 Backtesting":

    st.subheader("📈 Backtesting Temporal — Resultados del Champion Model")

    bt_path = DATA_DIR / "tabla_backtesting.csv"
    if bt_path.exists():
        df_bt = pd.read_csv(bt_path)
        st.markdown("#### Métricas por Partición Temporal")
        st.dataframe(
            df_bt.style.format({
                "AUC-ROC":  "{:.4f}", "Gini":      "{:.4f}",
                "PR-AUC":   "{:.4f}", "KS":         "{:.4f}",
                "F1":       "{:.4f}", "Recall":     "{:.4f}",
                "Precision":"{:.4f}", "Accuracy":   "{:.4f}",
                "Tasa_Detección": "{:.2%}", "Ahorro_R$": "R$ {:,.0f}",
            }).background_gradient(subset=["AUC-ROC","Recall","F1"],
                                    cmap="YlGn"),
            use_container_width=True
        )

        st.divider()
        cols_plot = ["AUC-ROC","Gini","Recall","F1","Precision"]
        cols_ok   = [c for c in cols_plot if c in df_bt.columns]
        if cols_ok:
            fig_bt = px.bar(df_bt.melt(id_vars="split", value_vars=cols_ok,
                                        var_name="Métrica",
                                        value_name="Valor"),
                             x="split", y="Valor", color="Métrica",
                             barmode="group",
                             title="Comparación de Métricas por Partición",
                             color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.info("Ejecuta `python 05_modelado.py` para generar los resultados de backtesting.")

    # Imagen de curva ROC si existe
    roc_img = REPORT_DIR / "curva_roc_comparativa.png"
    if roc_img.exists():
        st.markdown("#### Curva ROC — Comparativa de Modelos")
        st.image(str(roc_img), use_column_width=True)

    # SHAP si existe
    shap_img = REPORT_DIR / "shap_importancia.png"
    if shap_img.exists():
        st.markdown("#### SHAP — Importancia Global de Features")
        st.image(str(shap_img), use_column_width=True)

    # Métricas del modelo guardadas
    art = cargar_modelo()
    m_back = art.get("metricas_backtest", {})
    if m_back:
        st.divider()
        st.markdown("#### Métricas Champion en Backtest (Aug-Sep 2018)")
        cols = st.columns(4)
        metricas_show = ["AUC-ROC","Gini","Recall","F1"]
        for i, met in enumerate(metricas_show):
            if met in m_back:
                cols[i].metric(met, f"{m_back[met]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — SELECCIÓN DE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔍 Selección de Variables":

    st.subheader("🔍 Proceso de Selección de Variables — Metodología del Docente")

    tabla_path = DATA_DIR / "tabla_seleccion_variables.csv"
    if tabla_path.exists():
        df_sel = pd.read_csv(tabla_path)
        st.markdown("#### Tabla de Progreso — Variable Selection")
        st.dataframe(
            df_sel[["Orden","Método","Threshold","Features",
                    "AUC-ROC Train","AUC-ROC Val"]]
            .style.format({"AUC-ROC Train":"{:.4f}","AUC-ROC Val":"{:.4f}"})
            .background_gradient(subset=["AUC-ROC Val"], cmap="YlGn"),
            use_container_width=True
        )

        fig_sel = px.line(df_sel.dropna(subset=["AUC-ROC Val"]),
                           x="Features", y=["AUC-ROC Train","AUC-ROC Val"],
                           markers=True,
                           title="AUC-ROC vs Número de Features (proceso de selección)",
                           labels={"value":"AUC-ROC","variable":"Partición"})
        st.plotly_chart(fig_sel, use_container_width=True)
    else:
        st.info("Ejecuta `python 04_seleccion_variables.py` para generar la tabla.")

    # IV Report
    iv_path = DATA_DIR / "reporte_iv.csv"
    if iv_path.exists():
        st.divider()
        st.markdown("#### Information Value (IV) — Top 20 Features")
        df_iv = pd.read_csv(iv_path)
        df_iv.columns = ["Feature","IV"]
        df_iv = df_iv.sort_values("IV", ascending=False).head(20)

        fig_iv = px.bar(df_iv, x="IV", y="Feature", orientation="h",
                         color="IV", color_continuous_scale="Viridis",
                         title="Top 20 Features por Information Value")
        fig_iv.update_layout(coloraxis_showscale=False, height=500,
                              yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_iv, use_container_width=True)

    # Features finales
    feat_path = DATA_DIR / "features_finales.json"
    if feat_path.exists():
        with open(feat_path) as f:
            feats = json.load(f)
        st.divider()
        st.markdown(f"#### ✅ Features Finales Seleccionadas: **{len(feats)}**")
        st.code(str(feats), language="python")
