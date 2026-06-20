"""
pages/02_simulador_pedido.py
==============================
FIX: usa categorías REALES de Olist (portugués), y agrega controles
de "Historial del vendedor" que SÍ modulan el riesgo de forma visible
— porque la categoría del producto, por un comportamiento real del
pipeline, no varía el score.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.predict import (
    cargar_champion, cargar_transformaciones, predecir_proba,
    score_a_nivel, construir_pedido_simulado, descomponer_riesgo,
    generar_recomendaciones, CATEGORIAS_REALES_OLIST,
)
from utils.plots import grafico_semaforo

st.set_page_config(page_title="Simulador de Pedido", page_icon="🎯", layout="wide")

st.markdown("# 🎯 Simulador de Pedido")
st.markdown("**Evaluá el riesgo de incumplimiento de un pedido antes de confirmarlo**")
st.caption("⚠️ Herramienta interna de operaciones — no se muestra al comprador final")
st.markdown("---")

art   = cargar_champion()
trans = cargar_transformaciones()

if trans is None:
    st.error("⚠️ No se pudo cargar transformaciones.pkl.")
    st.stop()

ESTADOS_BR = ["SP","RJ","MG","RS","PR","SC","BA","GO","PE","CE","DF","ES",
              "PA","MT","MS","MA","PB","PI","RN","AL","SE","TO","RO","AM","AC","RR","AP"]

col_form, col_result = st.columns([1, 1.2])

with col_form:
    st.markdown('<div class="section-title">Parámetros del pedido</div>', unsafe_allow_html=True)

    customer_state = st.selectbox("Estado del cliente (destino)", ESTADOS_BR, index=0)
    seller_state   = st.selectbox("Estado del vendedor (origen)", ESTADOS_BR,
                                   index=ESTADOS_BR.index("AM"))
    main_category  = st.selectbox("Categoría del producto", CATEGORIAS_REALES_OLIST, index=3)

    peso_g = st.slider("Peso del paquete (gramos)", 100, 30000, 12000, step=100)
    n_items = st.slider("Número de ítems en el pedido", 1, 10, 3)

    estimated_delivery_days = st.slider("Promesa de entrega (días)", 1, 60, 7)

    col1, col2 = st.columns(2)
    with col1:
        dia_semana = st.selectbox("Día de la semana",
            ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"], index=4)
    with col2:
        hora_compra = st.slider("Hora de compra", 0, 23, 18)
    purchase_weekday = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"].index(dia_semana)

    main_payment_type = st.selectbox("Método de pago",
        ["credit_card","boleto","voucher","debit_card"], index=1)
    max_installments = st.slider("Número de cuotas", 1, 24, 1)
    price_per_item   = st.number_input("Precio por ítem (R$)", 10.0, 5000.0, 800.0, step=10.0)

    with st.expander("⚙️ Historial del vendedor (variable de mayor impacto en SHAP)"):
        seller_late_rate = st.slider(
            "Tasa histórica de retraso del vendedor", 0.0, 1.0, 0.111, step=0.01,
            help="0.111 = media global (vendedor sin historial). Valores altos "
                 "simulan un vendedor con mal historial de entregas."
        )
        seller_orders = st.slider(
            "Pedidos previos del vendedor", 0, 200, 0,
            help="0 = vendedor nuevo sin historial"
        )

    evaluar = st.button("🔍 Evaluar Riesgo", type="primary", use_container_width=True)


with col_result:
    st.markdown('<div class="section-title">Firma de riesgo</div>', unsafe_allow_html=True)

    if evaluar:
        X_sim, factores = construir_pedido_simulado(
            customer_state, seller_state, main_category, peso_g, n_items,
            estimated_delivery_days, purchase_weekday, hora_compra,
            main_payment_type, max_installments, price_per_item,
            trans,
            seller_late_rate_hist=seller_late_rate,
            seller_total_orders_hist=seller_orders,
        )

        with st.expander("🔧 Debug — ver vector de entrada al modelo"):
            st.dataframe(X_sim.T, use_container_width=True)

        proba = predecir_proba(X_sim)[0]
        resultado = score_a_nivel(proba)

        st.plotly_chart(grafico_semaforo(resultado["score"], resultado["nivel"]),
                        use_container_width=True)

        badge_color = resultado["color"]
        st.markdown(f"""
        <div style="text-align:center; margin-top:-10px;">
            <span style="background:{badge_color}22; color:{badge_color};
                  padding:6px 18px; border-radius:20px; font-weight:500;
                  font-size:0.95rem;">
                Riesgo {resultado['nivel']}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.6rem">{resultado['probabilidad']*100:.1f}%</div>
                <div class="metric-label">Prob. de incumplimiento</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            plazo_ajustado = estimated_delivery_days + (2 if resultado["score"]>50 else 0)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.6rem">{plazo_ajustado} días</div>
                <div class="metric-label">Plazo ajustado sugerido</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Descomposición del riesgo</div>', unsafe_allow_html=True)
        descomp = descomponer_riesgo(factores)
        for nombre, valor in descomp.items():
            color_barra = ("#1D9E75" if valor<=30 else "#EF9F27" if valor<=60 else "#D85A30")
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
                    <span>{nombre}</span><span style="color:{color_barra}; font-weight:500">{valor:.0f}</span>
                </div>
                <div style="background:#1e2130; border-radius:6px; height:8px;">
                    <div style="background:{color_barra}; width:{valor}%; height:8px; border-radius:6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0f1117; border-left:3px solid #00d4aa;
             border-radius:0 8px 8px 0; padding:11px 15px;">
        📅 <b>Días colchón recomendados:</b> {2 if resultado['score']>50 else 0} días<br>
        📦 <b>Distancia estimada:</b> {factores['distance_km']} km<br>
        🔄 <b>Misma región origen-destino:</b> {"Sí" if customer_state==seller_state else "No"}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recomendaciones</div>', unsafe_allow_html=True)
        for r in generar_recomendaciones(resultado["score"], factores, descomp):
            st.markdown(f"- {r}")
    else:
        st.info("👈 Completá los parámetros del pedido y presioná **Evaluar Riesgo**")
        st.markdown("""
        **💡 Probá estos dos extremos para ver el rango completo:**

        **Riesgo bajo:** Cliente `SP`, vendedor `SP`, promesa `25 días`,
        lunes `10hs`, tarjeta de crédito, vendedor con `100` pedidos previos
        y tasa de retraso `0.05`.

        **Riesgo alto:** Cliente `PA`, vendedor `SP`, promesa `5 días`,
        viernes `18hs`, boleto, vendedor con `0` pedidos previos
        y tasa de retraso `0.40`.
        """)
