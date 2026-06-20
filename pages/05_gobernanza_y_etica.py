"""
pages/05_gobernanza_y_etica.py
=================================
Documentación de sesgos, limitaciones, privacidad y gobernanza
del modelo. 
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Gobernanza y Ética", page_icon="⚖️", layout="wide")

st.markdown("# ⚖️ Gobernanza y Ética")
st.markdown("**Transparencia sobre cómo se construyó el modelo, sus límites y riesgos**")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">Variables utilizadas (50)</div>', unsafe_allow_html=True)
    st.markdown("""
    Agrupadas en 6 categorías: temporales, geográficas, producto, pago,
    historial del vendedor (rolling, sin leakage) y score de riesgo compuesto.
    Ninguna variable usa datos personales identificables del comprador
    (nombre, email, teléfono, dirección exacta).
    """)

    st.markdown('<div class="section-title">Variables excluidas</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Geolocalización exacta** — se usó proxy por región/estado, no coordenadas
    - **Reviews del cliente** — serían data leakage (se generan después de la entrega)
    - **Fecha real de entrega y derivadas** — el modelo nunca ve el resultado
    - **Nombre, email, teléfono del comprador o vendedor**
    """)

with col2:
    st.markdown('<div class="section-title">Posibles sesgos identificados</div>', unsafe_allow_html=True)
    st.markdown("""
    **Sesgo geográfico:** estados con pocos pedidos históricos (ej. Amapá, Roraima)
    tienen tasas de retraso poco confiables estadísticamente pero el modelo las
    incorpora vía WOE encoding igual que estados con miles de pedidos.

    **Sesgo temporal:** el modelo aprendió de un período de fuerte crecimiento de
    Olist (2016–2018). Patrones de comportamiento actuales podrían diferir.

    **Sesgo de "vendedor nuevo":** vendedores sin historial reciben la tasa media
    global como imputación — puede penalizar o favorecer injustamente a quienes
    recién empiezan a vender en la plataforma.
    """)

    st.markdown('<div class="section-title">Riesgos del modelo</div>', unsafe_allow_html=True)
    st.markdown("""
    - Falsos positivos generan intervenciones innecesarias (costo operativo)
    - Falsos negativos dejan pasar retrasos reales sin intervención
    - Con umbral 0.50 la precisión es baja (~13%) — la mayoría de las alertas
      no se concretan en un retraso real
    """)

st.markdown("---")

st.markdown('<div class="section-title">Limitaciones declaradas</div>', unsafe_allow_html=True)
st.markdown("""
- El modelo se entrenó con datos de Sep 2016 a Mar 2018 y se validó hasta Jul 2018.
  No incorpora eventos posteriores a esa fecha.
- Agosto 2018 fue excluido completamente por censura de datos — el sistema no fue
  evaluado en ese período.
- El AUC-ROC en Live (0.66) es notablemente menor al de Validación (0.75) — el
  modelo pierde poder discriminativo en datos más recientes, posible señal de drift.
- Los costos de negocio usados para calcular ROI (R$45 por reclamo, R$8 por
  intervención) son supuestos académicos, no datos confirmados por Olist.
""")

st.markdown('<div class="section-title">Privacidad</div>', unsafe_allow_html=True)
st.markdown("""
El modelo opera exclusivamente sobre variables agregadas y categóricas de
ubicación (estado, no dirección), características del producto y comportamiento
transaccional. No se procesa ni almacena información de identificación personal
directa. El identificador de pedido y cliente se usa solo para trazabilidad
interna, nunca como variable predictiva.
""")

st.markdown('<div class="section-title">Gobernanza y monitoreo</div>', unsafe_allow_html=True)
st.markdown("""
- **Versionado:** el modelo se registra con número de versión (`v3.0.0`) y fecha
  de entrenamiento, trazable en `champion_resumen_v3.json`.
- **Monitoreo continuo:** se calcula PSI mensualmente sobre variables clave
  comparando contra la distribución de entrenamiento (ver página Monitor del Modelo).
- **Política de reentrenamiento:** si el PSI supera 0.25 en variables críticas,
  se recomienda reentrenar antes de continuar en producción.
- **Explicabilidad:** todas las predicciones pueden descomponerse en SHAP values
  o coeficientes de la regresión logística — el modelo es interpretable por diseño,
  no es una caja negra.
- **Responsable humano:** el modelo asiste la decisión, no la reemplaza. Toda
  intervención final (renegociar plazo, contactar vendedor) la ejecuta una persona.
""")

st.markdown("---")
st.caption("⚖️ Esta sección documenta honestamente las limitaciones del sistema. "
           "Un modelo de ML responsable reconoce lo que no sabe tanto como lo que sabe.")
