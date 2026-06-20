"""
pages/04_storytelling_demo_day.py
====================================
Narrativa ejecutiva del proyecto completo, para presentar a
directivos de Olist en el Demo Day.
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Storytelling Demo Day", page_icon="🎤", layout="wide")

st.markdown("# 🎤 Storytelling — Demo Day")
st.markdown("**La historia completa del proyecto, para presentar a directivos de Olist**")
st.markdown("---")

tabs = st.tabs(["1. Problema", "2. Hipótesis", "3. Hallazgos", "4. Construcción",
                 "5. Resultados", "6. Impacto", "7. Próximos pasos"])

with tabs[0]:
    st.markdown("""
    ## El problema de negocio

    Olist opera con una estimación de tiempo de entrega **genérica**, igual para
    todos los pedidos, sin considerar la dificultad real de cada operación.

    Esto genera tres consecuencias críticas:

    - **Promesas irreales** — se comprometen fechas que no corresponden a la realidad operativa
    - **Mala asignación de recursos** — pedidos simples y complejos se gestionan igual
    - **Reacción en lugar de prevención** — la empresa se en
    tera del retraso cuando el cliente ya reclamó

    De **89,802 pedidos analizados** (Sep 2016 – Jul 2018), el **7.9%** llegó tarde.
    Eso son más de 7,000 clientes insatisfechos sin ninguna intervención previa.
    """)

with tabs[1]:
    st.markdown("""
    ## Hipótesis del Sprint 1 — todas con respaldo empírico final

    **H1 — La complejidad logística, no la distancia, define el riesgo**
    Confirmada: `seller_state`, `customer_state` y `same_state` superan en importancia
    a la distancia pura en el modelo final.

    **H2 — La promesa de entrega es irreal desde el origen**
    Confirmada: `estimated_delivery_days` e `is_short_promise` están en el top 7
    de variables más importantes.

    **H3 — El día y hora de compra impactan el resultado**
    Confirmada parcialmente: `is_festivo` e `is_peak_season` tienen peso moderado.

    **H4 — Existen perfiles de riesgo diferenciados**
    Confirmada con fuerza: `tiene_varios_vendedores` es la variable **#1** del modelo.
    """)

with tabs[2]:
    st.markdown("""
    ## Hallazgos y casuísticas encontradas

    **Censura de datos:** el dataset se extrajo el 29-Ago-2018. Pedidos de agosto con
    entrega estimada posterior a ese corte tenían target inválido — se eliminaron
    completamente para no contaminar el modelo.

    **Irregularidad en los splits temporales:** los meses elegidos para evaluación
    (Abr–Jul 2018) resultaron ser los más tranquilos del dataset, mientras que los
    picos de retraso (Feb 16%, Mar 21%) quedaron en entrenamiento. Aunque Junio tuvo el
    % de tardio más bajo de todo el historial de entregas con retraso solo 1.4%.

    **Sesgo geográfico real:** Ceará tiene 48.6% de tardíos pero con solo 177 pedidos
    — poco confiable estadísticamente. Río de Janeiro con 36.4% y 1,743 pedidos
    sí es una señal robusta.

    **São Paulo como hub saturado:** concentra el mayor volumen de vendedores pero
    también una tasa de retraso no despreciable (11–20% según el corte) — confirma
    que el problema no es solo geográfico sino de capacidad operativa.
    """)

with tabs[3]:
    st.markdown("""
    ## Construcción del sistema

    **Feature Engineering:** 67 variables iniciales en 6 grupos (temporales, geográficas,
    producto, pago, historial del vendedor, score de riesgo compuesto), reducidas a
    **50 variables finales** tras un pipeline de selección estadística (Missing → PSI ≤0.15 → Correlación).

    **Sin data leakage:** el historial del vendedor se calculó con ventana rolling
    expanding — solo información pasada, nunca el resultado del pedido actual ni futuros.

    **Modelado:** se evaluaron 4 algoritmos (Regresión Logística, Random Forest,
    LightGBM, XGBoost). Se tunearon con Optuna (200 trials, CV=5, penalización
    de sobreajuste) los dos más prometedores.
    """)

with tabs[4]:
    st.markdown("""
    ## Resultados del modelo campeón

    **Modelo:** Regresión Logística Tuneada (WOE Encoding)

    | Métrica | Train | Validación | Backtest | Live |
    |---|---|---|---|---|
    | AUC-ROC | 0.72 | 0.75 | 0.74 | 0.66 |
    | Recall  | 0.65 | 73.8% | 34.9% | 50.4% |

    **¿Por qué Regresión Logística y no un modelo más complejo?**
    Es el único modelo sin sobreajuste — generaliza mejor de lo que memoriza.
    Random Forest llegó a AUC 1.0 en entrenamiento (memorización total).
    LightGBM y XGBoost mostraron brechas de 0.17 a 0.29 entre entrenamiento y validación.
    """)

with tabs[5]:
    st.markdown("""
    ## Impacto esperado para Olist

    - Detección temprana de pedidos en riesgo **antes** de que salgan del depósito
    - Posibilidad de ajustar la promesa de entrega mostrada al cliente de forma proactiva
    - Alertas a vendedores con mal historial para intervención preventiva
    - Reducción potencial de reclamos y compensaciones reactivas

    **Uso recomendado:** panel interno de operaciones y vendedores — **no** debe
    mostrarse al comprador final. El 90% de los compradores de Olist son de compra
    única, por lo que no aplica un modelo de relación de largo plazo con el cliente,
    y exponer una "probabilidad de incumplimiento" generaría ansiedad sin una acción clara para él.
    """)

with tabs[6]:
    st.markdown("""
    ## Próximos pasos

    - Validar los supuestos de costo de negocio (R$45 por reclamo, R$8 por intervención)
      con el equipo real de operaciones.
    - Investigar ventana deslizante para capturar mejor el drift estacional.
    - Monitorear el PSI mensualmente y reentrenar si supera los límites establecidos.
    - Expandir el sistema a un modelo de scoring escalonado (alerta leve vs intervención profunda)
    - Evaluar la integración con el sistema de checkout para ajuste dinámico de promesas
    """)