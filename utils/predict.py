"""
utils/predict.py
=================
Se documenta esto en la UI para que sea transparente.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

MODEL_PATH = Path("models/champion_model_v3.pkl")
DATA_DIR   = Path("data")

# Mismas columnas numéricas escaladas en 03_preparar_datos.py
VARS_NUMERICAS = [
    "estimated_delivery_days", "distance_km",
    "total_weight_g", "avg_weight_g",
    "volume_cm3", "density_gcm3",
    "total_price", "total_freight", "total_payment",
    "freight_ratio", "price_per_item", "freight_per_kg",
    "log_total_weight", "log_volume_cm3", "log_total_payment",
    "risk_score", "region_x_complexity", "distance_x_complexity",
    "weekday_x_hour", "seller_late_rate_hist",
    "seller_total_orders_hist", "max_dimension_cm",
    "approval_ratio",
]

# Categorías EXACTAS del código real (02_feature_engineering.py línea 279-292)
# — están en inglés y NO matchean con las categorías reales en portugués
CAT_ALTA_CODIGO = [
    "furniture_decor", "home_confort", "home_comfort", "electronics",
    "construction_tools_lights", "housewares", "office_furniture",
    "small_appliances", "air_conditioning", "computers", "garden_tools"
]
CAT_BAJA_CODIGO = [
    "books_general_interest", "stationery", "arts_and_craftmanship",
    "perfumery", "fashion_male_clothing", "fashion_female_clothing",
    "toys", "food_drink", "baby", "health_beauty"
]

# Categorías REALES de Olist (portugués) — para el selector del simulador.
# Se muestran al usuario pero, igual que en el modelo entrenado, SIEMPRE
# producen logistic_complexity=2 porque no matchean el código en inglés.
CATEGORIAS_REALES_OLIST = [
    "beleza_saude", "relogios_presentes", "esporte_lazer",
    "moveis_decoracao", "fashion_calcados", "brinquedos",
    "consoles_games", "utilidades_domesticas", "telefonia_fixa",
    "climatizacao", "automotivo", "perfumaria", "cool_stuff",
    "cama_mesa_banho", "informatica_acessorios", "moveis_escritorio",
    "eletrodomesticos", "construcao_ferramentas_construcao",
]

# Mapeo geográfico EXACTO (02_feature_engineering.py línea 238-244)
REGION_ALTO  = ["AM", "PA", "RO", "AC", "RR", "AP", "TO"]
REGION_MEDIO = ["BA", "PE", "MA", "PI", "CE", "RN", "PB", "AL", "SE"]
HUBS_SP_RJ_MG = ["SP", "RJ", "MG"]
DIST_PROXY = {1: 500, 2: 1800, 3: 2800}

# risk_score: suma_max teórica con flags en 1 y scores en su máximo (3,3)
# region_risk_score(3) * 2.5 + logistic_complexity(3) * 1.5 + flags(1 c/u)
# = 7.5 + 4.5 + 2.5 + 1.5 + 1.0 + 1.0 + 1.5 + 0.5 + 0.5 = 20.5
# El código real usa df["risk_score"].max() DEL DATASET DE ENTRENAMIENTO
# completo (no de esta fila), que no podemos conocer exactamente desde
# el simulador. Verificado empíricamente en la muestra real: el máximo
# observado en el dataset completo es ~19.03 (no el teórico 20.5).
RISK_SCORE_MAX_DATASET = 19.03


@st.cache_resource
def cargar_champion():
    if not MODEL_PATH.exists():
        st.error(f"No se encontró {MODEL_PATH}.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_resource
def cargar_transformaciones():
    path = DATA_DIR / "transformaciones.pkl"
    if not path.exists():
        st.warning("⚠️ No se encontró transformaciones.pkl")
        return None
    return joblib.load(path)


def predecir_proba(X: pd.DataFrame) -> np.ndarray:
    art      = cargar_champion()
    modelo   = art["modelo"]
    features = art["features_finales"]
    for f in features:
        if f not in X.columns:
            X[f] = 0
    return modelo.predict_proba(X[features].fillna(0))[:, 1]


def score_a_nivel(proba: float) -> dict:
    score = round(proba * 100, 1)
    if score <= 30:
        nivel, color = "Bajo", "#1D9E75"
    elif score <= 60:
        nivel, color = "Medio", "#EF9F27"
    elif score <= 80:
        nivel, color = "Alto", "#D85A30"
    else:
        nivel, color = "Crítico", "#E24B4A"
    return {"score": score, "nivel": nivel, "color": color,
            "probabilidad": round(proba, 4)}


def construir_pedido_simulado(
    customer_state: str, seller_state: str, main_category: str,
    peso_g: float, n_items: int, estimated_delivery_days: int,
    purchase_weekday: int, purchase_hour: int,
    main_payment_type: str, max_installments: int, price_per_item: float,
    transformaciones: dict,
    avg_length_cm: float = 25.0, avg_height_cm: float = 18.0,
    avg_width_cm: float = 22.0,
    time_to_approve_hours: float = 18.0,
    mediana_aprob_global: float = 12.0,
    unique_sellers: int = 1,
    seller_late_rate_hist: float = 0.111,
    seller_total_orders_hist: int = 0,
) -> tuple:
    """
    Reconstruye un pedido EXACTAMENTE con la misma lógica de
    02_feature_engineering.py, paso a paso, sin inventar fórmulas.
    """
    woe_maps = transformaciones.get("woe_maps", {}) if transformaciones else {}
    scaler   = transformaciones.get("scaler")       if transformaciones else None
    limites  = transformaciones.get("limites", {})  if transformaciones else {}

    # ── GEOGRÁFICAS (líneas 241-256 del código real) ───────────────────────
    same_state = 1 if customer_state == seller_state else 0
    interstate_flag = 1 - same_state

    if customer_state in REGION_ALTO:
        region_risk_score = 3
    elif customer_state in REGION_MEDIO:
        region_risk_score = 2
    else:
        region_risk_score = 1

    complex_route = 1 if (seller_state in HUBS_SP_RJ_MG and
                          customer_state in REGION_ALTO) else 0

    distance_km = DIST_PROXY[region_risk_score]
    is_long_distance = 1 if distance_km > 1000 else 0

    # ── PRODUCTO (líneas 266-297) ───────────────────────────────────────────
    volume_cm3 = avg_length_cm * avg_height_cm * avg_width_cm
    avg_weight_g = peso_g / max(n_items, 1)
    total_weight_g = peso_g
    density_gcm3 = total_weight_g / volume_cm3 if volume_cm3 > 0 else np.nan
    max_dimension_cm = max(avg_length_cm, avg_height_cm, avg_width_cm)
    has_oversized_item = 1 if max_dimension_cm > 60 else 0
    is_heavy = 1 if avg_weight_g > 5000 else 0

    # FIX honesto: reproduce el bug real — categorías en portugués
    # nunca matchean cat_alta/cat_baja en inglés → siempre 2.
    if main_category in CAT_ALTA_CODIGO:
        logistic_complexity = 3
    elif main_category in CAT_BAJA_CODIGO:
        logistic_complexity = 1
    else:
        logistic_complexity = 2  # esto SIEMPRE pasa con categorías reales de Olist

    carrito_complejo = 1 if n_items > 2 else 0
    tiene_varios_vendedores = 1 if unique_sellers > 1 else 0
    log_total_weight = np.log1p(total_weight_g)
    log_volume_cm3   = np.log1p(volume_cm3)

    # ── TEMPORALES (líneas 171-228) ─────────────────────────────────────────
    bin_hora = (0 if purchase_hour <= 7 else
                1 if purchase_hour <= 12 else
                2 if purchase_hour <= 17 else 3)
    is_weekend = 1 if purchase_weekday in [5,6] else 0
    is_off_hours = 1 if (purchase_hour < 8 or purchase_hour >= 18) else 0
    high_risk_day = 1 if (purchase_weekday in [5,6] or
                          (purchase_weekday == 4 and purchase_hour >= 16)) else 0
    is_short_promise = 1 if estimated_delivery_days < 10 else 0

    approval_ratio = time_to_approve_hours / max(mediana_aprob_global, 1)
    approval_delay_flag = 1 if approval_ratio > 3.0 else 0
    approval_bin = (0 if approval_ratio <= 0.5 else
                    1 if approval_ratio <= 2.0 else 2)
    weekday_x_hour = purchase_weekday * purchase_hour
    semana_del_mes = 2  # neutro — no afecta materialmente

    # ── PAGO (líneas 307-326) ───────────────────────────────────────────────
    total_price   = price_per_item * n_items
    # Proxy de flete calibrado contra el dato REAL del dataset:
    # freight_ratio mediana observada = 0.20 (ver master_table_dirty.csv)
    total_freight = price_per_item * n_items * 0.20
    total_payment = total_price + total_freight
    freight_ratio = total_freight / total_price if total_price > 0 else 0
    is_boleto = 1 if main_payment_type == "boleto" else 0
    is_high_installments = 1 if max_installments > 6 else 0
    log_total_payment = np.log1p(total_payment)
    high_freight_flag = 1 if freight_ratio > 0.5 else 0
    freight_category = (1 if freight_ratio <= 0.2 else
                        2 if freight_ratio <= 0.5 else 3)
    freight_per_kg = (total_freight / (total_weight_g/1000)
                      if total_weight_g > 0 else np.nan)

    # ── HISTORIAL VENDEDOR — parametrizable desde el simulador ─────────────
    # seller_late_rate_hist=0.111 es la media global real (sin historial).
    # Permitir que el usuario simule un vendedor "confiable" (bajo) o
    # "problemático" (alto) — esta es de las variables más predictivas
    # según SHAP, junto con estimated_delivery_days.
    seller_is_experienced = 1 if seller_total_orders_hist > 50 else 0

    # ── RISK SCORE (líneas 374-387) ─────────────────────────────────────────
    risk_score_raw = (
        region_risk_score * 2.5 + logistic_complexity * 1.5 +
        approval_delay_flag * 2.5 + high_risk_day * 1.5 +
        is_long_distance * 1.0 + is_boleto * 1.0 +
        complex_route * 1.5 + has_oversized_item * 0.5 +
        high_freight_flag * 0.5
    )
    risk_score = round(risk_score_raw / RISK_SCORE_MAX_DATASET * 10, 2)
    region_x_complexity   = region_risk_score * logistic_complexity
    distance_x_complexity = distance_km * logistic_complexity

    fila_cruda = {
        "customer_state": customer_state, "seller_state": seller_state,
        "main_category": main_category, "main_payment_type": main_payment_type,
        "total_items": n_items, "unique_products": n_items,
        "price_per_item": price_per_item,
        "total_weight_g": total_weight_g, "avg_weight_g": avg_weight_g,
        "avg_length_cm": avg_length_cm, "avg_height_cm": avg_height_cm,
        "avg_width_cm": avg_width_cm,
        "payment_methods": 1, "max_installments": max_installments,
        "payment_rows": 1, "purchase_weekday": purchase_weekday,
        "semana_del_mes": semana_del_mes, "bin_hora": bin_hora,
        "is_weekend": is_weekend, "is_off_hours": is_off_hours,
        "high_risk_day": high_risk_day, "is_peak_season": 0, "is_festivo": 0,
        "estimated_delivery_days": estimated_delivery_days,
        "is_short_promise": is_short_promise, "approval_bin": approval_bin,
        "weekday_x_hour": weekday_x_hour, "same_state": same_state,
        "complex_route": complex_route, "is_long_distance": is_long_distance,
        "volume_cm3": volume_cm3, "density_gcm3": density_gcm3,
        "max_dimension_cm": max_dimension_cm,
        "has_oversized_item": has_oversized_item, "is_heavy": is_heavy,
        "logistic_complexity": logistic_complexity,
        "carrito_complejo": carrito_complejo,
        "tiene_varios_vendedores": tiene_varios_vendedores,
        "log_total_weight": log_total_weight, "log_volume_cm3": log_volume_cm3,
        "freight_ratio": freight_ratio, "is_boleto": is_boleto,
        "is_high_installments": is_high_installments,
        "log_total_payment": log_total_payment,
        "high_freight_flag": high_freight_flag,
        "freight_category": freight_category, "freight_per_kg": freight_per_kg,
        "seller_total_orders_hist": seller_total_orders_hist,
        "seller_is_experienced": seller_is_experienced,
        "risk_score": risk_score, "total_price": total_price,
        "total_freight": total_freight, "total_payment": total_payment,
        "distance_km": distance_km, "region_x_complexity": region_x_complexity,
        "distance_x_complexity": distance_x_complexity,
        "seller_late_rate_hist": seller_late_rate_hist,
        "approval_ratio": approval_ratio, "interstate_flag": interstate_flag,
    }

    df = pd.DataFrame([fila_cruda])

    for col, (p1, p99) in limites.items():
        if col in df.columns:
            df[col] = df[col].clip(p1, p99)

    for col, woe_map in woe_maps.items():
        if col in df.columns:
            df[col] = df[col].map(woe_map).fillna(0)

    cols_escalar = [c for c in VARS_NUMERICAS if c in df.columns]
    if scaler is not None and cols_escalar:
        df[cols_escalar] = scaler.transform(df[cols_escalar].fillna(0))

    factores = {
        "region_risk_score": region_risk_score,
        "logistic_complexity": logistic_complexity,
        "complex_route": complex_route, "is_long_distance": is_long_distance,
        "high_risk_day": high_risk_day, "is_short_promise": is_short_promise,
        "distance_km": distance_km, "has_oversized_item": has_oversized_item,
        "tiene_varios_vendedores": tiene_varios_vendedores,
    }
    return df, factores


def descomponer_riesgo(factores: dict) -> dict:
    complejidad = min(
        (factores["logistic_complexity"] / 3 * 50) +
        (factores["complex_route"] * 30) +
        (10 if factores.get("has_oversized_item") else 0), 100
    )
    geografico = min(
        (factores["region_risk_score"] / 3 * 60) +
        (factores["is_long_distance"] * 30), 100
    )
    calendario = min(factores["high_risk_day"] * 70 + 10, 100) if factores["high_risk_day"] else 12
    comercial  = min(factores["is_short_promise"] * 65 + 15, 100) if factores["is_short_promise"] else 25
    return {
        "Complejidad logística": round(complejidad, 0),
        "Factor geográfico":     round(geografico, 0),
        "Factor calendario":     round(calendario, 0),
        "Factor comercial":      round(comercial, 0),
    }


def generar_recomendaciones(score: float, factores: dict, descomp: dict) -> list:
    recs = []
    if score > 60:
        recs.append("⚠️ Activar seguimiento preventivo del pedido")
    if descomp["Factor comercial"] > 50:
        recs.append("📅 Agregar 2-3 días de margen a la promesa de entrega")
    if descomp["Factor geográfico"] > 50:
        recs.append("🚚 Verificar disponibilidad de transportista para la región")
    if descomp["Complejidad logística"] > 50:
        recs.append("📦 Revisar empaque — producto de alta complejidad logística")
    if descomp["Factor calendario"] > 50:
        recs.append("🗓️ Compra en horario crítico — el despacho puede iniciar en el siguiente turno operativo")
    if factores.get("tiene_varios_vendedores"):
        recs.append("👥 Pedido multi-vendedor — coordinar tiempos de despacho")
    if score <= 30:
        recs.append("✅ Riesgo bajo — proceder con flujo estándar")
    if not recs:
        recs.append("👀 Monitorear despacho de forma estándar")
    return recs
