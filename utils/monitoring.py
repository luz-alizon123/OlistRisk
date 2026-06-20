"""
utils/monitoring.py
====================
Regla de reentrenamiento implementada EXACTAMENTE como se pidió:

  SI AUC < 0.65  O  PSI > 0.25 en MÁS DE 3 variables críticas
  ⇒ Recomendación: REENTRENAR

En cualquier otro caso ⇒ Modelo estable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.metrics import calcular_psi, estado_psi

DATA_DIR = Path("data")

VARIABLES_MONITOREO = [
    "estimated_delivery_days", "total_weight_g", "price_per_item",
    "freight_ratio", "risk_score", "total_payment",
]

UMBRAL_PSI_CRITICO = 0.25
UMBRAL_AUC_MINIMO  = 0.65
MAX_VARIABLES_INESTABLES_PERMITIDAS = 3


def cargar_splits_monitoreo() -> dict:
    splits = {}
    for nombre in ["train", "val", "backtest", "live"]:
        path = DATA_DIR / f"split_{nombre}.parquet"
        if path.exists():
            splits[nombre] = pd.read_parquet(path)
    return splits


def calcular_psi_monitoreo(splits: dict) -> pd.DataFrame:
    if "train" not in splits:
        return pd.DataFrame()

    filas = []
    train = splits["train"]
    for split_nombre in ["val", "backtest", "live"]:
        if split_nombre not in splits:
            continue
        actual = splits[split_nombre]
        for var in VARIABLES_MONITOREO:
            if var not in train.columns or var not in actual.columns:
                continue
            psi = calcular_psi(train[var].dropna(), actual[var].dropna())
            est = estado_psi(psi)
            filas.append({
                "Split": split_nombre.capitalize(),
                "Variable": var,
                "PSI": round(psi, 4),
                "Estado": est["estado"],
                "_color": est["color"],
            })
    return pd.DataFrame(filas)


def estilizar_tabla_psi(df_psi: pd.DataFrame):
    def color_fila(row):
        color = row.get("_color", "#888780")
        return [f"background-color:{color}22; color:{color}; font-weight:600"
                if col == "Estado" else "" for col in row.index]
    df_visible = df_psi.drop(columns=["_color"], errors="ignore")
    return df_visible.style.apply(color_fila, axis=1)


def evaluar_regla_reentrenamiento(df_psi: pd.DataFrame, auc_actual: float,
                                   split_referencia: str = "Live") -> dict:
    """
    Regla de negocio EXACTA:
      SI AUC < 0.65 O PSI > 0.25 en MÁS DE 3 variables críticas
      ⇒ REENTRENAR
      EN CUALQUIER OTRO CASO ⇒ Modelo estable

    Se evalúa sobre el split de referencia (por defecto: Live, el
    período más reciente / más parecido a producción real).
    """
    df_ref = df_psi[df_psi["Split"] == split_referencia] if not df_psi.empty else df_psi

    n_vars_inestables = int((df_ref["PSI"] > UMBRAL_PSI_CRITICO).sum()) if not df_ref.empty else 0
    vars_inestables = (df_ref[df_ref["PSI"] > UMBRAL_PSI_CRITICO]["Variable"].tolist()
                       if not df_ref.empty else [])

    condicion_auc = auc_actual < UMBRAL_AUC_MINIMO
    condicion_psi = n_vars_inestables > MAX_VARIABLES_INESTABLES_PERMITIDAS

    requiere_reentrenamiento = condicion_auc or condicion_psi

    if requiere_reentrenamiento:
        razones = []
        if condicion_auc:
            razones.append(f"AUC actual ({auc_actual:.4f}) < umbral mínimo ({UMBRAL_AUC_MINIMO})")
        if condicion_psi:
            razones.append(f"{n_vars_inestables} variables con PSI > {UMBRAL_PSI_CRITICO} "
                           f"(máximo permitido: {MAX_VARIABLES_INESTABLES_PERMITIDAS}) — "
                           f"variables: {', '.join(vars_inestables)}")
        return {
            "decision": "REENTRENAR",
            "color": "#E24B4A",
            "razones": razones,
            "auc_actual": round(auc_actual, 4),
            "n_vars_inestables": n_vars_inestables,
            "vars_inestables": vars_inestables,
        }
    else:
        return {
            "decision": "MODELO ESTABLE",
            "color": "#1D9E75",
            "razones": [
                f"AUC actual ({auc_actual:.4f}) ≥ umbral mínimo ({UMBRAL_AUC_MINIMO})",
                f"Solo {n_vars_inestables} variable(s) con PSI > {UMBRAL_PSI_CRITICO} "
                f"(máximo permitido: {MAX_VARIABLES_INESTABLES_PERMITIDAS})",
            ],
            "auc_actual": round(auc_actual, 4),
            "n_vars_inestables": n_vars_inestables,
            "vars_inestables": vars_inestables,
        }


def diagnostico_global(df_psi: pd.DataFrame) -> dict:
    """Mantenido por compatibilidad — usa la escala estándar de PSI."""
    if df_psi.empty:
        return {"estado": "Sin datos", "accion": "No se pudo calcular PSI", "color": "#888780"}
    df_live = df_psi[df_psi["Split"] == "Live"]
    if df_live.empty:
        df_live = df_psi
    psi_max = df_live["PSI"].max()
    return {"psi_max": round(psi_max, 4), **estado_psi(psi_max)}
