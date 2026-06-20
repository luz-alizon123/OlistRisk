"""
utils/metrics.py
=================
Funciones de cálculo de métricas
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, recall_score,
    precision_score, accuracy_score, average_precision_score,
    confusion_matrix,
)


def calcular_ks(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluar_split(y_true, y_prob, umbral: float = 0.50) -> dict:
    """Calcula el set completo de métricas para un split dado."""
    y_pred = (y_prob >= umbral).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "umbral":    umbral,
        "AUC-ROC":   round(auc, 4),
        "Gini":      round(2*auc - 1, 4),
        "PR-AUC":    round(average_precision_score(y_true, y_prob), 4),
        "KS":        round(calcular_ks(y_true, y_prob), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Tasa_Deteccion": round(tp / max(int(y_true.sum()), 1), 4),
        "N": len(y_true),
        "Tasa_real_%": round(y_true.mean()*100, 1),
    }


def calcular_psi(serie_train: pd.Series, serie_actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index entre una variable de referencia (train)
    y la misma variable en un período nuevo (monitoreo de drift).
    """
    try:
        breaks = pd.qcut(serie_train, q=bins, retbins=True, duplicates="drop")[1]
        breaks[0]  = -np.inf
        breaks[-1] = np.inf

        def dist(s):
            c = pd.cut(s, bins=breaks).value_counts(sort=False)
            return (c / len(s)).replace(0, 1e-4)

        d_ref = dist(serie_train)
        d_act = dist(serie_actual)
        psi = float(np.sum((d_ref - d_act) * np.log(d_ref / d_act)))
        return psi if np.isfinite(psi) else 999.0
    except Exception:
        return 0.0


def estado_psi(psi: float) -> dict:
    """Clasifica el PSI según el umbral del docente (0.15)."""
    if psi < 0.15:
        return {"estado": "Estable", "color": "#1D9E75", "accion": "Modelo estable"}
    elif psi < 0.25:
        return {"estado": "Moderado", "color": "#EF9F27", "accion": "Monitorear de cerca"}
    else:
        return {"estado": "Inestable", "color": "#E24B4A", "accion": "Reentrenar modelo"}


def calcular_roi(tp: int, fp: int, costo_reclamo: float = 45,
                  costo_intervencion: float = 8,
                  tasa_exito_intervencion: float = 0.70) -> dict:
    """
    Calcula el ROI estimado del modelo en producción.
    Todos los costos son supuestos declarados — deben validarse
    con el equipo de operaciones real de Olist.
    """
    reclamos_evitados  = tp * tasa_exito_intervencion
    ahorro_bruto       = reclamos_evitados * costo_reclamo
    costo_total        = (tp + fp) * costo_intervencion
    ahorro_neto        = ahorro_bruto - costo_total
    roi_pct            = (ahorro_neto / costo_total * 100) if costo_total > 0 else 0
    return {
        "reclamos_evitados": round(reclamos_evitados, 1),
        "ahorro_bruto":      round(ahorro_bruto, 2),
        "costo_intervencion":round(costo_total, 2),
        "ahorro_neto":       round(ahorro_neto, 2),
        "roi_pct":           round(roi_pct, 1),
    }