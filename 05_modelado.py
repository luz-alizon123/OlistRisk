"""
05_modelado.py — SPRINT 2 (sin tuneo)
======================================
Incluye:
  - Logistic Regression  (baseline)
  - Random Forest
  - XGBoost
  - LightGBM
  - Stacking Ensemble
  - Metricas completas + backtesting + SHAP + curva ROC
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, StackingClassifier
from sklearn.metrics         import (
    roc_auc_score, roc_curve, f1_score, recall_score,
    precision_score, accuracy_score, average_precision_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠️  xgboost no instalado — se omite")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  ⚠️  lightgbm no instalado — se omite")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  ⚠️  shap no instalado — se omite")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models");  MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)

TARGET = "is_late_delivery"
SEED   = 42
np.random.seed(SEED)

# ─── CARGA ────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  MODELADO — OLIST (sin tuneo)")
print("=" * 65)

train    = pd.read_parquet(DATA_DIR / "split_train.parquet")
val      = pd.read_parquet(DATA_DIR / "split_val.parquet")
backtest = pd.read_parquet(DATA_DIR / "split_backtest.parquet")
live     = pd.read_parquet(DATA_DIR / "split_live.parquet")

features_finales = joblib.load(DATA_DIR / "features_finales.pkl")
features_finales = [f for f in features_finales if f in train.columns]
print("\nVERIFICACION DE FEATURES")
print("Total features:", len(features_finales))
print(f"\n  Features del modelo: {len(features_finales)}")
print(f"  {features_finales}")

X_train = train[features_finales].fillna(0)
y_train = train[TARGET]
X_val   = val[features_finales].fillna(0)
y_val   = val[TARGET]
X_back  = backtest[features_finales].fillna(0)
y_back  = backtest[TARGET]
X_live  = live[features_finales].fillna(0)
y_live  = live[TARGET]

scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
print(f"\n  Desbalance — scale_pos_weight: {scale_pos:.2f}")
print(f"  Positivos (tardíos): {y_train.sum():,}  |  Negativos: {(y_train==0).sum():,}")
print("\nVERIFICACION DE SPLITS")
print("Train :", X_train.shape)
print("Val   :", X_val.shape)
print("Back  :", X_back.shape)
print("Live  :", X_live.shape)


# ─── MÉTRICAS ─────────────────────────────────────────────────────────────────

def calcular_ks(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def evaluar(nombre, modelo, X, y, umbral=0.40):
    y_prob = modelo.predict_proba(X)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)

    auc    = roc_auc_score(y, y_prob)
    gini   = 2 * auc - 1
    pr_auc = average_precision_score(y, y_prob)
    ks     = calcular_ks(y, y_prob)
    f1     = f1_score(y, y_pred, zero_division=0)
    rec    = recall_score(y, y_pred, zero_division=0)
    prec   = precision_score(y, y_pred, zero_division=0)
    acc    = accuracy_score(y, y_pred)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    ahorro         = tp * 45 * 0.7
    reclamos_evit  = tp * 0.7

    return {
        "modelo": nombre,
        "AUC-ROC": round(auc, 4), "Gini": round(gini, 4),
        "PR-AUC": round(pr_auc, 4), "KS": round(ks, 4),
        "F1": round(f1, 4), "Recall": round(rec, 4),
        "Precision": round(prec, 4), "Accuracy": round(acc, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Tasa_Deteccion": round(tp / max(y.sum(), 1), 4),
        "Ahorro_R$": round(ahorro, 0),
        "Reclamos_Evit": round(reclamos_evit, 0),
        "y_prob": y_prob,
    }

def imprimir(r, split_name=""):
    print(f"\n  {'─'*55}")
    print(f"  {r['modelo']}  [{split_name}]")
    print(f"  {'─'*55}")
    print(f"  AUC-ROC : {r['AUC-ROC']}   Gini   : {r['Gini']}")
    print(f"  PR-AUC  : {r['PR-AUC']}   KS     : {r['KS']}")
    print(f"  F1      : {r['F1']}   Recall : {r['Recall']}")
    print(f"  Precision:{r['Precision']}   Accuracy:{r['Accuracy']}")
    print(f"  TP={r['TP']}  FP={r['FP']}  TN={r['TN']}  FN={r['FN']}")
    print(f"  Tasa Detección: {r['Tasa_Deteccion']*100:.1f}%")
    print(f"  Ahorro potencial: R$ {r['Ahorro_R$']:,.0f}")

def optimizar_umbral(y_true, y_prob):
    mejor_u, mejor_f1 = 0.5, 0
    for u in np.arange(0.10, 0.80, 0.01):
        yp = (y_prob >= u).astype(int)
        f  = f1_score(y_true, yp, zero_division=0)
        if f > mejor_f1:
            mejor_f1 = f
            mejor_u  = u
    return round(mejor_u, 2)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 1 — LOGISTIC REGRESSION (Baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  MODELO 1 — LOGISTIC REGRESSION (Baseline)")
print("═"*65)

lr = LogisticRegression(
    C=0.05, class_weight="balanced",
    solver="lbfgs", max_iter=2000, random_state=SEED
)
lr.fit(X_train, y_train)
r_lr_val = evaluar("LogReg", lr, X_val, y_val)
imprimir(evaluar("LogReg", lr, X_train, y_train), "TRAIN")
imprimir(r_lr_val, "VAL")


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 2 — RANDOM FOREST (parámetros fijos)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  MODELO 2 — RANDOM FOREST")
print("═"*65)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced",
    random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
r_rf_val = evaluar("RandomForest", rf, X_val, y_val)
imprimir(evaluar("RandomForest", rf, X_train, y_train), "TRAIN")
imprimir(r_rf_val, "VAL")


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 3 — XGBOOST (parámetros fijos)
# ══════════════════════════════════════════════════════════════════════════════
xgb = None
r_xgb_val = None
if HAS_XGB:
    print("\n" + "═"*65)
    print("  MODELO 3 — XGBOOST")
    print("═"*65)

    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="auc", random_state=SEED, n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    r_xgb_val = evaluar("XGBoost", xgb, X_val, y_val)
    imprimir(evaluar("XGBoost", xgb, X_train, y_train), "TRAIN")
    imprimir(r_xgb_val, "VAL")


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 4 — LIGHTGBM (parámetros fijos)
# ══════════════════════════════════════════════════════════════════════════════
lgbm = None
r_lgbm_val = None
if HAS_LGB:
    print("\n" + "═"*65)
    print("  MODELO 4 — LIGHTGBM")
    print("═"*65)

    lgbm = LGBMClassifier(
        n_estimators=500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=SEED, n_jobs=-1, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    r_lgbm_val = evaluar("LightGBM", lgbm, X_val, y_val)
    imprimir(evaluar("LightGBM", lgbm, X_train, y_train), "TRAIN")
    imprimir(r_lgbm_val, "VAL")


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 5 — STACKING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  MODELO 5 — STACKING ENSEMBLE")
print("═"*65)

estimadores = [("rf", rf)]
if HAS_XGB and xgb is not None:
    estimadores.append(("xgb", xgb))
if HAS_LGB and lgbm is not None:
    estimadores.append(("lgbm", lgbm))

meta = LogisticRegression(C=0.1, class_weight="balanced",
                           max_iter=1000, random_state=SEED)

stacking = StackingClassifier(
    estimators=estimadores,
    final_estimator=meta,
    cv=5,                    # ← CORREGIDO: entero en lugar de TimeSeriesSplit
    stack_method="predict_proba",
    n_jobs=-1,
)
stacking.fit(X_train, y_train)
print("  Stacking Ensemble entrenado")

r_stack_val = evaluar("Stacking", stacking, X_val, y_val)
imprimir(evaluar("Stacking", stacking, X_train, y_train), "TRAIN")
imprimir(r_stack_val, "VAL")


# ══════════════════════════════════════════════════════════════════════════════
# TABLA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  TABLA COMPARATIVA — VALIDACIÓN")
print("="*65)

resultados_val = [r_lr_val, r_rf_val]
if r_xgb_val:  resultados_val.append(r_xgb_val)
if r_lgbm_val: resultados_val.append(r_lgbm_val)
resultados_val.append(r_stack_val)

cols = ["modelo","AUC-ROC","Gini","PR-AUC","KS","F1","Recall","Precision","Accuracy"]
df_comp = pd.DataFrame([{k: v for k, v in r.items() if k in cols}
                         for r in resultados_val])
print(df_comp.to_string(index=False))
df_comp.to_csv(DATA_DIR / "tabla_comparativa_modelos.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# CHAMPION MODEL — criterio: mayor AUC-ROC en validación
# ══════════════════════════════════════════════════════════════════════════════
candidatos = {
    "Stacking":     (stacking,  r_stack_val["AUC-ROC"]),
    "RandomForest": (rf,        r_rf_val["AUC-ROC"]),
}
if r_xgb_val:  candidatos["XGBoost"]  = (xgb,  r_xgb_val["AUC-ROC"])
if r_lgbm_val: candidatos["LightGBM"] = (lgbm, r_lgbm_val["AUC-ROC"])

nombre_champion, (champion, _) = max(candidatos.items(), key=lambda x: x[1][1])
print(f"\n  ✅ CHAMPION MODEL: {nombre_champion}  "
      f"(AUC-ROC Val: {candidatos[nombre_champion][1]:.4f})")


# ── Umbral óptimo ─────────────────────────────────────────────────────────────
print("\n▶ Optimizando umbral (criterio: F1 en validación)...")
proba_val = champion.predict_proba(X_val)[:, 1]
umbral    = optimizar_umbral(y_val, proba_val)
print(f"  Umbral óptimo: {umbral}")
imprimir(evaluar(nombre_champion, champion, X_val, y_val, umbral=umbral),
         f"VAL (umbral={umbral})")


# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  BACKTESTING TEMPORAL")
print("="*65)

resultados_bt = []
for nombre_split, X_s, y_s in [
    ("VAL",      X_val,  y_val),
    ("BACKTEST", X_back, y_back),
    ("LIVE",     X_live, y_live),
]:
    r = evaluar(nombre_champion, champion, X_s, y_s, umbral=umbral)
    r["split"] = nombre_split
    imprimir(r, nombre_split)
    resultados_bt.append(r)

df_bt = pd.DataFrame([{k: v for k, v in r.items() if k != "y_prob"}
                       for r in resultados_bt])
df_bt.to_csv(DATA_DIR / "tabla_backtesting.csv", index=False)
print("\n  RESUMEN BACKTESTING:")
print(df_bt[["split","AUC-ROC","Gini","Recall","F1",
             "Tasa_Deteccion","Ahorro_R$"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# CURVA ROC
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Generando curva ROC...")

fig, ax = plt.subplots(figsize=(9, 6))
modelos_plot = [("LogReg", lr), ("RandomForest", rf)]
if xgb:  modelos_plot.append(("XGBoost", xgb))
if lgbm: modelos_plot.append(("LightGBM", lgbm))
modelos_plot.append((nombre_champion, champion))

colores = ["#95A5A6","#27AE60","#2E86AB","#F39C12","#E74C3C"]
for (nom, m), color in zip(modelos_plot, colores):
    prob = m.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, prob)
    auc = roc_auc_score(y_val, prob)
    lw  = 3 if nom == nombre_champion else 1.5
    ax.plot(fpr, tpr, label=f"{nom} (AUC={auc:.3f})", color=color, linewidth=lw)

ax.plot([0,1],[0,1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Curva ROC — Comparación de Modelos (Validación)")
ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR / "curva_roc_comparativa.png", dpi=150)
plt.close()
print("  ✅ Guardado: reports/curva_roc_comparativa.png")


# ══════════════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════════════
if HAS_SHAP and lgbm is not None:
    print("\n▶ Calculando SHAP values (LightGBM)...")
    try:
        explainer   = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(X_val)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        fig_s, ax_s = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_val, feature_names=features_finales,
                          plot_type="bar", show=False, max_display=25)
        plt.title("SHAP — Importancia Global de Features")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_importancia.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✅ Guardado: reports/shap_importancia.png")

        joblib.dump({"explainer": explainer, "feature_names": features_finales},
                    MODEL_DIR / "shap_explainer.pkl")
    except Exception as e:
        print(f"  ⚠️  SHAP falló: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CHAMPION MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Guardando Champion Model...")

artefactos = {
    "modelo":              champion,
    "modelo_lgbm":         lgbm,
    "features_finales":    features_finales,
    "umbral_optimo":       float(umbral),
    "nombre_champion":     nombre_champion,
    "version":             "1.0.0",
    "fecha_entrenamiento": str(datetime.now().date()),
    "periodo_train":       "2016-09-15 / 2018-05-31",
    "metricas_val": {
        k: v for k, v in
        evaluar(nombre_champion, champion, X_val, y_val, umbral=umbral).items()
        if k not in ["y_prob", "modelo"]
    },
    "metricas_backtest": {
        k: v for k, v in resultados_bt[1].items()
        if k not in ["y_prob", "modelo", "split"]
    },
}

joblib.dump(artefactos, MODEL_DIR / "champion_model_v1.pkl")
print("  ✅ Guardado: models/champion_model_v1.pkl")

resumen = {k: v for k, v in artefactos.items() if k not in ["modelo","modelo_lgbm"]}
with open(MODEL_DIR / "champion_resumen.json", "w") as f:
    json.dump(resumen, f, indent=2, default=str)
print("  ✅ Guardado: models/champion_resumen.json")

print("\n" + "="*65)
print("  SPRINT 3 COMPLETADO ✅")
print("  Champion:", nombre_champion)
print(f"  AUC-ROC Val: {candidatos[nombre_champion][1]:.4f}")
print("="*65)