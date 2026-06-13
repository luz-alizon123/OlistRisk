"""
06_tuneo_v2.py 
==========================================

  1. CV=5 en Optuna (StratifiedKFold sobre train)
     Antes: evaluaba directo sobre Val → overfitting al val set
     Ahora: promedio de 5 folds → evaluación más honesta y robusta

  2. Objetivo penaliza sobreajuste directamente
     objetivo = AUC_cv - max(0, gap - 0.10) * 0.5
     Si el gap Train→Val supera 0.10, se descuenta del score.
     Esto fuerza a Optuna a buscar modelos generalizables.

  3. Espacio de búsqueda LightGBM restringido
     num_leaves:        10–50   (antes 10–100, producía 96 → sobreajuste)
     min_child_samples: 50–500  (antes 20–300, producía 21 → sobreajuste)
     reg_lambda:        0.1–10  (antes 0.0001–10, casi sin regularización)
     n_estimators:      100–500 (antes 100–1000, producía 934 → memorización)

  4. CSV de variables finales con PSI, decisión e importancia del modelo
  5. Tabla de hiperparámetros: qué mejoró vs empeoró entre base y tuneado
  6. Umbral calibrado para maximizar Recall sin sacrificarlo
  7. Importancia de variables del champion exportada post-backtest

Modelos: Regresión Logística + LightGBM
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    roc_auc_score, roc_curve, f1_score, recall_score,
    precision_score, accuracy_score, average_precision_score,
    confusion_matrix,
)

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  ⚠️  pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  ⚠️  pip install optuna")

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
    from imblearn.combine       import SMOTEENN
    HAS_IMBL = True
except ImportError:
    HAS_IMBL = False
    print("  ⚠️  pip install imbalanced-learn")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models");  MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)

TARGET   = "is_late_delivery"
SEED     = 42
N_TRIALS = 200
CV_FOLDS = 5          # CV sobre train — más robusto que evaluar directo en Val
GAP_MAX  = 0.10       # Gap máximo permitido antes de penalizar

np.random.seed(SEED)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# ─── CARGA ────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  TUNEO — SPRINT 3 (versión 2)")
print(f"  LogReg + LightGBM | CV={CV_FOLDS} | Optuna {N_TRIALS} trials")
print(f"  Penalización gap > {GAP_MAX} | Espacio LightGBM restringido")
print("=" * 65)

train    = pd.read_parquet(DATA_DIR / "split_train.parquet")
val      = pd.read_parquet(DATA_DIR / "split_val.parquet")
backtest = pd.read_parquet(DATA_DIR / "split_backtest.parquet")
live     = pd.read_parquet(DATA_DIR / "split_live.parquet")

features = joblib.load(DATA_DIR / "features_finales.pkl")
features = [f for f in features if f in train.columns]

X_train = train[features].fillna(0);    y_train = train[TARGET]
X_val   = val[features].fillna(0);      y_val   = val[TARGET]
X_back  = backtest[features].fillna(0); y_back  = backtest[TARGET]
X_live  = live[features].fillna(0);     y_live  = live[TARGET]

scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

print(f"\n  Features: {len(features)}")
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | "
      f"Back: {len(X_back):,} | Live: {len(X_live):,}")
print(f"  Desbalance scale_pos_weight: {scale_pos:.2f}")

# Índices de columnas categóricas/binarias para SMOTENC
COLS_CAT_IDX = [
    features.index(f) for f in [
        "is_weekend","is_off_hours","high_risk_day","is_peak_season",
        "is_festivo","is_short_promise","same_state","complex_route",
        "is_long_distance","has_oversized_item","is_heavy",
        "carrito_complejo","tiene_varios_vendedores","is_boleto",
        "is_high_installments","high_freight_flag","seller_is_experienced",
        "bin_hora","approval_bin","logistic_complexity","freight_category",
    ]
    if f in features
]


# ─── MÉTRICAS ─────────────────────────────────────────────────────────────────

def calcular_ks(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def evaluar(nombre, modelo, X, y, umbral=0.50):
    y_prob = modelo.predict_proba(X)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)
    auc    = roc_auc_score(y, y_prob)
    ks     = calcular_ks(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    f1     = f1_score(y, y_pred, zero_division=0)
    rec    = recall_score(y, y_pred, zero_division=0)
    prec   = precision_score(y, y_pred, zero_division=0)
    acc    = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    return {
        "modelo": nombre, "umbral": umbral,
        "AUC-ROC": round(auc, 4), "Gini": round(2*auc-1, 4),
        "PR-AUC":  round(pr_auc, 4), "KS": round(ks, 4),
        "F1":      round(f1, 4), "Recall": round(rec, 4),
        "Precision": round(prec, 4), "Accuracy": round(acc, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Tasa_Deteccion": round(tp / max(int(y.sum()), 1), 4),
        "y_prob": y_prob,
    }

def imprimir(r, split_name=""):
    print(f"\n  {'─'*55}")
    print(f"  {r['modelo']}  [{split_name}]  umbral={r['umbral']}")
    print(f"  {'─'*55}")
    print(f"  AUC-ROC : {r['AUC-ROC']}   Gini    : {r['Gini']}")
    print(f"  PR-AUC  : {r['PR-AUC']}   KS      : {r['KS']}")
    print(f"  F1      : {r['F1']}   Recall  : {r['Recall']}")
    print(f"  Precision:{r['Precision']}   Accuracy: {r['Accuracy']}")
    print(f"  TP={r['TP']}  FP={r['FP']}  TN={r['TN']}  FN={r['FN']}")
    print(f"  Tasa Detección: {r['Tasa_Deteccion']*100:.1f}%")

def umbral_max_recall(y_true, y_prob, precision_minima=0.08):
    """
    Encuentra el umbral que maximiza Recall con Precisión >= precision_minima.
    No sacrifica Recall — solo limita cuántas falsas alarmas son aceptables.
    """
    mejor_u, mejor_rec = 0.50, 0
    for u in np.arange(0.10, 0.80, 0.01):
        yp   = (y_prob >= u).astype(int)
        rec  = recall_score(y_true, yp, zero_division=0)
        prec = precision_score(y_true, yp, zero_division=0)
        if prec >= precision_minima and rec > mejor_rec:
            mejor_rec = rec
            mejor_u   = u
    return round(mejor_u, 2)

def optimizar_umbral_f1(y_true, y_prob):
    mejor_u, mejor_f1 = 0.50, 0
    for u in np.arange(0.10, 0.80, 0.01):
        yp = (y_prob >= u).astype(int)
        f  = f1_score(y_true, yp, zero_division=0)
        if f > mejor_f1:
            mejor_f1 = f
            mejor_u  = u
    return round(mejor_u, 2)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — COMPARATIVA DE TÉCNICAS DE BALANCEO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FASE 1 — COMPARATIVA TÉCNICAS DE BALANCEO")
print("="*65)

def aplicar_balanceo(nombre, X, y):
    if not HAS_IMBL or nombre == "sin_balanceo":
        return X.copy(), y.copy()
    elif nombre == "RandomOverSampler":
        return RandomOverSampler(random_state=SEED).fit_resample(X, y)
    elif nombre == "SMOTE":
        return SMOTE(random_state=SEED, k_neighbors=5).fit_resample(X, y)
    elif nombre == "SMOTENC":
        if not COLS_CAT_IDX:
            return SMOTE(random_state=SEED, k_neighbors=5).fit_resample(X, y)
        return SMOTENC(categorical_features=COLS_CAT_IDX,
                       random_state=SEED, k_neighbors=5).fit_resample(X, y)
    elif nombre == "SMOTEENN":
        return SMOTEENN(random_state=SEED).fit_resample(X, y)
    return X.copy(), y.copy()

tecnicas = (["sin_balanceo","RandomOverSampler","SMOTE","SMOTENC","SMOTEENN"]
            if HAS_IMBL else ["sin_balanceo"])

lr_base = LogisticRegression(C=1.0, class_weight="balanced",
                              solver="lbfgs", max_iter=1000, random_state=SEED)
lgbm_base = (LGBMClassifier(n_estimators=200, scale_pos_weight=scale_pos,
                              random_state=SEED, n_jobs=-1, verbose=-1)
             if HAS_LGB else None)

resultados_balanceo = []

for tecnica in tecnicas:
    print(f"\n  ── {tecnica} ──")
    try:
        Xr, yr = aplicar_balanceo(tecnica, X_train, y_train)
        n1 = int((yr == 1).sum())
        print(f"     Clase 0: {int((yr==0).sum()):,} | Clase 1: {n1:,} | "
              f"Total: {len(yr):,}")

        # LogReg — CV sobre train
        lr_base.fit(Xr, yr)
        auc_tr = roc_auc_score(y_train, lr_base.predict_proba(X_train)[:,1])
        auc_v  = roc_auc_score(y_val,   lr_base.predict_proba(X_val)[:,1])
        print(f"     LogReg   → Train: {auc_tr:.4f} | Val: {auc_v:.4f} | "
              f"Gap: {auc_tr-auc_v:.4f}")
        resultados_balanceo.append({"tecnica": tecnica, "modelo": "LogReg",
            "AUC_train": round(auc_tr,4), "AUC_val": round(auc_v,4),
            "gap": round(auc_tr-auc_v,4), "n_clase1": n1})

        if lgbm_base:
            lgbm_base.set_params(scale_pos_weight=1.0)
            lgbm_base.fit(Xr, yr)
            auc_tr = roc_auc_score(y_train, lgbm_base.predict_proba(X_train)[:,1])
            auc_v  = roc_auc_score(y_val,   lgbm_base.predict_proba(X_val)[:,1])
            print(f"     LightGBM → Train: {auc_tr:.4f} | Val: {auc_v:.4f} | "
                  f"Gap: {auc_tr-auc_v:.4f}")
            resultados_balanceo.append({"tecnica": tecnica, "modelo": "LightGBM",
                "AUC_train": round(auc_tr,4), "AUC_val": round(auc_v,4),
                "gap": round(auc_tr-auc_v,4), "n_clase1": n1})
    except Exception as e:
        print(f"     ⚠️  Error: {e}")

df_bal = pd.DataFrame(resultados_balanceo)
df_bal.to_csv(DATA_DIR / "tabla_balanceo.csv", index=False)

print(f"\n  RESUMEN:")
print(df_bal.to_string(index=False))

mejor_lr   = (df_bal[df_bal["modelo"]=="LogReg"]
              .sort_values("AUC_val", ascending=False).iloc[0]["tecnica"])
mejor_lgbm = "sin_balanceo"
if HAS_LGB:
    mejor_lgbm = (df_bal[df_bal["modelo"]=="LightGBM"]
                  .sort_values("AUC_val", ascending=False).iloc[0]["tecnica"])

print(f"\n  ✅ Mejor para LogReg:    {mejor_lr}")
print(f"  ✅ Mejor para LightGBM:  {mejor_lgbm}")

X_train_lr,   y_train_lr   = aplicar_balanceo(mejor_lr,   X_train, y_train)
X_train_lgbm, y_train_lgbm = aplicar_balanceo(mejor_lgbm, X_train, y_train)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — OPTUNA CON CV=5 Y PENALIZACIÓN DE GAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"  FASE 2 — OPTUNA ({N_TRIALS} trials | CV={CV_FOLDS} | penaliza gap>{GAP_MAX})")
print("="*65)

if not HAS_OPTUNA:
    print("  ⚠️  pip install optuna")
    lr_tuned = lgbm_tuned = None
else:
    # ── 2A. LOGISTIC REGRESSION ───────────────────────────────────────────────
    print("\n▶ Tuneando Regresión Logística...")

    def objetivo_lr(trial):
        C      = trial.suggest_float("C", 1e-4, 10.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs","saga","liblinear"])
        penalty= "l2"
        if solver == "saga":
            penalty = trial.suggest_categorical("penalty_saga", ["l1","l2"])
        elif solver == "liblinear":
            penalty = trial.suggest_categorical("penalty_ll", ["l1","l2"])

        modelo = LogisticRegression(
            C=C, solver=solver, penalty=penalty,
            class_weight="balanced", max_iter=2000, random_state=SEED
        )
        # CV=5 sobre train — evalúa en 5 particiones distintas
        auc_cv = cross_val_score(
            modelo, X_train_lr, y_train_lr,
            cv=cv, scoring="roc_auc", n_jobs=-1
        ).mean()

        # Penalizar gap si el modelo sobreajusta
        modelo.fit(X_train_lr, y_train_lr)
        auc_tr  = roc_auc_score(y_train_lr, modelo.predict_proba(X_train_lr)[:,1])
        auc_val = roc_auc_score(y_val,       modelo.predict_proba(X_val)[:,1])
        gap     = auc_tr - auc_val
        penalizacion = max(0, gap - GAP_MAX) * 0.5

        return auc_cv - penalizacion

    study_lr = optuna.create_study(
        direction="maximize", study_name="LogReg_v2",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study_lr.optimize(objetivo_lr, n_trials=N_TRIALS, show_progress_bar=True)

    bp_lr = study_lr.best_params
    print(f"\n  Mejores parámetros LogReg:")
    for k, v in bp_lr.items():
        print(f"    {k}: {v}")
    print(f"  Mejor AUC CV: {study_lr.best_value:.4f}")

    penalty_final = bp_lr.get("penalty_saga", bp_lr.get("penalty_ll", "l2"))
    lr_tuned = LogisticRegression(
        C=bp_lr["C"], solver=bp_lr["solver"], penalty=penalty_final,
        class_weight="balanced", max_iter=2000, random_state=SEED
    )
    lr_tuned.fit(X_train_lr, y_train_lr)

    study_lr.trials_dataframe().to_csv(
        DATA_DIR / "optuna_historial_logreg.csv", index=False
    )
    print(f"  ✅ data/optuna_historial_logreg.csv")

    # ── 2B. LIGHTGBM — ESPACIO RESTRINGIDO ────────────────────────────────────
    lgbm_tuned = None
    if HAS_LGB:
        print("\n▶ Tuneando LightGBM (espacio restringido)...")
        print("  Restricciones aplicadas:")
        print("    num_leaves:        10–50   (antes 10–100)")
        print("    min_child_samples: 50–500  (antes 20–300)")
        print("    reg_lambda:        0.1–10  (antes 0.0001–10)")
        print("    n_estimators:      100–500 (antes 100–1000)")

        def objetivo_lgbm(trial):
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                "num_leaves":        trial.suggest_int("num_leaves", 10, 50),
                "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq":      1,
                "scale_pos_weight":  1.0,
                "random_state":      SEED,
                "n_jobs":            -1,
                "verbose":           -1,
            }
            modelo = LGBMClassifier(**params)

            # CV=5 sobre train
            auc_cv = cross_val_score(
                modelo, X_train_lgbm, y_train_lgbm,
                cv=cv, scoring="roc_auc", n_jobs=-1
            ).mean()

            # Penalizar gap
            modelo.fit(X_train_lgbm, y_train_lgbm)
            auc_tr  = roc_auc_score(y_train_lgbm, modelo.predict_proba(X_train_lgbm)[:,1])
            auc_val = roc_auc_score(y_val,         modelo.predict_proba(X_val)[:,1])
            gap     = auc_tr - auc_val
            penalizacion = max(0, gap - GAP_MAX) * 0.5

            return auc_cv - penalizacion

        study_lgbm = optuna.create_study(
            direction="maximize", study_name="LightGBM_v2",
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
        study_lgbm.optimize(objetivo_lgbm, n_trials=N_TRIALS,
                            show_progress_bar=True)

        bp_lgbm = study_lgbm.best_params
        print(f"\n  Mejores parámetros LightGBM:")
        for k, v in bp_lgbm.items():
            print(f"    {k}: {v}")
        print(f"  Mejor AUC CV: {study_lgbm.best_value:.4f}")

        lgbm_tuned = LGBMClassifier(
            **bp_lgbm, bagging_freq=1,
            scale_pos_weight=1.0, random_state=SEED, n_jobs=-1, verbose=-1
        )
        lgbm_tuned.fit(X_train_lgbm, y_train_lgbm)

        study_lgbm.trials_dataframe().to_csv(
            DATA_DIR / "optuna_historial_lightgbm.csv", index=False
        )
        print(f"  ✅ data/optuna_historial_lightgbm.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — TABLA DE HIPERPARÁMETROS: BASE vs TUNEADO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FASE 3 — HIPERPARÁMETROS: BASE vs TUNEADO")
print("="*65)

filas_hiper = []

# LogReg
params_base_lr = {"C": 1.0, "solver": "lbfgs", "penalty": "l2",
                   "class_weight": "balanced", "max_iter": 1000}
if HAS_OPTUNA and lr_tuned:
    params_tuned_lr = lr_tuned.get_params()
    for param in ["C", "solver", "penalty"]:
        v_base = params_base_lr.get(param, "—")
        v_tune = params_tuned_lr.get(param, "—")
        cambio = "✅ MEJORÓ" if param == "C" and v_tune != v_base else (
                 "→ igual" if v_base == v_tune else "↔ cambió")
        filas_hiper.append({
            "Modelo": "LogReg", "Hiperparámetro": param,
            "Base": v_base, "Tuneado": v_tune, "Cambio": cambio
        })

# LightGBM
params_base_lgbm = {"n_estimators": 200, "num_leaves": 31,
                     "learning_rate": 0.1, "min_child_samples": 20,
                     "reg_alpha": 0.0, "reg_lambda": 0.0,
                     "feature_fraction": 1.0, "bagging_fraction": 1.0}
if HAS_LGB and HAS_OPTUNA and lgbm_tuned:
    params_tuned_lgbm = lgbm_tuned.get_params()
    for param in ["n_estimators","num_leaves","learning_rate",
                  "min_child_samples","reg_alpha","reg_lambda",
                  "feature_fraction","bagging_fraction"]:
        v_base = params_base_lgbm.get(param, "—")
        v_tune = params_tuned_lgbm.get(param, "—")
        # Determinar si el cambio es hacia más regularización (mejora)
        regularizacion_mayor = False
        if param in ["reg_alpha","reg_lambda"] and isinstance(v_tune, float):
            regularizacion_mayor = v_tune > v_base
        elif param == "num_leaves" and isinstance(v_tune, (int,float)):
            regularizacion_mayor = v_tune < v_base
        elif param == "min_child_samples" and isinstance(v_tune, (int,float)):
            regularizacion_mayor = v_tune > v_base
        cambio = ("✅ más regularización" if regularizacion_mayor else
                  "→ igual" if v_base == v_tune else "↔ cambió")
        filas_hiper.append({
            "Modelo": "LightGBM", "Hiperparámetro": param,
            "Base": v_base, "Tuneado": round(v_tune,6) if isinstance(v_tune,float) else v_tune,
            "Cambio": cambio
        })

df_hiper = pd.DataFrame(filas_hiper)
if not df_hiper.empty:
    print(df_hiper.to_string(index=False))
    df_hiper.to_csv(DATA_DIR / "tabla_hiperparametros.csv", index=False)
    print(f"\n  ✅ data/tabla_hiperparametros.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — COMPARATIVA BASE vs TUNEADO EN VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FASE 4 — COMPARATIVA BASE vs TUNEADO (Validación)")
print("="*65)

lr_base_final = LogisticRegression(C=1.0, class_weight="balanced",
                                    solver="lbfgs", max_iter=1000,
                                    random_state=SEED)
lr_base_final.fit(X_train, y_train)

modelos_comp = [("LogReg BASE", lr_base_final, X_train, y_train)]
if HAS_OPTUNA and lr_tuned:
    modelos_comp.append(("LogReg TUNEADO", lr_tuned, X_train_lr, y_train_lr))
if HAS_LGB:
    lgbm_base_f = LGBMClassifier(n_estimators=200, scale_pos_weight=scale_pos,
                                  random_state=SEED, n_jobs=-1, verbose=-1)
    lgbm_base_f.fit(X_train, y_train)
    modelos_comp.append(("LightGBM BASE", lgbm_base_f, X_train, y_train))
    if HAS_OPTUNA and lgbm_tuned:
        modelos_comp.append(("LightGBM TUNEADO", lgbm_tuned,
                             X_train_lgbm, y_train_lgbm))

resultados_comp = []
print(f"\n  {'Modelo':<28} {'Train':>8} {'Val':>8} {'Gap':>8}  Estado")
print(f"  {'-'*65}")
for nombre, modelo, Xtr, ytr in modelos_comp:
    r = evaluar(nombre, modelo, X_val, y_val)
    auc_tr = roc_auc_score(ytr, modelo.predict_proba(Xtr)[:,1])
    gap    = auc_tr - r["AUC-ROC"]
    estado = "✅ OK" if gap < 0.10 else ("⚠️ Moderado" if gap < 0.20 else "❌ Sobreajuste")
    print(f"  {nombre:<28} {auc_tr:>8.4f} {r['AUC-ROC']:>8.4f} {gap:>8.4f}  {estado}")
    resultados_comp.append({k:v for k,v in r.items() if k!="y_prob"})

df_comp = pd.DataFrame(resultados_comp)
df_comp.to_csv(DATA_DIR / "tabla_comparativa_tuneo.csv", index=False)
cols_show = ["modelo","AUC-ROC","Gini","PR-AUC","KS","F1","Recall","Precision"]
print(f"\n  MÉTRICAS DETALLADAS VAL:")
print(df_comp[cols_show].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5 — CHAMPION + UMBRAL PARA MAXIMIZAR RECALL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FASE 5 — CHAMPION Y UMBRAL")
print("="*65)

candidatos = {"LogReg BASE": (lr_base_final, roc_auc_score(y_val, lr_base_final.predict_proba(X_val)[:,1]))}
if HAS_OPTUNA and lr_tuned:
    candidatos["LogReg TUNEADO"] = (lr_tuned, roc_auc_score(y_val, lr_tuned.predict_proba(X_val)[:,1]))
if HAS_LGB:
    candidatos["LightGBM BASE"]  = (lgbm_base_f, roc_auc_score(y_val, lgbm_base_f.predict_proba(X_val)[:,1]))
    if HAS_OPTUNA and lgbm_tuned:
        candidatos["LightGBM TUNEADO"] = (lgbm_tuned, roc_auc_score(y_val, lgbm_tuned.predict_proba(X_val)[:,1]))

champ_nombre, (champ_modelo, _) = max(candidatos.items(), key=lambda x: x[1][1])
print(f"\n  Champion: {champ_nombre}")

# Dos umbrales: F1 óptimo y Recall máximo
proba_v = champ_modelo.predict_proba(X_val)[:,1]
u_f1    = optimizar_umbral_f1(y_val, proba_v)
u_rec   = umbral_max_recall(y_val, proba_v, precision_minima=0.08)

print(f"\n  Umbral óptimo F1:           {u_f1}  (balance precisión/recall)")
print(f"  Umbral máximo Recall:       {u_rec}  (prioriza detección, prec≥8%)")

print(f"\n  Con umbral F1 ({u_f1}):")
imprimir(evaluar(champ_nombre, champ_modelo, X_val, y_val, u_f1), "VAL")

print(f"\n  Con umbral Recall ({u_rec}):")
imprimir(evaluar(champ_nombre, champ_modelo, X_val, y_val, u_rec), "VAL")

# Para el negocio logístico se prioriza Recall — usar u_rec
umbral_final = u_rec
print(f"\n  ✅ Umbral final elegido: {umbral_final} (maximiza detección de tardíos)")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 6 — BACKTESTING COMPLETO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FASE 6 — BACKTESTING TEMPORAL")
print("="*65)

resultados_bt = []
for split_name, X_s, y_s in [
    ("VAL",      X_val,  y_val),
    ("BACKTEST", X_back, y_back),
    ("LIVE",     X_live, y_live),
]:
    r = evaluar(champ_nombre, champ_modelo, X_s, y_s, umbral=umbral_final)
    r["split"] = split_name
    imprimir(r, split_name)
    resultados_bt.append(r)

df_bt = pd.DataFrame([{k:v for k,v in r.items() if k!="y_prob"} for r in resultados_bt])
df_bt.to_csv(DATA_DIR / "tabla_backtesting_tuneado.csv", index=False)

print(f"\n  RESUMEN:")
print(df_bt[["split","AUC-ROC","Gini","Recall","F1","Tasa_Deteccion"]].to_string(index=False))

# Comparativa base vs tuneado completa
print(f"\n  BASE vs TUNEADO — Val/Backtest/Live:")
print(f"  {'Modelo':<28} {'Val':>8} {'Back':>8} {'Live':>8}")
print(f"  {'-'*58}")
for nombre, modelo, _, __ in modelos_comp:
    aucs = [roc_auc_score(y_s, modelo.predict_proba(X_s)[:,1])
            for X_s, y_s in [(X_val,y_val),(X_back,y_back),(X_live,y_live)]]
    print(f"  {nombre:<28} {aucs[0]:>8.4f} {aucs[1]:>8.4f} {aucs[2]:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 7 — IMPORTANCIA DE VARIABLES DEL CHAMPION (post-backtest)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FASE 7 — IMPORTANCIA DE VARIABLES DEL CHAMPION")
print("="*65)

importancias_modelo = {}

if "LightGBM" in champ_nombre and hasattr(champ_modelo, "feature_importances_"):
    imps = champ_modelo.feature_importances_
    importancias_modelo = dict(zip(features, imps))
    print("  Método: feature_importances_ (LightGBM gain)")

elif "LogReg" in champ_nombre:
    # Para LogReg: coeficientes absolutos como proxy de importancia
    coefs = np.abs(champ_modelo.coef_[0])
    importancias_modelo = dict(zip(features, coefs))
    print("  Método: |coeficientes| de la Regresión Logística")

if importancias_modelo:
    df_imp_modelo = pd.Series(importancias_modelo).sort_values(ascending=False)
    print(f"\n  Top 20 variables más importantes ({champ_nombre}):")
    print(f"  {'Variable':<40} {'Importancia':>12}")
    print(f"  {'-'*55}")
    for feat, imp in df_imp_modelo.head(20).items():
        print(f"  {feat:<40} {imp:>12.4f}")

# Cargar PSI del 04_seleccion_variables para cruzar información
psi_path = DATA_DIR / "reporte_psi.csv"
psi_data = {}
if psi_path.exists():
    df_psi = pd.read_csv(psi_path, index_col=0)
    psi_data = df_psi.iloc[:,0].to_dict()

# CSV completo: variable | PSI | importancia_modelo | ranking
filas_var = []
for i, feat in enumerate(features):
    imp = importancias_modelo.get(feat, 0)
    filas_var.append({
        "Variable":            feat,
        "PSI":                 round(psi_data.get(feat, 0), 4),
        "Importancia_Modelo":  round(float(imp), 6),
        "Ranking_Importancia": 0,
    })

df_vars = pd.DataFrame(filas_var).sort_values(
    "Importancia_Modelo", ascending=False
).reset_index(drop=True)
df_vars["Ranking_Importancia"] = df_vars.index + 1

df_vars.to_csv(DATA_DIR / "variables_finales_importancia.csv", index=False)
print(f"\n  ✅ data/variables_finales_importancia.csv  ({len(df_vars)} variables)")
print("  Columnas: Variable | PSI | Importancia_Modelo | Ranking_Importancia")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 8 — GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Generando gráficas...")

# Curva ROC base vs tuneado
fig, ax = plt.subplots(figsize=(10, 6))
paleta = {"LogReg BASE":      "#95A5A6",
          "LogReg TUNEADO":   "#E74C3C",
          "LightGBM BASE":    "#2E86AB",
          "LightGBM TUNEADO": "#F39C12"}

for nombre, modelo, _, __ in modelos_comp:
    prob = modelo.predict_proba(X_val)[:,1]
    fpr, tpr, _ = roc_curve(y_val, prob)
    auc  = roc_auc_score(y_val, prob)
    lw   = 2.5 if "TUNEADO" in nombre else 1.2
    ls   = "-"  if "TUNEADO" in nombre else "--"
    ax.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.4f})",
            color=paleta.get(nombre,"#333"), linewidth=lw, linestyle=ls)

ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Aleatorio (0.50)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Curva ROC — Base vs Tuneado (Validación)")
ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR/"curva_roc_tuneado.png", dpi=150)
plt.close()
print("  ✅ reports/curva_roc_tuneado.png")

# Historial Optuna
if HAS_OPTUNA:
    estudios = [("LogReg", study_lr)]
    if HAS_LGB and lgbm_tuned:
        estudios.append(("LightGBM", study_lgbm))

    fig_op, axes_op = plt.subplots(1, len(estudios),
                                    figsize=(7*len(estudios), 5))
    if len(estudios) == 1:
        axes_op = [axes_op]

    for ax_o, (nom, study) in zip(axes_op, estudios):
        vals = [t.value for t in study.trials if t.value is not None]
        ax_o.scatter(range(1,len(vals)+1), vals,
                     alpha=0.3, s=10, color="#6366f1")
        if len(vals) >= 10:
            mm = pd.Series(vals).rolling(10).mean()
            ax_o.plot(range(1,len(vals)+1), mm,
                      color="#E74C3C", linewidth=2, label="Media móvil 10")
        best = max(v for v in vals if v)
        ax_o.axhline(y=best, linestyle="--", color="#27AE60", alpha=0.7,
                     label=f"Mejor: {best:.4f}")
        ax_o.set_title(f"Optuna — {nom} ({len(vals)} trials)")
        ax_o.set_xlabel("Trial"); ax_o.set_ylabel("Score (AUC CV - penalización)")
        ax_o.legend(fontsize=8); ax_o.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORT_DIR/"optuna_historial.png", dpi=150)
    plt.close()
    print("  ✅ reports/optuna_historial.png")

# Importancia de variables del champion
if importancias_modelo:
    top20 = df_vars.head(20)
    fig_i, ax_i = plt.subplots(figsize=(10, 8))
    colores_bar = ["#E74C3C" if i < 3 else "#6366f1" if i < 10 else "#95A5A6"
                   for i in range(len(top20))]
    ax_i.barh(top20["Variable"][::-1], top20["Importancia_Modelo"][::-1],
               color=colores_bar[::-1])
    ax_i.set_xlabel("Importancia")
    ax_i.set_title(f"Top 20 Variables — {champ_nombre}")
    ax_i.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(REPORT_DIR/"importancia_variables_champion.png", dpi=150)
    plt.close()
    print("  ✅ reports/importancia_variables_champion.png")

# Comparativa balanceo
if HAS_IMBL and len(resultados_balanceo) > 0:
    fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 5))
    for ax_b, mod_nm, c_tr, c_v in [
        (axes_b[0], "LogReg",   "#6366f1", "#E74C3C"),
        (axes_b[1], "LightGBM", "#2E86AB", "#F39C12"),
    ]:
        df_m = df_bal[df_bal["modelo"] == mod_nm]
        if df_m.empty:
            continue
        x = range(len(df_m))
        ax_b.bar([i-0.2 for i in x], df_m["AUC_train"],
                  width=0.35, label="Train", color=c_tr, alpha=0.7)
        ax_b.bar([i+0.2 for i in x], df_m["AUC_val"],
                  width=0.35, label="Val",   color=c_v,  alpha=0.7)
        ax_b.set_xticks(list(x))
        ax_b.set_xticklabels(df_m["tecnica"], rotation=20, ha="right")
        ax_b.set_title(f"Balanceo — {mod_nm}"); ax_b.set_ylabel("AUC-ROC")
        ax_b.set_ylim([0.5, 1.05]); ax_b.legend(); ax_b.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(REPORT_DIR/"comparativa_balanceo.png", dpi=150)
    plt.close()
    print("  ✅ reports/comparativa_balanceo.png")

# SHAP si el champion es LightGBM
if HAS_SHAP and "LightGBM" in champ_nombre:
    try:
        explainer   = shap.TreeExplainer(champ_modelo)
        shap_values = explainer.shap_values(X_val)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        fig_s, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_val, feature_names=features,
                          plot_type="bar", show=False, max_display=25)
        plt.title(f"SHAP — {champ_nombre}")
        plt.tight_layout()
        plt.savefig(REPORT_DIR/"shap_tuneado.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✅ reports/shap_tuneado.png")
    except Exception as e:
        print(f"  ⚠️  SHAP: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CHAMPION
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Guardando Champion...")

r_v    = evaluar(champ_nombre, champ_modelo, X_val,  y_val,  umbral_final)
r_back = evaluar(champ_nombre, champ_modelo, X_back, y_back, umbral_final)
r_live = evaluar(champ_nombre, champ_modelo, X_live, y_live, umbral_final)

artefactos = {
    "modelo":              champ_modelo,
    "features_finales":    features,
    "umbral_optimo":       float(umbral_final),
    "umbral_f1":           float(u_f1),
    "umbral_recall":       float(u_rec),
    "nombre_champion":     champ_nombre,
    "mejor_tecnica_lr":    mejor_lr,
    "mejor_tecnica_lgbm":  mejor_lgbm if HAS_LGB else "N/A",
    "version":             "3.0.0",
    "fecha_entrenamiento": str(datetime.now().date()),
    "periodo_train":       "2016-09-15 / 2018-03-31",
    "psi_umbral":          0.15,
    "cv_folds_optuna":     CV_FOLDS,
    "n_trials_optuna":     N_TRIALS,
    "gap_max_penalizacion":GAP_MAX,
    "metricas_val":     {k:v for k,v in r_v.items()    if k not in ["y_prob","modelo"]},
    "metricas_backtest":{k:v for k,v in r_back.items() if k not in ["y_prob","modelo"]},
    "metricas_live":    {k:v for k,v in r_live.items() if k not in ["y_prob","modelo"]},
}

joblib.dump(artefactos, MODEL_DIR/"champion_model_v3.pkl")
resumen = {k:v for k,v in artefactos.items() if k != "modelo"}
with open(MODEL_DIR/"champion_resumen_v3.json","w") as f:
    json.dump(resumen, f, indent=2, default=str)

print("  ✅ models/champion_model_v3.pkl")
print("  ✅ models/champion_resumen_v3.json")

print("\n" + "="*65)
print("  SPRINT 3 — TUNEO COMPLETADO ✅")
print(f"  Champion:         {champ_nombre}")
print(f"  AUC-ROC Val:      {r_v['AUC-ROC']}")
print(f"  AUC-ROC Backtest: {r_back['AUC-ROC']}")
print(f"  AUC-ROC Live:     {r_live['AUC-ROC']}")
print(f"  Recall Val:       {r_v['Recall']}")
print(f"  Umbral final:     {umbral_final} (maximiza Recall, prec≥8%)")
print(f"  CV Optuna:        {CV_FOLDS} folds")
print(f"  Penalización gap: activada si gap > {GAP_MAX}")
print(f"  Balanceo LogReg:  {mejor_lr}")
if HAS_LGB:
    print(f"  Balanceo LightGBM:{mejor_lgbm}")
print("="*65)

print("\n  Archivos CSV generados para el docente:")
print("  ✅ data/tabla_balanceo.csv           — técnicas vs AUC")
print("  ✅ data/tabla_comparativa_tuneo.csv  — base vs tuneado")
print("  ✅ data/tabla_hiperparametros.csv    — qué cambió en cada param")
print("  ✅ data/tabla_backtesting_tuneado.csv— val/back/live")
print("  ✅ data/variables_finales_importancia.csv — variables + PSI + importancia")
print("  ✅ data/optuna_historial_logreg.csv  — 200 trials LogReg")
if HAS_LGB:
    print("  ✅ data/optuna_historial_lightgbm.csv — 200 trials LightGBM")   