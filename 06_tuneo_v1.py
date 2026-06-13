"""
06_tuneo.py
=======================
Modelos a tunear:
  1. Regresión Logística  (champion Sprint 3 — AUC Val 0.7528)
  2. LightGBM             (más prometedor — gap moderado 0.17)

Pipeline:
  FASE 1 — Comparativa de técnicas de balanceo (4 técnicas × 2 modelos)
           RandomOverSampler, SMOTE, SMOTENC, SMOTEENN
           Se evalúa cada combinación con parámetros base sobre Val.
           Se elige la mejor técnica por modelo.

  FASE 2 — Tuneo con Optuna (200 trials por modelo)
           LogReg:   C, solver, max_iter
           LightGBM: num_leaves, learning_rate, n_estimators,
                     reg_alpha, reg_lambda, min_child_samples,
                     feature_fraction, bagging_fraction
           Early stopping en LightGBM: eval_set sobre Val,
           early_stopping_rounds=50.

  FASE 3 — Backtesting completo
           Champion tuneado vs champion base (LogReg Sprint 2)
           sobre Val → Backtest → Live.

  FASE 4 — Gráficas y artefactos
           Curva ROC, curvas de aprendizaje, SHAP, historial Optuna.
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
from sklearn.metrics         import (
    roc_auc_score, roc_curve, f1_score, recall_score,
    precision_score, accuracy_score, average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
    from imblearn.over_sampling  import RandomOverSampler, SMOTE, SMOTENC
    from imblearn.combine        import SMOTEENN
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

# ─── RUTAS ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models");  MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)

TARGET    = "is_late_delivery"
SEED      = 42
N_TRIALS  = 200
CV_FOLDS  = 3

np.random.seed(SEED)

# ─── CARGA ────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  TUNEO — SPRINT 3")
print("  LogReg + LightGBM | 4 técnicas de balanceo | Optuna 200 trials")
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
print(f"  Tasa tardíos — Train: {y_train.mean()*100:.1f}% | "
      f"Val: {y_val.mean()*100:.1f}% | "
      f"Back: {y_back.mean()*100:.1f}% | "
      f"Live: {y_live.mean()*100:.1f}%")

# Identificar columnas categóricas para SMOTENC
# (las que tienen pocos valores únicos enteros — categóricas WOE-encoded
#  se tratan como continuas; las binarias/ordinales como categóricas)
COLS_CATEGORICAS_IDX = [
    features.index(f) for f in features
    if f in [
        "is_weekend", "is_off_hours", "high_risk_day", "is_peak_season",
        "is_festivo", "is_short_promise", "same_state", "complex_route",
        "is_long_distance", "has_oversized_item", "is_heavy",
        "carrito_complejo", "tiene_varios_vendedores", "is_boleto",
        "is_high_installments", "high_freight_flag", "seller_is_experienced",
        "bin_hora", "approval_bin", "logistic_complexity",
        "freight_category",
    ]
    and f in features
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
        "PR-AUC": round(pr_auc, 4), "KS": round(ks, 4),
        "F1": round(f1, 4), "Recall": round(rec, 4),
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

def optimizar_umbral(y_true, y_prob):
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
print("  FASE 1 — COMPARATIVA DE TÉCNICAS DE BALANCEO")
print("="*65)
print("""
  Técnicas evaluadas:
  ┌─────────────────┬─────────────────────────────────────────────┐
  │ Sin balanceo    │ Solo class_weight='balanced'                │
  │ RandomOverSample│ Duplica aleatoriamente muestras minoritarias│
  │ SMOTE           │ Genera ejemplos sintéticos interpolando     │
  │ SMOTENC         │ SMOTE para datasets con features categóricas│
  │ SMOTEENN        │ SMOTE + limpieza con Edited Nearest Neighbor│
  └─────────────────┴─────────────────────────────────────────────┘
  Nota: todas las técnicas se aplican SOLO sobre X_train.
        Val, Backtest y Live no se modifican.
""")

if not HAS_IMBL:
    print("  ⚠️  imbalanced-learn no disponible. Salteando Fase 1.")
    print("  Ejecutar: pip install imbalanced-learn")
    mejor_tecnica_lr   = "sin_balanceo"
    mejor_tecnica_lgbm = "sin_balanceo"
    X_train_lr   = X_train.copy()
    y_train_lr   = y_train.copy()
    X_train_lgbm = X_train.copy()
    y_train_lgbm = y_train.copy()
else:
    def aplicar_balanceo(nombre, X, y):
        """
        Aplica la técnica de balanceo indicada sobre X_train.
        Retorna X_resampled, y_resampled.
        """
        if nombre == "sin_balanceo":
            return X.copy(), y.copy()
        elif nombre == "RandomOverSampler":
            ros = RandomOverSampler(random_state=SEED)
            return ros.fit_resample(X, y)
        elif nombre == "SMOTE":
            sm = SMOTE(random_state=SEED, k_neighbors=5)
            return sm.fit_resample(X, y)
        elif nombre == "SMOTENC":
            # SMOTENC requiere índices de columnas categóricas
            if len(COLS_CATEGORICAS_IDX) == 0:
                print("    ⚠️  Sin columnas categóricas detectadas, "
                      "usando SMOTE en su lugar")
                sm = SMOTE(random_state=SEED, k_neighbors=5)
                return sm.fit_resample(X, y)
            smtnc = SMOTENC(
                categorical_features=COLS_CATEGORICAS_IDX,
                random_state=SEED,
                k_neighbors=5,
            )
            return smtnc.fit_resample(X, y)
        elif nombre == "SMOTEENN":
            smenn = SMOTEENN(random_state=SEED)
            return smenn.fit_resample(X, y)
        else:
            return X.copy(), y.copy()

    tecnicas = ["sin_balanceo", "RandomOverSampler",
                "SMOTE", "SMOTENC", "SMOTEENN"]

    # Modelos base (sin tuneo) para comparar balanceo
    lr_base   = LogisticRegression(
        C=1.0, class_weight="balanced",
        solver="lbfgs", max_iter=1000, random_state=SEED
    )
    lgbm_base = None
    if HAS_LGB:
        lgbm_base = LGBMClassifier(
            n_estimators=200,
            scale_pos_weight=scale_pos,
            random_state=SEED, n_jobs=-1, verbose=-1
        )

    resultados_balanceo = []

    for tecnica in tecnicas:
        print(f"\n  ── {tecnica} ──")
        try:
            Xr, yr = aplicar_balanceo(tecnica, X_train, y_train)
            n0 = (yr == 0).sum()
            n1 = (yr == 1).sum()
            print(f"     Clase 0: {n0:,}  |  Clase 1: {n1:,}  "
                  f"|  Total: {len(yr):,}")

            # LogReg
            lr_base.fit(Xr, yr)
            auc_lr_tr = roc_auc_score(
                y_train, lr_base.predict_proba(X_train)[:, 1]
            )
            auc_lr_v = roc_auc_score(
                y_val, lr_base.predict_proba(X_val)[:, 1]
            )
            print(f"     LogReg  → AUC Train: {auc_lr_tr:.4f} | "
                  f"AUC Val: {auc_lr_v:.4f} | "
                  f"Gap: {auc_lr_tr - auc_lr_v:.4f}")

            fila = {
                "tecnica": tecnica,
                "modelo": "LogReg",
                "AUC_train": round(auc_lr_tr, 4),
                "AUC_val":   round(auc_lr_v, 4),
                "gap":       round(auc_lr_tr - auc_lr_v, 4),
                "n_clase1":  int(n1),
            }
            resultados_balanceo.append(fila)

            # LightGBM
            if HAS_LGB and lgbm_base is not None:
                lgbm_base.set_params(scale_pos_weight=1.0)
                lgbm_base.fit(Xr, yr)
                auc_lgbm_tr = roc_auc_score(
                    y_train, lgbm_base.predict_proba(X_train)[:, 1]
                )
                auc_lgbm_v = roc_auc_score(
                    y_val, lgbm_base.predict_proba(X_val)[:, 1]
                )
                print(f"     LightGBM → AUC Train: {auc_lgbm_tr:.4f} | "
                      f"AUC Val: {auc_lgbm_v:.4f} | "
                      f"Gap: {auc_lgbm_tr - auc_lgbm_v:.4f}")
                resultados_balanceo.append({
                    "tecnica": tecnica,
                    "modelo":  "LightGBM",
                    "AUC_train": round(auc_lgbm_tr, 4),
                    "AUC_val":   round(auc_lgbm_v,  4),
                    "gap":       round(auc_lgbm_tr - auc_lgbm_v, 4),
                    "n_clase1":  int(n1),
                })
        except Exception as e:
            print(f"     ⚠️  Error con {tecnica}: {e}")

    df_balanceo = pd.DataFrame(resultados_balanceo)
    df_balanceo.to_csv(DATA_DIR / "tabla_balanceo.csv", index=False)

    print("\n" + "─"*65)
    print("  RESUMEN COMPARATIVA DE BALANCEO")
    print("─"*65)
    print(df_balanceo.to_string(index=False))

    # Elegir mejor técnica por modelo (mayor AUC Val)
    mejor_tecnica_lr = (
        df_balanceo[df_balanceo["modelo"] == "LogReg"]
        .sort_values("AUC_val", ascending=False)
        .iloc[0]["tecnica"]
    )
    print(f"\n  ✅ Mejor técnica para LogReg:    {mejor_tecnica_lr}")

    mejor_tecnica_lgbm = "sin_balanceo"
    if HAS_LGB:
        mejor_tecnica_lgbm = (
            df_balanceo[df_balanceo["modelo"] == "LightGBM"]
            .sort_values("AUC_val", ascending=False)
            .iloc[0]["tecnica"]
        )
        print(f"  ✅ Mejor técnica para LightGBM:  {mejor_tecnica_lgbm}")

    # Preparar datasets con la mejor técnica para cada modelo
    X_train_lr, y_train_lr     = aplicar_balanceo(mejor_tecnica_lr, X_train, y_train)
    X_train_lgbm, y_train_lgbm = aplicar_balanceo(mejor_tecnica_lgbm, X_train, y_train)

    print(f"\n  Train LogReg  post-balanceo:  {len(y_train_lr):,} filas")
    if HAS_LGB:
        print(f"  Train LightGBM post-balanceo: {len(y_train_lgbm):,} filas")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — TUNEO CON OPTUNA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"  FASE 2 — TUNEO CON OPTUNA ({N_TRIALS} trials por modelo)")
print("="*65)

if not HAS_OPTUNA:
    print("  ⚠️  Optuna no disponible. Salteando Fase 2.")
    print("  Ejecutar: pip install optuna")
    lr_tuned   = None
    lgbm_tuned = None
else:
    # ── 2A. OPTUNA — LOGISTIC REGRESSION ─────────────────────────────────────
    print("\n▶ Tuneando Regresión Logística...")
    print(f"  Espacio de búsqueda: C, solver, max_iter")
    print(f"  Criterio: AUC-ROC sobre Val (CV {CV_FOLDS} folds en train)")

    def objetivo_lr(trial):
        C       = trial.suggest_float("C", 1e-4, 10.0, log=True)
        solver  = trial.suggest_categorical(
            "solver", ["lbfgs", "saga", "liblinear"]
        )
        # liblinear no soporta multinomial — solo para binario
        penalty = "l2"
        if solver == "saga":
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        elif solver == "liblinear":
            penalty = trial.suggest_categorical("penalty_ll", ["l1", "l2"])
            penalty = penalty  # liblinear soporta l1 y l2

        modelo = LogisticRegression(
            C=C,
            solver=solver,
            penalty=penalty if solver != "liblinear" else trial.params.get("penalty_ll", "l2"),
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )
        modelo.fit(X_train_lr, y_train_lr)
        return roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])

    study_lr = optuna.create_study(
        direction="maximize",
        study_name="LogReg_tuneo",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study_lr.optimize(objetivo_lr, n_trials=N_TRIALS, show_progress_bar=True)

    best_lr_params = study_lr.best_params
    print(f"\n  Mejores parámetros LogReg:")
    for k, v in best_lr_params.items():
        print(f"    {k}: {v}")
    print(f"  Mejor AUC Val: {study_lr.best_value:.4f}")

    # Reentrenar con los mejores parámetros
    solver_best  = best_lr_params.get("solver", "lbfgs")
    penalty_best = best_lr_params.get(
        "penalty",
        best_lr_params.get("penalty_ll", "l2")
    )
    lr_tuned = LogisticRegression(
        C=best_lr_params["C"],
        solver=solver_best,
        penalty=penalty_best,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    lr_tuned.fit(X_train_lr, y_train_lr)

    # Guardar historial Optuna LogReg
    df_hist_lr = study_lr.trials_dataframe()
    df_hist_lr.to_csv(DATA_DIR / "optuna_historial_logreg.csv", index=False)
    print(f"  ✅ data/optuna_historial_logreg.csv")

    # ── 2B. OPTUNA — LIGHTGBM ─────────────────────────────────────────────────
    lgbm_tuned = None
    if HAS_LGB:
        print("\n▶ Tuneando LightGBM...")
        print(f"  Espacio de búsqueda: num_leaves, lr, reg_alpha, reg_lambda,")
        print(f"                       min_child_samples, feature_fraction,")
        print(f"                       bagging_fraction, n_estimators")
        print(f"  Early stopping: 50 rounds sobre Val")

        def objetivo_lgbm(trial):
            params = {
                "n_estimators":      trial.suggest_int(
                    "n_estimators", 100, 1000
                ),
                "num_leaves":        trial.suggest_int(
                    "num_leaves", 10, 100
                ),
                "learning_rate":     trial.suggest_float(
                    "learning_rate", 0.005, 0.1, log=True
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 20, 300
                ),
                "reg_alpha":         trial.suggest_float(
                    "reg_alpha", 1e-4, 10.0, log=True
                ),
                "reg_lambda":        trial.suggest_float(
                    "reg_lambda", 1e-4, 10.0, log=True
                ),
                "feature_fraction":  trial.suggest_float(
                    "feature_fraction", 0.5, 1.0
                ),
                "bagging_fraction":  trial.suggest_float(
                    "bagging_fraction", 0.5, 1.0
                ),
                "bagging_freq":      1,
                "scale_pos_weight":  1.0,   # sin peso extra — ya balanceamos
                "random_state":      SEED,
                "n_jobs":            -1,
                "verbose":           -1,
            }

            modelo = LGBMClassifier(**params)
            modelo.fit(
                X_train_lgbm, y_train_lgbm,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[
                    # Early stopping: para si no mejora en 50 rondas
                    optuna.integration.lightgbm.early_stopping(
                        stopping_rounds=50, verbose=False
                    )
                    if hasattr(optuna.integration, "lightgbm")
                    else None
                ],
            )
            return roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])

        study_lgbm = optuna.create_study(
            direction="maximize",
            study_name="LightGBM_tuneo",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study_lgbm.optimize(
            objetivo_lgbm, n_trials=N_TRIALS, show_progress_bar=True
        )

        best_lgbm_params = study_lgbm.best_params
        print(f"\n  Mejores parámetros LightGBM:")
        for k, v in best_lgbm_params.items():
            print(f"    {k}: {v}")
        print(f"  Mejor AUC Val: {study_lgbm.best_value:.4f}")

        # Reentrenar con mejores parámetros (sin early stopping)
        lgbm_tuned = LGBMClassifier(
            **best_lgbm_params,
            bagging_freq=1,
            scale_pos_weight=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm_tuned.fit(X_train_lgbm, y_train_lgbm)

        df_hist_lgbm = study_lgbm.trials_dataframe()
        df_hist_lgbm.to_csv(
            DATA_DIR / "optuna_historial_lightgbm.csv", index=False
        )
        print(f"  ✅ data/optuna_historial_lightgbm.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — COMPARATIVA: BASE vs TUNEADO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FASE 3 — COMPARATIVA BASE vs TUNEADO (Validación)")
print("="*65)

# Modelos base del Sprint 4 para comparar
lr_base_final = LogisticRegression(
    C=1.0, class_weight="balanced",
    solver="lbfgs", max_iter=1000, random_state=SEED
)
lr_base_final.fit(X_train, y_train)

modelos_comparar = [
    ("LogReg BASE (Sprint 2)", lr_base_final),
]
if lr_tuned:
    modelos_comparar.append(("LogReg TUNEADO", lr_tuned))
if HAS_LGB:
    lgbm_base_final = LGBMClassifier(
        n_estimators=200, scale_pos_weight=scale_pos,
        random_state=SEED, n_jobs=-1, verbose=-1
    )
    lgbm_base_final.fit(X_train, y_train)
    modelos_comparar.append(("LightGBM BASE (Sprint 2)", lgbm_base_final))
    if lgbm_tuned:
        modelos_comparar.append(("LightGBM TUNEADO", lgbm_tuned))

resultados_comp = []
for nombre, modelo in modelos_comparar:
    r = evaluar(nombre, modelo, X_val, y_val)
    imprimir(r, "VAL")
    resultados_comp.append({k: v for k, v in r.items() if k != "y_prob"})

df_comp = pd.DataFrame(resultados_comp)
cols_show = ["modelo","AUC-ROC","Gini","PR-AUC","KS","F1","Recall","Precision"]
print(f"\n  RESUMEN COMPARATIVA:")
print(df_comp[cols_show].to_string(index=False))
df_comp.to_csv(DATA_DIR / "tabla_comparativa_tuneo.csv", index=False)

# Brecha sobreajuste post-tuneo
print(f"\n  BRECHA SOBREAJUSTE (Train → Val) POST-TUNEO:")
print(f"  {'Modelo':<30} {'Train':>8} {'Val':>8} {'Gap':>8}  Estado")
print(f"  {'-'*65}")
for nombre, modelo in modelos_comparar:
    X_tr_uso = X_train_lr if "LogReg" in nombre else X_train_lgbm
    y_tr_uso = y_train_lr if "LogReg" in nombre else y_train_lgbm
    auc_tr = roc_auc_score(y_tr_uso, modelo.predict_proba(X_tr_uso)[:, 1])
    auc_v  = roc_auc_score(y_val,    modelo.predict_proba(X_val)[:, 1])
    gap    = auc_tr - auc_v
    estado = "✅ OK" if gap < 0.10 else ("⚠️ Moderado" if gap < 0.20 else "❌ Sobreajuste")
    print(f"  {nombre:<30} {auc_tr:>8.4f} {auc_v:>8.4f} {gap:>8.4f}  {estado}")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3B — ELEGIR CHAMPION TUNEADO Y BACKTESTING COMPLETO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  BACKTESTING TEMPORAL — MODELO GANADOR TUNEADO")
print("="*65)

# El champion tuneado es el de mayor AUC Val entre los tuneados
candidatos_tuneados = {}
if lr_tuned:
    auc_lr_v = roc_auc_score(y_val, lr_tuned.predict_proba(X_val)[:, 1])
    candidatos_tuneados["LogReg TUNEADO"] = (lr_tuned, auc_lr_v)
if HAS_LGB and lgbm_tuned:
    auc_lgbm_v = roc_auc_score(y_val, lgbm_tuned.predict_proba(X_val)[:, 1])
    candidatos_tuneados["LightGBM TUNEADO"] = (lgbm_tuned, auc_lgbm_v)

if not candidatos_tuneados:
    print("  ⚠️  No hay modelos tuneados. Usando LogReg base.")
    champion_nombre = "LogReg BASE"
    champion_modelo = lr_base_final
else:
    champion_nombre, (champion_modelo, _) = max(
        candidatos_tuneados.items(), key=lambda x: x[1][1]
    )

print(f"\n  ✅ Champion tuneado: {champion_nombre}")

# Umbral óptimo sobre Val
proba_val   = champion_modelo.predict_proba(X_val)[:, 1]
umbral_opt  = optimizar_umbral(y_val, proba_val)
print(f"  Umbral óptimo (F1 sobre Val): {umbral_opt}")

resultados_bt = []
for split_name, X_s, y_s in [
    ("VAL",      X_val,  y_val),
    ("BACKTEST", X_back, y_back),
    ("LIVE",     X_live, y_live),
]:
    r = evaluar(champion_nombre, champion_modelo, X_s, y_s, umbral=umbral_opt)
    r["split"] = split_name
    imprimir(r, split_name)
    resultados_bt.append(r)

df_bt = pd.DataFrame([
    {k: v for k, v in r.items() if k != "y_prob"}
    for r in resultados_bt
])
df_bt.to_csv(DATA_DIR / "tabla_backtesting_tuneado.csv", index=False)

print(f"\n  RESUMEN BACKTESTING:")
print(df_bt[["split","AUC-ROC","Gini","Recall","F1",
             "Tasa_Deteccion"]].to_string(index=False))

# Comparativa base vs tuneado en backtesting
print(f"\n  COMPARATIVA VAL → BACKTEST → LIVE:")
print(f"  {'Modelo':<30} {'Val':>8} {'Backtest':>9} {'Live':>7}")
print(f"  {'-'*58}")
for nombre, modelo in [
    ("LogReg BASE",   lr_base_final),
    (champion_nombre, champion_modelo),
]:
    aucs = []
    for X_s, y_s in [(X_val, y_val), (X_back, y_back), (X_live, y_live)]:
        aucs.append(roc_auc_score(y_s, modelo.predict_proba(X_s)[:, 1]))
    print(f"  {nombre:<30} {aucs[0]:>8.4f} {aucs[1]:>9.4f} {aucs[2]:>7.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Generando gráficas...")

# ── Curva ROC comparativa ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
paleta = {
    "LogReg BASE (Sprint 2)":   "#95A5A6",
    "LogReg TUNEADO":           "#E74C3C",
    "LightGBM BASE (Sprint 2)": "#2E86AB",
    "LightGBM TUNEADO":         "#F39C12",
}
for nombre, modelo in modelos_comparar:
    prob = modelo.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, prob)
    auc = roc_auc_score(y_val, prob)
    lw  = 2.5 if "TUNEADO" in nombre else 1.2
    ls  = "-" if "TUNEADO" in nombre else "--"
    color = paleta.get(nombre, "#333333")
    ax.plot(fpr, tpr,
            label=f"{nombre} (AUC={auc:.4f})",
            color=color, linewidth=lw, linestyle=ls)

ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Aleatorio (0.50)")
ax.set_xlabel("False Positive Rate (1 - Especificidad)")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("Curva ROC — Base vs Tuneado (Validación)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR / "curva_roc_tuneado.png", dpi=150)
plt.close()
print("  ✅ reports/curva_roc_tuneado.png")

# ── Historial Optuna ─────────────────────────────────────────────────────────
if HAS_OPTUNA:
    fig_op, axes_op = plt.subplots(
        1, 2 if HAS_LGB and lgbm_tuned else 1,
        figsize=(14 if HAS_LGB and lgbm_tuned else 7, 5)
    )
    if not isinstance(axes_op, np.ndarray):
        axes_op = [axes_op]

    estudios = [("LogReg", study_lr)]
    if HAS_LGB and lgbm_tuned:
        estudios.append(("LightGBM", study_lgbm))

    for ax_o, (nom, study) in zip(axes_op, estudios):
        vals = [t.value for t in study.trials if t.value is not None]
        ax_o.plot(range(1, len(vals)+1), vals,
                  alpha=0.4, color="#6366f1", linewidth=0.8)
        # Media móvil
        if len(vals) >= 10:
            mm = pd.Series(vals).rolling(10).mean()
            ax_o.plot(range(1, len(vals)+1), mm,
                      color="#E74C3C", linewidth=2, label="Media móvil 10")
        ax_o.axhline(y=max(v for v in vals if v),
                     linestyle="--", color="#27AE60", alpha=0.7,
                     label=f"Mejor: {max(v for v in vals if v):.4f}")
        ax_o.set_title(f"Historial Optuna — {nom}")
        ax_o.set_xlabel("Trial")
        ax_o.set_ylabel("AUC-ROC Val")
        ax_o.legend(fontsize=8)
        ax_o.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "optuna_historial.png", dpi=150)
    plt.close()
    print("  ✅ reports/optuna_historial.png")

# ── SHAP del champion tuneado ────────────────────────────────────────────────
if HAS_SHAP and "LightGBM" in champion_nombre:
    try:
        explainer   = shap.TreeExplainer(champion_modelo)
        shap_values = explainer.shap_values(X_val)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        fig_s, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_val, feature_names=features,
                          plot_type="bar", show=False, max_display=25)
        plt.title(f"SHAP — {champion_nombre} (Validación)")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_tuneado.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✅ reports/shap_tuneado.png")
    except Exception as e:
        print(f"  ⚠️  SHAP falló: {e}")

# ── Curvas de balanceo ───────────────────────────────────────────────────────
if HAS_IMBL and len(resultados_balanceo) > 0:
    fig_b, (ax_lr, ax_lgbm) = plt.subplots(1, 2, figsize=(14, 5))

    df_b = pd.DataFrame(resultados_balanceo)
    for ax_b, modelo_nm, color_tr, color_v in [
        (ax_lr,   "LogReg",   "#6366f1", "#E74C3C"),
        (ax_lgbm, "LightGBM", "#2E86AB", "#F39C12"),
    ]:
        df_m = df_b[df_b["modelo"] == modelo_nm]
        if df_m.empty:
            continue
        x = range(len(df_m))
        ax_b.bar([i-0.2 for i in x], df_m["AUC_train"],
                  width=0.35, label="Train", color=color_tr, alpha=0.7)
        ax_b.bar([i+0.2 for i in x], df_m["AUC_val"],
                  width=0.35, label="Val",   color=color_v,  alpha=0.7)
        ax_b.set_xticks(list(x))
        ax_b.set_xticklabels(df_m["tecnica"], rotation=20, ha="right", fontsize=9)
        ax_b.set_title(f"Técnicas de balanceo — {modelo_nm}")
        ax_b.set_ylabel("AUC-ROC")
        ax_b.set_ylim([0.5, 1.05])
        ax_b.legend()
        ax_b.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "comparativa_balanceo.png", dpi=150)
    plt.close()
    print("  ✅ reports/comparativa_balanceo.png")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CHAMPION TUNEADO
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Guardando Champion Tuneado...")

r_champ_v    = evaluar(champion_nombre, champion_modelo, X_val,  y_val,  umbral_opt)
r_champ_back = evaluar(champion_nombre, champion_modelo, X_back, y_back, umbral_opt)
r_champ_live = evaluar(champion_nombre, champion_modelo, X_live, y_live, umbral_opt)

artefactos = {
    "modelo":              champion_modelo,
    "features_finales":    features,
    "umbral_optimo":       float(umbral_opt),
    "nombre_champion":     champion_nombre,
    "mejor_tecnica_lr":    mejor_tecnica_lr,
    "mejor_tecnica_lgbm":  mejor_tecnica_lgbm if HAS_LGB else "N/A",
    "version":             "3.0.0",
    "fecha_entrenamiento": str(datetime.now().date()),
    "periodo_train":       "2016-09-15 / 2018-03-31",
    "psi_umbral":          0.15,
    "n_trials_optuna":     N_TRIALS,
    "metricas_val": {
        k: v for k, v in r_champ_v.items()
        if k not in ["y_prob", "modelo"]
    },
    "metricas_backtest": {
        k: v for k, v in r_champ_back.items()
        if k not in ["y_prob", "modelo"]
    },
    "metricas_live": {
        k: v for k, v in r_champ_live.items()
        if k not in ["y_prob", "modelo"]
    },
}

joblib.dump(artefactos, MODEL_DIR / "champion_model_v3.pkl")
resumen = {k: v for k, v in artefactos.items() if k != "modelo"}
with open(MODEL_DIR / "champion_resumen_v3.json", "w") as f:
    json.dump(resumen, f, indent=2, default=str)

print("  ✅ models/champion_model_v3.pkl")
print("  ✅ models/champion_resumen_v3.json")

print("\n" + "="*65)
print("  SPRINT 3 — TUNEO COMPLETADO ✅")
print(f"  Champion: {champion_nombre}")
print(f"  AUC-ROC Val:      {r_champ_v['AUC-ROC']}")
print(f"  AUC-ROC Backtest: {r_champ_back['AUC-ROC']}")
print(f"  AUC-ROC Live:     {r_champ_live['AUC-ROC']}")
print(f"  Umbral óptimo:    {umbral_opt}")
print(f"  Técnica balanceo LogReg:   {mejor_tecnica_lr}")
if HAS_LGB:
    print(f"  Técnica balanceo LightGBM: {mejor_tecnica_lgbm}")
print("="*65)