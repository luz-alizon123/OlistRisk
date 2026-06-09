"""
05_modelado.py — SPRINT 2 (v2 CORREGIDO)
==========================================
Correcciones aplicadas:
  • Regularización fuerte en LightGBM y XGBoost (anti-overfitting)
  • Stacking reemplazado por Soft Voting
  • Curvas de aprendizaje para demostrar control de sobreajuste
  • Métricas ampliadas: Curva de Ganancia, Curva de Elevación, KS
  • Comparación Soft Voting vs modelos individuales
  • Backtesting sobre val / backtest / live
  • SHAP para interpretabilidad
"""
 
import pandas as pd
import numpy as np
import warnings
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
from pathlib import Path
from datetime import datetime
 
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (RandomForestClassifier,
                                     VotingClassifier)
from sklearn.model_selection import learning_curve
from sklearn.metrics         import (
    roc_auc_score, roc_curve, f1_score, recall_score,
    precision_score, accuracy_score, average_precision_score,
    confusion_matrix, precision_recall_curve,
)
 
warnings.filterwarnings("ignore")
 
try:
    from xgboost  import XGBClassifier;  HAS_XGB  = True
except ImportError:
    HAS_XGB  = False; print("  ⚠️  xgboost no instalado")
 
try:
    from lightgbm import LGBMClassifier; HAS_LGB  = True
except ImportError:
    HAS_LGB  = False; print("  ⚠️  lightgbm no instalado")
 
try:
    import shap;                          HAS_SHAP = True
except ImportError:
    HAS_SHAP = False; print("  ⚠️  shap no instalado")
 
# ─── RUTAS ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models");  MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)
 
TARGET = "is_late_delivery"
SEED   = 42
np.random.seed(SEED)
 
# ─── CARGA ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  MODELADO v2 — OLIST (con regularización + Soft Voting)")
print("=" * 70)
 
# Para entrenamiento: usar train balanceado (undersampled)
# Para reportes/backtesting: usar train_original (distribución real)
train_bal  = pd.read_parquet(DATA_DIR / "split_train.parquet")
try:
    train_orig = pd.read_parquet(DATA_DIR / "split_train_original.parquet")
except FileNotFoundError:
    train_orig = train_bal.copy()
    print("  ⚠️  train_original no encontrado, usando train balanceado")
 
val      = pd.read_parquet(DATA_DIR / "split_val.parquet")
backtest = pd.read_parquet(DATA_DIR / "split_backtest.parquet")
live     = pd.read_parquet(DATA_DIR / "split_live.parquet")
 
features_finales = joblib.load(DATA_DIR / "features_finales.pkl")
features_finales = [f for f in features_finales if f in train_bal.columns]
print(f"\n  Features del modelo: {len(features_finales)}")
print(f"  {features_finales}")
 
X_train = train_bal[features_finales].fillna(0)
y_train = train_bal[TARGET]
 
# Para métricas en train usar la distribución original (más representativa)
X_train_o = train_orig[features_finales].fillna(0)
y_train_o = train_orig[TARGET]
 
X_val   = val[features_finales].fillna(0)
y_val   = val[TARGET]
X_back  = backtest[features_finales].fillna(0)
y_back  = backtest[TARGET]
X_live  = live[features_finales].fillna(0)
y_live  = live[TARGET]
 
scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f"\n  Train balanceado: {X_train.shape}  | Tasa tardíos: {y_train.mean()*100:.1f}%")
print(f"  Train original:   {X_train_o.shape} | Tasa tardíos: {y_train_o.mean()*100:.1f}%")
print(f"  Val:              {X_val.shape}      | Tasa tardíos: {y_val.mean()*100:.1f}%")
print(f"  Backtest:         {X_back.shape}     | Tasa tardíos: {y_back.mean()*100:.1f}%")
print(f"  Live:             {X_live.shape}     | Tasa tardíos: {y_live.mean()*100:.1f}%")
 
 
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
    gap = f" [{split_name}]" if split_name else ""
    print(f"\n  {'─'*60}")
    print(f"  {r['modelo']}{gap}")
    print(f"  {'─'*60}")
    print(f"  AUC-ROC : {r['AUC-ROC']}   Gini   : {r['Gini']}")
    print(f"  PR-AUC  : {r['PR-AUC']}   KS     : {r['KS']}")
    print(f"  F1      : {r['F1']}   Recall : {r['Recall']}")
    print(f"  Precision:{r['Precision']}   Accuracy:{r['Accuracy']}")
    print(f"  TP={r['TP']}  FP={r['FP']}  TN={r['TN']}  FN={r['FN']}")
    print(f"  Tasa Detección: {r['Tasa_Deteccion']*100:.1f}%")
    print(f"  Ahorro potencial: R$ {r['Ahorro_R$']:,.0f}")
 
def optimizar_umbral(y_true, y_prob):
    mejor_u, mejor_f1 = 0.5, 0
    for u in np.arange(0.10, 0.90, 0.01):
        yp = (y_prob >= u).astype(int)
        f  = f1_score(y_true, yp, zero_division=0)
        if f > mejor_f1:
            mejor_f1 = f
            mejor_u  = u
    return round(mejor_u, 2)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MODELO 1 — LOGISTIC REGRESSION (Baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  MODELO 1 — LOGISTIC REGRESSION (Baseline)")
print("═"*70)
 
lr = LogisticRegression(
    C=0.05, class_weight="balanced",
    solver="lbfgs", max_iter=3000, random_state=SEED
)
lr.fit(X_train, y_train)
imprimir(evaluar("LogReg", lr, X_train_o, y_train_o), "TRAIN")
r_lr_val = evaluar("LogReg", lr, X_val, y_val)
imprimir(r_lr_val, "VAL")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MODELO 2 — RANDOM FOREST (regularizado)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  MODELO 2 — RANDOM FOREST")
print("═"*70)
 
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced",
    random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
imprimir(evaluar("RandomForest", rf, X_train_o, y_train_o), "TRAIN")
r_rf_val = evaluar("RandomForest", rf, X_val, y_val)
imprimir(r_rf_val, "VAL")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MODELO 3 — XGBOOST (regularización fuerte anti-overfitting)
# ══════════════════════════════════════════════════════════════════════════════
# ⚠️ CORRECCIÓN: En v1, XGBoost tenia AUC Train 0.86 vs Val 0.56 (brecha 0.30)
# Se aplica: max_depth reducido, min_child_weight alto, subsample bajo,
# reg_alpha y reg_lambda fuertes, early stopping implícito via n_estimators bajo
xgb = None
r_xgb_val = None
if HAS_XGB:
    print("\n" + "═"*70)
    print("  MODELO 3 — XGBOOST (regularizado)")
    print("═"*70)
 
    xgb = XGBClassifier(
        n_estimators=200,          # ↓ de 400 a 200
        max_depth=3,               # ↓ de 5 a 3 (árboles poco profundos)
        learning_rate=0.05,
        subsample=0.7,             # ↓ más subsampling
        colsample_bytree=0.7,
        min_child_weight=10,       # ↑ mínimo de muestras por hoja
        gamma=1.0,                 # regularización por ganancia mínima
        reg_alpha=0.5,             # L1
        reg_lambda=2.0,            # L2
        scale_pos_weight=1.0,      # ya balanceado por undersampling
        eval_metric="auc",
        random_state=SEED, n_jobs=-1, verbosity=0
    )
    xgb.fit(X_train, y_train)
    imprimir(evaluar("XGBoost", xgb, X_train_o, y_train_o), "TRAIN")
    r_xgb_val = evaluar("XGBoost", xgb, X_val, y_val)
    imprimir(r_xgb_val, "VAL")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MODELO 4 — LIGHTGBM (regularización fuerte anti-overfitting)
# ══════════════════════════════════════════════════════════════════════════════
# ⚠️ CORRECCIÓN: En v1, LightGBM tenia AUC Train 0.94 vs Val 0.49 (brecha 0.45!)
# Sobreajuste extremo. Se aplica regularización fuerte + num_leaves bajo.
lgbm = None
r_lgbm_val = None
if HAS_LGB:
    print("\n" + "═"*70)
    print("  MODELO 4 — LIGHTGBM (regularizado)")
    print("═"*70)
 
    lgbm = LGBMClassifier(
        n_estimators=300,          # ↓ de 500 a 300
        num_leaves=15,             # ↓↓ de 63 a 15 (clave para reducir overfitting)
        max_depth=4,               # ↓ nuevo límite de profundidad
        learning_rate=0.05,        # ↑ ligeramente para compensar menos árboles
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=30,      # ↑ mínimo de muestras por hoja
        reg_alpha=1.0,             # ↑ L1 fuerte
        reg_lambda=2.0,            # ↑ L2 fuerte
        scale_pos_weight=1.0,      # ya balanceado
        random_state=SEED, n_jobs=-1, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    imprimir(evaluar("LightGBM", lgbm, X_train_o, y_train_o), "TRAIN")
    r_lgbm_val = evaluar("LightGBM", lgbm, X_val, y_val)
    imprimir(r_lgbm_val, "VAL")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MODELO 5 — SOFT VOTING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
# ⚠️ CORRECCIÓN: Stacking v1 estaba roto (TN=0, predice todo positivo).
# Soft Voting promedia las probabilidades de todos los modelos base.
# Es más robusto que Stacking porque no requiere meta-modelo entrenado
# con CV sobre datos temporales (que era el punto de fallo).
 
print("\n" + "═"*70)
print("  MODELO 5 — SOFT VOTING ENSEMBLE")
print("═"*70)
print("  Estrategia: promedio ponderado de probabilidades (soft)")
print("  Ventaja vs Stacking: sin riesgo de data leakage temporal")
print("  Pesos: LightGBM×2, XGBoost×2, RandomForest×1.5, LogReg×1")
 
estimadores_vote = [("lr", lr), ("rf", rf)]
pesos = [1.0, 1.5]
if HAS_XGB and xgb is not None:
    estimadores_vote.append(("xgb", xgb))
    pesos.append(2.0)
if HAS_LGB and lgbm is not None:
    estimadores_vote.append(("lgbm", lgbm))
    pesos.append(2.0)
 
soft_voting = VotingClassifier(
    estimators=estimadores_vote,
    voting="soft",
    weights=pesos,
    n_jobs=-1,
)
# VotingClassifier re-entrena los estimadores internamente
# Para evitar eso, usamos una versión manual (predict_proba directo)
class ManualSoftVoting:
    """Soft voting manual para evitar re-entrenamiento."""
    def __init__(self, estimadores, pesos):
        self.estimadores = estimadores
        self.pesos       = np.array(pesos) / np.sum(pesos)
 
    def predict_proba(self, X):
        probas = np.array([
            m.predict_proba(X)[:, 1] for _, m in self.estimadores
        ])
        avg = np.average(probas, axis=0, weights=self.pesos)
        return np.column_stack([1 - avg, avg])
 
    def predict(self, X, umbral=0.5):
        return (self.predict_proba(X)[:, 1] >= umbral).astype(int)
 
soft_v = ManualSoftVoting(estimadores_vote, pesos)
r_sv_val = evaluar("SoftVoting", soft_v, X_val, y_val)
imprimir(evaluar("SoftVoting", soft_v, X_train_o, y_train_o), "TRAIN")
imprimir(r_sv_val, "VAL")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TABLA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TABLA COMPARATIVA — VALIDACIÓN")
print("="*70)
 
resultados_val = [r_lr_val, r_rf_val]
if r_xgb_val:  resultados_val.append(r_xgb_val)
if r_lgbm_val: resultados_val.append(r_lgbm_val)
resultados_val.append(r_sv_val)
 
cols_tabla = ["modelo","AUC-ROC","Gini","PR-AUC","KS","F1","Recall","Precision","Accuracy"]
df_comp = pd.DataFrame([{k: v for k, v in r.items() if k in cols_tabla}
                         for r in resultados_val])
print(df_comp.to_string(index=False))
df_comp.to_csv(DATA_DIR / "tabla_comparativa_modelos_v2.csv", index=False)
 
# ── Análisis de sobreajuste ──────────────────────────────────────────────────
print("\n  ANÁLISIS DE SOBREAJUSTE (AUC Train original vs AUC Val):")
print(f"  {'Modelo':15s}  {'Train':>8s}  {'Val':>8s}  {'Brecha':>8s}  {'Estado':>12s}")
print(f"  {'─'*60}")
for nombre, modelo in [("LogReg", lr), ("RandomForest", rf)] + \
                       ([("XGBoost", xgb)] if xgb else []) + \
                       ([("LightGBM", lgbm)] if lgbm else []) + \
                       [("SoftVoting", soft_v)]:
    r_tr = evaluar(nombre, modelo, X_train_o, y_train_o)
    r_va = evaluar(nombre, modelo, X_val, y_val)
    brecha = r_tr["AUC-ROC"] - r_va["AUC-ROC"]
    estado = "✅ OK" if brecha < 0.10 else ("⚠️ Moderado" if brecha < 0.20 else "❌ Overfitting")
    print(f"  {nombre:15s}  {r_tr['AUC-ROC']:>8.4f}  {r_va['AUC-ROC']:>8.4f}  {brecha:>8.4f}  {estado:>12s}")
 
 
# ── Champion Model ──────────────────────────────────────────────────────────
candidatos = {"RandomForest": (rf, r_rf_val["AUC-ROC"]),
              "SoftVoting":   (soft_v, r_sv_val["AUC-ROC"])}
if r_xgb_val:  candidatos["XGBoost"]  = (xgb,  r_xgb_val["AUC-ROC"])
if r_lgbm_val: candidatos["LightGBM"] = (lgbm, r_lgbm_val["AUC-ROC"])
 
nombre_champion, (champion, auc_champion) = max(
    candidatos.items(), key=lambda x: x[1][1]
)
print(f"\n  ✅ CHAMPION MODEL: {nombre_champion}  (AUC-ROC Val: {auc_champion:.4f})")
 
# Umbral óptimo
proba_val = champion.predict_proba(X_val)[:, 1]
umbral    = optimizar_umbral(y_val, proba_val)
print(f"  Umbral óptimo (F1 en val): {umbral}")
imprimir(evaluar(nombre_champion, champion, X_val, y_val, umbral=umbral),
         f"VAL (umbral={umbral})")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  BACKTESTING TEMPORAL")
print("="*70)
 
resultados_bt = []
for split_name, X_s, y_s in [
    ("VAL",      X_val,  y_val),
    ("BACKTEST", X_back, y_back),
    ("LIVE",     X_live, y_live),
]:
    r = evaluar(nombre_champion, champion, X_s, y_s, umbral=umbral)
    r["split"] = split_name
    imprimir(r, split_name)
    resultados_bt.append(r)
 
df_bt = pd.DataFrame([{k: v for k, v in r.items() if k != "y_prob"}
                       for r in resultados_bt])
df_bt.to_csv(DATA_DIR / "tabla_backtesting_v2.csv", index=False)
print("\n  RESUMEN BACKTESTING:")
print(df_bt[["split","AUC-ROC","Gini","Recall","F1",
             "Tasa_Deteccion","Ahorro_R$"]].to_string(index=False))
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CURVAS DE APRENDIZAJE — demostrar control de sobreajuste
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Generando curvas de aprendizaje (Train vs Val por tamaño)...")
 
modelos_lc = [
    ("LogReg",       lr,   "#95A5A6"),
    ("RandomForest", rf,   "#27AE60"),
]
if xgb:  modelos_lc.append(("XGBoost",  xgb,  "#2E86AB"))
if lgbm: modelos_lc.append(("LightGBM", lgbm, "#F39C12"))
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
 
for idx, (nom, m, color) in enumerate(modelos_lc):
    if idx >= 4: break
    ax = axes[idx]
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            m, X_train, y_train,
            cv=3, scoring="roc_auc",
            train_sizes=np.linspace(0.2, 1.0, 6),
            n_jobs=-1, shuffle=True, random_state=SEED
        )
        tr_mean = train_scores.mean(axis=1)
        tr_std  = train_scores.std(axis=1)
        vl_mean = val_scores.mean(axis=1)
        vl_std  = val_scores.std(axis=1)
 
        ax.plot(train_sizes, tr_mean, "o-", color=color, label="Train AUC")
        ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color=color)
        ax.plot(train_sizes, vl_mean, "s--", color="red", label="Val AUC")
        ax.fill_between(train_sizes, vl_mean-vl_std, vl_mean+vl_std, alpha=0.15, color="red")
        ax.set_title(f"{nom} — Curva de Aprendizaje")
        ax.set_xlabel("Tamaño de entrenamiento")
        ax.set_ylabel("AUC-ROC")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        brecha = tr_mean[-1] - vl_mean[-1]
        ax.set_title(f"{nom}\n(Brecha final: {brecha:.3f})")
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
 
plt.suptitle("Curvas de Aprendizaje — Control de Sobreajuste\n"
             "Brecha Train–Val < 0.10 indica buen control", fontsize=12)
plt.tight_layout()
plt.savefig(REPORT_DIR / "curvas_aprendizaje.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Guardado: reports/curvas_aprendizaje.png")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CURVA ROC COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
print("▶ Generando curva ROC comparativa...")
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
 
modelos_plot = [("LogReg", lr, "#95A5A6"), ("RandomForest", rf, "#27AE60")]
if xgb:  modelos_plot.append(("XGBoost",   xgb,  "#2E86AB"))
if lgbm: modelos_plot.append(("LightGBM",  lgbm, "#F39C12"))
modelos_plot.append(("SoftVoting", soft_v, "#E74C3C"))
 
# ROC en Validación
for nom, m, color in modelos_plot:
    prob = m.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, prob)
    auc = roc_auc_score(y_val, prob)
    lw  = 3 if nom == nombre_champion else 1.5
    ax1.plot(fpr, tpr, label=f"{nom} (AUC={auc:.3f})", color=color, linewidth=lw)
 
ax1.plot([0,1],[0,1], "k--", alpha=0.4)
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("Curva ROC — Validación"); ax1.legend(loc="lower right"); ax1.grid(True, alpha=0.3)
 
# Curva Precisión-Recall
for nom, m, color in modelos_plot:
    prob = m.predict_proba(X_val)[:, 1]
    prec, rec, _ = precision_recall_curve(y_val, prob)
    pr_auc = average_precision_score(y_val, prob)
    lw  = 3 if nom == nombre_champion else 1.5
    ax2.plot(rec, prec, label=f"{nom} (PR-AUC={pr_auc:.3f})", color=color, linewidth=lw)
 
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Curva Precisión-Recall — Validación")
ax2.legend(loc="upper right"); ax2.grid(True, alpha=0.3)
 
plt.suptitle("Comparación de Modelos — Validación", fontsize=13)
plt.tight_layout()
plt.savefig(REPORT_DIR / "curvas_roc_pr_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Guardado: reports/curvas_roc_pr_v2.png")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CURVA DE GANANCIA Y ELEVACIÓN
# ══════════════════════════════════════════════════════════════════════════════
print("▶ Generando curvas de ganancia y elevación...")
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
 
for nom, m, color in modelos_plot:
    prob = m.predict_proba(X_val)[:, 1]
    df_g = pd.DataFrame({"y": y_val.values, "p": prob})
    df_g = df_g.sort_values("p", ascending=False).reset_index(drop=True)
    df_g['acum_pos'] = df_g['y'].cumsum() / df_g['y'].sum()
    df_g['pct_pob']  = (df_g.index + 1) / len(df_g)
    lw = 3 if nom == nombre_champion else 1.5
    ax1.plot(df_g['pct_pob'], df_g['acum_pos'],
             label=nom, color=color, linewidth=lw)
 
ax1.plot([0,1],[0,1], "k--", alpha=0.4, label="Aleatorio")
ax1.set_xlabel("% Población contactada"); ax1.set_ylabel("% Retrasos detectados")
ax1.set_title("Curva de Ganancia (Lift Acumulado)"); ax1.legend(); ax1.grid(True, alpha=0.3)
 
# Curva de Elevación
for nom, m, color in modelos_plot:
    prob = m.predict_proba(X_val)[:, 1]
    df_g = pd.DataFrame({"y": y_val.values, "p": prob})
    df_g = df_g.sort_values("p", ascending=False).reset_index(drop=True)
    df_g['acum_pos'] = df_g['y'].cumsum() / df_g['y'].sum()
    df_g['pct_pob']  = (df_g.index + 1) / len(df_g)
    df_g['elevacion'] = df_g['acum_pos'] / df_g['pct_pob'].clip(0.001)
    lw = 3 if nom == nombre_champion else 1.5
    ax2.plot(df_g['pct_pob'], df_g['elevacion'],
             label=nom, color=color, linewidth=lw)
 
ax2.axhline(1, color="k", linestyle="--", alpha=0.4, label="Base (sin modelo)")
ax2.set_xlabel("% Población contactada"); ax2.set_ylabel("Factor de Elevación")
ax2.set_title("Curva de Elevación"); ax2.legend(); ax2.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig(REPORT_DIR / "curvas_ganancia_elevacion.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Guardado: reports/curvas_ganancia_elevacion.png")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════════════
shap_modelo = lgbm if (HAS_SHAP and lgbm is not None) else (rf if HAS_SHAP else None)
shap_nombre = "LightGBM" if lgbm else "RandomForest"
 
if HAS_SHAP and shap_modelo is not None:
    print(f"\n▶ Calculando SHAP values ({shap_nombre})...")
    try:
        explainer   = shap.TreeExplainer(shap_modelo)
        shap_values = explainer.shap_values(X_val)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
 
        fig_s, ax_s = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_val, feature_names=features_finales,
                          plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP — Importancia Global ({shap_nombre})")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_importancia_v2.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✅ Guardado: reports/shap_importancia_v2.png")
        joblib.dump({"explainer": explainer, "feature_names": features_finales},
                    MODEL_DIR / "shap_explainer_v2.pkl")
    except Exception as e:
        print(f"  ⚠️  SHAP falló: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CHAMPION MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Guardando Champion Model v2...")
 
artefactos = {
    "modelo":              champion,
    "modelo_lgbm":         lgbm,
    "modelo_rf":           rf,
    "modelo_xgb":          xgb,
    "modelo_lr":           lr,
    "soft_voting":         soft_v,
    "features_finales":    features_finales,
    "umbral_optimo":       float(umbral),
    "nombre_champion":     nombre_champion,
    "version":             "2.0.0",
    "fecha_entrenamiento": str(datetime.now().date()),
    "periodo_train":       "2016-09-15 / 2018-01-31",
    "metricas_val": {
        k: v for k, v in
        evaluar(nombre_champion, champion, X_val, y_val, umbral=umbral).items()
        if k not in ["y_prob", "modelo"]
    },
    "metricas_backtest": {
        k: v for k, v in resultados_bt[1].items()
        if k not in ["y_prob", "modelo", "split"]
    },
    "analisis_overfitting": {
        "LightGBM":  "Regularizado: num_leaves=15, max_depth=4, reg_alpha=1.0",
        "XGBoost":   "Regularizado: max_depth=3, min_child_weight=10, gamma=1.0",
        "SoftVoting": "Sin stacking (riesgo temporal eliminado)"
    }
}
 
joblib.dump(artefactos, MODEL_DIR / "champion_model_v2.pkl")
print("  ✅ Guardado: models/champion_model_v2.pkl")
 
resumen = {k: v for k, v in artefactos.items()
           if k not in ["modelo","modelo_lgbm","modelo_rf","modelo_xgb","modelo_lr","soft_voting"]}
with open(MODEL_DIR / "champion_resumen_v2.json", "w") as f:
    json.dump(resumen, f, indent=2, default=str)
print("  ✅ Guardado: models/champion_resumen_v2.json")
 
print("\n" + "="*70)
print(f"  SPRINT 2 v2 COMPLETADO ✅")
print(f"  Champion: {nombre_champion}")
print(f"  AUC-ROC Val: {auc_champion:.4f}")
print(f"  Umbral óptimo: {umbral}")
print("="*70)
print("\n  Correcciones aplicadas:")
print("  ✅ PSI umbral reducido a 0.15")
print("  ✅ Variables inestables reconstruidas (approval_ratio, bin_hora, etc.)")
print("  ✅ Splits temporales ajustados (val con tasa de tardíos realista)")
print("  ✅ Undersampling aplicado (ratio 1:3)")
print("  ✅ LightGBM regularizado: num_leaves 63→15, max_depth 4, reg_alpha 1.0")
print("  ✅ XGBoost regularizado:  max_depth 5→3, min_child_weight 10, gamma 1.0")
print("  ✅ Stacking reemplazado por Soft Voting")
print("  ✅ Curvas de aprendizaje, ganancia y elevación generadas")
print("  ✅ Análisis de sobreajuste Train vs Val documentado")
