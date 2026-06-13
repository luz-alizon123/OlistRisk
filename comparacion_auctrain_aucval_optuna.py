"""
generar_excel_gap_optuna.py
===========================
Genera un Excel con el gap Train-Val de cada uno de los 200 trials
de Optuna, tanto para LogReg como para LightGBM.

El docente quiere ver:
  - Por cada trial: qué parámetros se usaron
  - Cuál fue el AUC en CV (lo que Optuna optimizó)
  - Cuál fue el AUC en Val real (evaluación honesta)
  - Cuál fue el AUC en Train
  - Cuál fue el gap Train → Val
  - Si ese gap era aceptable (< 0.10) o era sobreajuste

Ejecutar:
  python generar_excel_gap_optuna.py
"""

import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from imblearn.over_sampling import SMOTE
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    print("  ⚠️  Algunas librerías no están instaladas")

DATA_DIR   = Path("data")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "is_late_delivery"
SEED   = 42
np.random.seed(SEED)

print("=" * 65)
print("  EXCEL GAP TRAIN-VAL — HISTORIAL OPTUNA")
print("  200 trials LogReg + 200 trials LightGBM")
print("=" * 65)

# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────
features = joblib.load(DATA_DIR / "features_finales.pkl")

train = pd.read_parquet(DATA_DIR / "split_train.parquet")
val   = pd.read_parquet(DATA_DIR / "split_val.parquet")

X_train = train[features].fillna(0); y_train = train[TARGET]
X_val   = val[features].fillna(0);   y_val   = val[TARGET]

scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

# SMOTE para LightGBM (la mejor técnica según Fase 1)
if HAS_LIBS:
    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_lgbm, y_train_lgbm = sm.fit_resample(X_train, y_train)
    print(f"  Train original: {len(X_train):,} | Train SMOTE: {len(X_train_lgbm):,}")
else:
    X_train_lgbm, y_train_lgbm = X_train, y_train

# ─── CARGAR HISTORIALES ───────────────────────────────────────────────────────
path_lr   = DATA_DIR / "optuna_historial_logreg.csv"
path_lgbm = DATA_DIR / "optuna_historial_lightgbm.csv"

if not path_lr.exists() or not path_lgbm.exists():
    print("  ⚠️  No se encontraron los historiales de Optuna.")
    print("  Asegurate de haber ejecutado 06_tuneo.py primero.")
    exit()

df_lr   = pd.read_csv(path_lr)
df_lgbm = pd.read_csv(path_lgbm)

print(f"\n  Trials LogReg:    {len(df_lr)}")
print(f"  Trials LightGBM:  {len(df_lgbm)}")


# ══════════════════════════════════════════════════════════════════════════════
# RECALCULAR GAP PARA CADA TRIAL — LOGREG
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Recalculando gap por trial — LogReg...")
print("  (puede tardar ~5 minutos — 200 modelos × 2 evaluaciones)")

filas_lr = []
for i, row in df_lr.iterrows():
    if i % 20 == 0:
        print(f"  Trial {i}/200...")

    try:
        # Recuperar parámetros del trial
        C      = row.get("params_C", 1.0)
        solver = row.get("params_solver", "lbfgs")

        # Determinar penalty según solver
        if solver == "saga":
            penalty = row.get("params_penalty_saga", "l2")
        elif solver == "liblinear":
            penalty = row.get("params_penalty_ll", "l2")
        else:
            penalty = "l2"

        modelo = LogisticRegression(
            C=C, solver=solver, penalty=penalty,
            class_weight="balanced", max_iter=2000, random_state=SEED
        )
        modelo.fit(X_train, y_train)

        auc_train = roc_auc_score(y_train, modelo.predict_proba(X_train)[:,1])
        auc_val   = roc_auc_score(y_val,   modelo.predict_proba(X_val)[:,1])
        gap       = auc_train - auc_val
        auc_cv    = row.get("value", np.nan)

        estado_gap = ("✅ Estable"      if gap < 0.05 else
                      "✅ OK"           if gap < 0.10 else
                      "⚠️ Moderado"    if gap < 0.20 else
                      "❌ Sobreajuste")

        filas_lr.append({
            "Trial":        int(row["number"]),
            "AUC_CV_Optuna": round(auc_cv, 6)  if not np.isnan(auc_cv) else None,
            "AUC_Train":    round(auc_train, 6),
            "AUC_Val":      round(auc_val, 6),
            "Gap_Train_Val":round(gap, 6),
            "Estado_Gap":   estado_gap,
            "C":            round(C, 6),
            "solver":       solver,
            "penalty":      penalty,
            "Duracion_seg": row.get("duration", "").split(" ")[-1] if isinstance(row.get("duration",""), str) else "",
        })
    except Exception as e:
        filas_lr.append({
            "Trial": int(row["number"]),
            "AUC_CV_Optuna": None, "AUC_Train": None,
            "AUC_Val": None, "Gap_Train_Val": None,
            "Estado_Gap": f"Error: {e}",
            "C": None, "solver": None, "penalty": None,
        })

df_gap_lr = pd.DataFrame(filas_lr)
print(f"  ✅ Gap calculado para {len(df_gap_lr)} trials de LogReg")


# ══════════════════════════════════════════════════════════════════════════════
# RECALCULAR GAP PARA CADA TRIAL — LIGHTGBM
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Recalculando gap por trial — LightGBM...")
print("  (puede tardar ~10 minutos — 200 modelos × 2 evaluaciones)")

filas_lgbm = []
for i, row in df_lgbm.iterrows():
    if i % 20 == 0:
        print(f"  Trial {i}/200...")

    try:
        params = {
            "n_estimators":      int(row.get("params_n_estimators", 200)),
            "num_leaves":        int(row.get("params_num_leaves", 31)),
            "learning_rate":     float(row.get("params_learning_rate", 0.1)),
            "min_child_samples": int(row.get("params_min_child_samples", 20)),
            "reg_alpha":         float(row.get("params_reg_alpha", 0.0)),
            "reg_lambda":        float(row.get("params_reg_lambda", 0.0)),
            "feature_fraction":  float(row.get("params_feature_fraction", 1.0)),
            "bagging_fraction":  float(row.get("params_bagging_fraction", 1.0)),
            "bagging_freq":      1,
            "scale_pos_weight":  1.0,
            "random_state":      SEED,
            "n_jobs":            -1,
            "verbose":           -1,
        }

        modelo = LGBMClassifier(**params)
        modelo.fit(X_train_lgbm, y_train_lgbm)

        auc_train = roc_auc_score(y_train_lgbm, modelo.predict_proba(X_train_lgbm)[:,1])
        auc_val   = roc_auc_score(y_val,         modelo.predict_proba(X_val)[:,1])
        gap       = auc_train - auc_val
        auc_cv    = row.get("value", np.nan)

        estado_gap = ("✅ Estable"      if gap < 0.05 else
                      "✅ OK"           if gap < 0.10 else
                      "⚠️ Moderado"    if gap < 0.20 else
                      "❌ Sobreajuste")

        filas_lgbm.append({
            "Trial":             int(row["number"]),
            "AUC_CV_Optuna":     round(auc_cv, 6) if not np.isnan(auc_cv) else None,
            "AUC_Train":         round(auc_train, 6),
            "AUC_Val":           round(auc_val, 6),
            "Gap_Train_Val":     round(gap, 6),
            "Estado_Gap":        estado_gap,
            "n_estimators":      params["n_estimators"],
            "num_leaves":        params["num_leaves"],
            "learning_rate":     round(params["learning_rate"], 6),
            "min_child_samples": params["min_child_samples"],
            "reg_alpha":         round(params["reg_alpha"], 6),
            "reg_lambda":        round(params["reg_lambda"], 6),
            "feature_fraction":  round(params["feature_fraction"], 4),
            "bagging_fraction":  round(params["bagging_fraction"], 4),
        })
    except Exception as e:
        filas_lgbm.append({
            "Trial": int(row["number"]),
            "AUC_CV_Optuna": None, "AUC_Train": None,
            "AUC_Val": None, "Gap_Train_Val": None,
            "Estado_Gap": f"Error: {e}",
        })

df_gap_lgbm = pd.DataFrame(filas_lgbm)
print(f"  ✅ Gap calculado para {len(df_gap_lgbm)} trials de LightGBM")


# ══════════════════════════════════════════════════════════════════════════════
# ESTADÍSTICAS RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  ESTADÍSTICAS DEL GAP — 200 TRIALS")
print("="*65)

for nombre, df_g in [("LogReg", df_gap_lr), ("LightGBM", df_gap_lgbm)]:
    gaps = df_g["Gap_Train_Val"].dropna()
    print(f"\n  {nombre}:")
    print(f"    Gap promedio:  {gaps.mean():.4f}")
    print(f"    Gap mínimo:    {gaps.min():.4f}")
    print(f"    Gap máximo:    {gaps.max():.4f}")
    print(f"    Gap mediana:   {gaps.median():.4f}")
    print(f"    Trials gap<0.10 (OK):      {(gaps<0.10).sum()}")
    print(f"    Trials gap 0.10-0.20 (mod):{((gaps>=0.10)&(gaps<0.20)).sum()}")
    print(f"    Trials gap>0.20 (sobre):   {(gaps>=0.20).sum()}")

    # Mejor trial por AUC Val (no por CV)
    mejor = df_g.loc[df_g["AUC_Val"].idxmax()]
    print(f"\n    Mejor trial por AUC Val:")
    print(f"      Trial #{int(mejor['Trial'])} | "
          f"AUC Val={mejor['AUC_Val']:.4f} | "
          f"AUC Train={mejor['AUC_Train']:.4f} | "
          f"Gap={mejor['Gap_Train_Val']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTAR EXCEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n▶ Exportando Excel...")

output_path = REPORT_DIR / "gap_historial_optuna.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

    # ── Hoja 1: LogReg completo ───────────────────────────────────────────────
    df_gap_lr.to_excel(writer, sheet_name="LogReg_200trials", index=False)

    # ── Hoja 2: LightGBM completo ─────────────────────────────────────────────
    df_gap_lgbm.to_excel(writer, sheet_name="LightGBM_200trials", index=False)

    # ── Hoja 3: Resumen comparativo ───────────────────────────────────────────
    resumen_rows = []
    for nombre, df_g in [("LogReg", df_gap_lr), ("LightGBM", df_gap_lgbm)]:
        gaps = df_g["Gap_Train_Val"].dropna()
        mejor = df_g.loc[df_g["AUC_Val"].dropna().idxmax()]
        resumen_rows.append({
            "Modelo":                nombre,
            "Trials_total":          len(df_g),
            "AUC_Val_promedio":      round(df_g["AUC_Val"].mean(), 4),
            "AUC_Val_maximo":        round(df_g["AUC_Val"].max(), 4),
            "AUC_CV_maximo":         round(df_g["AUC_CV_Optuna"].max(), 4),
            "Gap_promedio":          round(gaps.mean(), 4),
            "Gap_minimo":            round(gaps.min(), 4),
            "Gap_maximo":            round(gaps.max(), 4),
            "Trials_gap_OK_<0.10":   int((gaps < 0.10).sum()),
            "Trials_gap_mod_0.10-0.20": int(((gaps>=0.10)&(gaps<0.20)).sum()),
            "Trials_gap_sobre_>0.20":int((gaps >= 0.20).sum()),
            "Mejor_trial_nro":       int(mejor["Trial"]),
            "Mejor_trial_AUC_Val":   round(mejor["AUC_Val"], 4),
            "Mejor_trial_Gap":       round(mejor["Gap_Train_Val"], 4),
        })

    pd.DataFrame(resumen_rows).to_excel(
        writer, sheet_name="Resumen_Comparativo", index=False
    )

    # ── Hoja 4: Solo los top 20 trials por AUC Val de cada modelo ─────────────
    top_lr   = df_gap_lr.nlargest(20, "AUC_Val")
    top_lgbm = df_gap_lgbm.nlargest(20, "AUC_Val")
    top_lr["Modelo"]   = "LogReg"
    top_lgbm["Modelo"] = "LightGBM"
    pd.concat([top_lr, top_lgbm]).to_excel(
        writer, sheet_name="Top20_por_AUC_Val", index=False
    )

    # ── Hoja 5: Explicación para el docente ───────────────────────────────────
    explicacion = pd.DataFrame([
        {"Columna": "Trial",          "Descripción": "Número del intento de Optuna (0 a 199)"},
        {"Columna": "AUC_CV_Optuna",  "Descripción": "Score que Optuna optimizó: AUC promedio de CV=5 folds sobre train, penalizado si gap > 0.10"},
        {"Columna": "AUC_Train",      "Descripción": "AUC del modelo entrenado sobre TODO el conjunto de train (64,152 pedidos)"},
        {"Columna": "AUC_Val",        "Descripción": "AUC del modelo evaluado sobre el conjunto de validación (13,415 pedidos — datos no vistos)"},
        {"Columna": "Gap_Train_Val",  "Descripción": "AUC_Train - AUC_Val. Gap negativo = generaliza mejor que entrena (ideal). Gap > 0.20 = sobreajuste"},
        {"Columna": "Estado_Gap",     "Descripción": "Clasificación del gap: Estable (<0.05), OK (<0.10), Moderado (<0.20), Sobreajuste (>=0.20)"},
        {"Columna": "C (LogReg)",     "Descripción": "Parámetro de regularización. C bajo = más regularización = modelo más conservador"},
        {"Columna": "solver",         "Descripción": "Algoritmo de optimización interno de la regresión logística"},
        {"Columna": "penalty",        "Descripción": "Tipo de regularización: l1 produce modelos sparse, l2 distribuye el peso entre variables"},
        {"Columna": "n_estimators",   "Descripción": "Número de árboles en LightGBM. Más árboles = más complejo"},
        {"Columna": "num_leaves",     "Descripción": "Máximo de hojas por árbol. Más hojas = árboles más profundos = más riesgo de sobreajuste"},
        {"Columna": "learning_rate",  "Descripción": "Velocidad de aprendizaje. Bajo = aprende lento pero generaliza mejor"},
        {"Columna": "min_child_samples","Descripción":"Mínimo de muestras por hoja. Alto = menos divisiones = más regularización"},
        {"Columna": "reg_alpha",      "Descripción": "Regularización L1. Penaliza coeficientes absolutos. Produce modelos más simples"},
        {"Columna": "reg_lambda",     "Descripción": "Regularización L2. Penaliza cuadrado de coeficientes. Evita valores extremos"},
        {"Columna": "feature_fraction","Descripción":"Fracción de variables usadas por árbol. <1.0 = subsampling = menos sobreajuste"},
        {"Columna": "bagging_fraction","Descripción":"Fracción de filas usadas por árbol. <1.0 = subsampling = menos sobreajuste"},
    ])
    explicacion.to_excel(writer, sheet_name="Glosario_Columnas", index=False)

    # ── Formato: ajustar ancho de columnas ────────────────────────────────────
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value else 0
                for cell in col
            )
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 40)

print(f"  ✅ {output_path}")
print(f"\n  Hojas del Excel:")
print(f"    1. LogReg_200trials      — 200 trials con gap por trial")
print(f"    2. LightGBM_200trials    — 200 trials con gap por trial")
print(f"    3. Resumen_Comparativo   — estadísticas del gap por modelo")
print(f"    4. Top20_por_AUC_Val     — los 20 mejores trials de cada modelo")
print(f"    5. Glosario_Columnas     — qué significa cada columna")

print("\n" + "="*65)
print("  ✅ Excel generado: reports/gap_historial_optuna.xlsx")
print("="*65)
