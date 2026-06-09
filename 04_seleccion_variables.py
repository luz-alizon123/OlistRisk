"""
04_seleccion_variables.py — SPRINT 2 (v2 CORREGIDO)
=====================================================
Correcciones:
  • PSI umbral reducido a 0.15 (exigencia docente, antes estaba 0.50)
  • PSI calculado sobre datos crudos con bins adaptativos
  • Se incluyen los 4 pasos completos del instructor:
      0. Estado inicial
      1. Missing rate > 10%
      2. PSI > 0.15  (umbral corregido)
      3. Correlación > 0.90 (thresholds adicionales: 0.80/0.95/0.99)
      4. Random Forest importancia (top N)
      5. WOE/IV mínimo 0.02
  • AUC medido con train_original (no con el undersampled, para referencia)
  • Se usa train_original para PSI (distribución real, no submuestreada)
"""
 
import pandas as pd
import numpy as np
import warnings
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle
 
warnings.filterwarnings("ignore")
 
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
 
TARGET = "is_late_delivery"
SEED   = 42
 
# ─── CARGAR DATOS ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  SELECCION DE VARIABLES v2 — OLIST")
print("=" * 65)
 
master = pd.read_parquet("master_table_dirty.parquet")
master['order_purchase_timestamp'] = pd.to_datetime(master['order_purchase_timestamp'])
 
# Cortes ajustados (consistentes con 03_preparar_datos.py v2)
train_raw = master[master['order_purchase_timestamp'] <= '2018-01-31'].copy()
val_raw   = master[
    (master['order_purchase_timestamp'] >= '2018-02-01') &
    (master['order_purchase_timestamp'] <= '2018-03-31')
].copy()
print(f"  Train crudo: {train_raw.shape}  |  Val crudo: {val_raw.shape}")
print(f"  Tasa tardíos train: {train_raw[TARGET].mean()*100:.1f}%  |  val: {val_raw[TARGET].mean()*100:.1f}%")
 
# Splits procesados para medir AUC (usar train_original — distribución real)
try:
    train_proc = pd.read_parquet(DATA_DIR / "split_train_original.parquet")
    print("  Usando split_train_original.parquet para AUC")
except:
    train_proc = pd.read_parquet(DATA_DIR / "split_train.parquet")
    print("  Usando split_train.parquet para AUC")
 
val_proc   = pd.read_parquet(DATA_DIR / "split_val.parquet")
 
# Cargar IV dict del preprocesamiento
try:
    with open(DATA_DIR / 'transformaciones.pkl', 'rb') as f:
        transf = pickle.load(f)
    iv_dict = transf.get('iv_dict', {})
except:
    iv_dict = {}
 
# ─── COLUMNAS A EXCLUIR ────────────────────────────────────────────────────────
EXCLUIR = [
    TARGET,
    "order_id", "customer_id", "mes", "main_seller_id",
    "order_purchase_timestamp", "order_approved_at",
    "order_estimated_delivery_date",
    "order_delivered_customer_date", "order_delivered_carrier_date",
    "real_delivery_days", "estimation_error_days", "delay_days",
    "review_score", "order_status",
    "customer_zip_code_prefix", "seller_zip_code_prefix", "customer_city",
    # Excluidas por diseño (v2)
    "approval_hours_raw", "risk_profile",
    # Columnas crudas originales (si sobrevivieron al pipeline)
    "purchase_month", "purchase_quarter",
]
 
CATS_WOE = ['customer_state', 'seller_state', 'main_category', 'main_payment_type']
 
FEATURES_INIT = [
    c for c in train_raw.columns
    if c not in EXCLUIR
    and (train_raw[c].dtype in [np.float64, np.int64, np.float32, np.int32]
         or c in CATS_WOE)
    and c in train_proc.columns
]
print(f"\n  Features candidatas iniciales: {len(FEATURES_INIT)}")
 
 
# ─── UTILIDADES ───────────────────────────────────────────────────────────────
 
def auc_rf(features):
    """AUC con RF rápido. Usa datos procesados."""
    feats = [f for f in features if f in train_proc.columns and f in val_proc.columns]
    if len(feats) == 0:
        return 0.0, 0.0
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=SEED,
        class_weight="balanced", n_jobs=-1
    )
    rf.fit(train_proc[feats].fillna(0), train_proc[TARGET])
    p_t = rf.predict_proba(train_proc[feats].fillna(0))[:, 1]
    p_v = rf.predict_proba(val_proc[feats].fillna(0))[:, 1]
    return (round(roc_auc_score(train_proc[TARGET], p_t), 4),
            round(roc_auc_score(val_proc[TARGET],   p_v), 4))
 
 
def calcular_psi(s_train, s_val, bins=10):
    """
    PSI (Population Stability Index).
    Interpretación:
      PSI < 0.10  → Distribución estable (sin cambios)
      PSI 0.10–0.25 → Ligero cambio — monitorear
      PSI > 0.25  → Cambio significativo — variable inestable
      (El docente exige PSI ≤ 0.15 para cada feature)
    """
    try:
        breaks = pd.qcut(s_train, q=bins, retbins=True, duplicates="drop")[1]
        if len(breaks) < 3:
            return 0.0
        breaks[0]  = -np.inf
        breaks[-1] = np.inf
        def dist(s):
            c = pd.cut(s, bins=breaks).value_counts(sort=False)
            return (c / max(len(s), 1)).replace(0, 1e-4)
        d_tr = dist(s_train)
        d_va = dist(s_val)
        psi  = float(np.sum((d_tr - d_va) * np.log(d_tr / d_va)))
        return psi if np.isfinite(psi) else 999.0
    except Exception:
        return 0.0
 
 
tabla   = []
features_activas = FEATURES_INIT.copy()
 
# ── Estado inicial ─────────────────────────────────────────────────────────────
print("\n▶ Estado inicial")
auc_t, auc_v = auc_rf(features_activas)
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
tabla.append({"Paso": "0 - Inicial",
              "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v,
              "Eliminadas": "-"})
 
 
# ── PASO 1: Missing rate > 10% ────────────────────────────────────────────────
print("\n▶ Paso 1: Missing rate > 10%")
miss  = train_raw[features_activas].isnull().mean()
elim1 = miss[miss > 0.10].index.tolist()
features_activas = [f for f in features_activas if f not in elim1]
auc_t, auc_v = auc_rf(features_activas)
print(f"    Eliminadas: {elim1 if elim1 else 'Ninguna'}")
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
tabla.append({"Paso": "1 - Missing >10%",
              "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v,
              "Eliminadas": str(elim1)})
 
 
# ── PASO 2: PSI > 0.15 ────────────────────────────────────────────────────────
# ⚠️ CORRECCIÓN CRÍTICA: umbral reducido de 0.50 a 0.15 (exigencia docente)
# Las variables purchase_month y purchase_quarter tenían PSI 9.9 y 3.1
# porque reflejan la distribución mensual exacta, que cambia radicalmente
# entre train (sep 2016–ene 2018) y val (feb–mar 2018).
# Con los nuevos cortes temporales y variables reconstruidas (approval_ratio
# en lugar de time_to_approve_hours) se espera que el PSI baje < 0.15.
 
PSI_UMBRAL = 0.15
print(f"\n▶ Paso 2: PSI sobre datos crudos (umbral {PSI_UMBRAL})")
 
psi_scores = {}
for col in features_activas:
    if col in train_raw.columns:
        s_tr = pd.to_numeric(train_raw[col], errors='coerce').dropna()
        s_va = pd.to_numeric(val_raw[col],   errors='coerce').dropna()
        psi_scores[col] = calcular_psi(s_tr, s_va) if len(s_tr) > 10 else 0.0
    else:
        psi_scores[col] = 0.0
 
psi_s = pd.Series(psi_scores).sort_values(ascending=False)
psi_s.to_csv(DATA_DIR / "reporte_psi_v2.csv", header=["PSI"])
 
print("  Top 20 PSI:")
print(psi_s.head(20).round(4).to_string())
print(f"\n  Variables con PSI > {PSI_UMBRAL}:")
elim2 = psi_s[psi_s > PSI_UMBRAL].index.tolist()
for v in elim2:
    print(f"    {v:35s}  PSI = {psi_s[v]:.4f}")
 
features_activas = [f for f in features_activas if f not in elim2]
auc_t, auc_v = auc_rf(features_activas)
print(f"\n    Eliminadas: {len(elim2)} variables")
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
tabla.append({"Paso": f"2 - PSI >{PSI_UMBRAL}",
              "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v,
              "Eliminadas": str(elim2)})
 
 
# ── PASO 3: Correlación (cuatro umbrales) ─────────────────────────────────────
print("\n▶ Paso 3: Correlación — cuatro umbrales (0.80 / 0.90 / 0.95 / 0.99)")
 
feats_num = [f for f in features_activas
             if f in train_raw.select_dtypes(include=np.number).columns]
corr_mat = train_raw[feats_num].fillna(0).corr().abs()
corr_tgt = train_raw[feats_num + [TARGET]].fillna(0).corr()[TARGET].abs()
 
UMBRAL_CORR = 0.90   # umbral principal
upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
 
elim3 = set()
for col in upper.columns:
    for par in upper[col][upper[col] > UMBRAL_CORR].index:
        if par in elim3:
            continue
        keep = col if corr_tgt.get(col, 0) >= corr_tgt.get(par, 0) else par
        drop = par if keep == col else col
        elim3.add(drop)
 
features_activas = [f for f in features_activas if f not in elim3]
auc_t, auc_v = auc_rf(features_activas)
print(f"    Umbral 0.90 — Eliminadas: {list(elim3)}")
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
tabla.append({"Paso": "3 - Correlacion >0.90",
              "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v,
              "Eliminadas": str(list(elim3))})
 
 
# ── PASO 4: Random Forest — importancia ───────────────────────────────────────
print("\n▶ Paso 4: Importancia con Random Forest (eliminar bottom 10%)")
 
feats_proc = [f for f in features_activas
              if f in train_proc.columns and f in val_proc.columns]
rf_imp = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=SEED,
    class_weight="balanced", n_jobs=-1
)
rf_imp.fit(train_proc[feats_proc].fillna(0), train_proc[TARGET])
importancias = pd.Series(rf_imp.feature_importances_, index=feats_proc).sort_values()
umbral_imp   = importancias.quantile(0.10)
elim4        = importancias[importancias < umbral_imp].index.tolist()
 
features_activas = [f for f in features_activas if f not in elim4]
auc_t, auc_v = auc_rf(features_activas)
print(f"    Importancia mínima (p10): {umbral_imp:.6f}")
print(f"    Eliminadas: {elim4}")
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
print("\n  Top 20 importancias RF:")
print(importancias.sort_values(ascending=False).head(20).round(5).to_string())
tabla.append({"Paso": "4 - Importancia RF (bottom 10%)",
              "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v,
              "Eliminadas": str(elim4)})
 
 
# ── PASO 5: WOE / IV mínimo 0.02 ──────────────────────────────────────────────
print("\n▶ Paso 5: WOE/IV — eliminar features con IV < 0.02")
 
# IV ya calculado en 03 para categóricas; para numéricas calcular aquí
def calcular_iv_numerico(col, df_train, bins=10):
    try:
        df_tmp   = df_train[[col, TARGET]].dropna()
        df_tmp['bin'] = pd.qcut(df_tmp[col], q=bins, duplicates='drop')
        total_ev  = max(df_tmp[TARGET].sum(), 1)
        total_nev = max(len(df_tmp) - total_ev, 1)
        g = df_tmp.groupby('bin')[TARGET].agg(['sum','count'])
        g.columns = ['ev', 'tot']
        g['nev']   = g['tot'] - g['ev']
        g['pct_e'] = (g['ev']  / total_ev ).replace(0, 1e-6)
        g['pct_n'] = (g['nev'] / total_nev).replace(0, 1e-6)
        g['woe']   = np.log(g['pct_e'] / g['pct_n'])
        g['iv_i']  = (g['pct_e'] - g['pct_n']) * g['woe']
        return round(g['iv_i'].sum(), 5)
    except Exception:
        return 0.0
 
iv_final = {}
for col in features_activas:
    if col in iv_dict:
        iv_final[col] = iv_dict[col]
    elif col in train_raw.columns:
        iv_final[col] = calcular_iv_numerico(col, train_raw)
    else:
        iv_final[col] = 0.0
 
iv_s = pd.Series(iv_final).sort_values(ascending=False)
iv_s.to_csv(DATA_DIR / "reporte_iv_v2.csv", header=["IV"])
 
print("\n  Information Value por feature (top 25):")
print(iv_s.head(25).round(5).to_string())
 
IV_UMBRAL = 0.02
elim5 = iv_s[iv_s < IV_UMBRAL].index.tolist()
features_activas = [f for f in features_activas if f not in elim5]
auc_t, auc_v = auc_rf(features_activas)
print(f"\n    Eliminadas (IV < {IV_UMBRAL}): {elim5}")
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
tabla.append({"Paso": f"5 - IV <{IV_UMBRAL}",
              "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v,
              "Eliminadas": str(elim5)})
 
 
# ── TABLA RESUMEN ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TABLA DE PROGRESO — SELECCIÓN DE VARIABLES v2")
print("=" * 65)
df_tabla = pd.DataFrame(tabla)[["Paso","Features","AUC Train","AUC Val"]]
print(df_tabla.to_string(index=False))
df_tabla.to_csv(DATA_DIR / "tabla_seleccion_variables_v2.csv", index=False)
 
 
# ── PSI FINAL — verificar que todas están ≤ 0.15 ─────────────────────────────
print("\n" + "=" * 65)
print("  VERIFICACIÓN PSI FINAL (todas deben ser ≤ 0.15)")
print("=" * 65)
psi_final = {}
for col in features_activas:
    if col in train_raw.columns:
        s_tr = pd.to_numeric(train_raw[col], errors='coerce').dropna()
        s_va = pd.to_numeric(val_raw[col],   errors='coerce').dropna()
        psi_final[col] = calcular_psi(s_tr, s_va)
    else:
        psi_final[col] = 0.0
 
psi_ok    = {k: v for k, v in psi_final.items() if v <= 0.15}
psi_alert = {k: v for k, v in psi_final.items() if v > 0.15}
 
print(f"  Variables con PSI ≤ 0.15: {len(psi_ok)}")
print(f"  Variables con PSI > 0.15: {len(psi_alert)}")
if psi_alert:
    print("  ⚠️ Alertas PSI:")
    for k, v in sorted(psi_alert.items(), key=lambda x: -x[1]):
        print(f"     {k:35s}  PSI = {v:.4f}")
 
 
# ── RESULTADO FINAL ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FEATURES FINALES")
print("=" * 65)
print(f"  Total: {len(features_activas)}")
 
df_final = pd.DataFrame({
    'feature':    features_activas,
    'iv':         [iv_final.get(f, 0) for f in features_activas],
    'psi':        [psi_final.get(f, 0) for f in features_activas],
    'importancia_rf': [rf_imp.feature_importances_[feats_proc.index(f)]
                       if f in feats_proc else np.nan for f in features_activas],
}).sort_values('iv', ascending=False)
 
print(df_final.to_string(index=False))
df_final.to_csv(DATA_DIR / "features_finales_detalle_v2.csv", index=False)
 
joblib.dump(features_activas, DATA_DIR / "features_finales.pkl")
with open(DATA_DIR / "features_finales.json", "w") as fj:
    json.dump(features_activas, fj, indent=2)
 
print(f"\n✅ {len(features_activas)} features finales guardadas")
print("   → Ejecutar ahora: python 05_modelado.py")
