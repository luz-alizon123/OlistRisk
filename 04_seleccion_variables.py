"""
04_seleccion_variables.py
=========================
Pasos:
  1. Missing rate  > 10%
  2. PSI sobre datos crudos (umbral 0.50)
  3. Correlacion   > 0.90
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TARGET = "is_late_delivery"
SEED   = 42

# ─── CARGAR DATOS ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  SELECCION DE VARIABLES — OLIST")
print("=" * 60)

master = pd.read_parquet("master_table_dirty.parquet")
master['order_purchase_timestamp'] = pd.to_datetime(master['order_purchase_timestamp'])

# Datos crudos para PSI (distribuciones originales)
train_raw = master[master['order_purchase_timestamp'] <= '2018-05-31'].copy()
val_raw   = master[(master['order_purchase_timestamp'] >= '2018-06-01') &
                   (master['order_purchase_timestamp'] <= '2018-06-30')].copy()
print(f"  Train crudo: {train_raw.shape}  |  Val crudo: {val_raw.shape}")

# Splits procesados (WOE + scaled) solo para medir AUC
train_proc = pd.read_parquet("data/split_train.parquet")
val_proc   = pd.read_parquet("data/split_val.parquet")

# ─── COLUMNAS EXCLUIDAS ────────────────────────────────────────────────────────
EXCLUIR = [
    TARGET,
    "order_id", "customer_id", "mes", "main_seller_id",
    "order_purchase_timestamp", "order_approved_at",
    "order_estimated_delivery_date",
    "order_delivered_customer_date", "order_delivered_carrier_date",
    "real_delivery_days", "estimation_error_days", "delay_days",
    "review_score",
    "customer_zip_code_prefix", "seller_zip_code_prefix", "customer_city",
]

CATS_WOE = ['customer_state', 'seller_state', 'main_category',
            'main_payment_type', 'risk_profile']

FEATURES_INIT = [
    c for c in train_raw.columns
    if c not in EXCLUIR
    and (train_raw[c].dtype in [np.float64, np.int64, np.float32, np.int32]
         or c in CATS_WOE)
    and c in train_proc.columns
]
print(f"\n  Features candidatas: {len(FEATURES_INIT)}")


# ─── UTILIDADES ───────────────────────────────────────────────────────────────

def auc_rf(features):
    feats = [f for f in features if f in train_proc.columns]
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
    try:
        breaks = pd.qcut(s_train, q=bins, retbins=True, duplicates="drop")[1]
        breaks[0] = -np.inf; breaks[-1] = np.inf
        def dist(s):
            c = pd.cut(s, bins=breaks).value_counts(sort=False)
            return (c / len(s)).replace(0, 1e-4)
        d_tr = dist(s_train); d_va = dist(s_val)
        psi = float(np.sum((d_tr - d_va) * np.log(d_tr / d_va)))
        return psi if np.isfinite(psi) else 999
    except:
        return 0.0


tabla = []
features_activas = FEATURES_INIT.copy()

# ── Estado inicial ─────────────────────────────────────────────────────────────
print("\n▶ Estado inicial")
auc_t, auc_v = auc_rf(features_activas)
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
tabla.append({"Paso": "0 - Inicial", "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v})


# ── PASO 1: Missing rate ───────────────────────────────────────────────────────
print("\n▶ Paso 1: Missing rate > 10%")
miss  = train_raw[features_activas].isnull().mean()
elim  = miss[miss > 0.10].index.tolist()
features_activas = [f for f in features_activas if f not in elim]
auc_t, auc_v = auc_rf(features_activas)
print(f"    Eliminadas: {elim}")
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
tabla.append({"Paso": "1 - Missing >10%", "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v})


# ── PASO 2: PSI sobre datos crudos ────────────────────────────────────────────
print("\n▶ Paso 2: PSI sobre datos crudos (umbral 0.50)")
psi_scores = {}
for col in features_activas:
    if col in train_raw.columns:
        s_tr = pd.to_numeric(train_raw[col], errors='coerce').dropna()
        s_va = pd.to_numeric(val_raw[col],   errors='coerce').dropna()
        psi_scores[col] = calcular_psi(s_tr, s_va) if len(s_tr) > 0 else 0.0
    else:
        psi_scores[col] = 0.0

psi_s = pd.Series(psi_scores).sort_values(ascending=False)
psi_s.to_csv(DATA_DIR / "reporte_psi.csv", header=["PSI"])

print("  Top 15 PSI:")
print(psi_s.head(15).round(3).to_string())

elim_psi = psi_s[psi_s > 0.50].index.tolist()
features_activas = [f for f in features_activas if f not in elim_psi]
auc_t, auc_v = auc_rf(features_activas)
print(f"\n    Eliminadas: {elim_psi}")
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
tabla.append({"Paso": "2 - PSI >0.25", "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v})


# ── PASO 3: Correlacion > 0.90 ────────────────────────────────────────────────
print("\n▶ Paso 3: Correlacion > 0.90")
feats_num = [f for f in features_activas
             if f in train_raw.select_dtypes(include=np.number).columns]
corr_mat = train_raw[feats_num].fillna(0).corr().abs()
corr_tgt = train_raw[feats_num + [TARGET]].fillna(0).corr()[TARGET].abs()
upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))

elim_corr = set()
for col in upper.columns:
    for par in upper[col][upper[col] > 0.90].index:
        if par in elim_corr: continue
        if corr_tgt.get(par, 0) <= corr_tgt.get(col, 0):
            elim_corr.add(par)
        else:
            elim_corr.add(col)

features_activas = [f for f in features_activas if f not in elim_corr]
auc_t, auc_v = auc_rf(features_activas)
print(f"    Eliminadas: {list(elim_corr)}")
print(f"    {len(features_activas)} features | AUC Train {auc_t} | AUC Val {auc_v}")
tabla.append({"Paso": "3 - Correlacion >0.90", "Features": len(features_activas),
              "AUC Train": auc_t, "AUC Val": auc_v})


# ── TABLA RESUMEN ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TABLA DE PROGRESO")
print("=" * 60)
df_tabla = pd.DataFrame(tabla)
print(df_tabla.to_string(index=False))
df_tabla.to_csv(DATA_DIR / "tabla_seleccion_variables.csv", index=False)


# ── RESULTADO FINAL ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FEATURES FINALES")
print("=" * 60)
print(f"  Total: {len(features_activas)}")
for f in features_activas:
    print(f"  {f}")

joblib.dump(features_activas, DATA_DIR / "features_finales.pkl")
with open(DATA_DIR / "features_finales.json", "w") as fj:
    json.dump(features_activas, fj, indent=2)

print(f"\n✅ {len(features_activas)} features finales guardadas")
print("   → Ejecutar ahora: python 05_modelado.py")