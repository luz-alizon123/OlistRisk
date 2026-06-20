import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

# Cargar
live = pd.read_parquet("data/split_live.parquet")
master = pd.read_parquet("master_table_dirty.parquet")
master['order_purchase_timestamp'] = pd.to_datetime(master['order_purchase_timestamp'])
master['order_estimated_delivery_date'] = pd.to_datetime(master['order_estimated_delivery_date'])

CORTE = pd.Timestamp('2018-08-29')

# Pedidos con target confiable (entrega estimada ANTES del corte)
live_valido = master[
    (master['order_purchase_timestamp'] >= '2018-08-01') &
    (master['order_estimated_delivery_date'] <= CORTE)
][['order_id']].copy()

live_clean = live.merge(live_valido, on='order_id', how='inner')
print(f"Live original:  {len(live):,} pedidos")
print(f"Live limpio:    {len(live_clean):,} pedidos ({len(live_clean)/len(live)*100:.1f}%)")
print(f"Tasa tardíos:   {live_clean['is_late_delivery'].mean()*100:.1f}%")

# Re-evaluar el champion
artefactos = joblib.load("models/champion_model_v1.pkl")
champion   = artefactos['modelo']
features   = artefactos['features_finales']
umbral     = artefactos['umbral_optimo']

X_live_clean = live_clean[features].fillna(0)
y_live_clean = live_clean['is_late_delivery']

proba = champion.predict_proba(X_live_clean)[:, 1]
auc   = roc_auc_score(y_live_clean, proba)
print(f"\nAUC-ROC Live CORREGIDO: {auc:.4f}")