import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

live   = pd.read_parquet("data/split_live.parquet")
master = pd.read_parquet("master_table_dirty.parquet")
master['order_purchase_timestamp']      = pd.to_datetime(master['order_purchase_timestamp'])
master['order_estimated_delivery_date'] = pd.to_datetime(master['order_estimated_delivery_date'])

CORTE = pd.Timestamp('2018-08-29')
live_valido = master[
    (master['order_purchase_timestamp'] >= '2018-08-01') &
    (master['order_estimated_delivery_date'] <= CORTE)
][['order_id']]

live_clean = live.merge(live_valido, on='order_id', how='inner')
print(f"Live limpio:  {len(live_clean):,} pedidos")
print(f"Tasa tardios: {live_clean['is_late_delivery'].mean()*100:.1f}%")

art      = joblib.load("models/champion_model_v1.pkl")
features = art['features_finales']

proba = art['modelo'].predict_proba(live_clean[features].fillna(0))[:, 1]
auc   = roc_auc_score(live_clean['is_late_delivery'], proba)
print(f"AUC-ROC Live CORREGIDO: {auc:.4f}")