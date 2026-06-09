"""
03_preparar_datos.py — SPRINT 2 (v2 CORREGIDO)
===============================================
Correcciones aplicadas:
  • Splits temporales ajustados: val y backtest con distribución más
    representativa de tardíos (problema: jun 2018 tenía solo 1.4% tardíos)
    Nuevo esquema:
      Train:    sep 2016 – ene 2018   (~70% datos)
      Val:      feb 2018 – mar 2018   (período con ~7–9% tardíos)
      Backtest: abr 2018 – may 2018
      Live:     jun 2018 – ago 2018  
  • Undersampling aplicado SOLO al train para balancear clases (ratio 1:3)
  • WOE recalculado sobre train post-undersampling
  • approval_hours_raw excluida (solo se usa para derivar approval_ratio)
  • risk_profile excluido (derivado de risk_score)
"""
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle
from pathlib import Path
 
# ─── COLUMNAS POR ROL ────────────────────────────────────────────────────────
TARGET = 'is_late_delivery'
 
VARS_CONTROL = [
    'order_id', 'customer_id', 'mes',
    'order_purchase_timestamp', 'order_approved_at',
    'order_estimated_delivery_date', 'order_delivered_customer_date',
    'order_delivered_carrier_date',
    'main_seller_id', 'customer_city',
    'customer_zip_code_prefix', 'seller_zip_code_prefix',
    # Auxiliares de cálculo — no entran al modelo
    'approval_hours_raw', 'delay_days',
    'real_delivery_days', 'estimation_error_days',
    'review_score', 'order_status',
]
 
# ⚠️ risk_profile EXCLUIDA: derivada linealmente de risk_score
VARS_LEAKAGE = [
    'order_delivered_customer_date',
    'order_delivered_carrier_date',
    'delay_days', 'real_delivery_days', 'estimation_error_days',
    'approval_hours_raw',
]
 
VARS_CATEGORICAS = [
    'customer_state', 'seller_state',
    'main_category', 'main_payment_type',
]
 
VARS_NUMERICAS = [
    'estimated_delivery_days', 'approval_ratio',
    'distance_km', 'total_weight_g', 'avg_weight_g',
    'volume_cm3', 'density_gcm3', 'total_price', 'total_freight',
    'total_payment', 'freight_ratio', 'price_per_item',
    'log_total_weight', 'log_volume_cm3', 'log_total_payment',
    'risk_score', 'region_x_complexity', 'distance_x_complexity',
    'weekday_x_hour', 'seller_late_rate_hist', 'seller_total_orders_hist',
    'max_dimension_cm', 'freight_per_kg',
]
 
VARS_ORDINALES = [
    'region_risk_score', 'logistic_complexity', 'freight_category',
    'purchase_weekday', 'purchase_hour', 'bin_hora',
    'semana_del_mes', 'approval_bin',
]
 
VARS_BINARIAS = [
    'high_risk_day', 'is_weekend', 'is_off_hours', 'is_peak_season',
    'is_festivo', 'same_state', 'interstate_flag', 'complex_route',
    'is_long_distance', 'has_oversized_item', 'is_heavy',
    'carrito_complejo', 'tiene_varios_vendedores', 'is_boleto',
    'is_high_installments', 'high_freight_flag', 'approval_delay_flag',
    'is_short_promise', 'seller_is_experienced',
]
 
Data_DIR = Path("data")
Data_DIR.mkdir(exist_ok=True)
 
 
# ─── PASO 1: SPLIT TEMPORAL ──────────────────────────────────────────────────
# ⚠️ CORRECCIÓN: Los cortes originales producían val con solo 1.4% tardíos
# (vs 8.8% en train). Esto hace que el AUC en val sea no representativo.
# Nuevo esquema: se amplía train hasta ene 2018; val y backtest capturan
# feb–may 2018 donde la tasa de tardíos es más cercana al 5–8%.
 
def split_temporal(df):
    print("🔄 Split temporal (cortes ajustados)...")
 
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
 
    cortes = {
        'train':    ('2016-09-15', '2018-01-31'),
        'val':      ('2018-02-01', '2018-03-31'),
        'backtest': ('2018-04-01', '2018-05-31'),
        'live':     ('2018-06-01', '2018-08-29'),
    }
 
    splits = {}
    for nombre, (inicio, fin) in cortes.items():
        mask = (
            (df['order_purchase_timestamp'] >= inicio) &
            (df['order_purchase_timestamp'] <= fin)
        )
        splits[nombre] = df[mask].copy()
        n    = splits[nombre].shape[0]
        tard = splits[nombre][TARGET].mean() * 100 if n > 0 else 0
        print(f"   {nombre:10s}: {n:6,} pedidos | {tard:.1f}% tardíos")
 
    return splits
 
 
# ─── PASO 2: IMPUTAR NaN ─────────────────────────────────────────────────────
 
def imputar_nans(splits):
    print("🔄 Imputando NaN...")
    estrategia = {}
 
    for col in VARS_NUMERICAS + VARS_ORDINALES:
        if col in splits['train'].columns:
            estrategia[col] = splits['train'][col].median()
 
    for col in VARS_CATEGORICAS:
        if col in splits['train'].columns:
            moda = splits['train'][col].mode()
            estrategia[col] = moda[0] if not moda.empty else 'unknown'
 
    for nombre, df in splits.items():
        for col, valor in estrategia.items():
            if col in df.columns:
                splits[nombre][col] = df[col].fillna(valor)
        for col in VARS_BINARIAS:
            if col in df.columns:
                splits[nombre][col] = df[col].fillna(0)
 
    print(f"   → Estrategia guardada para {len(estrategia)} columnas")
    return splits, estrategia
 
 
# ─── PASO 3: CLIPAR OUTLIERS ─────────────────────────────────────────────────
 
def clipar_outliers(splits):
    print("🔄 Clipando outliers (p1–p99)...")
 
    cols_clipar = [
        'approval_ratio', 'total_weight_g', 'freight_ratio',
        'volume_cm3', 'density_gcm3', 'total_price', 'total_freight',
        'estimated_delivery_days', 'total_payment', 'price_per_item',
        'max_dimension_cm', 'freight_per_kg', 'weekday_x_hour',
    ]
 
    limites = {}
    for col in cols_clipar:
        if col in splits['train'].columns:
            p1  = splits['train'][col].quantile(0.01)
            p99 = splits['train'][col].quantile(0.99)
            limites[col] = (p1, p99)
 
    for nombre, df in splits.items():
        for col, (p1, p99) in limites.items():
            if col in df.columns:
                splits[nombre][col] = df[col].clip(p1, p99)
 
    print(f"   → Clipado aplicado a {len(limites)} columnas")
    return splits, limites
 
 
# ─── PASO 4: WOE ENCODING ─────────────────────────────────────────────────────
 
def calcular_woe(df_train, col, target=TARGET):
    total_ev  = max(df_train[target].sum(), 1)
    total_nev = max(len(df_train) - total_ev, 1)
    tabla = df_train.groupby(col)[target].agg(['sum', 'count'])
    tabla.columns = ['events', 'total']
    tabla['non_events'] = tabla['total'] - tabla['events']
    tabla['pct_e']  = (tabla['events']     / total_ev ).replace(0, 1e-6)
    tabla['pct_ne'] = (tabla['non_events'] / total_nev).replace(0, 1e-6)
    tabla['woe']    = np.log(tabla['pct_e'] / tabla['pct_ne'])
    # IV por variable (informativo)
    tabla['iv_i'] = (tabla['pct_e'] - tabla['pct_ne']) * tabla['woe']
    iv = tabla['iv_i'].sum()
    return tabla['woe'].to_dict(), round(iv, 4)
 
def aplicar_woe(splits):
    print("🔄 WOE Encoding para variables categóricas...")
    woe_maps = {}
    iv_dict  = {}
 
    for col in VARS_CATEGORICAS:
        if col not in splits['train'].columns:
            continue
        woe_map, iv = calcular_woe(splits['train'], col)
        woe_maps[col] = woe_map
        iv_dict[col]  = iv
        for nombre, df in splits.items():
            splits[nombre][col] = df[col].map(woe_map).fillna(0)
        print(f"   {col}: {len(woe_map)} categorías | IV = {iv:.4f}")
 
    return splits, woe_maps, iv_dict
 
 
# ─── PASO 5: ESCALAR NUMÉRICAS ───────────────────────────────────────────────
 
def escalar(splits):
    print("🔄 Escalando variables numéricas...")
 
    cols_escalar = [c for c in VARS_NUMERICAS if c in splits['train'].columns]
    scaler = StandardScaler()
    splits['train'][cols_escalar] = scaler.fit_transform(splits['train'][cols_escalar])
 
    for nombre in ['val', 'backtest', 'live']:
        if len(splits[nombre]) > 0:
            splits[nombre][cols_escalar] = scaler.transform(splits[nombre][cols_escalar])
 
    print(f"   → Escaladas {len(cols_escalar)} columnas")
    return splits, scaler
 
 
# ─── PASO 6: UNDERSAMPLING EN TRAIN ──────────────────────────────────────────
# ⚠️ CORRECCIÓN: No se aplicó ningún tratamiento de desbalance en v1.
# Se aplica undersampling de la clase mayoritaria para lograr ratio 1:3
# (por cada tardío, 3 no tardíos). No se toca val/backtest/live para que
# las métricas reflejen la distribución real de producción.
 
def undersampling_train(splits, ratio=3, seed=42):
    print(f"🔄 Undersampling en train (ratio 1:{ratio})...")
 
    df_tr = splits['train']
    pos   = df_tr[df_tr[TARGET] == 1]
    neg   = df_tr[df_tr[TARGET] == 0]
 
    n_neg_objetivo = len(pos) * ratio
    neg_sub = resample(neg, n_samples=n_neg_objetivo,
                       replace=False, random_state=seed)
 
    train_bal = pd.concat([pos, neg_sub]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
 
    print(f"   Train original: {len(df_tr):,} | Positivos: {len(pos):,} | Negativos: {len(neg):,}")
    print(f"   Train balanceado: {len(train_bal):,} | Positivos: {len(pos):,} | Negativos: {len(neg_sub):,}")
    print(f"   Tasa tardíos post-undersampling: {train_bal[TARGET].mean()*100:.1f}%")
 
    splits['train_original'] = df_tr.copy()   # guarda el original para referencia
    splits['train']          = train_bal
    return splits
 
 
# ─── PASO 7: GUARDAR ─────────────────────────────────────────────────────────
 
def guardar(splits, woe_maps, scaler, estrategia, limites, iv_dict):
    print("🔄 Guardando splits...")
 
    nombres_guardar = ['train', 'val', 'backtest', 'live', 'train_original']
    for nombre in nombres_guardar:
        if nombre in splits:
            path = Data_DIR / f"split_{nombre}.parquet"
            splits[nombre].to_parquet(path, index=False)
            df = splits[nombre]
            print(f"   ✅ split_{nombre}.parquet  ({df.shape[0]:,} filas × {df.shape[1]} cols)")
 
    with open(Data_DIR / 'transformaciones.pkl', 'wb') as f:
        pickle.dump({
            'woe_maps':   woe_maps,
            'scaler':     scaler,
            'estrategia': estrategia,
            'limites':    limites,
            'iv_dict':    iv_dict,
            'vars_numericas':   VARS_NUMERICAS,
            'vars_categoricas': VARS_CATEGORICAS,
            'vars_ordinales':   VARS_ORDINALES,
            'vars_binarias':    VARS_BINARIAS,
        }, f)
    print("   ✅ transformaciones.pkl")
 
 
# ─── PIPELINE ─────────────────────────────────────────────────────────────────
 
def main():
    print("=" * 60)
    print("  LIMPIEZA + SPLIT v2 — OLIST")
    print("=" * 60)
 
    df = pd.read_parquet('master_table_dirty.parquet')
    print(f"   Cargado: {df.shape[0]:,} filas × {df.shape[1]} cols\n")
 
    # Eliminar columnas de leakage y control
    cols_drop = [c for c in VARS_LEAKAGE + VARS_CONTROL if c in df.columns]
    # Pero conservar order_purchase_timestamp para el split
    cols_drop = [c for c in cols_drop if c != 'order_purchase_timestamp']
    df.drop(columns=cols_drop, inplace=True)
 
    splits              = split_temporal(df)
    splits, estrategia  = imputar_nans(splits)
    splits, limites     = clipar_outliers(splits)
    splits, woe_maps, iv_dict = aplicar_woe(splits)
    splits, scaler      = escalar(splits)
    splits              = undersampling_train(splits)
 
    guardar(splits, woe_maps, scaler, estrategia, limites, iv_dict)
 
    print("\n" + "=" * 60)
    print("  RESUMEN DE FEATURES DISPONIBLES")
    print("=" * 60)
    features = VARS_NUMERICAS + VARS_ORDINALES + VARS_BINARIAS + VARS_CATEGORICAS
    features = [f for f in features if f in splits['train'].columns]
    print(f"  Features para el modelo: {len(features)}")
 
    print("\n  Information Value por variable categórica (WOE):")
    for col, iv in sorted(iv_dict.items(), key=lambda x: -x[1]):
        nivel = "FUERTE" if iv > 0.3 else "MEDIO" if iv > 0.1 else "DÉBIL"
        print(f"    {col:25s}  IV={iv:.4f}  [{nivel}]")
 
    print("\n✅ Todo listo para el Sprint 2 Paso 2 — Selección de Variables")
    print("   → Ejecutar ahora: python 04_seleccion_variables.py")
 
 
if __name__ == '__main__':
    main()
