"""
LIMPIEZA + SPLIT TEMPORAL — PROYECTO OLIST
1. Imputa NaN
2. Clipa outliers (p1–p99)
3. WOE Encoding para categóricas
4. StandardScaler para numéricas
5. Split cronológico Train / Val / Backtest / Live
6. Guarda los 4 conjuntos como parquet
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# ─── COLUMNAS POR ROL ────────────────────────────────────────────────────────

TARGET = 'is_late_delivery'

# No son features — solo para identificar y partir en el tiempo
VARS_CONTROL = [
    'order_id', 'customer_id', 'mes',
    'order_purchase_timestamp', 'order_approved_at',
    'order_estimated_delivery_date', 'main_seller_id',
    'customer_city', 'customer_zip_code_prefix', 'seller_zip_code_prefix'
]

# Data leakage — NUNCA entran al modelo
VARS_LEAKAGE = [
    'order_delivered_customer_date',
    'order_delivered_carrier_date',
]

# Categóricas → WOE Encoding manual
VARS_CATEGORICAS = [
    'customer_state', 'seller_state',
    'main_category', 'main_payment_type', 'risk_profile'
]

# Numéricas continuas → escalar
VARS_NUMERICAS = [
    'estimated_delivery_days', 'time_to_approve_hours',
    'distance_km', 'total_weight_g', 'avg_weight_g',
    'volume_cm3', 'density_gcm3', 'total_price', 'total_freight',
    'total_payment', 'freight_ratio', 'price_per_item',
    'log_total_weight', 'log_volume_cm3', 'log_total_payment',
    'risk_score', 'region_x_complexity', 'distance_x_complexity',
    'weekday_x_hour', 'seller_late_rate_hist', 'seller_total_orders_hist',
    'max_dimension_cm'
]

# Ordinales y binarias → sin escalar
VARS_ORDINALES = [
    'region_risk_score', 'logistic_complexity', 'freight_category',
    'purchase_weekday', 'purchase_hour', 'purchase_month', 'purchase_quarter'
]
VARS_BINARIAS = [
    'high_risk_day', 'is_weekend', 'is_off_hours', 'is_peak_season',
    'same_state', 'interstate_flag', 'complex_route', 'is_long_distance',
    'has_oversized_item', 'is_heavy', 'carrito_complejo',
    'tiene_varios_vendedores', 'is_boleto', 'is_high_installments',
    'high_freight_flag', 'approval_delay_flag', 'is_short_promise',
    'seller_is_experienced'
]


# ─── PASO 1: SPLIT TEMPORAL ──────────────────────────────────────────────────

def split_temporal(df):
    print("🔄 Split temporal...")

    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    cortes = {
        'train':    ('2016-09-15', '2018-05-31'),
        'val':      ('2018-06-01', '2018-06-30'),
        'backtest': ('2018-07-01', '2018-07-31'),
        'live':     ('2018-08-01', '2018-08-29'),
    }

    splits = {}
    for nombre, (inicio, fin) in cortes.items():
        mask = (
            (df['order_purchase_timestamp'] >= inicio) &
            (df['order_purchase_timestamp'] <= fin)
        )
        splits[nombre] = df[mask].copy()
        n    = splits[nombre].shape[0]
        tard = splits[nombre][TARGET].mean() * 100
        print(f"   {nombre:10s}: {n:6,} pedidos | {tard:.1f}% tardíos")

    return splits


# ─── PASO 2: IMPUTAR NaN ─────────────────────────────────────────────────────

def imputar_nans(splits):
    print("🔄 Imputando NaN...")

    estrategia = {}

    # Calcular medianas y modas SOLO en train
    for col in VARS_NUMERICAS + VARS_ORDINALES:
        if col in splits['train'].columns:
            estrategia[col] = splits['train'][col].median()

    for col in VARS_CATEGORICAS:
        if col in splits['train'].columns:
            moda = splits['train'][col].mode()
            estrategia[col] = moda[0] if not moda.empty else 'unknown'

    # Aplicar a todos los splits
    for nombre, df in splits.items():
        for col, valor in estrategia.items():
            if col in df.columns:
                splits[nombre][col] = df[col].fillna(valor)

    # Binarias: imputar con 0
    for nombre, df in splits.items():
        for col in VARS_BINARIAS:
            if col in df.columns:
                splits[nombre][col] = df[col].fillna(0)

    print(f"   → Estrategia guardada para {len(estrategia)} columnas")
    return splits, estrategia


# ─── PASO 3: CLIPAR OUTLIERS ─────────────────────────────────────────────────

def clipar_outliers(splits):
    print("🔄 Clipando outliers (p1–p99)...")

    cols_clipar = [
        'time_to_approve_hours', 'total_weight_g', 'freight_ratio',
        'volume_cm3', 'density_gcm3', 'total_price', 'total_freight',
        'estimated_delivery_days', 'total_payment', 'price_per_item',
        'max_dimension_cm'
    ]

    limites = {}

    # Calcular límites SOLO en train
    for col in cols_clipar:
        if col in splits['train'].columns:
            p1  = splits['train'][col].quantile(0.01)
            p99 = splits['train'][col].quantile(0.99)
            limites[col] = (p1, p99)

    # Aplicar a todos
    for nombre, df in splits.items():
        for col, (p1, p99) in limites.items():
            if col in df.columns:
                splits[nombre][col] = df[col].clip(p1, p99)

    print(f"   → Clipado aplicado a {len(limites)} columnas")
    return splits, limites


# ─── PASO 4: WOE ENCODING MANUAL ─────────────────────────────────────────────

def calcular_woe(df_train, col, target=TARGET):
    """Calcula WOE por categoría usando solo train."""
    total_events     = df_train[target].sum()
    total_non_events = len(df_train) - total_events

    tabla = df_train.groupby(col)[target].agg(['sum', 'count'])
    tabla.columns = ['events', 'total']
    tabla['non_events'] = tabla['total'] - tabla['events']

    tabla['pct_e']  = (tabla['events']     / total_events).replace(0, 1e-6)
    tabla['pct_ne'] = (tabla['non_events'] / total_non_events).replace(0, 1e-6)
    tabla['woe']    = np.log(tabla['pct_e'] / tabla['pct_ne'])

    return tabla['woe'].to_dict()

def aplicar_woe(splits):
    print("🔄 WOE Encoding para variables categóricas...")

    woe_maps = {}

    for col in VARS_CATEGORICAS:
        if col not in splits['train'].columns:
            continue

        woe_map = calcular_woe(splits['train'], col)
        woe_maps[col] = woe_map

        for nombre, df in splits.items():
            # Categorías no vistas en train → WOE = 0 (neutro)
            splits[nombre][col] = df[col].map(woe_map).fillna(0)

        print(f"   {col}: {len(woe_map)} categorías codificadas")

    return splits, woe_maps


# ─── PASO 5: ESCALAR NUMÉRICAS ────────────────────────────────────────────────

def escalar(splits):
    print("🔄 Escalando variables numéricas...")

    cols_escalar = list(dict.fromkeys(
        [c for c in VARS_NUMERICAS if c in splits ['train'].columns]
    ))

    scaler = StandardScaler()
    splits['train'][cols_escalar]    = scaler.fit_transform(splits['train'][cols_escalar])

    for nombre in ['val', 'backtest', 'live']:

        if len(splits[nombre]) > 0:
            splits[nombre][cols_escalar] = scaler.transform(splits[nombre][cols_escalar]
                                                            )
        else:
            print(f"⚠️ {nombre} vacío, se omite escalado")

    splits['val'][cols_escalar]      = scaler.transform(splits['val'][cols_escalar])
    splits['backtest'][cols_escalar] = scaler.transform(splits['backtest'][cols_escalar])
    splits['live'][cols_escalar]     = scaler.transform(splits['live'][cols_escalar])

    print(f"   → Escaladas {len(cols_escalar)} columnas")
    return splits, scaler


# ─── PASO 6: GUARDAR ─────────────────────────────────────────────────────────

def guardar(splits, woe_maps, scaler, estrategia, limites):
    print("🔄 Guardando splits...")

    for nombre, df in splits.items():
        path = f"split_{nombre}.parquet"
        df.to_parquet(path, index=False)
        print(f"   ✅ {path}  ({df.shape[0]:,} filas × {df.shape[1]} cols)")

    # Guardar objetos de transformación para usar en predicción futura
    with open('transformaciones.pkl', 'wb') as f:
        pickle.dump({
            'woe_maps':   woe_maps,
            'scaler':     scaler,
            'estrategia': estrategia,
            'limites':    limites
        }, f)
    print("   ✅ transformaciones.pkl")


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  LIMPIEZA + SPLIT — OLIST")
    print("=" * 55)

    # Cargar Master Table Sucia
    df = pd.read_parquet('master_table_dirty.parquet')
    print(f"   Cargado: {df.shape[0]:,} filas × {df.shape[1]} cols\n")

    # Eliminar columnas de leakage
    df.drop(columns=[c for c in VARS_LEAKAGE if c in df.columns], inplace=True)

    splits              = split_temporal(df)
    splits, estrategia  = imputar_nans(splits)
    splits, limites     = clipar_outliers(splits)
    splits, woe_maps    = aplicar_woe(splits)
    splits, scaler      = escalar(splits)

    guardar(splits, woe_maps, scaler, estrategia, limites)

    print("\n" + "=" * 55)
    print("  RESUMEN DE FEATURES DISPONIBLES")
    print("=" * 55)
    features = VARS_NUMERICAS + VARS_ORDINALES + VARS_BINARIAS + VARS_CATEGORICAS
    features = [f for f in features if f in splits['train'].columns]
    print(f"  Features para el modelo: {len(features)}")
    print(f"  {features}")

    print("\n✅ Todo listo para el Sprint 3 — Modelado")


if __name__ == '__main__':
    main()