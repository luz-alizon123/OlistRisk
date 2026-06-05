"""
FEATURE ENGINEERING — PROYECTO OLIST
Genera la Master Table Sucia con ~50 features.
Granularidad: 1 fila = 1 order_id
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
ENGINE = create_engine(
    "postgresql://postgres:2001@localhost:5432/OlistRisk"
)

OUTPUT_PATH = "master_table_dirty.parquet"


# ─── PASO 1: CARGAR DATOS ─────────────────────────────────────────────────────

def cargar_datos(engine):
    print("🔄 Cargando datos desde PostgreSQL...")

    base = pd.read_sql("SELECT * FROM v_base_unificada", engine)
    items = pd.read_sql("SELECT * FROM v_items_agg", engine)
    pagos = pd.read_sql("SELECT * FROM v_payments_agg", engine)

    df = base.merge(items, on='order_id', how='left')
    df = df.merge(pagos, on='order_id', how='left')

    print(f"   → {len(df):,} pedidos | {df.shape[1]} columnas iniciales")
    return df


# ─── PASO 2: FEATURES TEMPORALES ─────────────────────────────────────────────

def features_temporales(df):
    print("🔄 Features temporales...")

    df['order_purchase_timestamp']      = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_approved_at']             = pd.to_datetime(df['order_approved_at'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

    df['purchase_weekday']  = df['order_purchase_timestamp'].dt.weekday  # 0=Lun
    df['purchase_hour']     = df['order_purchase_timestamp'].dt.hour
    df['purchase_month']    = df['order_purchase_timestamp'].dt.month
    df['purchase_quarter']  = df['order_purchase_timestamp'].dt.quarter

    # Viernes ≥ 16h o fin de semana → riesgo operativo
    df['high_risk_day'] = np.where(
        (df['purchase_weekday'].isin([5, 6])) |
        ((df['purchase_weekday'] == 4) & (df['purchase_hour'] >= 16)),
        1, 0
    )
    df['is_weekend']     = np.where(df['purchase_weekday'].isin([5, 6]), 1, 0)
    df['is_off_hours']   = np.where(
        (df['purchase_hour'] < 8) | (df['purchase_hour'] >= 18), 1, 0
    )
    df['is_peak_season'] = np.where(df['purchase_month'].isin([11, 12]), 1, 0)

    # Días prometidos al cliente (disponible al momento de la compra)
    df['estimated_delivery_days'] = (
        df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
    ).dt.days

    # Horas hasta aprobación del pago
    df['time_to_approve_hours'] = (
        df['order_approved_at'] - df['order_purchase_timestamp']
    ).dt.total_seconds() / 3600

    # Aprobación lenta: más del doble de la mediana
    mediana = df['time_to_approve_hours'].median()
    df['approval_delay_flag'] = np.where(
        df['time_to_approve_hours'] > mediana * 2.0, 1, 0
    )

    # Promesa agresiva: menos de 10 días prometidos
    df['is_short_promise'] = np.where(df['estimated_delivery_days'] < 10, 1, 0)

    return df


# ─── PASO 3: FEATURES GEOGRÁFICAS (sin geolocation) ──────────────────────────

def features_geograficas(df):
    print("🔄 Features geográficas (proxy por región)...")

    region_alto  = ['AM', 'PA', 'RO', 'AC', 'RR', 'AP', 'TO']
    region_medio = ['BA', 'PE', 'MA', 'PI', 'CE', 'RN', 'PB', 'AL', 'SE']

    df['region_risk_score'] = np.where(
        df['customer_state'].isin(region_alto), 3,
        np.where(df['customer_state'].isin(region_medio), 2, 1)
    )

    df['same_state']      = np.where(df['customer_state'] == df['seller_state'], 1, 0)
    df['interstate_flag'] = 1 - df['same_state']

    # Ruta compleja: hub principal → región Norte
    hubs = ['SP', 'RJ', 'MG']
    df['complex_route'] = np.where(
        (df['seller_state'].isin(hubs)) &
        (df['customer_state'].isin(region_alto)),
        1, 0
    )

    # Distancia proxy basada en región (km promedio estimado)
    dist_proxy = {1: 500, 2: 1800, 3: 2800}
    df['distance_km'] = df['region_risk_score'].map(dist_proxy)
    df['is_long_distance'] = np.where(df['distance_km'] > 1000, 1, 0)

    return df


# ─── PASO 4: FEATURES DEL PRODUCTO ───────────────────────────────────────────

def features_producto(df):
    print("🔄 Features del producto...")

    df['volume_cm3'] = (
        df['avg_length_cm'] * df['avg_height_cm'] * df['avg_width_cm']
    ).fillna(0)

    df['density_gcm3'] = np.where(
        df['volume_cm3'] > 0,
        df['total_weight_g'] / df['volume_cm3'],
        np.nan
    )

    df['max_dimension_cm']  = df[['max_length_cm','max_height_cm','max_width_cm']].max(axis=1)
    df['has_oversized_item']= np.where(df['max_dimension_cm'] > 60, 1, 0)
    df['is_heavy']          = np.where(df['avg_weight_g'] > 5000, 1, 0)

    # Complejidad logística por categoría (en español, sin traducción)
    categorias_alta = [
        'furniture_decor','home_confort','home_comfort','electronics',
        'construction_tools_lights','housewares','office_furniture',
        'small_appliances','air_conditioning','computers','garden_tools'
    ]
    categorias_baja = [
        'books_general_interest','stationery','arts_and_craftmanship',
        'perfumery','fashion_male_clothing','fashion_female_clothing',
        'toys','food_drink','baby','health_beauty'
    ]

    df['logistic_complexity'] = np.where(
        df['main_category'].isin(categorias_alta), 3,
        np.where(df['main_category'].isin(categorias_baja), 1, 2)
    )

    df['carrito_complejo']        = np.where(df['total_items'] > 2, 1, 0)
    df['tiene_varios_vendedores'] = np.where(df['unique_sellers'] > 1, 1, 0)

    df['log_total_weight'] = np.log1p(df['total_weight_g'].fillna(0))
    df['log_volume_cm3']   = np.log1p(df['volume_cm3'].fillna(0))

    return df


# ─── PASO 5: FEATURES DE PAGO ────────────────────────────────────────────────

def features_pago(df):
    print("🔄 Features de pago...")

    df['freight_ratio'] = np.where(
        df['total_price'] > 0,
        df['total_freight'] / df['total_price'],
        0
    )
    df['price_per_item']       = np.where(
        df['total_items'] > 0, df['total_price'] / df['total_items'], 0
    )
    df['is_boleto']            = np.where(df['main_payment_type'] == 'boleto', 1, 0)
    df['is_high_installments'] = np.where(df['max_installments'] > 6, 1, 0)
    df['log_total_payment']    = np.log1p(df['total_payment'].fillna(0))
    df['high_freight_flag']    = np.where(df['freight_ratio'] > 0.5, 1, 0)

    df['freight_category'] = pd.cut(
        df['freight_ratio'],
        bins=[-np.inf, 0.2, 0.5, np.inf],
        labels=[1, 2, 3]
    ).astype(float)

    return df


# ─── PASO 6: HISTORIAL DEL VENDEDOR (rolling, sin data leakage) ──────────────

def features_vendedor(df):
    print("🔄 Historial del vendedor (rolling window)...")

    df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)

    resultados = []
    for seller_id, grupo in df.groupby('main_seller_id'):
        grupo = grupo.sort_values('order_purchase_timestamp').copy()
        grupo['seller_late_rate_hist']    = grupo['is_late_delivery'].expanding().mean().shift(1)
        grupo['seller_total_orders_hist'] = grupo['is_late_delivery'].expanding().count().shift(1)
        resultados.append(
            grupo[['order_id', 'seller_late_rate_hist', 'seller_total_orders_hist']]
        )

    hist = pd.concat(resultados)
    df   = df.merge(hist, on='order_id', how='left')

    media_global = df['is_late_delivery'].mean()
    df['seller_late_rate_hist']    = df['seller_late_rate_hist'].fillna(media_global)
    df['seller_total_orders_hist'] = df['seller_total_orders_hist'].fillna(0)
    df['seller_is_experienced']    = np.where(df['seller_total_orders_hist'] > 50, 1, 0)

    return df


# ─── PASO 7: SCORE DE RIESGO COMPUESTO ───────────────────────────────────────

def features_riesgo(df):
    print("🔄 Score de riesgo compuesto...")

    df['risk_score'] = (
        df['region_risk_score']   * 2.5 +
        df['logistic_complexity'] * 1.5 +
        df['approval_delay_flag'] * 2.5 +
        df['high_risk_day']       * 1.5 +
        df['is_long_distance']    * 1.0 +
        df['is_boleto']           * 1.0 +
        df['complex_route']       * 1.5 +
        df['has_oversized_item']  * 0.5 +
        df['high_freight_flag']   * 0.5
    )

    df['risk_score'] = (df['risk_score'] / df['risk_score'].max() * 10).round(2)

    df['risk_profile'] = pd.cut(
        df['risk_score'],
        bins=[-np.inf, 3.5, 6.5, np.inf],
        labels=['BAJO', 'MEDIO', 'ALTO']
    ).astype(str)

    df['region_x_complexity']   = df['region_risk_score'] * df['logistic_complexity']
    df['distance_x_complexity'] = df['distance_km'] * df['logistic_complexity']
    df['weekday_x_hour']        = df['purchase_weekday'] * df['purchase_hour']

    return df


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  FEATURE ENGINEERING — OLIST")
    print("=" * 55)

    df = cargar_datos(ENGINE)
    df = features_temporales(df)
    df = features_geograficas(df)
    df = features_producto(df)
    df = features_pago(df)
    df = features_vendedor(df)
    df = features_riesgo(df)

    print("\n" + "=" * 55)
    print("  VALIDACIÓN")
    print("=" * 55)
    print(f"  Filas:           {df.shape[0]:,}")
    print(f"  Columnas:        {df.shape[1]}")
    print(f"  Duplicados:      {df['order_id'].duplicated().sum()}")
    print(f"  Tasa de retraso: {df['is_late_delivery'].mean()*100:.1f}%")
    print(f"  Missing máx:     {df.isnull().mean().max()*100:.1f}%")
    print(f"  Risk score:      {df['risk_score'].min():.1f} – {df['risk_score'].max():.1f}")
    print(f"\n  Distribución risk_profile:")
    print(df['risk_profile'].value_counts().to_string())

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n✅ Guardado en: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()