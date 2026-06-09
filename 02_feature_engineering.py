2 feature engineering · PY
"""
02_feature_engineering.py — SPRINT 2 (v2 CORREGIDO)
=====================================================
Correcciones aplicadas respecto a v1:
  • Variable objetivo: solo órdenes con fecha de entrega real confirmada
  • Variables temporales reconstruidas como features ESTABLES (sin PSI alto):
      - purchase_month → bin_mes_Q (quarter del año, 4 categorías)
      - purchase_quarter → eliminado (redundante con bin_mes_Q)
      - time_to_approve_hours → approval_ratio (relativo a la mediana train)
        + approval_bin (bajo / normal / alto — 3 bins estables)
  • Se agregan: semana del mes, día del mes bin, flag festivos Brasil
  • Se elimina risk_profile (derivado de risk_score → multicolinealidad segura)
  • Todas las variables se documentan con nombre / tipo / lógica / significado
Granularidad: 1 fila = 1 order_id
"""
 
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
 
# ─── CONFIG ──────────────────────────────────────────────────────────────────
ENGINE      = create_engine("postgresql://postgres:2001@localhost:5432/OlistRisk")
OUTPUT_PATH = "master_table_dirty.parquet"
SEED        = 42
 
# Festivos nacionales Brasil (fechas fijas más relevantes)
FESTIVOS_BR = {
    (1,  1), (4, 21), (5,  1), (9,  7),
    (10, 12), (11, 2), (11, 15), (12, 25),
}
 
# ─── PASO 0: CARGAR ──────────────────────────────────────────────────────────
 
def cargar_datos():
    print("🔄 Cargando datos desde PostgreSQL...")
    base  = pd.read_sql("SELECT * FROM v_base_unificada", ENGINE)
    items = pd.read_sql("SELECT * FROM v_items_agg",      ENGINE)
    pagos = pd.read_sql("SELECT * FROM v_payments_agg",   ENGINE)
    df = base.merge(items, on='order_id', how='left')
    df = df.merge(pagos,  on='order_id', how='left')
    print(f"   → {len(df):,} pedidos | {df.shape[1]} columnas iniciales")
    return df
 
 
# ─── PASO 1: VARIABLE OBJETIVO Y FILTRADO ───────────────────────────────────
# ⚠️ CORRECCIÓN CRÍTICA: solo órdenes cerradas (con fecha de entrega real).
# Los registros sin fecha de entrega (cancelados, en tránsito) introducen
# ruido en la target y deben excluirse ANTES de construir cualquier feature.
 
def construir_target(df):
    print("🔄 Construyendo variable objetivo (is_late_delivery)...")
 
    cols_fecha = [
        'order_purchase_timestamp', 'order_estimated_delivery_date',
        'order_delivered_customer_date', 'order_approved_at'
    ]
    for c in cols_fecha:
        df[c] = pd.to_datetime(df[c], errors='coerce')
 
    n_inicial = len(df)
 
    # Exclusión 1: sin fecha de entrega real → pedido no cerrado
    df = df[df['order_delivered_customer_date'].notna()].copy()
    print(f"   → Excluidos sin entrega real: {n_inicial - len(df):,}")
 
    # Exclusión 2: sin fecha estimada → no hay promesa que comparar
    df = df[df['order_estimated_delivery_date'].notna()].copy()
 
    # Exclusión 3: fechas ilógicas (entrega antes de la compra)
    mask_logica = df['order_delivered_customer_date'] >= df['order_purchase_timestamp']
    df = df[mask_logica].copy()
 
    # Exclusión 4: pedidos cancelados (status != delivered)
    if 'order_status' in df.columns:
        df = df[df['order_status'] == 'delivered'].copy()
 
    # Definición de retraso: entrega real > fecha estimada
    df['is_late_delivery'] = (
        df['order_delivered_customer_date'] > df['order_estimated_delivery_date']
    ).astype(int)
 
    # Variable auxiliar: días de retraso (positivo = tarde, negativo = adelantado)
    df['delay_days'] = (
        df['order_delivered_customer_date'] - df['order_estimated_delivery_date']
    ).dt.days
 
    print(f"   → Pedidos válidos: {len(df):,}")
    print(f"   → Tasa de retraso: {df['is_late_delivery'].mean()*100:.1f}%")
    return df
 
 
# ─── PASO 2: FEATURES TEMPORALES (RECONSTRUIDAS PARA ESTABILIDAD PSI) ───────
# Problema original: purchase_month (PSI=9.92) y purchase_quarter (PSI=3.08)
# eran inestables porque el periodo de validación (jun 2018) cae fuera del
# rango estacional del train (sep 2016–may 2018).
# Solución: transformar en BINS CÍCLICOS o FLAGS que tengan la misma
# distribución esperada en cualquier mes del año.
 
def features_temporales(df):
    print("🔄 Features temporales (v2 — estables)...")
 
    ts = df['order_purchase_timestamp']
 
    # ── Componentes crudas (no entran al modelo, solo para derivar features) ──
    mes   = ts.dt.month
    hora  = ts.dt.hour
    dia_s = ts.dt.dayofweek      # 0=Lun … 6=Dom
    dia_m = ts.dt.day            # 1–31
 
    # ── Features estables ────────────────────────────────────────────────────
 
    # 1. purchase_weekday — 0–6 (lunes–domingo) → ORDINAL
    #    Captura patrón operativo semanal (distribución plana ≈ uniforme)
    df['purchase_weekday'] = dia_s
 
    # 2. purchase_hour — 0–23 → ORDINAL
    df['purchase_hour'] = hora
 
    # 3. bin_hora — agrupa hora en 3 turnos estables (noche/día/tarde)
    #    Más estable que hora exacta; PSI esperado < 0.05
    df['bin_hora'] = pd.cut(hora, bins=[-1, 6, 14, 18, 24],
                            labels=[0, 1, 2, 3]).astype(float)
 
    # 4. is_weekend — flag binario (distribución ~2/7 siempre)
    df['is_weekend'] = (dia_s >= 5).astype(int)
 
    # 5. is_off_hours — fuera del horario laboral (< 8h o ≥ 18h)
    df['is_off_hours'] = ((hora < 8) | (hora >= 18)).astype(int)
 
    # 6. high_risk_day — viernes tarde o fin de semana
    df['high_risk_day'] = (
        (dia_s.isin([5, 6])) |
        ((dia_s == 4) & (hora >= 16))
    ).astype(int)
 
    # 7. semana_del_mes — 1ª a 5ª semana (distribución estable cualquier mes)
    df['semana_del_mes'] = ((dia_m - 1) // 7 + 1).clip(1, 5)
 
    # 8. is_peak_season — solo noviembre–diciembre (Black Friday / Navidad)
    #    Reemplaza purchase_month; PSI bajo porque es flag 0/1
    df['is_peak_season'] = mes.isin([11, 12]).astype(int)
 
    # 9. is_festivo — festivo nacional Brasil
    #    Operaciones se acumulan/retrasan alrededor de festivos
    df['is_festivo'] = ts.apply(
        lambda t: int((t.month, t.day) in FESTIVOS_BR)
    )
 
    # 10. Días prometidos al cliente (disponible en el momento de la compra)
    df['estimated_delivery_days'] = (
        df['order_estimated_delivery_date'] - ts
    ).dt.days.clip(0, 120)
 
    # 11. is_short_promise — promesa agresiva < 10 días
    df['is_short_promise'] = (df['estimated_delivery_days'] < 10).astype(int)
 
    # 12. approval_hours — horas hasta aprobación de pago (raw, para derivar)
    df['approval_hours_raw'] = (
        df['order_approved_at'] - ts
    ).dt.total_seconds() / 3600
 
    # 13. approval_ratio — tiempo de aprobación / mediana global
    #     REEMPLAZA time_to_approve_hours (que tenía PSI=0.52).
    #     Al dividir por la mediana se vuelve una medida relativa estable.
    mediana_aprov = df['approval_hours_raw'].median()
    df['approval_ratio'] = (
        df['approval_hours_raw'] / max(mediana_aprov, 0.1)
    ).clip(0, 20).fillna(1.0)
 
    # 14. approval_bin — versión discreta (3 bins) para WOE posterior
    df['approval_bin'] = pd.cut(
        df['approval_ratio'],
        bins=[-np.inf, 0.5, 2.0, np.inf],
        labels=[0, 1, 2]
    ).astype(float)
 
    # 15. approval_delay_flag — aprobación muy lenta (ratio > 3×)
    df['approval_delay_flag'] = (df['approval_ratio'] > 3.0).astype(int)
 
    # 16. weekday_x_hour — interacción (captura ej: domingos nocturnos)
    df['weekday_x_hour'] = dia_s * hora
 
    return df
 
 
# ─── PASO 3: FEATURES GEOGRÁFICAS ────────────────────────────────────────────
 
def features_geograficas(df):
    print("🔄 Features geográficas...")
 
    region_alto  = ['AM', 'PA', 'RO', 'AC', 'RR', 'AP', 'TO']
    region_medio = ['BA', 'PE', 'MA', 'PI', 'CE', 'RN', 'PB', 'AL', 'SE']
 
    # region_risk_score — 1/2/3 según región destino
    df['region_risk_score'] = np.where(
        df['customer_state'].isin(region_alto), 3,
        np.where(df['customer_state'].isin(region_medio), 2, 1)
    )
 
    df['same_state']      = (df['customer_state'] == df['seller_state']).astype(int)
    df['interstate_flag'] = 1 - df['same_state']
 
    hubs = ['SP', 'RJ', 'MG']
    df['complex_route'] = (
        df['seller_state'].isin(hubs) &
        df['customer_state'].isin(region_alto)
    ).astype(int)
 
    dist_proxy = {1: 500, 2: 1800, 3: 2800}
    df['distance_km']      = df['region_risk_score'].map(dist_proxy)
    df['is_long_distance'] = (df['distance_km'] > 1000).astype(int)
 
    return df
 
 
# ─── PASO 4: FEATURES DEL PRODUCTO ───────────────────────────────────────────
 
def features_producto(df):
    print("🔄 Features del producto...")
 
    df['volume_cm3'] = (
        df['avg_length_cm'] * df['avg_height_cm'] * df['avg_width_cm']
    ).fillna(0)
 
    df['density_gcm3'] = np.where(
        df['volume_cm3'] > 0,
        df['total_weight_g'] / df['volume_cm3'], np.nan
    )
 
    df['max_dimension_cm']  = df[['max_length_cm','max_height_cm','max_width_cm']].max(axis=1)
    df['has_oversized_item']= (df['max_dimension_cm'] > 60).astype(int)
    df['is_heavy']          = (df['avg_weight_g'] > 5000).astype(int)
 
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
 
    df['carrito_complejo']        = (df['total_items'] > 2).astype(int)
    df['tiene_varios_vendedores'] = (df['unique_sellers'] > 1).astype(int)
    df['log_total_weight']        = np.log1p(df['total_weight_g'].fillna(0))
    df['log_volume_cm3']          = np.log1p(df['volume_cm3'].fillna(0))
 
    return df
 
 
# ─── PASO 5: FEATURES DE PAGO ────────────────────────────────────────────────
 
def features_pago(df):
    print("🔄 Features de pago...")
 
    df['freight_ratio'] = np.where(
        df['total_price'] > 0, df['total_freight'] / df['total_price'], 0
    )
    df['price_per_item']       = np.where(
        df['total_items'] > 0, df['total_price'] / df['total_items'], 0
    )
    df['is_boleto']            = (df['main_payment_type'] == 'boleto').astype(int)
    df['is_high_installments'] = (df['max_installments'] > 6).astype(int)
    df['log_total_payment']    = np.log1p(df['total_payment'].fillna(0))
    df['high_freight_flag']    = (df['freight_ratio'] > 0.5).astype(int)
 
    df['freight_category'] = pd.cut(
        df['freight_ratio'],
        bins=[-np.inf, 0.2, 0.5, np.inf],
        labels=[1, 2, 3]
    ).astype(float)
 
    # Flete por kg — normaliza el peso en la ecuación de costo
    df['freight_per_kg'] = np.where(
        df['total_weight_g'] > 0,
        df['total_freight'] / (df['total_weight_g'] / 1000), 0
    ).clip(0, 500)
 
    return df
 
 
# ─── PASO 6: HISTORIAL DEL VENDEDOR (rolling sin data leakage) ───────────────
 
def features_vendedor(df):
    print("🔄 Historial del vendedor (rolling window)...")
 
    df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
    resultados = []
    for seller_id, grupo in df.groupby('main_seller_id'):
        grupo = grupo.sort_values('order_purchase_timestamp').copy()
        grupo['seller_late_rate_hist']    = grupo['is_late_delivery'].expanding().mean().shift(1)
        grupo['seller_total_orders_hist'] = grupo['is_late_delivery'].expanding().count().shift(1)
        resultados.append(grupo[['order_id','seller_late_rate_hist','seller_total_orders_hist']])
 
    hist = pd.concat(resultados)
    df   = df.merge(hist, on='order_id', how='left')
 
    media_global = df['is_late_delivery'].mean()
    df['seller_late_rate_hist']    = df['seller_late_rate_hist'].fillna(media_global)
    df['seller_total_orders_hist'] = df['seller_total_orders_hist'].fillna(0)
    df['seller_is_experienced']    = (df['seller_total_orders_hist'] > 50).astype(int)
 
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
 
    # ⚠️ risk_profile ELIMINADO: es derivado lineal de risk_score
    # → multicolinealidad garantizada → lo elimina el paso de correlación.
    # Se mantiene risk_score (numérico, continuo) como feature.
 
    df['region_x_complexity']   = df['region_risk_score'] * df['logistic_complexity']
    df['distance_x_complexity'] = df['distance_km'] * df['logistic_complexity']
 
    return df
 
 
# ─── PIPELINE ─────────────────────────────────────────────────────────────────
 
def main():
    print("=" * 60)
    print("  FEATURE ENGINEERING v2 — OLIST")
    print("=" * 60)
 
    df = cargar_datos()
    df = construir_target(df)
    df = features_temporales(df)
    df = features_geograficas(df)
    df = features_producto(df)
    df = features_pago(df)
    df = features_vendedor(df)
    df = features_riesgo(df)
 
    # ── Documentación completa de features ────────────────────────────────────
    CATALOGO = """
╔══════════════════════════════════════════════════════════════════════╗
║              CATÁLOGO DE VARIABLES — OLIST v2                        ║
╠══════════════════════════════════════════════════════════════════════╣
║ GRUPO 1 — VARIABLES TEMPORALES (16 features)                         ║
║  purchase_weekday      Num/Ordinal  0=Lun … 6=Dom                    ║
║  purchase_hour         Num/Ordinal  0–23h                            ║
║  bin_hora              Num/Ordinal  0=noche 1=día 2=tarde 3=noche2   ║
║  is_weekend            Binaria      1 si sábado o domingo            ║
║  is_off_hours          Binaria      1 si < 8h o ≥ 18h               ║
║  high_risk_day         Binaria      1 si viernes tarde o fin semana  ║
║  semana_del_mes        Num/Ordinal  1–5 (semana dentro del mes)      ║
║  is_peak_season        Binaria      1 si nov–dic                     ║
║  is_festivo            Binaria      1 si festivo nacional Brasil      ║
║  estimated_delivery_days Numérica   días prometidos desde la compra  ║
║  is_short_promise      Binaria      1 si prometidos < 10 días        ║
║  approval_ratio        Numérica     horas_aprobación / mediana_global║
║  approval_bin          Num/Ordinal  0=rápido 1=normal 2=lento        ║
║  approval_delay_flag   Binaria      1 si ratio > 3× la mediana       ║
║  weekday_x_hour        Numérica     interacción día×hora             ║
╠══════════════════════════════════════════════════════════════════════╣
║ GRUPO 2 — VARIABLES GEOGRÁFICAS (6 features)                         ║
║  region_risk_score     Num/Ordinal  1=baja 2=media 3=alta (destino)  ║
║  same_state            Binaria      1 si comprador y vendedor mismo   ║
║  interstate_flag       Binaria      1 si envío interestatal           ║
║  complex_route         Binaria      1 si hub SP/RJ/MG → Norte        ║
║  distance_km           Numérica     km proxy por región destino       ║
║  is_long_distance      Binaria      1 si distance_km > 1000          ║
╠══════════════════════════════════════════════════════════════════════╣
║ GRUPO 3 — VARIABLES DE PRODUCTO / LOGÍSTICA (10 features)            ║
║  volume_cm3            Numérica     largo×alto×ancho promedio         ║
║  density_gcm3          Numérica     peso_total / volumen              ║
║  max_dimension_cm      Numérica     mayor dimensión del pedido        ║
║  has_oversized_item    Binaria      1 si alguna dim > 60 cm           ║
║  is_heavy              Binaria      1 si peso_promedio > 5 kg         ║
║  logistic_complexity   Num/Ordinal  1=baja 2=media 3=alta (categoría)║
║  carrito_complejo      Binaria      1 si total_items > 2              ║
║  tiene_varios_vendedores Binaria    1 si unique_sellers > 1           ║
║  log_total_weight      Numérica     log(1 + peso_total_g)            ║
║  log_volume_cm3        Numérica     log(1 + volumen_cm3)             ║
╠══════════════════════════════════════════════════════════════════════╣
║ GRUPO 4 — VARIABLES DE PAGO (8 features)                             ║
║  freight_ratio         Numérica     flete / precio_total             ║
║  price_per_item        Numérica     precio_total / n_items           ║
║  is_boleto             Binaria      1 si pago en boleto bancário      ║
║  is_high_installments  Binaria      1 si cuotas > 6                  ║
║  log_total_payment     Numérica     log(1 + total_pago)              ║
║  high_freight_flag     Binaria      1 si freight_ratio > 0.5         ║
║  freight_category      Num/Ordinal  1=bajo 2=medio 3=alto            ║
║  freight_per_kg        Numérica     flete por kg enviado             ║
╠══════════════════════════════════════════════════════════════════════╣
║ GRUPO 5 — HISTORIAL DEL VENDEDOR (3 features)                        ║
║  seller_late_rate_hist  Numérica    tasa retraso histórica (rolling)  ║
║  seller_total_orders_hist Numérica  órdenes previas del vendedor      ║
║  seller_is_experienced  Binaria     1 si más de 50 órdenes previas   ║
╠══════════════════════════════════════════════════════════════════════╣
║ GRUPO 6 — SCORE E INTERACCIONES (3 features)                         ║
║  risk_score            Numérica     score compuesto 0–10             ║
║  region_x_complexity   Numérica     region_risk × logistic_complexity ║
║  distance_x_complexity Numérica     distance_km × logistic_complexity ║
╚══════════════════════════════════════════════════════════════════════╝
  TOTAL: ~46 features de ingeniería + columnas originales necesarias
"""
    print(CATALOGO)
 
    print("=" * 60)
    print("  VALIDACIÓN")
    print("=" * 60)
    print(f"  Filas:           {df.shape[0]:,}")
    print(f"  Columnas:        {df.shape[1]}")
    print(f"  Duplicados:      {df['order_id'].duplicated().sum()}")
    print(f"  Tasa de retraso: {df['is_late_delivery'].mean()*100:.1f}%")
    print(f"  Missing máx:     {df.isnull().mean().max()*100:.1f}%")
    print(f"  Risk score:      {df['risk_score'].min():.1f} – {df['risk_score'].max():.1f}")
 
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n✅ Guardado en: {OUTPUT_PATH}")
    print("   → Ejecutar ahora: python 03_preparar_datos.py")
 
 
if __name__ == '__main__':
    main()
