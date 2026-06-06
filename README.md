# Proyecto Olist - Predicción de Retrasos en Entregas

## Descripción del Proyecto

Este proyecto tiene como objetivo predecir si una orden de compra será entregada con retraso utilizando técnicas de Machine Learning sobre el dataset público de Olist.

La solución fue desarrollada siguiendo una arquitectura de datos completa, desde la extracción y almacenamiento en PostgreSQL hasta la construcción de modelos predictivos y visualización de resultados mediante Streamlit.

## Arquitectura General

```mermaid
graph TD
    A[Dataset CSV Olist] --> B[PostgreSQL]
    B --> C[01_conexion_postgres.py]
    C --> D[05_Master_Table.md<br>(master_table_dirty.parquet)]
    D --> E[02_feature_engineering.py]
    E --> F[03_preparar_datos.py]
    F --> G[04_seleccion_variables.py]
    G --> H[05_modelado.py]
    H --> I[Champion Model]
    I --> J[Dashboard Streamlit]
```

## Tecnologías Utilizadas

- **Python 3.14**
- **PostgreSQL**
- **Pandas**
- **NumPy**
- **Scikit-Learn**
- **XGBoost**
- **LightGBM**
- **SHAP**
- **Matplotlib**
- **Seaborn**
- **Joblib**
- **Streamlit**

## Estructura del Proyecto

```
olist/
├── data/
│   ├── split_train.parquet
│   ├── split_val.parquet
│   ├── split_backtest.parquet
│   ├── split_live.parquet
│   └── features_finales.pkl
├── datasets/
│   ├── olist_customers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   ├── olist_order_items_dataset
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── product_category_name_translation.csv
│   └── olist_sellers_dataset.csv

├── models/
│   ├── champion_model_v1.pkl
│   └── champion_resumen.json
├── reports/
│   ├── curva_roc_comparativa.png
│   └── shap_importancia.png
├── 01_conexion_postgres.py
├── 02_feature_engineering.py
├── 03_preparar_datos.py
├── 04_seleccion_variables.py
├── 05_modelado.py
├── app.py
└── README.md
```

## 1. Carga y Consolidación de Datos (PostgreSQL)

### Objetivo
Centralizar toda la información del negocio en una única base de datos relacional para facilitar el análisis y modelado.

### Proceso
Se cargaron los archivos CSV originales de Olist en PostgreSQL:

- `customers`
- `orders`
- `order_items`
- `order_payments`
- `order_reviews`
- `products`
- `sellers`
- `geolocation`

Posteriormente se construyó una vista consolidada: **`v_base_unificada`**

Esta vista integra toda la información necesaria para el proyecto mediante JOINs entre tablas.

### Validación de conexión

```python
from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql://postgres:2001@localhost:5432/OlistRisk"
)

with engine.connect() as conn:
    result = conn.execute(
        text("SELECT COUNT(*) FROM v_base_unificada")
    )
    print(result.fetchone()[0])
```

**Resultado:** 96,470 registros

## 2. Feature Engineering

**Archivo:** `02_feature_engineering.py`

### Objetivo
Transformar los datos operacionales en variables predictivas capaces de capturar patrones asociados a retrasos logísticos.

### Variables generadas

#### Temporales
- `purchase_month`
- `purchase_quarter`
- `purchase_weekday`
- `purchase_hour`
- `is_weekend`
- `is_off_hours`
- `is_peak_season`

#### Logísticas
- `estimated_delivery_days`
- `approval_delay_flag`
- `is_short_promise`
- `interstate_flag`
- `distance_km`
- `is_long_distance`
- `complex_route`

#### Producto
- `total_weight_g`
- `avg_weight_g`
- `volume_cm3`
- `density_gcm3`
- `max_dimension_cm`
- `has_oversized_item`
- `is_heavy`

#### Carrito
- `total_items`
- `unique_products`
- `carrito_complejo`
- `tiene_varios_vendedores`

#### Pago
- `total_payment`
- `log_total_payment`
- `payment_methods`
- `main_payment_type`
- `max_installments`
- `is_boleto`

#### Riesgo
- `seller_total_orders_hist`
- `seller_late_rate_hist`
- `seller_is_experienced`
- `risk_score`
- `risk_profile`

**Resultado:** `master_table_dirty.parquet` con **76 columnas**.

## 3. Preparación de Datos

**Archivo:** `03_preparar_datos.py`

### Objetivo
Preparar los datos para entrenamiento evitando fugas de información (Data Leakage).

### Variable objetivo
`is_late_delivery` = `order_delivered_customer_date > order_estimated_delivery_date`

### División temporal
- **Train**: Hasta 2018-04-29
- **Validation**: 2018-05-01 a 2018-05-31
- **Backtest**: 2018-06-01 a 2018-06-29
- **Live**: 2018-07-01 en adelante

### Transformaciones aplicadas
- **Missing Values**: `SimpleImputer`
- **Variables categóricas**: WOE Encoder
- **Variables numéricas**: `RobustScaler`

**Archivos generados:**
- `split_train.parquet`
- `split_val.parquet`
- `split_backtest.parquet`
- `split_live.parquet`

## 4. Selección de Variables

**Archivo:** `04_seleccion_variables.py`

### Objetivo
Eliminar variables redundantes o inestables manteniendo el máximo desempeño predictivo.

### Proceso
- **Estado inicial**: 64 variables (AUC Train=0.7802 / Val=0.6360)
- **Missing Rate** (>10%): 0 eliminadas
- **PSI** (>0.25): Eliminadas `purchase_month`, `purchase_quarter`, `time_to_approve_hours`, `seller_late_rate_hist`, `total_freight`
- **Correlación** (>0.90): Eliminadas 11 variables
- **Resultado final**: **48 variables** (AUC Train=0.7340 / Val=0.6330)

## 5. Modelado Predictivo

**Archivo:** `05_modelado.py`

### Modelos evaluados
| Modelo                | AUC Validation |
|-----------------------|----------------|
| Logistic Regression   | 0.5748        |
| Random Forest         | **0.6067**    |
| XGBoost               | 0.5565        |
| LightGBM              | 0.4856        |
| Stacking Ensemble     | 0.5549        |

### Modelo Champion
**Random Forest**

**Métricas principales:**
- AUC-ROC = 0.6067
- Gini = 0.2133
- KS = 0.1855
- Recall = 73.49% (umbral 0.5)

**Umbral óptimo:** 0.56 → Recall=15.66%, Precision=4.32%, F1=0.0677

### Backtesting Temporal
- Validation: AUC = 0.6067
- Backtest: AUC = 0.6266
- Live: AUC = 0.4958 (degradación observada)

### Interpretabilidad
Se utilizó **SHAP**. Variables más importantes:
- `estimated_delivery_days`
- `interstate_flag`
- `seller_total_orders_hist`
- `risk_score`
- `freight_ratio`
- `log_total_payment`

## Dashboard Streamlit

**Archivo:** `app.py`

Visualiza:
- Métricas del Champion Model
- Curva ROC
- Importancia SHAP
- Resultados de Backtesting
- Comparación entre modelos
- KPIs del negocio

**Ejecución:**
```bash
streamlit run app.py
```

## Próximos Pasos

- Hyperparameter Tuning
- Cross Validation Temporal
- Feature Selection Avanzada
- Calibración de Probabilidades
- Monitoreo de Drift
- Despliegue en Producción
- Integración con APIs
- Automatización ETL

---

**Proyecto completado hasta la fase de modelado base.**
