import pandas as pd

df = pd.read_parquet("master_table_dirty.parquet")

df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)

df["mes"] = (
    df["order_purchase_timestamp"]
      .dt.to_period("M")
      .astype(str)
)

#1. Ver estados con más tardíos en Febrero-Marzo 2018
tmp = df[df["mes"].isin(["2018-02", "2018-03"])]

resultado = (
    tmp.groupby("customer_state")
       .agg(
           pedidos=("order_id","count"),
           tasa_tardios=("is_late_delivery","mean")
       )
)

resultado["tasa_tardios"] = (
    resultado["tasa_tardios"] * 100
).round(1)

print(
    resultado.sort_values(
        "tasa_tardios",
        ascending=False
    ).head(20)
)

#2. Ver estados de vendedores con más retrasos
tmp = df[df["mes"].isin(["2018-02", "2018-03"])]

resultado = (
    tmp.groupby("seller_state")
       .agg(
           pedidos=("order_id","count"),
           tasa_tardios=("is_late_delivery","mean")
       )
)

resultado["tasa_tardios"] = (
    resultado["tasa_tardios"] * 100
).round(1)

print(
    resultado.sort_values(
        "tasa_tardios",
        ascending=False
    )
)

#3. Analizar si aumentó la distancia
df.groupby("mes")["distance_km"] \
  .mean() \
  .round(1)

print(
    df.groupby("mes")
      .agg(
          promesa_media=("estimated_delivery_days","mean")
      )
      .round(1)
)

#4. Analizar si aumentó la promesa de entrega
print(
    df.groupby("mes")
      .agg(
          promesa_media=("estimated_delivery_days","mean")
      )
      .round(1)
)

#5. Analizar volumen de pedidos
print(
    df.groupby("mes")
      .agg(
          pedidos=("order_id","count")
      )
)