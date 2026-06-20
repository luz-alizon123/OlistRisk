"""
utils/plots.py
===============
Funciones de visualización — Plotly para todo
el dashboard, estilo oscuro consistente.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

TEMA = dict(
    plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
    font=dict(color="#d1d5db", family="DM Sans"),
)
GRID = dict(gridcolor="#1e2130")

COLOR_NIVEL = {
    "Bajo": "#1D9E75", "Medio": "#EF9F27",
    "Alto": "#D85A30", "Crítico": "#E24B4A",
}


def grafico_semaforo(score: float, nivel: str) -> go.Figure:
    """Gauge tipo semáforo para el score de riesgo 0-100."""
    color = COLOR_NIVEL.get(nivel, "#888780")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
            "bar": {"color": color},
            "bgcolor": "#1e2130",
            "steps": [
                {"range": [0, 30],  "color": "#0f3d2e"},
                {"range": [30, 60], "color": "#3d2f0a"},
                {"range": [60, 80], "color": "#3d1f0a"},
                {"range": [80, 100],"color": "#3d0a0a"},
            ],
        },
    ))
    fig.update_layout(**TEMA, height=260, margin=dict(t=20, b=10))
    return fig


def grafico_barras_perfil(conteos: dict) -> go.Figure:
    """Barras de pedidos por perfil de riesgo."""
    niveles = list(conteos.keys())
    valores = list(conteos.values())
    colores = [COLOR_NIVEL.get(n, "#888780") for n in niveles]
    fig = go.Figure(go.Bar(
        x=niveles, y=valores, marker_color=colores,
        text=valores, textposition="outside",
    ))
    fig.update_layout(**TEMA, yaxis=GRID, xaxis=GRID, height=320,
                       margin=dict(t=20, b=20), showlegend=False)
    return fig


def grafico_evolucion_metricas(df_bt: pd.DataFrame) -> go.Figure:
    """Evolución de AUC, Recall, F1 por split temporal."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df_bt["split"], y=df_bt["Tasa_real_%"],
        name="Tasa real %", marker_color="#1e2d40", opacity=0.85,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df_bt["split"], y=df_bt["AUC-ROC"],
        name="AUC-ROC", mode="lines+markers",
        line=dict(color="#00d4aa", width=3), marker=dict(size=10),
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=df_bt["split"], y=df_bt["Recall"],
        name="Recall", mode="lines+markers",
        line=dict(color="#f59e0b", width=2, dash="dot"), marker=dict(size=8),
    ), secondary_y=True)
    fig.update_layout(**TEMA, height=380, margin=dict(t=20,b=20),
                       legend=dict(bgcolor="#0f1117"))
    fig.update_yaxes(title_text="Tasa real %", secondary_y=False, **GRID)
    fig.update_yaxes(title_text="AUC-ROC / Recall", range=[0,1.05],
                      secondary_y=True, **GRID)
    return fig


def grafico_ranking(df: pd.DataFrame, col_label: str, col_valor: str,
                     titulo: str, color: str = "#D85A30", top_n: int = 10) -> go.Figure:
    """Ranking horizontal genérico (top estados, top categorías)."""
    d = df.nlargest(top_n, col_valor).sort_values(col_valor)
    fig = go.Figure(go.Bar(
        x=d[col_valor], y=d[col_label], orientation="h",
        marker_color=color,
        text=[f"{v:.1f}%" if v < 100 else f"{v:.0f}" for v in d[col_valor]],
        textposition="outside",
    ))
    fig.update_layout(**TEMA, xaxis=GRID, yaxis=GRID, height=380,
                       margin=dict(t=30,b=10), title=titulo)
    return fig


def grafico_barras_drift(df_psi: pd.DataFrame) -> go.Figure:
    """Barras de PSI por variable y split, coloreadas por estado."""
    fig = go.Figure()
    for split in df_psi["Split"].unique():
        d = df_psi[df_psi["Split"] == split]
        fig.add_trace(go.Bar(
            x=d["Variable"], y=d["PSI"], name=split,
            marker_color=d["Color"],
        ))
    fig.add_hline(y=0.15, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Umbral 0.15")
    fig.add_hline(y=0.25, line_dash="dash", line_color="#ef4444",
                  annotation_text="Umbral 0.25")
    fig.update_layout(**TEMA, xaxis=GRID, yaxis=GRID, height=380,
                       margin=dict(t=20,b=60), barmode="group",
                       legend=dict(bgcolor="#0f1117"))
    return fig
