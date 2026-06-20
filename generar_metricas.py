"""
generar_metricas_umbral_050.py
================================
Script independiente que:
1. Carga champion_model_v3.pkl
2. Carga los 4 splits (train, val, backtest, live)
3. Recalcula métricas con umbral = 0.50
4. Guarda metrics_threshold_050.csv y .json
5. Genera reports/metricas_modelo.png

Ejecutar: python generar_metricas_umbral_050.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, recall_score,
    precision_score, accuracy_score, average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

TARGET  = "is_late_delivery"
UMBRAL  = 0.50
SEED    = 42

print("=" * 60)
print("  MÉTRICAS CON UMBRAL 0.50 — SIN REENTRENAR")
print("=" * 60)

# ── CARGAR MODELO ─────────────────────────────────────────────
print("\n▶ Cargando champion_model_v3.pkl...")
art      = joblib.load(MODEL_DIR / "champion_model_v3.pkl")
modelo   = art["modelo"]
features = art["features_finales"]

print(f"  Modelo:   {art['nombre_champion']}")
print(f"  Versión:  {art['version']}")
print(f"  Umbral:   {UMBRAL} (default, no el guardado en pkl)")
print(f"  Features: {len(features)}")

# ── CARGAR SPLITS ─────────────────────────────────────────────
print("\n▶ Cargando splits...")
splits_info = {
    "Train":    ("split_train.parquet",    "Sep 2016–Mar 2018"),
    "Val":      ("split_val.parquet",      "Abr–May 2018"),
    "Backtest": ("split_backtest.parquet", "Jun 2018"),
    "Live":     ("split_live.parquet",     "Jul 2018"),
}

def calcular_ks(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

# ── CALCULAR MÉTRICAS ─────────────────────────────────────────
print(f"\n▶ Calculando métricas con umbral={UMBRAL}...")
resultados = []

for nombre, (archivo, periodo) in splits_info.items():
    path = DATA_DIR / archivo
    if not path.exists():
        print(f"  ⚠️  {archivo} no encontrado — saltando")
        continue

    df    = pd.read_parquet(path)
    X     = df[features].fillna(0)
    y     = df[TARGET]
    proba = modelo.predict_proba(X)[:, 1]
    y_pred= (proba >= UMBRAL).astype(int)

    auc    = roc_auc_score(y, proba)
    ks     = calcular_ks(y, proba)
    pr_auc = average_precision_score(y, proba)
    f1     = f1_score(y, y_pred, zero_division=0)
    rec    = recall_score(y, y_pred, zero_division=0)
    prec   = precision_score(y, y_pred, zero_division=0)
    acc    = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()

    r = {
        "Split":         nombre,
        "Periodo":       periodo,
        "N_pedidos":     len(y),
        "Tasa_real_%":   round(y.mean()*100, 1),
        "Umbral":        UMBRAL,
        "AUC-ROC":       round(auc, 4),
        "Gini":          round(2*auc-1, 4),
        "PR-AUC":        round(pr_auc, 4),
        "KS":            round(ks, 4),
        "F1":            round(f1, 4),
        "Recall":        round(rec, 4),
        "Precision":     round(prec, 4),
        "Accuracy":      round(acc, 4),
        "TP":            int(tp), "FP": int(fp),
        "TN":            int(tn), "FN": int(fn),
        "Tasa_Deteccion":round(tp/max(int(y.sum()),1), 4),
        "y_prob":        proba,
        "y_true":        y,
    }
    resultados.append(r)

    print(f"\n  {nombre} ({periodo})")
    print(f"    Pedidos: {len(y):,} | Tasa tardíos: {y.mean()*100:.1f}%")
    print(f"    AUC-ROC: {auc:.4f} | KS: {ks:.4f} | Gini: {2*auc-1:.4f}")
    print(f"    Recall:  {rec:.4f} | Precisión: {prec:.4f} | F1: {f1:.4f}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")

# ── GUARDAR CSV Y JSON ────────────────────────────────────────
cols_guardar = [k for k in resultados[0].keys() if k not in ["y_prob","y_true"]]
df_out = pd.DataFrame([{k:v for k,v in r.items() if k in cols_guardar}
                        for r in resultados])
df_out.to_csv(DATA_DIR / "metrics_threshold_050.csv", index=False)

with open(DATA_DIR / "metrics_threshold_050.json", "w") as f:
    json.dump(df_out.to_dict(orient="records"), f, indent=2, default=str)

print(f"\n  ✅ data/metrics_threshold_050.csv")
print(f"  ✅ data/metrics_threshold_050.json")

# ── GRÁFICO COMPLETO ──────────────────────────────────────────
print("\n▶ Generando reports/metricas_modelo.png...")

fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("#0f1117")
fig.suptitle(f"Métricas del Modelo Champion — Umbral {UMBRAL}\n"
             "LogReg Tuneado · Sprint 4 MVP",
             fontsize=14, color="#d1d5db", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

splits_nombres = [r["Split"] for r in resultados]
colores_split  = {"Train":"#6366f1","Val":"#f59e0b","Backtest":"#10b981","Live":"#ef4444"}
colores_lista  = [colores_split.get(s,"#888780") for s in splits_nombres]

metricas_barra = [
    ("AUC-ROC",   [r["AUC-ROC"]   for r in resultados], gs[0,0]),
    ("Recall",    [r["Recall"]    for r in resultados], gs[0,1]),
    ("Precision", [r["Precision"] for r in resultados], gs[0,2]),
    ("F1-Score",  [r["F1"]        for r in resultados], gs[1,0]),
    ("KS",        [r["KS"]        for r in resultados], gs[1,1]),
    ("Gini",      [r["Gini"]      for r in resultados], gs[1,2]),
]

for titulo, valores, pos in metricas_barra:
    ax = fig.add_subplot(pos)
    ax.set_facecolor("#0f1117")
    bars = ax.bar(splits_nombres, valores, color=colores_lista, alpha=0.85)
    for bar, v in zip(bars, valores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, color="#d1d5db", fontweight="bold")
    ax.set_title(titulo, color="#d1d5db", fontsize=11, pad=8)
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#9ca3af")
    ax.spines["bottom"].set_color("#1e2130")
    ax.spines["left"].set_color("#1e2130")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_tick_params(labelcolor="#9ca3af")
    ax.xaxis.set_tick_params(labelcolor="#9ca3af")
    ax.grid(True, alpha=0.2, color="#1e2130", axis="y")
    ymax = max(valores) * 1.2 if max(valores) > 0 else 0.1
    ax.set_ylim([0, min(ymax, 1.05)])

# ── Curva ROC ─────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[2, 0:2])
ax_roc.set_facecolor("#0f1117")
ax_roc.tick_params(colors="#9ca3af")
for k in ["bottom","left"]:
    ax_roc.spines[k].set_color("#1e2130")
for k in ["top","right"]:
    ax_roc.spines[k].set_visible(False)

for r in resultados:
    fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
    color = colores_split.get(r["Split"],"#888780")
    lw    = 2.5 if r["Split"] == "Val" else 1.5
    ax_roc.plot(fpr, tpr,
                label=f"{r['Split']} (AUC={r['AUC-ROC']})",
                color=color, linewidth=lw)

ax_roc.plot([0,1],[0,1],"--",color="#6b7280",alpha=0.5,label="Aleatorio")
ax_roc.axhline(y=0.75, linestyle=":", color="#6366f1", alpha=0.5, label="Objetivo 0.75")
ax_roc.set_xlabel("False Positive Rate", color="#9ca3af")
ax_roc.set_ylabel("True Positive Rate", color="#9ca3af")
ax_roc.set_title("Curva ROC por Período", color="#d1d5db", fontsize=11, pad=8)
ax_roc.legend(loc="lower right", fontsize=9,
               facecolor="#1e2130", labelcolor="#d1d5db")
ax_roc.grid(True, alpha=0.2, color="#1e2130")

# ── Tabla resumen ─────────────────────────────────────────────
ax_tab = fig.add_subplot(gs[2, 2])
ax_tab.set_facecolor("#0f1117")
ax_tab.axis("off")
tab_datos = [[r["Split"], r["N_pedidos"], f"{r['Tasa_real_%']}%",
              r["AUC-ROC"], r["Recall"], r["F1"]] for r in resultados]
tabla = ax_tab.table(
    cellText=tab_datos,
    colLabels=["Split","N","Tasa","AUC","Recall","F1"],
    loc="center", cellLoc="center"
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(8)
tabla.scale(1.1, 1.8)
for (i,j), cell in tabla.get_celld().items():
    cell.set_facecolor("#1e2130" if i > 0 else "#0a0f1a")
    cell.set_edgecolor("#2d3748")
    cell.set_text_props(color="#d1d5db")
ax_tab.set_title("Resumen", color="#d1d5db", fontsize=10, pad=8)

plt.savefig(REPORT_DIR / "metricas_modelo.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  ✅ reports/metricas_modelo.png")

print("\n" + "="*60)
print("  RESUMEN FINAL")
print("="*60)
print(f"  {'Split':<10} {'AUC':>7} {'Recall':>8} {'Prec':>8} {'F1':>7}")
print(f"  {'-'*45}")
for r in resultados:
    print(f"  {r['Split']:<10} {r['AUC-ROC']:>7.4f} {r['Recall']:>8.4f} "
          f"{r['Precision']:>8.4f} {r['F1']:>7.4f}")
print("\n  ✅ Completado. No se reentrenó ningún modelo.")