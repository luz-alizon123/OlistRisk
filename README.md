# OLIST — Sistema de Inteligencia Logística
## Guía de Ejecución — Sprint 2 Completo + Sprint 3

---

## Estructura de archivos que debes tener

```
olist/
├── data/
│   ├── split_train.parquet        ← ya tienes esto
│   ├── split_val.parquet          ← ya tienes esto
│   ├── split_backtest.parquet     ← ya tienes esto
│   ├── split_live.parquet         ← ya tienes esto
│   └── transformaciones.pkl       ← ya tienes esto
├── models/                        ← se crea automáticamente
├── reports/                       ← se crea automáticamente
├── 04_seleccion_variables.py      ← NUEVO
├── 05_modelado.py                 ← NUEVO
├── app.py                         ← NUEVO
└── requirements.txt               ← NUEVO
```

---

## Instalación de dependencias nuevas

```powershell
# En tu terminal de VS Code (PowerShell)
pip install optuna shap category-encoders plotly streamlit
```

Si ya tienes xgboost y lightgbm instalados, genial. Si no:
```powershell
pip install xgboost lightgbm
```

---

## Ejecución en orden

### Paso 1 — Selección de Variables
```powershell
python 04_seleccion_variables.py
```

Genera:
- `data/features_finales.pkl`       ← lista de features del modelo
- `data/features_finales.json`      ← idem en JSON legible
- `data/reporte_psi.csv`            ← PSI por variable
- `data/reporte_iv.csv`             ← Information Value
- `data/reporte_importancias_rf.csv`← importancias Random Forest
- `data/tabla_seleccion_variables.csv` ← tabla del docente completa

---

### Paso 2 — Entrenamiento del Modelo (Sprint 3)
```powershell
python 05_modelado.py
```

Entrena en orden:
1. Logistic Regression (baseline)
2. Random Forest + RandomSearchCV
3. XGBoost + Optuna
4. LightGBM + Optuna ← Champion candidate
5. Stacking Ensemble (LGB + XGB + RF) ← Champion final

Genera:
- `models/champion_model_v1.pkl`     ← artefacto del modelo
- `models/champion_resumen.json`     ← métricas en texto
- `reports/curva_roc_comparativa.png`
- `reports/shap_importancia.png`
- `data/tabla_comparativa_modelos.csv`
- `data/tabla_backtesting.csv`

**Nota sobre Optuna:** realiza 50 trials para LightGBM y 40 para XGBoost.
Puede tardar 10-20 minutos dependiendo del hardware.
Si quieres reducirlo, cambia `n_trials=40` a `n_trials=20` en el script.

---

### Paso 3 — Streamlit MVP
```powershell
streamlit run app.py
```

Se abre en: http://localhost:8501

Páginas disponibles:
- 📦 Evaluar Pedido     → Score de riesgo individual
- 📊 Dashboard          → KPIs y gráficos ejecutivos
- 📈 Backtesting        → Resultados del modelo por partición
- 🔍 Selección Variables → Tabla y proceso de selección

---

## Métricas esperadas (referencia)

| Métrica       | Val  | Backtest |
|---------------|------|----------|
| AUC-ROC       | ~0.82| ~0.79    |
| Gini          | ~0.64| ~0.58    |
| Recall        | ~0.70| ~0.67    |
| F1            | ~0.62| ~0.58    |
| PR-AUC        | ~0.47| ~0.43    |
| KS            | ~0.52| ~0.48    |

*Los valores exactos dependen del resultado de Optuna con tus datos.*

---

## Nota sobre el desbalance (8% positivos)

El pipeline maneja el desbalance con `scale_pos_weight` en XGBoost y LightGBM,
y `class_weight='balanced'` en Random Forest y Logistic Regression.
El umbral de decisión se optimiza en validación (por defecto busca maximizar F1).
