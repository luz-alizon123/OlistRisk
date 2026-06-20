"""
main.py — Pipeline automatizado Olist Risk
==========================================
Ejecuta automáticamente los pasos del pipeline SOLO si sus
archivos de salida no existen. NUNCA reentrena el modelo.

Scripts que SÍ ejecuta (si faltan outputs):
  01_conexion_postgres.py     → verificación de conexión
  02_feature_engineering.py  → master_table_dirty.parquet
  03_preparar_datos.py        → split_*.parquet
  04_seleccion_variables.py   → features_finales.pkl

Scripts que NUNCA ejecuta:
  05_modelado.py     ← requiere entrenamiento
  06_tuneo.py        ← requiere entrenamiento
  07_reporte_sprint3.py ← opcional, separado

Ejecutar:
  python main.py
  python main.py --force   (fuerza re-ejecución de todo)
  python main.py --app     (lanza Streamlit al final)
"""

import subprocess
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")

# ── Lo que necesita cada script para considerarse "hecho" ────────────────────
PIPELINE = [
    {
        "nombre":    "Verificación de conexión PostgreSQL",
        "script":    "01_conexion_postgres.py",
        "outputs":   [],               # no genera archivo, siempre se verifica
        "siempre":   True,
        "critico":   True,
    },
    {
        "nombre":    "Feature Engineering",
        "script":    "02_feature_engineering.py",
        "outputs":   ["master_table_dirty.parquet"],
        "siempre":   False,
        "critico":   True,
    },
    {
        "nombre":    "Preparación de datos",
        "script":    "03_preparar_datos.py",
        "outputs":   [
            "data/split_train.parquet",
            "data/split_val.parquet",
            "data/split_backtest.parquet",
            "data/split_live.parquet",
        ],
        "siempre":   False,
        "critico":   True,
    },
    {
        "nombre":    "Selección de variables",
        "script":    "04_seleccion_variables.py",
        "outputs":   [
            "data/features_finales.pkl",
            "data/features_finales.json",
            "data/reporte_psi.csv",
        ],
        "siempre":   False,
        "critico":   True,
    },
]

# Scripts que NUNCA deben ejecutarse automáticamente
SCRIPTS_PROHIBIDOS = ["05_modelado.py", "06_tuneo.py"]


def log(msg: str, tipo: str = "info"):
    icons = {"info": "  ▶", "ok": "  ✅", "skip": "  ⏭️ ", "error": "  ❌", "warn": "  ⚠️ "}
    print(f"{icons.get(tipo, '  ')} {msg}")


def verificar_modelo():
    """Verifica que el modelo champion existe antes de continuar."""
    path = MODEL_DIR / "champion_model_v3.pkl"
    if not path.exists():
        log("No se encontró champion_model_v3.pkl en /models", "error")
        log("Este pipeline NUNCA reentrena el modelo.", "warn")
        log("Copiá manualmente el archivo champion_model_v3.pkl a /models/", "warn")
        return False
    log("champion_model_v3.pkl encontrado", "ok")
    return True


def outputs_existen(outputs: list) -> bool:
    """True si TODOS los outputs ya existen."""
    return all(Path(o).exists() for o in outputs)


def ejecutar_script(script: str) -> bool:
    """Ejecuta un script Python y retorna True si tuvo éxito."""
    if script in SCRIPTS_PROHIBIDOS:
        log(f"{script} está prohibido en el pipeline automático", "error")
        return False

    path = Path(script)
    if not path.exists():
        log(f"{script} no encontrado", "error")
        return False

    resultado = subprocess.run(
        [sys.executable, script],
        capture_output=False,
    )
    return resultado.returncode == 0


def guardar_estado(estado: dict):
    """Guarda un registro de la última ejecución del pipeline."""
    path = Path("pipeline_estado.json")
    estado["timestamp"] = str(datetime.now())
    with open(path, "w") as f:
        json.dump(estado, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Pipeline Olist Risk")
    parser.add_argument("--force", action="store_true",
                        help="Forzar re-ejecución aunque los outputs existan")
    parser.add_argument("--app",   action="store_true",
                        help="Lanzar Streamlit al final")
    parser.add_argument("--metricas", action="store_true",
                        help="Generar métricas con umbral 0.50 al final")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  OLIST RISK — PIPELINE AUTOMATIZADO")
    print("  NO reentrena modelos. Solo prepara datos.")
    print("="*60)

    # Verificar modelo
    if not verificar_modelo():
        sys.exit(1)

    estado_ejecucion = {}
    todo_ok = True

    for paso in PIPELINE:
        nombre  = paso["nombre"]
        script  = paso["script"]
        outputs = paso["outputs"]
        siempre = paso.get("siempre", False)
        critico = paso.get("critico", True)

        print(f"\n{'─'*60}")
        log(f"Paso: {nombre}")

        if not siempre and not args.force and outputs and outputs_existen(outputs):
            log(f"Ya existe — omitiendo ({', '.join(outputs)})", "skip")
            estado_ejecucion[script] = "omitido"
            continue

        log(f"Ejecutando {script}...")
        ok = ejecutar_script(script)

        if ok:
            log(f"{nombre} completado", "ok")
            estado_ejecucion[script] = "ejecutado"
        else:
            log(f"{nombre} falló", "error")
            estado_ejecucion[script] = "error"
            if critico:
                log("Paso crítico fallido — abortando pipeline", "error")
                todo_ok = False
                break

    # Métricas con umbral 0.50 (opcional)
    if args.metricas or todo_ok:
        print(f"\n{'─'*60}")
        log("Generando métricas con umbral 0.50...")
        ok_m = ejecutar_script("generar_metricas_umbral_050.py")
        if ok_m:
            log("Métricas generadas", "ok")
        else:
            log("No se pudieron generar las métricas", "warn")

    guardar_estado({"pasos": estado_ejecucion, "exitoso": todo_ok})

    print(f"\n{'='*60}")
    if todo_ok:
        print("  ✅ PIPELINE COMPLETADO")
    else:
        print("  ❌ PIPELINE CON ERRORES — revisá los logs")
    print("="*60)

    if args.app and todo_ok:
        print("\n▶ Lanzando Streamlit...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()