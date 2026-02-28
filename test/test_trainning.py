import json
from pathlib import Path
import sys
import pytest
from train_model import train_model


def test_training_metrics_regression(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    baseline_path = project_root / "metrics" / "model_metrics.json"

    if not baseline_path.exists():
        pytest.skip(f"No se encontró la baseline en {baseline_path}. Ejecuta primero el entrenamiento baseline.")

    # Cargar métricas baseline
    with open(baseline_path) as f:
        baseline = json.load(f)

    # Hacer importable src y cargar la función de entrenamiento
    sys.path.insert(0, str(project_root / "src"))

    # Ejecutar entrenamiento con rutas aisladas (no sobrescribir baseline)
    data_path = project_root / "data" / "processed" / "bank-processed.csv"
    model_out = tmp_path / "model.pkl"
    prep_out = tmp_path / "preprocessor.pkl"
    metrics_out = tmp_path / "train_metrics.json"

    metrics = train_model(
        data_path=str(data_path),
        model_output_path=str(model_out),
        preprocessor_output_path=str(prep_out),
        metrics_output_path=str(metrics_out),
    )

    # Comparar claves y valores con tolerancia pequeña
    assert set(metrics.keys()) == set(baseline.keys())
    atol = 1e-9

    for k in baseline.keys():
        assert metrics[k] == pytest.approx(baseline[k], rel=0, abs=atol), \
            f"Métrica {k} cambió: baseline={baseline[k]} nueva={metrics[k]}"