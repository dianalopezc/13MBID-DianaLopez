import pandas as pd
from pandera.pandas import DataFrameSchema, Column
import pytest


@pytest.fixture
def datos_banco():
    """Fixture para cargar los datos del banco desde un archivo csv"""
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")
    return df

def test_esquema(datos_banco):
    """ Test de esquema para el Dataframe de datos_banco"""
    esquema = DataFrameSchema({
        "age": Column(int, nullable=False),
        "job": Column(str, nullable=False),
        "marital": Column(str, nullable=False),
        "education": Column(str, nullable=False),
        "default": Column(str, nullable=True),
        "housing": Column(str, nullable=False),
        "loan": Column(str, nullable=False),
        "contact": Column(str, nullable=False),
        "month": Column(str, nullable=False),
        "day_of_week": Column(str, nullable=False),
        "duration": Column(int, nullable=False),
        "campaign": Column(int, nullable=False),
        "pdays": Column(int, nullable=False),
        "previous": Column(int, nullable=False),
        "poutcome": Column(str, nullable=False),
        "emp.var.rate": Column(float, nullable=False),
        "cons.price.idx": Column(float, nullable=False),
        "cons.conf.idx": Column(float, nullable=False),
        "euribor3m": Column(float, nullable=False),
        "nr.employed": Column(float, nullable=False),
        "y": Column(str, nullable=False)
    })

    esquema.validate(datos_banco)

def test_basico(datos_banco):
    """
    Test básico para verificar que el DataFrame de datos_banco no está vacío
    y contiene las columnas esperadas.

    Args:
        datos_banco (pd.DataFrame): DataFrame que contiene los datos del banco.
    """
    df = datos_banco

    # Verificar que el DataFrame no está vacío
    assert not df.empty, "El DataFrame está vacío."
    # Verificar nulos
    assert df.isnull().sum().sum() == 0, "El DataFrame contiene valores nulos."
    # Verificar duplicados
    # assert df.duplicated().sum() == 0, "El DataFrame contiene filas duplicadas."
    # Verificar cantidad de columnas
    assert df.shape[1] == 21, f"El DataFrame debería tener 21 columnas, pero tiene {df.shape[1]}."


if __name__ == "__main__":
    try:
        test_esquema(datos_banco())
        test_basico(datos_banco())
        print("Todos los tests pasaron exitosamente.")

        with open("docs/test_results/test_results.txt", "w") as f:
            f.write("Todos los tests pasaron exitosamente.\n")

    except AssertionError as e:
        print(f"Test fallido: {e}")

        with open("docs/test_results/test_results.txt", "w") as f:
            f.write(f"Test fallido: {e}\n")