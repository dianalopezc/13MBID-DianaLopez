import pandas as pd

def test_great_expectations():
    """
    Test para verificar que los datos cumplen con las expectativas definidas
    en un archivo de Great Expectations.

    Raises:
        AssertionError: Si alguna de las expectativas no se cumple.
    """

    # Cargar los datos
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")

    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count": 0, "total_count": 0}
    }


    def add_expectation(expectation_name, condition, message=""):
        results["statistics"]["total_count"] += 1

        if condition:
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
                "expectation": expectation_name,
                "success": True
            })
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": expectation_name,
                "success": False,
                "message": message
            })


    # Validaciones a verificar sobre los datos
    add_expectation(
        "age_range",
        df["age"].between(18, 100).all(),
        "La columna 'age' no está en el rango esperado (18-100)."
    )

    add_expectation(
        "target_values",
        df["y"].isin(["yes", "no"]).all(),
        "La columna 'y' contiene valores no válidos."
    )

    add_expectation(
        "campaign_min_value",
        (df["campaign"] >= 1).all(),
        "La columna 'campaign' contiene valores menores a 1."
    )

    add_expectation(
        "duration_positive",
        (df["duration"] > 0).all(),
        "La columna 'duration' contiene valores menores o iguales a 0."
    )