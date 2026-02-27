import pandas as pd
import numpy as np

INPUT_CSV = "data/raw/bank-additional-full.csv"
OUTPUT_CSV = "data/processed/bank-processed.csv"


def preprocess_data(input_path=INPUT_CSV, output_path=OUTPUT_CSV):

    # Load the library
    df = pd.read_csv(input_path, sep=";")

    # Adaptar nombres de columnas
    df.columns = df.columns.str.replace(".", "_")

    # Transformar los valores 'unknown' en NaN
    df.replace("unknown", np.nan, inplace=True)

    # Se elimina la columna 'default' ya que tiene muchos valores desconocidos
    df.drop(columns=["default"], inplace=True)

    # Se eliminan las filas que tienen valores nulos
    df.dropna(inplace=True)

    # Se hace un filtro para eliminar las filas duplicadas
    df.drop_duplicates(inplace=True)

    # Mapear la columna objetivo a valores binarios
    map = {"no": 0, "yes": 1}
    df["y"] = df["y"].map(map)

    # Guardar el dataset procesado
    df.to_csv(output_path, index=False)

    return df.shape

if __name__ == "__main__":
    dimensiones = preprocess_data()

    with open("docs/transformations.txt", "w") as f:
        f.write("Transformaciones realizadas:\n")
        f.write("- Se reemplazaron los valores 'unknown' por NaN\n")
        f.write("- Se eliminaron las filas con valores nulos\n")
        f.write("- Se eliminaron las filas duplicadas\n")
        f.write("- Se eliminó la columna 'default' debido a la alta cantidad de valores desconocidos\n")
        f.write(f"- Cantidad de filas finales: {dimensiones[0]}\n")
        f.write(f"- Cantidad de columnas finales: {dimensiones[1]}\n")