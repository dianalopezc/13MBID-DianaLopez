# Importación de librerías y supresión de advertencias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualizar_datos(fuente: str = "data/raw/bank-additional-full.csv",
                     salida: str = "docs/figures/"):
    """genera una serie de grafcios sobre los datos"""

    # creal el diccionario de salidad si no existe
    Path(salida).mkdir(parents=True, exist_ok=True)

    # Leer los datos
    df = pd.read_csv(fuente, sep=';')

    # Grafico 1: distribución de la variable objetivo
    plt.figure(figsize=(8,6))
    sns.countplot(x="y", data=df)
    plt.title("Distribución de la variable objetivo (suscripción al depósito)")
    plt.xlabel("¿Suscribió un depósito a plazo?")
    plt.ylabel("Cantidad de clientes")
    plt.savefig(f"{salida}/distribucion_target.png")
    plt.close()

    # Grafico 2: distribución del nivel educativo
    plt.figure(figsize=(8, 4))
    col = "education"
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(f"{salida}/distribucion_educacion.png")
    plt.close()


# Grafico 3: distribución del dia de la semana
    plt.figure(figsize=(8, 4))
    col = "day_of_week"
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(f"{salida}/distribucion_dia_semana.png")
    plt.close()


# Grafico 4: correlacion de variables
    num_df = df.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones')
    plt.savefig(f"{salida}/correlacion.png")
    plt.close()


if __name__ == "__main__":
    visualizar_datos()



