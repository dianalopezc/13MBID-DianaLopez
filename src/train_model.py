"""
Script para entrenar un modelo de clasificación utilizando la técnica con
que fuera seleccionada durante la experimentación.
"""

# Importaciones generales
import argparse
import json
import joblib
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

# Importaciones para el preprocesamiento y modelado
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample

# Importaciones para la evaluación - experimentación
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from mlflow.models.signature import infer_signature
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt


def load_data(path):
    """Función para cargar los datos desde un archivo CSV."""

    df = pd.read_csv(path)
    X = df.drop('y', axis=1)
    y = df['y']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def create_preprocessor(X_train):
    """Crea el preprocesador y convierte columnas enteras a float."""

    # Identificar columnas numéricas y categóricas
    X_train = X_train.copy()

    # Convertir columnas enteras a float
    int_columns = X_train.select_dtypes(include='int').columns
    for col in int_columns:
        X_train[col] = X_train[col].astype(float)

    numerical_columns = X_train.select_dtypes(exclude='object').columns
    categorical_columns = X_train.select_dtypes(include='object').columns

    # Pipeline para valores numéricos
    num_pipeline = Pipeline(steps=[
        ('RobustScaler', RobustScaler())
    ])

    # Pipeline para valores categóricos
    cat_pipeline = Pipeline(steps=[
        ('OneHotEncoder', OneHotEncoder(drop='first', sparse_output=False))
    ])

    # Configuración del preprocesador completo
    preprocessor_full = ColumnTransformer([
        ('num_pipeline', num_pipeline, numerical_columns),
        ('cat_pipeline', cat_pipeline, categorical_columns)
    ]).set_output(transform='pandas')

    return preprocessor_full, X_train  # FIX: retornaba tupla pero se desempaquetaba mal


def balance_data(X, y, random_state=42):
    """Balancea las clases usando undersampling de la clase mayoritaria."""

    balanced_data = pd.concat([X, y], axis=1)
    target_col = y.name

    # Separar clases
    majority_class = y.value_counts().idxmax()
    minority_class = y.value_counts().idxmin()

    df_majority = balanced_data[balanced_data[target_col] == majority_class]
    df_minority = balanced_data[balanced_data[target_col] == minority_class]

    # Undersampling de la clase mayoritaria
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=random_state
    )

    balanced_df = pd.concat([df_majority_downsampled, df_minority])

    X_resampled = balanced_df.drop(target_col, axis=1)
    y_resampled = balanced_df[target_col]

    return X_resampled, y_resampled


def train_model(data_path, model_output_path, preprocessor_output_path, metrics_output_path):
    """Método principal para entrenar el modelo de clasificación."""

    # Configuración de MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Proyecto 13MBID-ABR2526 - Producción")

    with mlflow.start_run(run_name="DecisionTree Production"):

        print("Cargando datos...")
        X_train, X_test, y_train, y_test = load_data(data_path)

        print("Creando preprocesador...")
        preprocessor, X_train = create_preprocessor(X_train)  # FIX: desempaquetar tupla correctamente

        # Convertir columnas enteras en X_test también
        X_test = X_test.copy()
        int_columns = X_test.select_dtypes(include=['int64', 'int32']).columns
        for col in int_columns:
            X_test[col] = X_test[col].astype('float64')

        print("Preprocesando datos...")
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep = preprocessor.transform(X_test)

        print("Balanceando datos...")
        X_train_balanced, y_train_balanced = balance_data(X_train_prep, y_train)

        print(f"Tamaño original: {len(X_train_prep)}")
        print(f"Tamaño balanceado: {len(X_train_balanced)}")
        print(f"Distribución: {y_train_balanced.value_counts().to_dict()}")

        print("\nEntrenando modelo Decision Tree...")
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_balanced, y_train_balanced)

        print("Evaluando modelo...")
        y_pred = model.predict(X_test_prep)

        # Crear pipeline completo
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Signatures
        pipeline_signature = infer_signature(X_train, y_pred)
        preprocessor_signature = infer_signature(X_train, X_train_prep)
        model_signature = infer_signature(X_train_prep, y_pred)

        # Calcular métricas
        metrics = {
            "f1_score": float(f1_score(y_test, y_pred)),
            "recall_score": float(recall_score(y_test, y_pred)),
            "precision_score": float(precision_score(y_test, y_pred)),
            "accuracy_score": float(accuracy_score(y_test, y_pred)),
        }

        # Registrar parámetros
        mlflow.log_params({
            "model_type": "DecisionTreeClassifier",
            "criterion": model.criterion,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "balancing_method": "undersampling",
            "train_samples": len(X_train_balanced),
            "test_samples": len(X_test),
            "random_state": 42
        })

        # Registrar métricas
        mlflow.log_metrics(metrics)

        # Registrar matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"]).plot(ax=ax)  # FIX: agregar .plot(ax=ax)
        plt.title("Confusion Matrix - Production Model")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        # Registrar pipeline completo
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            signature=pipeline_signature,
        )

        # Registrar preprocesador
        mlflow.sklearn.log_model(
            sk_model=preprocessor,
            artifact_path="preprocessor",
            signature=preprocessor_signature,
        )

        # Registrar modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="classifier",
            signature=model_signature,
        )

        # Guardar modelos localmente
        print("\nGuardando modelos...")
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_output_path)
        joblib.dump(preprocessor, preprocessor_output_path)

        # Guardar métricas
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return model, preprocessor, metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Entrenar modelo de producción")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/bank-processed.csv",
        help="Ruta al archivo de datos procesados"
    )

    parser.add_argument(
        "--model-output",
        type=str,
        default="models/decision_tree_model.pkl",
        help="Ruta donde guardar el modelo"
    )

    parser.add_argument(
        "--preprocessor-output",
        type=str,
        default="models/preprocessor.pkl",
        help="Ruta donde guardar el preprocesador"
    )

    parser.add_argument(
        "--metrics-output",
        type=str,
        default="metrics/model_metrics.json",
        help="Ruta donde guardar las métricas"
    )

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        model_output_path=args.model_output,
        preprocessor_output_path=args.preprocessor_output,
        metrics_output_path=args.metrics_output
    )