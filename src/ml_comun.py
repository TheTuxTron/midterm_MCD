"""
Piezas compartidas del modelado con scikit-learn: carga desde SQLite,
ingeniería de variables, partición temporal y pipelines candidatos.

Una transacción "cruda" (tal como llegaría al detector en tiempo real) trae
sólo `tiempo_seg`, `monto` y V1..V28. Todo lo demás se deriva DENTRO del
pipeline, así el modelo guardado recibe exactamente lo mismo en
entrenamiento y en producción.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config as C

SEMILLA = 42
COLS_ENTRADA = ["tiempo_seg", *C.COLS_PCA, "monto"]
OBJETIVO = "es_fraude"
VARIABLES_DERIVADAS = ["hora_sin", "hora_cos", "es_madrugada", "monto_log", "es_monto_cero"]
NOMBRES_CLASE = {0: "Normal", 1: "Fraudulenta"}


def cargar_transacciones() -> pd.DataFrame:
    with sqlite3.connect(C.DB_PATH) as con:
        df = pd.read_sql_query(f"SELECT * FROM {C.TABLA_TRANSACCIONES}", con)
    return df.sort_values("tiempo_seg", kind="stable").reset_index(drop=True)


def derivar_variables(X: pd.DataFrame) -> pd.DataFrame:
    """tiempo_seg y monto -> variables útiles; V1..V28 pasan tal cual.

    * hora_sin / hora_cos: la hora es cíclica (23 h está junto a 0 h).
    * es_madrugada: franja 1-6 h, donde la tasa de fraude es ~5 veces mayor (EDA).
    * monto_log: corrige la cola larga del monto.
    * es_monto_cero: posibles operaciones de prueba.
    """
    hora = (X["tiempo_seg"] % C.SEGUNDOS_DIA) / 3600
    salida = X[C.COLS_PCA].copy()
    salida["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    salida["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    salida["es_madrugada"] = ((hora >= 1) & (hora < 7)).astype(float)
    salida["monto_log"] = np.log1p(X["monto"])
    salida["es_monto_cero"] = (X["monto"] == 0).astype(float)
    return salida


def _nombres_derivados(transformador, nombres_entrada):
    return np.array([*C.COLS_PCA, *VARIABLES_DERIVADAS], dtype=object)


def construir_pipeline(modelo) -> Pipeline:
    return Pipeline([
        ("variables", FunctionTransformer(derivar_variables,
                                          feature_names_out=_nombres_derivados)),
        ("escalado", StandardScaler()),
        ("modelo", modelo),
    ])


def modelos_candidatos() -> dict[str, Pipeline]:
    """Tres familias distintas; todas compensan el desbalance con pesos de clase."""
    return {
        "Regresión logística": construir_pipeline(LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=SEMILLA)),
        "Random forest": construir_pipeline(RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEMILLA)),
        "Gradient boosting (HGB)": construir_pipeline(HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, class_weight="balanced",
            random_state=SEMILLA)),
    }


def dividir_temporal(df: pd.DataFrame, frac_prueba: float = 0.25):
    """Entrena con el pasado y prueba con el futuro: así se usaría en producción.
    Una partición aleatoria mezclaría operaciones posteriores en el entrenamiento."""
    corte = int(len(df) * (1 - frac_prueba))
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()


DECISIONES = ("APROBAR", "REVISAR", "BLOQUEAR")


def decidir(puntaje, umbrales: dict) -> np.ndarray:
    """Política de negocio sobre el puntaje del modelo:
    BLOQUEAR si puntaje >= umbral de bloqueo, REVISAR (analista) si supera el
    de revisión y APROBAR en otro caso."""
    puntaje = np.atleast_1d(puntaje)
    return np.select([puntaje >= umbrales["bloquear"], puntaje >= umbrales["revisar"]],
                     ["BLOQUEAR", "REVISAR"], "APROBAR")
