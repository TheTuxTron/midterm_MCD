"""
Parte 1 - Adquisición y limpieza de datos.

Analogía del proyecto
---------------------
El dataset "Credit Card Fraud Detection" (ULB / Worldline, Kaggle
`mlg-ulb/creditcardfraud`) contiene transacciones con tarjeta. Lo usamos como
un problema análogo al de transacciones interbancarias: en ambos casos hay
(1) un instante de la operación, (2) un monto, (3) atributos de comportamiento
anonimizados y (4) una etiqueta de fraude extremadamente minoritaria.
Por eso renombramos los campos al vocabulario interbancario:
    Time   -> tiempo_seg   (segundos transcurridos desde la primera operación)
    Amount -> monto
    Class  -> es_fraude    (1 = operación fraudulenta)
    V1..V28 se conservan: son componentes PCA anonimizados por el proveedor.

Pasos
-----
1. Adquisición: Kaggle (kagglehub) -> espejo público -> archivo local.
2. Limpieza documentada; cada decisión queda registrada en la tabla
   `log_limpieza` de SQLite y en `reportes/log_limpieza.txt`.
3. Carga a SQLite (`data/transacciones.db`).
"""
from __future__ import annotations

import sqlite3
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------
# 1. Adquisición
# --------------------------------------------------------------------------
def leer_datos() -> pd.DataFrame:
    """Lee el archivo de datos. Orden de preferencia:
    archivo local ya descargado -> Kaggle (kagglehub) -> espejo público."""
    if C.CSV_CRUDO.exists():
        print(f"[adquisición] Leyendo archivo local {C.CSV_CRUDO}")
        return pd.read_csv(C.CSV_CRUDO)

    C.DIR_DATOS.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        print(f"[adquisición] Descargando {C.KAGGLE_DATASET} con kagglehub…")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS, C.KAGGLE_DATASET, "creditcard.csv"
        )
    except Exception as e:
        print(f"[adquisición] Kaggle no disponible ({type(e).__name__}); "
              f"usando espejo {C.URL_ESPEJO}")
        urllib.request.urlretrieve(C.URL_ESPEJO, C.CSV_CRUDO)
        return pd.read_csv(C.CSV_CRUDO)

    df.to_csv(C.CSV_CRUDO, index=False)
    return df


# --------------------------------------------------------------------------
# 2. Limpieza
# --------------------------------------------------------------------------
class Bitacora:
    """Registra cada decisión de limpieza: qué se hizo, por qué y su impacto."""

    def __init__(self):
        self.pasos: list[dict] = []

    def registrar(self, paso: str, decision: str, justificacion: str,
                  filas_antes: int, filas_despues: int):
        self.pasos.append(dict(
            orden=len(self.pasos) + 1, paso=paso, decision=decision,
            justificacion=justificacion, filas_antes=filas_antes,
            filas_despues=filas_despues,
            filas_afectadas=filas_antes - filas_despues,
        ))
        print(f"  [{len(self.pasos):02d}] {paso}: {decision} "
              f"({filas_antes:,} -> {filas_despues:,})")

    def a_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.pasos)


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, Bitacora]:
    b = Bitacora()
    n = len(df)

    # 2.1 Esquema -------------------------------------------------------------
    faltantes = set(C.COLS_ORIGINALES) - set(df.columns)
    if faltantes:
        raise ValueError(f"El archivo no tiene las columnas esperadas: {faltantes}")
    df = df[C.COLS_ORIGINALES].copy()
    b.registrar(
        "Validación de esquema",
        "Se verifican las 31 columnas documentadas (Time, V1..V28, Amount, Class) "
        "y se descartan columnas ajenas.",
        "Garantiza que los pasos siguientes operan sobre el contrato de datos "
        "descrito en la documentación de Kaggle; falla temprano si el archivo cambia.",
        n, len(df))

    # 2.2 Tipos ---------------------------------------------------------------
    for col in ["Time", *C.COLS_PCA, "Amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    b.registrar(
        "Conversión de tipos",
        "Time, V1..V28 y Amount a float; Class a numérico (valores no "
        "convertibles pasan a NaN).",
        "El CSV puede traer texto (p.ej. '\"0\"'); forzar el tipo numérico expone "
        "valores corruptos como nulos para tratarlos en el paso siguiente.",
        len(df), len(df))

    # 2.3 Nulos ---------------------------------------------------------------
    nulos = int(df.isna().sum().sum())
    antes = len(df)
    df = df.dropna()
    b.registrar(
        "Valores nulos",
        f"Se encontraron {nulos} celdas nulas; se eliminan las filas con nulos.",
        "La etiqueta, el monto y el tiempo no pueden imputarse sin sesgar el "
        "análisis de fraude, y las V1..V28 son componentes PCA cuyo valor no se "
        "puede reconstruir. Eliminar es la opción más honesta (en este dataset "
        "el impacto es 0 filas).",
        antes, len(df))

    # 2.4 Dominio de valores ----------------------------------------------------
    antes = len(df)
    validas = (
        df["Class"].isin([0, 1])
        & (df["Amount"] >= 0)
        & df["Time"].between(0, 2 * C.SEGUNDOS_DIA)
    )
    df = df[validas].copy()
    df["Class"] = df["Class"].astype(int)
    b.registrar(
        "Reglas de dominio",
        "Se conservan solo filas con Class ∈ {0,1}, Amount ≥ 0 y "
        "0 ≤ Time ≤ 172 800 s (dos días).",
        "La documentación indica que la etiqueta es binaria, que los datos cubren "
        "dos días de septiembre de 2013 y un monto negativo no tiene sentido para "
        "una operación (las devoluciones no forman parte de este set).",
        antes, len(df))

    # 2.5 Duplicados ------------------------------------------------------------
    antes = len(df)
    dup_mask = df.duplicated(keep="first")
    dup_fraude = int(df.loc[dup_mask, "Class"].sum())
    df = df[~dup_mask].copy()
    b.registrar(
        "Duplicados exactos",
        f"Se eliminan {int(dup_mask.sum())} filas idénticas en las 31 columnas "
        f"(incluye {dup_fraude} fraudes); se conserva la primera aparición.",
        "Coincidir en tiempo, monto y 28 componentes PCA continuas es prácticamente "
        "imposible para operaciones distintas: lo tratamos como doble registro "
        "(en banca: retransmisión de un mensaje). Mantenerlas inflaría los conteos "
        "y filtraría información entre conjuntos de entrenamiento y prueba.",
        antes, len(df))

    # 2.6 Montos en cero (se marcan, no se eliminan) ---------------------------
    df["es_monto_cero"] = (df["Amount"] == 0).astype(int)
    b.registrar(
        "Montos en cero",
        f"Se marcan {int(df['es_monto_cero'].sum())} operaciones con monto 0 "
        "(columna es_monto_cero); no se eliminan.",
        "Suelen ser verificaciones de cuenta/autorizaciones; los defraudadores las "
        "usan para probar credenciales, así que contienen señal de fraude.",
        len(df), len(df))

    # 2.7 Atípicos de monto (se marcan, no se eliminan) ------------------------
    q1, q3 = df["Amount"].quantile([0.25, 0.75])
    limite = q3 + 1.5 * (q3 - q1)
    df["es_atipico_monto"] = (df["Amount"] > limite).astype(int)
    b.registrar(
        "Atípicos de monto (IQR)",
        f"Se marcan {int(df['es_atipico_monto'].sum())} montos > Q3 + 1.5·IQR "
        f"(= {limite:,.2f}) en es_atipico_monto; no se eliminan.",
        "En detección de fraude el fenómeno de interés ES una anomalía: borrar "
        "atípicos borraría justamente casos relevantes. Los montos altos son "
        "reales (la distribución es de cola larga).",
        len(df), len(df))

    # 2.8 Variables derivadas y renombrado --------------------------------------
    df = df.rename(columns=C.RENOMBRAR)
    df["dia"] = (df["tiempo_seg"] // C.SEGUNDOS_DIA).astype(int) + 1
    df["hora_dia"] = ((df["tiempo_seg"] % C.SEGUNDOS_DIA) // 3600).astype(int)
    df["monto_log"] = np.log1p(df["monto"])
    df.insert(0, "id_transaccion", np.arange(1, len(df) + 1))
    b.registrar(
        "Renombrado y variables derivadas",
        "Time→tiempo_seg, Amount→monto, Class→es_fraude; se crean id_transaccion, "
        "dia (1 o 2), hora_dia (0-23, relativa al inicio de la captura) y "
        "monto_log = ln(1+monto).",
        "El vocabulario interbancario hace el análisis legible. Time solo es útil "
        "convertido a ciclos horarios; el log de monto corrige su fuerte sesgo "
        "a la derecha para visualizarlo y modelarlo.",
        len(df), len(df))

    return df.reset_index(drop=True), b


# --------------------------------------------------------------------------
# 3. Carga a SQLite
# --------------------------------------------------------------------------
DESCRIPCIONES = {
    "id_transaccion": "Identificador secuencial asignado tras la limpieza.",
    "tiempo_seg": "Segundos entre esta operación y la primera del set (Time original).",
    "monto": "Monto de la operación en euros (Amount original). No escalado.",
    "es_fraude": "Etiqueta: 1 = fraude, 0 = legítima (Class original).",
    "es_monto_cero": "Bandera: 1 si el monto es 0 (posible verificación/prueba).",
    "es_atipico_monto": "Bandera: 1 si monto > Q3 + 1.5·IQR.",
    "dia": "Día de captura (1 o 2) derivado de tiempo_seg.",
    "hora_dia": "Hora del ciclo de 24 h desde el inicio de la captura (0-23).",
    "monto_log": "ln(1 + monto), versión des-sesgada del monto.",
    **{v: f"Componente principal {v[1:]} (PCA) de atributos confidenciales; "
          "media ≈ 0, ordenada por varianza explicada." for v in C.COLS_PCA},
}


def cargar_sqlite(df: pd.DataFrame, bitacora: Bitacora) -> None:
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(C.DB_PATH) as con:
        df.to_sql(C.TABLA_TRANSACCIONES, con, if_exists="replace", index=False,
                  chunksize=20_000)
        con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS ix_id ON "
                    f"{C.TABLA_TRANSACCIONES}(id_transaccion)")
        con.execute(f"CREATE INDEX IF NOT EXISTS ix_fraude ON "
                    f"{C.TABLA_TRANSACCIONES}(es_fraude)")
        con.execute(f"CREATE INDEX IF NOT EXISTS ix_hora ON "
                    f"{C.TABLA_TRANSACCIONES}(dia, hora_dia)")

        dicc = pd.DataFrame({
            "campo": df.columns,
            "tipo_sqlite": ["INTEGER" if pd.api.types.is_integer_dtype(t) else "REAL"
                            for t in df.dtypes],
            "descripcion": [DESCRIPCIONES.get(c, "") for c in df.columns],
        })
        dicc.to_sql(C.TABLA_DICCIONARIO, con, if_exists="replace", index=False)

        log = bitacora.a_dataframe()
        log["fecha_ejecucion"] = datetime.now().isoformat(timespec="seconds")
        log.to_sql(C.TABLA_LOG_LIMPIEZA, con, if_exists="replace", index=False)

        n_db = con.execute(f"SELECT COUNT(*) FROM {C.TABLA_TRANSACCIONES}").fetchone()[0]
    assert n_db == len(df), "El conteo en SQLite no coincide con el DataFrame"
    print(f"[sqlite] {n_db:,} filas en {C.DB_PATH.name}:{C.TABLA_TRANSACCIONES}")


def escribir_log_texto(b: Bitacora, n_inicial: int, n_final: int) -> None:
    C.DIR_REPORTES.mkdir(parents=True, exist_ok=True)
    lineas = ["BITÁCORA DE LIMPIEZA DE DATOS", "=" * 60,
              f"Filas iniciales: {n_inicial:,}", f"Filas finales:   {n_final:,}",
              f"Filas eliminadas: {n_inicial - n_final:,}", ""]
    for p in b.pasos:
        lineas += [f"{p['orden']:02d}. {p['paso']}",
                   f"    Decisión:      {p['decision']}",
                   f"    Justificación: {p['justificacion']}",
                   f"    Filas: {p['filas_antes']:,} -> {p['filas_despues']:,} "
                   f"(afectadas: {p['filas_afectadas']:,})", ""]
    (C.DIR_REPORTES / "log_limpieza.txt").write_text("\n".join(lineas), encoding="utf-8")


def main() -> pd.DataFrame:
    print("=== PARTE 1: Adquisición y limpieza ===")
    crudo = leer_datos()
    print(f"[adquisición] {crudo.shape[0]:,} filas × {crudo.shape[1]} columnas")
    print("[limpieza]")
    limpio, bitacora = limpiar(crudo)
    cargar_sqlite(limpio, bitacora)
    escribir_log_texto(bitacora, len(crudo), len(limpio))
    return limpio


if __name__ == "__main__":
    main()
