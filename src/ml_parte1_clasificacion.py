"""
Modelado - Parte 1: Clasificación de transacciones.

Enunciado original: "Clientes Normales" vs "Clientes Premium".
Adaptación: "Transacción Normal" vs "Transacción Fraudulenta".

Se comparan tres clasificadores de scikit-learn (más un clasificador trivial
como línea base) con validación cruzada estratificada sobre el conjunto de
entrenamiento. Las predicciones fuera de pliegue (OOF) se guardan para que la
Parte 3 elija los umbrales de decisión sin tocar el conjunto de prueba.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score,  # noqa: E402
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import config as C  # noqa: E402
from ml_comun import (COLS_ENTRADA, NOMBRES_CLASE, OBJETIVO, SEMILLA,  # noqa: E402
                      cargar_transacciones, construir_pipeline, dividir_temporal,
                      modelos_candidatos)
from parte3_visualizacion import estilo, guardar  # noqa: E402

N_PLIEGUES = 5
BASE = "Línea base (siempre 'Normal')"


def definir_clase(df: pd.DataFrame) -> pd.DataFrame:
    df["clase_transaccion"] = df[OBJETIVO].map(NOMBRES_CLASE)
    return df


def validacion_cruzada(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    modelos = {BASE: construir_pipeline(DummyClassifier(strategy="most_frequent")),
               **modelos_candidatos()}
    cv = StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=SEMILLA)
    filas, oof = [], {}
    for nombre, modelo in modelos.items():
        prob = np.zeros(len(y))
        for k, (i_tr, i_va) in enumerate(cv.split(X, y), start=1):
            m = clone(modelo).fit(X.iloc[i_tr], y.iloc[i_tr])
            p = m.predict_proba(X.iloc[i_va])[:, 1]
            prob[i_va] = p
            yv, pred = y.iloc[i_va], (p >= 0.5).astype(int)
            filas.append(dict(
                modelo=nombre, pliegue=k,
                auprc=average_precision_score(yv, p), roc_auc=roc_auc_score(yv, p),
                exactitud=accuracy_score(yv, pred),
                precision=precision_score(yv, pred, zero_division=0),
                recall=recall_score(yv, pred), f1=f1_score(yv, pred, zero_division=0)))
        oof[nombre] = prob
        r = pd.DataFrame(filas).query("modelo == @nombre")
        print(f"  {nombre:<32} AUPRC {r.auprc.mean():.3f} ± {r.auprc.std():.3f}")
    return pd.DataFrame(filas), oof


def vis_comparacion(res: pd.DataFrame, oof: dict, y: np.ndarray, elegido: str) -> None:
    resumen = res.groupby("modelo", sort=False)["auprc"].agg(["mean", "std"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.3})
    colores = [C.COLOR_LEGITIMA if m == elegido else C.COLOR_NEUTRO for m in resumen.index]
    pos = np.arange(len(resumen))[::-1]
    ax1.barh(pos, resumen["mean"], xerr=resumen["std"], color=colores, height=0.6,
             error_kw=dict(ecolor=C.COLOR_TEXTO_2, lw=1, capsize=3))
    ax1.set_yticks(pos, resumen.index)
    for p, v, d in zip(pos, resumen["mean"], resumen["std"].fillna(0)):
        ax1.text(v + d + 0.02, p, f"{v:.3f}", va="center", fontsize=9, color=C.COLOR_TEXTO)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel(f"AUPRC (media ± desv., {N_PLIEGUES} pliegues)")
    ax1.grid(axis="y", visible=False)
    ax1.set_title("(a) Precisión promedio por modelo", loc="left", fontsize=11)

    estilos = iter([(0, (5, 2)), (0, (1, 1.5)), (0, (3, 1, 1, 1))])
    for nombre, prob in oof.items():
        if nombre == BASE:
            continue
        prec, rec, _ = precision_recall_curve(y, prob)
        es = nombre == elegido
        ax2.plot(rec, prec, lw=2.2 if es else 1.4,
                 color=C.COLOR_LEGITIMA if es else C.COLOR_NEUTRO,
                 ls="-" if es else next(estilos), zorder=3 if es else 2,
                 label=f"{nombre} (AP {average_precision_score(y, prob):.3f})")
    ax2.axhline(y.mean(), color=C.COLOR_TEXTO_2, lw=0.8, ls=":")
    ax2.text(0.55, y.mean() + 0.02, f"azar = {y.mean():.4f}", fontsize=8.5,
             color=C.COLOR_TEXTO_2, ha="right")
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Recall (fraudes detectados / fraudes reales)")
    ax2.set_ylabel("Precisión (fraudes reales / alertas)")
    ax2.legend(loc="lower left", fontsize=9, bbox_to_anchor=(0, 0.06))
    ax2.set_title("(b) Curvas precisión-recall (predicciones fuera de pliegue)",
                  loc="left", fontsize=11)
    fig.suptitle("Clasificación Normal vs. Fraudulenta: comparación de modelos",
                 x=0.07, ha="left", fontsize=14, fontweight="bold", y=1.0)
    guardar(fig, "ml_01_comparacion_modelos.png")


def main() -> str:
    print("=== MODELADO 1: Clasificación ===")
    estilo()
    C.DIR_REPORTES_ML.mkdir(parents=True, exist_ok=True)
    C.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    df = definir_clase(cargar_transacciones())
    entreno, _ = dividir_temporal(df)
    X, y = entreno[COLS_ENTRADA], entreno[OBJETIVO]

    res, oof = validacion_cruzada(X, y)
    tabla = (res.drop(columns="pliegue").groupby("modelo", sort=False)
             .agg(["mean", "std"]))
    tabla.columns = [f"{m}_{s}" for m, s in tabla.columns]
    tabla = tabla.round(3).assign(exactitud_mean=tabla["exactitud_mean"].round(4))
    candidatos = tabla.drop(index=BASE)
    elegido = candidatos["auprc_mean"].idxmax()

    np.savez_compressed(C.OOF_PATH, y=y.to_numpy(),
                        **{f"oof_{i}": oof[n] for i, n in enumerate(oof)},
                        nombres=np.array(list(oof)), elegido=elegido)
    (C.DIR_REPORTES_ML / "seleccion_modelo.json").write_text(json.dumps(
        {"modelo_elegido": elegido, "criterio": "mayor AUPRC media en CV",
         "metricas_cv": tabla.reset_index().to_dict(orient="records")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    vis_comparacion(res, oof, y.to_numpy(), elegido)

    conteo = df["clase_transaccion"].value_counts()
    cols = ["auprc_mean", "auprc_std", "roc_auc_mean", "exactitud_mean",
            "precision_mean", "recall_mean", "f1_mean"]
    texto = f"""MODELADO - PARTE 1: CLASIFICACIÓN DE TRANSACCIONES
{'=' * 70}

1. Definición de las dos clases
-------------------------------
El enunciado original pide separar "Clientes Normales" de "Clientes Premium"
con un criterio a definir (volumen, cantidades...). En fraude transaccional la
pregunta equivalente es si una operación es NORMAL o FRAUDULENTA, y aquí la
clase no hay que inventarla: viene de la investigación posterior del banco
(contracargos, reclamos y casos confirmados), que es la columna es_fraude.

Nueva columna: clase_transaccion = "Normal" si es_fraude = 0
                                   "Fraudulenta" si es_fraude = 1
{conteo.to_string()}

¿Por qué no definirla con una regla (p. ej. "monto > X" o "de madrugada")?
Porque el EDA muestra que ninguna variable aislada separa las clases (el monto
tiene KS 0.27). Una regla simple etiquetaría mal a la mayoría. Las reglas sí
aparecen después, como POLÍTICA sobre el puntaje del modelo
(Aprobar / Revisar / Bloquear, Parte 3).

2. Protocolo de entrenamiento
-----------------------------
* Datos: el 75% más antiguo ({len(entreno):,} operaciones, {int(y.sum())} fraudes). El 25%
  más reciente queda reservado para la Parte 3 y no se usa aquí.
* Entrada del modelo: la transacción cruda (tiempo_seg, monto, V1..V28). El
  pipeline deriva hora_sin, hora_cos, es_madrugada, monto_log y es_monto_cero,
  y luego escala (StandardScaler).
* Validación cruzada estratificada de {N_PLIEGUES} pliegues, para que cada pliegue
  tenga la misma proporción de fraude.
* Desbalance: se compensa con class_weight="balanced" (pesos de clase)
  en lugar de sobremuestrear, porque así no se generan fraudes sintéticos.
* Métrica principal: AUPRC (área bajo la curva precisión-recall), la que
  recomienda la documentación del dataset para clases desbalanceadas.

3. Resultados (media de {N_PLIEGUES} pliegues; precisión/recall/F1 con umbral 0.5)
{'-' * 70}
{tabla[cols].to_string()}

4. Lectura
----------
* La línea base predice siempre "Normal" y logra {tabla.loc[BASE, 'exactitud_mean'] * 100:.2f}% de exactitud
  sin detectar un solo fraude (recall 0). Por eso la exactitud no sirve aquí.
* Modelo elegido: {elegido} (AUPRC {candidatos.loc[elegido, 'auprc_mean']:.3f}).
* La regresión logística tiene un recall alto con el umbral 0.5, pero una
  precisión muy baja: genera demasiadas alertas falsas. Los modelos de árboles
  capturan relaciones no lineales (las "islas" de fraude del mapa t-SNE).
* El umbral 0.5 es arbitrario. En la Parte 3 se eligen umbrales de negocio
  con estas mismas predicciones fuera de pliegue.

Gráfica: visualizaciones/ml_01_comparacion_modelos.png
"""
    (C.DIR_REPORTES_ML / "parte1_clasificacion.txt").write_text(texto, encoding="utf-8")
    print(f"[modelo] Elegido: {elegido}")
    return elegido


if __name__ == "__main__":
    main()
