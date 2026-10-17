"""
Modelado - Parte 0: Preparación.

La limpieza y el EDA completos están en parte1/2/3. Aquí se carga la tabla limpia
en pandas, se verifica que está lista para modelar y se completa el EDA con la
vista temporal que necesita el modelado (tendencia en el tiempo, picos y
dónde cae el corte entrenamiento/prueba).

Equivalencias con el enunciado de Online Retail II:
    pedidos cancelados / sin cliente -> duplicados, nulos y fuera de dominio (parte1)
    cantidad y precio de productos   -> monto de la operación
    productos más vendidos           -> rangos de monto más frecuentes y más defraudados
    patrones estacionales            -> ciclo horario y picos de actividad
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from ml_comun import cargar_transacciones, dividir_temporal  # noqa: E402
from parte3_visualizacion import estilo, guardar  # noqa: E402

RANGOS_MONTO = [-0.01, 0, 1, 10, 50, 100, 250, 500, 1000, 5000, np.inf]


def verificar(df: pd.DataFrame) -> list[str]:
    checks = {
        "Sin valores nulos": df.isna().sum().sum() == 0,
        "Sin duplicados en las variables de la transacción":
            not df.drop(columns="id_transaccion").duplicated().any(),
        "Etiqueta binaria (0/1)": set(df["es_fraude"].unique()) <= {0, 1},
        "Montos no negativos": (df["monto"] >= 0).all(),
        "Orden cronológico": df["tiempo_seg"].is_monotonic_increasing,
    }
    for nombre, ok in checks.items():
        if not ok:
            raise ValueError(f"Los datos no están listos para modelar: {nombre}")
    return [f"[OK] {n}" for n in checks]


def tabla_rangos_monto(df: pd.DataFrame) -> pd.DataFrame:
    rango = pd.cut(df["monto"], RANGOS_MONTO)
    t = df.groupby(rango, observed=True)["es_fraude"].agg(operaciones="size", fraudes="sum")
    t["tasa_fraude_por_mil"] = (t["fraudes"] / t["operaciones"] * 1000).round(2)
    t["pct_del_fraude"] = (t["fraudes"] / t["fraudes"].sum() * 100).round(1)
    t.index = [("0 (cero)" if iv.right == 0 else f"({iv.left:g}, {iv.right:g}]")
               for iv in t.index]
    return t


def vis_tendencia(df: pd.DataFrame, corte_seg: float) -> dict:
    hora_abs = (df["tiempo_seg"] // 3600).astype(int)
    por_hora = df.groupby(hora_abs)["es_fraude"].agg(operaciones="size", fraudes="sum")
    horas = por_hora.index.to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"hspace": 0.18})
    ax1.bar(horas, por_hora["operaciones"], width=0.85, color=C.COLOR_LEGITIMA)
    ax1.set_ylabel("Operaciones por hora")
    ax1.set_title("(a) Volumen de operaciones: ciclo día/noche", loc="left", fontsize=11)
    ax2.bar(horas, por_hora["fraudes"], width=0.85, color=C.COLOR_FRAUDE)
    ax2.set_ylabel("Fraudes por hora")
    ax2.set_title("(b) Fraudes: no siguen el ciclo; llegan en ráfagas",
                  loc="left", fontsize=11)
    ax2.set_xlabel("Horas desde el inicio de la captura")
    ax2.set_xticks(range(0, 49, 6))

    corte_h = corte_seg / 3600
    for ax in (ax1, ax2):
        ax.axvline(corte_h, color=C.COLOR_TEXTO_2, lw=1, ls=(0, (4, 3)))
        ax.axvspan(corte_h, 48, color=C.COLOR_GRID, alpha=0.35, lw=0, zorder=0)
    ax1.text(corte_h - 0.4, ax1.get_ylim()[1] * 0.93, "◀ entrenamiento (pasado)",
             ha="right", fontsize=9, color=C.COLOR_TEXTO_2)
    ax1.text(corte_h + 0.4, ax1.get_ylim()[1] * 0.93, "prueba (futuro) ▶",
             ha="left", fontsize=9, color=C.COLOR_TEXTO_2)
    pico_vol, pico_fra = int(por_hora["operaciones"].idxmax()), int(por_hora["fraudes"].idxmax())
    ax2.annotate(f"{int(por_hora['fraudes'].max())} fraudes", (pico_fra, por_hora["fraudes"].max()),
                 xytext=(pico_fra + 2, por_hora["fraudes"].max() * 0.95), fontsize=9,
                 arrowprops=dict(arrowstyle="-", color=C.COLOR_TEXTO_2))
    fig.suptitle("Tendencia temporal de las 48 h y partición entrenamiento / prueba",
                 x=0.07, ha="left", fontsize=14, fontweight="bold")
    guardar(fig, "ml_00_tendencia_temporal.png")
    return {"pico_volumen_h": pico_vol, "max_volumen": int(por_hora["operaciones"].max()),
            "pico_fraude_h": pico_fra, "max_fraudes": int(por_hora["fraudes"].max()),
            "min_volumen_h": int(por_hora["operaciones"].idxmin()),
            "min_volumen": int(por_hora["operaciones"].min())}


def main() -> None:
    print("=== MODELADO 0: Preparación ===")
    estilo()
    C.DIR_REPORTES_ML.mkdir(parents=True, exist_ok=True)
    df = cargar_transacciones()
    checks = verificar(df)
    entreno, prueba = dividir_temporal(df)
    rangos = tabla_rangos_monto(df)
    picos = vis_tendencia(df, prueba["tiempo_seg"].iloc[0])

    def resumen(d):
        return (f"{len(d):,} operaciones, {int(d.es_fraude.sum())} fraudes "
                f"({d.es_fraude.mean() * 100:.3f}%), horas {d.tiempo_seg.min() / 3600:.1f}"
                f"-{d.tiempo_seg.max() / 3600:.1f}")

    texto = f"""MODELADO - PARTE 0: PREPARACIÓN
{'=' * 70}

1. Carga en pandas
------------------
Tabla `{C.TABLA_TRANSACCIONES}` de {C.DB_PATH.name} -> DataFrame de {df.shape[0]:,} filas
× {df.shape[1]} columnas, ordenado cronológicamente por tiempo_seg.

2. Verificación de limpieza (detalle en reportes/log_limpieza.txt)
------------------------------------------------------------------
{chr(10).join(checks)}

Equivalencias con el enunciado original (Online Retail II):
* "Transacciones sin ID de cliente / pedidos canceladas" -> aquí no hay cliente
  ni cancelaciones; los registros irrelevantes son los duplicados exactos
  (1,081 eliminados) y los fuera de dominio (0).
* No hay valores faltantes que imputar.

3. Variables clave
------------------
* monto: mediana {df.monto.median():.2f} EUR, media {df.monto.mean():.2f} EUR, máximo
  {df.monto.max():,.2f} EUR (cola larga -> se modela como log(1+monto)).
* hora: ciclo de 24 h derivado de tiempo_seg.
* V1..V28: atributos de comportamiento anonimizados (PCA).

4. "Productos más vendidos" -> rangos de monto
---------------------------------------------
{rangos.to_string()}

La mayoría de las operaciones está entre 1 y 50 EUR. El fraude se concentra
en montos muy bajos (0-1 EUR: {rangos.iloc[:2]['pct_del_fraude'].sum():.0f}% de los fraudes, típicas
pruebas de credenciales) y su tasa vuelve a subir entre 250 y 1,000 EUR.

5. Tendencia temporal y períodos pico (ml_00_tendencia_temporal.png)
--------------------------------------------------------------------
* Pico de volumen: hora {picos['pico_volumen_h']} ({picos['max_volumen']:,} operaciones).
* Valle de volumen: hora {picos['min_volumen_h']} ({picos['min_volumen']:,} operaciones).
* Pico de fraude: hora {picos['pico_fraude_h']} ({picos['max_fraudes']} fraudes, un ataque concentrado).
El "patrón estacional" de este set es diario: el volumen legítimo sigue el
ciclo humano día/noche y el fraude no: aparece en ráfagas, también cuando casi
no hay actividad legítima (madrugadas), donde su TASA es máxima.

6. Partición para el modelado
-----------------------------
Partición TEMPORAL (no aleatoria): se entrena con el 75% más antiguo y se
prueba con el 25% más reciente, como pasaría al poner el modelo en producción.
* Entrenamiento: {resumen(entreno)}
* Prueba:        {resumen(prueba)}
"""
    (C.DIR_REPORTES_ML / "parte0_preparacion.txt").write_text(texto, encoding="utf-8")
    print(f"[reportes] {(C.DIR_REPORTES_ML / 'parte0_preparacion.txt').relative_to(C.RAIZ)}")


if __name__ == "__main__":
    main()
