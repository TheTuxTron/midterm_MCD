"""
Modelado - Parte 2: Segmentación con K-means y Mean Shift.

Enunciado original: segmentar CLIENTES. Este set no tiene identificador de
cliente ni de cuenta, así que se segmentan TRANSACCIONES en perfiles de
comportamiento. Después se mide cuánto fraude cae en cada perfil. El
agrupamiento NO usa la etiqueta: si un segmento resulta casi todo fraude, el
patrón se encontró sin supervisión.

Ambos algoritmos se ajustan sobre una muestra aleatoria (con la proporción
natural de fraude) y luego asignan las 283k operaciones con `predict`.
"""
from __future__ import annotations

import sqlite3
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth  # noqa: E402
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,  # noqa: E402
                             silhouette_score)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

import config as C  # noqa: E402
from ml_comun import COLS_ENTRADA, SEMILLA, cargar_transacciones, derivar_variables  # noqa: E402
from parte3_visualizacion import estilo, guardar  # noqa: E402

# Las 8 componentes más discriminantes del EDA + monto y hora
VARIABLES = ["V14", "V10", "V12", "V4", "V11", "V17", "V3", "V7",
             "monto_log", "hora_sin", "hora_cos"]
N_MUESTRA = 20_000
N_MUESTRA_MS = 10_000
RANGO_K = range(2, 11)
TOLERANCIA_SIL = 0.005
TAM_MIN_MS = 0.005            # clusters Mean Shift < 0.5% se agrupan en "otros"


def preparador() -> Pipeline:
    return Pipeline([
        ("variables", FunctionTransformer(derivar_variables)),
        ("seleccion", FunctionTransformer(lambda X: X[VARIABLES])),
        ("escalado", StandardScaler()),
    ])


def seleccionar_k(X: np.ndarray) -> tuple[pd.DataFrame, int]:
    filas = []
    for k in RANGO_K:
        km = KMeans(k, n_init=10, random_state=SEMILLA).fit(X)
        filas.append(dict(k=k, inercia=km.inertia_, silhouette=silhouette_score(
            X, km.labels_, sample_size=6000, random_state=SEMILLA)))
    t = pd.DataFrame(filas)
    candidatos = t[t["silhouette"] >= t["silhouette"].max() - TOLERANCIA_SIL]
    return t, int(candidatos["k"].max())


def describir(centro: pd.Series) -> str:
    rasgos = []
    for v, z in centro.drop(["hora_sin", "hora_cos"]).sort_values(key=abs, ascending=False).items():
        if abs(z) < 0.8 or len(rasgos) == 3:
            break
        nombre = "monto" if v == "monto_log" else v
        rasgos.append(f"{nombre} {'muy ' if abs(z) > 2 else ''}{'alto' if z > 0 else 'bajo'}")
    hora = (np.degrees(np.arctan2(centro["hora_sin"], centro["hora_cos"])) % 360) / 15
    if np.hypot(centro["hora_sin"], centro["hora_cos"]) > 0.6:
        rasgos.append(f"hora ≈ {hora:.0f} h")
    return ", ".join(rasgos) or "perfil promedio"


def perfilar(df: pd.DataFrame, etiquetas: np.ndarray, Z: np.ndarray, prefijo: str,
             agrupar_menores: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    etiquetas = etiquetas.copy()
    if agrupar_menores:
        tam = pd.Series(etiquetas).value_counts(normalize=True)
        menores = tam[tam < TAM_MIN_MS].index
        etiquetas = np.where(np.isin(etiquetas, menores), -1, etiquetas)
    base = df["es_fraude"].mean()
    filas, centros = [], []
    for c in sorted(set(etiquetas), key=lambda c: (c == -1, c)):
        m = etiquetas == c
        sub = df[m]
        centro = pd.Series(Z[m].mean(axis=0), index=VARIABLES)
        filas.append(dict(
            segmento=f"{prefijo}{c}" if c != -1 else f"{prefijo}-otros",
            operaciones=int(m.sum()), pct_operaciones=m.mean() * 100,
            fraudes=int(sub["es_fraude"].sum()),
            tasa_fraude_por_mil=sub["es_fraude"].mean() * 1000,
            veces_tasa_base=sub["es_fraude"].mean() / base,
            pct_del_fraude=sub["es_fraude"].sum() / df["es_fraude"].sum() * 100,
            monto_mediano=sub["monto"].median(),
            pct_madrugada=sub["hora_dia"].between(1, 6).mean() * 100,
            descripcion=describir(centro)))
        centros.append(centro.to_numpy())
    return pd.DataFrame(filas), np.array(centros)


def concentracion(perfil: pd.DataFrame, veces: float = 10) -> float:
    """% de los fraudes que caen en segmentos con tasa >= `veces` la tasa base."""
    return float(perfil.loc[perfil["veces_tasa_base"] >= veces, "pct_del_fraude"].sum())


def vis_seleccion_k(t: pd.DataFrame, k: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"wspace": 0.25})
    for ax, col, titulo in ((ax1, "inercia", "(a) Método del codo: inercia"),
                            (ax2, "silhouette", "(b) Coeficiente de silhouette")):
        ax.plot(t["k"], t[col], color=C.COLOR_LEGITIMA, lw=2, marker="o", ms=6,
                markeredgecolor=C.COLOR_SUPERFICIE, markeredgewidth=1.5)
        fila = t[t["k"] == k].iloc[0]
        ax.plot([k], [fila[col]], marker="o", ms=11, color=C.COLOR_FRAUDE,
                markeredgecolor=C.COLOR_SUPERFICIE, markeredgewidth=2, zorder=4)
        ax.annotate(f"k = {k}", (k, fila[col]), xytext=(10, 6 if col == "inercia" else -16), textcoords="offset points",
                    fontsize=9.5, color=C.COLOR_TEXTO)
        ax.set_xticks(list(RANGO_K))
        ax.set_xlabel("Número de segmentos (k)")
        ax.set_title(titulo, loc="left", fontsize=11)
    ax2.axhspan(t["silhouette"].max() - TOLERANCIA_SIL, t["silhouette"].max(),
                color=C.COLOR_GRID, alpha=0.6, lw=0)
    ax2.text(RANGO_K[-1], t["silhouette"].max() - TOLERANCIA_SIL / 2,
             f"a menos de {TOLERANCIA_SIL} del máximo", ha="right", va="center",
             fontsize=8.5, color=C.COLOR_TEXTO_2)
    fig.suptitle("K-means: selección del número de segmentos", x=0.07, ha="left",
                 fontsize=14, fontweight="bold", y=1.02)
    guardar(fig, "ml_02_kmeans_seleccion_k.png")


def vis_segmentos(perfiles: list[tuple[str, pd.DataFrame, np.ndarray]], base: float) -> None:
    filas_max = max(len(p) for _, p, _ in perfiles)
    fig, ejes = plt.subplots(2, 2, figsize=(15, 4 + 0.55 * filas_max * 2),
                             gridspec_kw={"width_ratios": [2.3, 1], "hspace": 0.45,
                                          "wspace": 0.05})
    cmap = LinearSegmentedColormap.from_list("div", C.DIVERGENTE)
    norm = TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4)
    for fila, (titulo, perfil, centros) in enumerate(perfiles):
        ax, axb = ejes[fila]
        im = ax.imshow(np.clip(centros, -4, 4), aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(VARIABLES)),
                      [v.replace("_", " ") for v in VARIABLES], rotation=35, ha="right")
        ax.set_yticks(range(len(perfil)),
                      [f"{s}  ({p:.1f}%)" for s, p in zip(perfil["segmento"],
                                                          perfil["pct_operaciones"])])
        ax.grid(False)
        for (i, j), v in np.ndenumerate(centros):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color=C.COLOR_SUPERFICIE if abs(v) > 2.5 else C.COLOR_TEXTO)
        ax.set_title(f"{titulo}: perfil promedio (z) de cada segmento", loc="left",
                     fontsize=11)

        tasa = perfil["tasa_fraude_por_mil"].clip(lower=0.05).to_numpy()
        y = np.arange(len(perfil))
        colores = [C.COLOR_FRAUDE if v >= 10 else C.COLOR_NEUTRO
                   for v in perfil["veces_tasa_base"]]
        axb.barh(y, tasa, color=colores, height=0.6)
        axb.set_xscale("log")
        axb.set_xlim(0.05, 30000)
        axb.set_ylim(len(perfil) - 0.5, -0.5)
        axb.set_yticks([])
        axb.axvline(base * 1000, color=C.COLOR_TEXTO_2, lw=0.8, ls=":")
        axb.text(base * 1000 * 1.15, len(perfil) - 0.55, f"base {base * 1000:.2f}‰",
                 fontsize=8, color=C.COLOR_TEXTO_2, va="bottom")
        for yi, v, f in zip(y, perfil["tasa_fraude_por_mil"], perfil["fraudes"]):
            axb.text(max(v, 0.05) * 1.3, yi, f"{v:.1f}‰ · {f} fraudes", va="center",
                     fontsize=8.5, color=C.COLOR_TEXTO)
        axb.set_title("Tasa de fraude (‰, escala log)", loc="left", fontsize=11)
        axb.grid(axis="y", visible=False)
    cb = fig.colorbar(im, ax=ejes[:, 0], location="bottom", fraction=0.025, pad=0.12)
    cb.set_label("Centroide estandarizado (azul = bajo, rojo = alto)", color=C.COLOR_TEXTO_2)
    cb.outline.set_visible(False)
    fig.suptitle("Segmentos de transacciones: K-means vs. Mean Shift", x=0.07, ha="left",
                 fontsize=14, fontweight="bold", y=0.995)
    guardar(fig, "ml_03_segmentos_kmeans_vs_meanshift.png")


def main() -> None:
    print("=== MODELADO 2: Segmentación ===")
    estilo()
    df = cargar_transacciones()
    prep = preparador().fit(df[COLS_ENTRADA])
    Z = prep.transform(df[COLS_ENTRADA])
    rng = np.random.RandomState(SEMILLA)
    idx = rng.choice(len(df), N_MUESTRA, replace=False)
    Xm = Z[idx]

    # K-means ----------------------------------------------------------------
    tabla_k, k = seleccionar_k(Xm)
    t0 = time.perf_counter()
    km = KMeans(k, n_init=10, random_state=SEMILLA).fit(Xm)
    t_km = time.perf_counter() - t0
    et_km = km.predict(Z)
    perfil_km, centros_km = perfilar(df, et_km, Z, "K")
    print(f"  K-means: k={k}")

    # Mean Shift -------------------------------------------------------------
    bw = estimate_bandwidth(Xm, quantile=0.1, n_samples=3000, random_state=SEMILLA)
    t0 = time.perf_counter()
    ms = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1).fit(Xm[:N_MUESTRA_MS])
    t_ms = time.perf_counter() - t0
    et_ms = ms.predict(Z)
    n_ms = len(np.unique(et_ms))
    perfil_ms, centros_ms = perfilar(df, et_ms, Z, "M", agrupar_menores=True)
    print(f"  Mean Shift: bandwidth={bw:.2f}, {n_ms} clusters")

    # Comparación --------------------------------------------------------------
    sub = rng.choice(len(df), 8000, replace=False)
    comp = pd.DataFrame({
        "K-means": {
            "n_segmentos": k, "parametro": "k elegido por silhouette",
            "silhouette": silhouette_score(Z[sub], et_km[sub]),
            "davies_bouldin": davies_bouldin_score(Z[sub], et_km[sub]),
            "pct_mayor_segmento": perfil_km["pct_operaciones"].max(),
            "pct_fraude_en_segmentos_10x": concentracion(perfil_km),
            "segundos_ajuste": t_km},
        "Mean Shift": {
            "n_segmentos": n_ms, "parametro": f"bandwidth {bw:.2f} (estimado)",
            "silhouette": silhouette_score(Z[sub], et_ms[sub]),
            "davies_bouldin": davies_bouldin_score(Z[sub], et_ms[sub]),
            "pct_mayor_segmento": perfil_ms["pct_operaciones"].max(),
            "pct_fraude_en_segmentos_10x": concentracion(perfil_ms),
            "segundos_ajuste": t_ms},
    })
    ari = adjusted_rand_score(et_km, et_ms)

    with sqlite3.connect(C.DB_PATH) as con:
        pd.DataFrame({"id_transaccion": df["id_transaccion"],
                      "segmento_kmeans": et_km, "segmento_meanshift": et_ms}
                     ).to_sql("segmentos", con, if_exists="replace", index=False)

    base = df["es_fraude"].mean()
    vis_seleccion_k(tabla_k, k)
    vis_segmentos([("K-means", perfil_km, centros_km),
                   ("Mean Shift", perfil_ms, centros_ms)], base)

    top_km = perfil_km.sort_values("tasa_fraude_por_mil", ascending=False).iloc[0]
    top_ms = perfil_ms.sort_values("tasa_fraude_por_mil", ascending=False).iloc[0]
    fmt = {"pct_operaciones": "{:.2f}".format, "tasa_fraude_por_mil": "{:.2f}".format,
           "veces_tasa_base": "{:.1f}".format, "pct_del_fraude": "{:.1f}".format,
           "monto_mediano": "{:.2f}".format, "pct_madrugada": "{:.1f}".format}
    cols = ["segmento", "operaciones", "pct_operaciones", "fraudes", "tasa_fraude_por_mil",
            "veces_tasa_base", "pct_del_fraude", "monto_mediano", "pct_madrugada"]

    def tabla(p):
        return (p[cols].to_string(index=False, formatters=fmt) + "\n\nDescripción:\n" +
                "\n".join(f"  {s}: {d}" for s, d in zip(p["segmento"], p["descripcion"])))

    texto = f"""MODELADO - PARTE 2: SEGMENTACIÓN DE TRANSACCIONES
{'=' * 70}

Adaptación: el enunciado pide segmentar clientes, pero este set no tiene
identificador de cliente ni de cuenta. Se segmentan TRANSACCIONES en perfiles
de comportamiento y luego se mide el fraude en cada perfil (la etiqueta no
participa en el agrupamiento).

Variables (estandarizadas): {', '.join(VARIABLES)}
Las 8 componentes más informativas del EDA, el log del monto y la hora
codificada como ciclo (seno/coseno). Usar las 28 componentes diluye las
distancias y perjudica sobre todo a Mean Shift.

Ajuste: muestra aleatoria de {N_MUESTRA:,} operaciones (proporción natural de
fraude); luego `predict` asigna las {len(df):,} operaciones.
Las asignaciones se guardaron en la tabla SQLite `segmentos`.

1. K-means
----------
Selección de k (ml_02_kmeans_seleccion_k.png):
{tabla_k.round(3).to_string(index=False)}

El silhouette es bajo y plano ({tabla_k.silhouette.min():.3f}-{tabla_k.silhouette.max():.3f}): las transacciones NO forman
grupos naturales bien separados, sino un continuo. El codo tampoco es
marcado. Regla usada: el k más granular cuyo silhouette queda a menos de
{TOLERANCIA_SIL} del mejor -> k = {k}.

Segmentos (tasa base de fraude = {base * 1000:.2f}‰):
{tabla(perfil_km)}

2. Mean Shift
-------------
No requiere fijar k: el ancho de banda (bandwidth = {bw:.2f}, estimado con
estimate_bandwidth, cuantil 0.1) define el tamaño de los "picos" de densidad.
Encontró {n_ms} clusters. Los menores al {TAM_MIN_MS * 100:.1f}% se agrupan en "M-otros"
para la tabla:
{tabla(perfil_ms)}

3. Comparación
--------------
{comp.T.to_string()}

Coincidencia entre ambas particiones (Adjusted Rand Index): {ari:.3f}

Lectura:
* K-means divide el grueso de las operaciones en segmentos amplios que se
  diferencian sobre todo por la franja horaria, y además aísla un segmento
  diminuto: {top_km['segmento']} ({top_km['operaciones']:,} operaciones, {top_km['pct_operaciones']:.2f}%) con una tasa de fraude de
  {top_km['tasa_fraude_por_mil']:.0f}‰ ({top_km['veces_tasa_base']:.0f} veces la base), que reúne el {top_km['pct_del_fraude']:.0f}% de todos los fraudes.
  El algoritmo aisló un perfil de ataque SIN ver la etiqueta. Su perfil:
  {top_km['descripcion']} (las mismas señales que mostró el EDA).
* Mean Shift encuentra un modo dominante, la operación "normal" ({perfil_ms['pct_operaciones'].max():.0f}% del
  total, con {perfil_ms.iloc[0]['tasa_fraude_por_mil']:.2f}‰ de fraude), y {n_ms - 1} clusters pequeños en la periferia que,
  sumados ({100 - perfil_ms.iloc[0]['pct_operaciones']:.1f}% de las operaciones), concentran el {100 - perfil_ms.iloc[0]['pct_del_fraude']:.0f}% del fraude.
  Funciona como un detector de anomalías: lo raro queda en clusters chicos.
* Métricas internas: Mean Shift obtiene mejor silhouette y Davies-Bouldin,
  pero en buena parte porque mete casi todo en un solo cluster. Con datos
  tan desbalanceados, esas métricas no bastan para comparar.
* ARI = {ari:.2f}: las dos técnicas dividen el espacio de forma muy distinta.
  K-means sirve para PERFILAR (pocos segmentos, descriptibles y accionables).
  Mean Shift sirve para SEPARAR LO ATÍPICO sin fijar k, pero depende mucho del
  bandwidth y escala peor: es O(n²), por eso se ajustó con {N_MUESTRA_MS:,}
  operaciones ({t_ms:.1f} s) frente a {N_MUESTRA:,} de K-means ({t_km:.1f} s).
* Uso práctico: el segmento puede servir como regla de priorización o como
  variable adicional del modelo supervisado. No reemplaza al clasificador:
  con K-means, el {100 - concentracion(perfil_km):.0f}% de los fraudes queda en segmentos de riesgo bajo.
"""
    (C.DIR_REPORTES_ML / "parte2_segmentacion.txt").write_text(texto, encoding="utf-8")
    print(f"[reportes] {(C.DIR_REPORTES_ML / 'parte2_segmentacion.txt').relative_to(C.RAIZ)}")


if __name__ == "__main__":
    main()
