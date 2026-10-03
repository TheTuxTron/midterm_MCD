"""
Parte 3 - Visualización.

Tres técnicas avanzadas, todas guardadas como PNG en `visualizaciones/`:

1. Ridgeline (joy plot) de densidades KDE: la "firma" del fraude en las
   componentes PCA, comparando las más y las menos discriminantes.
2. Reloj del riesgo: gráficos polares de 24 h + mapa de calor día × hora
   de la tasa de fraude.
3. Mapa t-SNE: proyección no lineal en 2D de las 28 componentes + monto, que
   muestra si el fraude forma regiones separables.

Además escribe `visualizaciones/historia_visualizaciones.txt` con la
narrativa que cuentan las gráficas.
"""
from __future__ import annotations

import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

import config as C  # noqa: E402

SEMILLA = 42


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": C.COLOR_SUPERFICIE, "axes.facecolor": C.COLOR_SUPERFICIE,
        "savefig.facecolor": C.COLOR_SUPERFICIE, "font.family": "DejaVu Sans",
        "font.size": 10, "text.color": C.COLOR_TEXTO,
        "axes.labelcolor": C.COLOR_TEXTO_2, "axes.edgecolor": C.COLOR_GRID,
        "xtick.color": C.COLOR_TEXTO_2, "ytick.color": C.COLOR_TEXTO_2,
        "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C.COLOR_GRID, "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def leer() -> pd.DataFrame:
    with sqlite3.connect(C.DB_PATH) as con:
        return pd.read_sql_query(f"SELECT * FROM {C.TABLA_TRANSACCIONES}", con)


def guardar(fig, nombre: str) -> None:
    C.DIR_VIS.mkdir(parents=True, exist_ok=True)
    ruta = C.DIR_VIS / nombre
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[vis] {ruta.relative_to(C.RAIZ)}")


# --------------------------------------------------------------------------
# 1. Ridgeline de densidades
# --------------------------------------------------------------------------
def ranking_ks(df: pd.DataFrame) -> pd.Series:
    """Estadístico KS fraude vs. legítima para cada componente PCA."""
    f = df["es_fraude"] == 1
    return pd.Series({v: stats.ks_2samp(df.loc[f, v], df.loc[~f, v]).statistic
                      for v in C.COLS_PCA}).sort_values(ascending=False)


def vis_ridgeline(df: pd.DataFrame, ks: pd.Series) -> dict:
    variables = list(ks.index[:6]) + list(ks.index[-2:])
    f = df["es_fraude"] == 1
    z = {v: (df[v] - df.loc[~f, v].mean()) / df.loc[~f, v].std() for v in variables}
    x_lo = min(min(z[v][~f].quantile(.001), z[v][f].quantile(.01)) for v in variables)
    x_hi = max(max(z[v][~f].quantile(.999), z[v][f].quantile(.99)) for v in variables)
    malla = np.linspace(x_lo, x_hi, 500)

    fig, ax = plt.subplots(figsize=(10, 9.5))
    altura, hueco = 0.85, 0.6
    for i, var in enumerate(variables):
        y0 = (len(variables) - 1 - i) + (hueco if i < 6 else 0)
        for clase, color, orden in ((0, C.COLOR_LEGITIMA, 3), (1, C.COLOR_FRAUDE, 4)):
            datos = z[var][df["es_fraude"] == clase]
            datos = datos.sample(min(len(datos), 20_000), random_state=SEMILLA)
            dens = stats.gaussian_kde(datos)(malla)
            dens = dens / dens.max() * altura
            ax.fill_between(malla, y0, y0 + dens, color=color, alpha=0.30, lw=0,
                            zorder=orden)
            ax.plot(malla, y0 + dens, color=color, lw=1.6, zorder=orden + 0.1)
        ax.axhline(y0, color=C.COLOR_GRID, lw=0.8, zorder=1)
        trans = ax.get_yaxis_transform()
        ax.text(-0.02, y0 + 0.30, var, transform=trans, ha="right", fontsize=11,
                fontweight="bold", color=C.COLOR_TEXTO)
        ax.text(-0.02, y0 + 0.05, f"KS {ks[var]:.2f}", transform=trans, ha="right",
                fontsize=8.5, color=C.COLOR_TEXTO_2)
    ax.axhline(2 + hueco / 2, color=C.COLOR_TEXTO_2, lw=0.8, ls=(0, (4, 3)))
    trans = ax.get_yaxis_transform()
    ax.text(1, 2 + hueco / 2 + 0.06, "▲ más discriminantes", transform=trans,
            ha="right", fontsize=8.5, color=C.COLOR_TEXTO_2)
    ax.text(1, 2 + hueco / 2 - 0.16, "▼ menos discriminantes", transform=trans,
            ha="right", fontsize=8.5, color=C.COLOR_TEXTO_2)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.15, len(variables) + hueco)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=C.COLOR_TEXTO_2, lw=0.6, alpha=0.5, zorder=2)
    ax.set_xlabel("Valor estandarizado respecto a las operaciones legítimas "
                  "(desviaciones estándar)")
    ax.set_title("La firma del fraude: densidad de las componentes PCA por clase",
                 loc="left", pad=46)
    ax.text(0, 1.055, "Seis componentes con mayor separación (KS) vs. las dos con menor. "
            "Cada curva normalizada a su propio máximo.", transform=ax.transAxes,
            color=C.COLOR_TEXTO_2, fontsize=9.5)
    ax.plot([], [], color=C.COLOR_LEGITIMA, lw=6, alpha=0.6, label="Legítima")
    ax.plot([], [], color=C.COLOR_FRAUDE, lw=6, alpha=0.6, label="Fraude")
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(-0.01, 0.99))
    guardar(fig, "01_ridgeline_firma_fraude.png")
    return {"top": list(ks.index[:6]), "bottom": list(ks.index[-2:]),
            "ks_top": float(ks.iloc[0]), "ks_bottom": float(ks.iloc[-1])}


# --------------------------------------------------------------------------
# 2. Reloj del riesgo (polar + mapa de calor)
# --------------------------------------------------------------------------
def vis_reloj(df: pd.DataFrame) -> dict:
    horas = np.arange(24)
    reparto = (df.groupby(["es_fraude", "hora_dia"]).size()
               .unstack(0).reindex(horas, fill_value=0))
    pct = reparto / reparto.sum() * 100
    tasa = df.groupby("hora_dia")["es_fraude"].mean().reindex(horas) * 1000

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 0.55], hspace=0.35, wspace=0.25)
    theta = 2 * np.pi * horas / 24
    ancho = 2 * np.pi / 24

    ax = fig.add_subplot(gs[0, 0], projection="polar")
    for clase, color, nombre in ((0, C.COLOR_LEGITIMA, "Legítimas"),
                                 (1, C.COLOR_FRAUDE, "Fraudes")):
        cerrado_t = np.append(theta, theta[0])
        cerrado_r = np.append(pct[clase].values, pct[clase].values[0])
        ax.plot(cerrado_t, cerrado_r, color=color, lw=2, label=nombre)
        ax.fill(cerrado_t, cerrado_r, color=color, alpha=0.15)
    _formato_polar(ax)
    ax.set_title("(a) ¿Cuándo ocurre cada clase?\n% de las operaciones de la clase en cada hora",
                 pad=22, fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.12, -0.12))

    ax = fig.add_subplot(gs[0, 1], projection="polar")
    cmap = LinearSegmentedColormap.from_list("naranja", C.SECUENCIAL_NARANJA)
    norm = plt.Normalize(0, tasa.max())
    ax.bar(theta, tasa.values, width=ancho * 0.9, color=cmap(norm(tasa.values)),
           edgecolor=C.COLOR_SUPERFICIE, linewidth=1)
    h_max = int(tasa.idxmax())
    ax.annotate(f"hora {h_max}: {tasa.max():.1f}‰", xy=(theta[h_max], tasa.max()),
                xytext=(theta[h_max] + 0.5, tasa.max() * 1.12), fontsize=9,
                color=C.COLOR_TEXTO, arrowprops=dict(arrowstyle="-", color=C.COLOR_TEXTO_2))
    _formato_polar(ax)
    ax.set_rlabel_position(200)
    ax.set_title("(b) ¿Cuándo es más riesgosa una operación?\nFraudes por cada 1 000 operaciones",
                 pad=22, fontsize=11)

    ax = fig.add_subplot(gs[1, :])
    mat = (df.groupby(["dia", "hora_dia"])["es_fraude"].mean().unstack() * 1000
           ).reindex(columns=horas)
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=0)
    ax.set_xticks(horas)
    ax.set_yticks([0, 1], [f"Día {d}" for d in mat.index])
    ax.set_xlabel("Hora relativa al inicio de la captura")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(mat.values):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                color=C.COLOR_SUPERFICIE if v > np.nanmax(mat.values) * 0.55 else C.COLOR_TEXTO)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("‰ fraude", color=C.COLOR_TEXTO_2)
    cb.outline.set_visible(False)
    ax.set_title("(c) Tasa de fraude (‰) por día y hora: ¿se repite el patrón?",
                 loc="left", fontsize=11)
    fig.suptitle("El reloj del riesgo: el fraude no duerme", x=0.06, ha="left",
                 fontsize=15, fontweight="bold", y=0.99)
    guardar(fig, "02_reloj_del_riesgo.png")

    nocturnas = tasa.loc[1:6].mean()
    diurnas = tasa.loc[9:22].mean()
    return {"h_max": h_max, "tasa_max": float(tasa.max()),
            "tasa_media": float(df["es_fraude"].mean() * 1000),
            "nocturna": float(nocturnas), "diurna": float(diurnas),
            "pct_leg_noche": float(pct[0].loc[1:6].sum()),
            "pct_fra_noche": float(pct[1].loc[1:6].sum())}


def _formato_polar(ax) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(2 * np.pi * np.arange(0, 24, 3) / 24,
                  [f"{h:02d}h" for h in range(0, 24, 3)])
    ax.tick_params(axis="y", labelsize=8, colors=C.COLOR_TEXTO_2)
    ax.grid(color=C.COLOR_GRID, lw=0.6)
    ax.spines["polar"].set_color(C.COLOR_GRID)


# --------------------------------------------------------------------------
# 3. Mapa t-SNE
# --------------------------------------------------------------------------
def vis_tsne(df: pd.DataFrame, n_legitimas: int = 6000) -> dict:
    fraudes = df[df["es_fraude"] == 1]
    legitimas = df[df["es_fraude"] == 0].sample(n_legitimas, random_state=SEMILLA)
    muestra = pd.concat([legitimas, fraudes])
    X = RobustScaler().fit_transform(muestra[C.COLS_PCA + ["monto_log"]])
    emb = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto",
               random_state=SEMILLA).fit_transform(X)
    y = muestra["es_fraude"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=9, color=C.COLOR_LEGITIMA,
               alpha=0.35, lw=0, label=f"Legítima (muestra de {n_legitimas:,})")
    ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=22, color=C.COLOR_FRAUDE,
               edgecolor=C.COLOR_SUPERFICIE, linewidth=0.8,
               label=f"Fraude (todas, {len(fraudes):,})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    nn = NearestNeighbors(n_neighbors=11).fit(emb)
    _, idx = nn.kneighbors(emb[y == 1])
    pureza = (y[idx[:, 1:]] == 1).mean(axis=1)
    pct_aislado = float((pureza >= 0.7).mean() * 100)

    ax.set_title("Mapa t-SNE: el fraude ocupa sus propias regiones", loc="left", pad=26)
    ax.text(0, 1.015, f"Proyección 2D no lineal de V1..V28 + log(monto). "
            f"{pct_aislado:.0f}% de los fraudes tiene ≥7 de sus 10 vecinos más cercanos "
            "también fraudulentos.", transform=ax.transAxes, color=C.COLOR_TEXTO_2,
            fontsize=9.5)
    ax.legend(loc="lower right", markerscale=1.8)
    guardar(fig, "03_tsne_mapa_fraude.png")
    return {"pct_aislado": pct_aislado, "n_leg": n_legitimas, "n_fra": len(fraudes)}


# --------------------------------------------------------------------------
# Narrativa
# --------------------------------------------------------------------------
def escribir_historia(r1: dict, r2: dict, r3: dict) -> None:
    texto = f"""LA HISTORIA QUE CUENTAN LAS GRÁFICAS
{'=' * 70}

01_ridgeline_firma_fraude.png — "La firma del fraude"
-----------------------------------------------------
Técnica: ridgeline (joy plot) de densidades por kernel (KDE), con el eje x
estandarizado contra las operaciones legítimas para comparar filas.

Historia: las operaciones legítimas forman un pico estrecho alrededor de 0 en
todas las componentes; son "aburridas" y homogéneas. Los fraudes, en cambio,
se desplazan entre 3 y 10 desviaciones estándar en las componentes
{', '.join(r1['top'])} (KS hasta {r1['ks_top']:.2f}): hacia valores negativos en
la mayoría y hacia positivos en V4 y V11, y con una dispersión muchísimo mayor
(el fraude es heterogéneo; la normalidad es homogénea). En contraste, en
{', '.join(r1['bottom'])} (KS ≈ {r1['ks_bottom']:.2f}) ambas curvas se superponen:
no aportan información. Conclusión: el fraude deja una huella clara pero
concentrada en pocas dimensiones; un sistema interbancario debería priorizar
esas señales y puede descartar ruido.

02_reloj_del_riesgo.png — "El reloj del riesgo: el fraude no duerme"
--------------------------------------------------------------------
Técnica: diagramas polares de 24 h (radar + barras radiales) y mapa de calor
día × hora con anotaciones.

Historia: (a) las operaciones legítimas siguen un ritmo humano: sólo el
{r2['pct_leg_noche']:.1f}% ocurre entre las horas relativas 1 y 6, mientras que el
{r2['pct_fra_noche']:.1f}% de los fraudes cae en esa franja. (b) Como el volumen
legítimo se desploma de madrugada pero el fraude sigue, la TASA de fraude se
dispara: {r2['nocturna']:.1f}‰ en promedio de 1 a 6 h frente a {r2['diurna']:.1f}‰ de
9 a 22 h, con un máximo de {r2['tasa_max']:.1f}‰ en la hora {r2['h_max']} (la media
global es {r2['tasa_media']:.2f}‰). (c) El mapa de calor muestra que el pico
aparece en las madrugadas de ambos días (más marcado en uno de ellos), por lo
que es un patrón y no un accidente. Lectura interbancaria: las ventanas de
baja actividad (noche, fines de semana, fuera del horario de compensación)
merecen umbrales de alerta más estrictos.

03_tsne_mapa_fraude.png — "Mapa t-SNE"
--------------------------------------
Técnica: reducción de dimensionalidad no lineal t-SNE sobre las 28
componentes y log(monto) escaladas robustamente (todos los fraudes + una
muestra aleatoria de {r3['n_leg']:,} legítimas, para que sea legible).

Historia: la nube azul de legítimas ocupa la mayor parte del mapa mientras que
los fraudes se agrupan en islas y bordes propios: el {r3['pct_aislado']:.0f}% de
los fraudes tiene al menos 7 de sus 10 vecinos también fraudulentos. Esto
indica que el problema es en gran parte SEPARABLE con modelos no lineales
(árboles, gradient boosting, k-NN), aunque queda un grupo de fraudes
"camuflados" dentro de la nube legítima: esos son los que generarán falsos
negativos y justifican combinar el modelo con reglas de negocio y revisión
humana.

Síntesis
--------
El fraude es raro (≈1.7 de cada 1 000 operaciones), se reconoce por una
firma concentrada en pocas variables latentes, se vuelve proporcionalmente
más frecuente cuando la actividad legítima baja y forma regiones propias en
el espacio de atributos. Las tres vistas apuntan al mismo diseño de sistema:
modelo no lineal + atributos temporales + métricas para clases desbalanceadas.
"""
    (C.DIR_VIS / "historia_visualizaciones.txt").write_text(texto, encoding="utf-8")


def main() -> None:
    print("=== PARTE 3: Visualización ===")
    estilo()
    df = leer()
    ks = ranking_ks(df)
    r1 = vis_ridgeline(df, ks)
    r2 = vis_reloj(df)
    r3 = vis_tsne(df)
    escribir_historia(r1, r2, r3)
    print(f"[vis] Narrativa en {(C.DIR_VIS / 'historia_visualizaciones.txt').relative_to(C.RAIZ)}")


if __name__ == "__main__":
    main()
