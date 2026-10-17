"""
Modelado - Parte 3: Predicción de fraude en operaciones futuras.

Enunciado original: predecir ventas futuras. Adaptación: predecir si las
transacciones FUTURAS son fraudulentas, que es lo que hace un detector en
producción.

1. Variables nuevas: hora cíclica, franja de madrugada, log del monto y
   monto cero (dentro del pipeline, ver ml_comun.derivar_variables).
2. Partición temporal: entrenamiento = 75% más antiguo, prueba = 25% más reciente.
3. Algoritmo: el ganador de la Parte 1 (validación cruzada).
4. Umbrales de decisión elegidos con las predicciones fuera de pliegue del
   entrenamiento (sin mirar la prueba).
5. Evaluación en el futuro, importancia de variables y guardado del detector.
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

import config as C  # noqa: E402
from ml_comun import (COLS_ENTRADA, DECISIONES, OBJETIVO, SEMILLA,  # noqa: E402
                      cargar_transacciones, decidir, derivar_variables,
                      dividir_temporal, modelos_candidatos)
from parte3_visualizacion import estilo, guardar  # noqa: E402

PRECISION_BLOQUEO = 0.90     # bloquear sólo si 9 de cada 10 bloqueos son fraude real
PRECISION_REVISION = 0.50    # el analista confirma al menos 1 de cada 2 alertas
COLOR_DECISION = {"APROBAR": C.COLOR_APROBAR, "REVISAR": C.COLOR_REVISAR,
                  "BLOQUEAR": C.COLOR_BLOQUEAR}


def elegir_umbrales(y: np.ndarray, puntaje: np.ndarray) -> dict:
    """Menor umbral que cumple cada precisión objetivo (menor umbral = más recall)."""
    prec, _, umbrales = precision_recall_curve(y, puntaje)
    prec = prec[:-1]

    def menor_que_cumple(objetivo):
        ok = np.where(prec >= objetivo)[0]
        return float(umbrales[ok[0]]) if len(ok) else 1.0

    return {"revisar": menor_que_cumple(PRECISION_REVISION),
            "bloquear": menor_que_cumple(PRECISION_BLOQUEO)}


def metricas_umbral(y, puntaje, umbral, montos) -> dict:
    pred = (puntaje >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(umbral=round(umbral, 4), alertas=int(pred.sum()), vp=int(tp), fp=int(fp),
                fn=int(fn), vn=int(tn), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                monto_fraude_detectado=float(montos[(pred == 1) & (y == 1)].sum()),
                monto_fraude_no_detectado=float(montos[(pred == 0) & (y == 1)].sum()))


def referencia_legitimas(X: pd.DataFrame, y: pd.Series) -> dict:
    """Media y desviación de las operaciones normales: el detector las usa para
    explicar POR QUÉ una operación es sospechosa (z-scores)."""
    D = derivar_variables(X[y == 0])
    return {"media": D.mean().to_dict(), "desv": D.std().replace(0, 1).to_dict()}


def vis_evaluacion(y, puntaje, umbrales, tabla_decision) -> None:
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    bins = np.linspace(0, 1, 51)
    for clase, color, nombre in ((0, C.COLOR_LEGITIMA, "Normal"), (1, C.COLOR_FRAUDE, "Fraudulenta")):
        ax.hist(puntaje[y == clase], bins=bins, color=color, alpha=0.75 if clase else 0.6,
                label=f"{nombre} (n={int((y == clase).sum()):,})", edgecolor=C.COLOR_SUPERFICIE,
                linewidth=0.5)
    ax.set_yscale("log")
    for nombre in ("revisar", "bloquear"):
        u = umbrales[nombre]
        ax.axvline(u, color=COLOR_DECISION[nombre.upper()], lw=1.8)
        ax.text(u + 0.01, ax.get_ylim()[1] * 0.4, f"{nombre.upper()}\n≥ {u:.2f}", fontsize=8.5,
                color=C.COLOR_TEXTO, va="top")
    ax.set_xlabel("Puntaje de fraude del modelo")
    ax.set_ylabel("Operaciones (escala log)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("(a) Puntajes en el período de prueba", loc="left", fontsize=11)

    ax = fig.add_subplot(gs[1])
    mat = tabla_decision.to_numpy()
    pct = mat / mat.sum(axis=0, keepdims=True) * 100
    ax.imshow(pct, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "a", C.SECUENCIAL_AZUL), vmin=0, vmax=100, aspect="auto")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}\n({pct[i, j]:.1f}%)", ha="center", va="center", fontsize=9.5,
                color=C.COLOR_SUPERFICIE if pct[i, j] > 55 else C.COLOR_TEXTO)
    ax.set_xticks([0, 1], ["Normal", "Fraudulenta"])
    ax.set_yticks(range(3), DECISIONES)
    ax.set_xlabel("Clase real")
    ax.grid(False)
    ax.set_title("(b) Decisión vs. realidad\n(% de cada clase real)", loc="left", fontsize=11)

    ax = fig.add_subplot(gs[2])
    prec, rec, _ = precision_recall_curve(y, puntaje)
    ax.plot(rec, prec, color=C.COLOR_LEGITIMA, lw=2)
    for nombre in ("revisar", "bloquear"):
        pred = puntaje >= umbrales[nombre]
        r, p = pred[y == 1].mean(), y[pred].mean()
        ax.plot(r, p, "o", ms=10, color=COLOR_DECISION[nombre.upper()],
                markeredgecolor=C.COLOR_SUPERFICIE, markeredgewidth=2, zorder=4)
        ax.annotate(f"{nombre.upper()}\nP={p:.2f} R={r:.2f}", (r, p), xytext=(-75, -38),
                    textcoords="offset points", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color=C.COLOR_TEXTO_2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title(f"(c) Curva P-R en prueba (AUPRC {average_precision_score(y, puntaje):.3f})",
                 loc="left", fontsize=11)
    fig.suptitle("Evaluación del detector en operaciones futuras (25% más reciente)",
                 x=0.07, ha="left", fontsize=14, fontweight="bold", y=1.04)
    guardar(fig, "ml_04_evaluacion_prueba.png")


def vis_importancia(imp: pd.DataFrame) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colores = [C.COLOR_LEGITIMA if v > 0.01 else C.COLOR_NEUTRO for v in top["media"]]
    ax.barh(top.index, top["media"], xerr=top["desv"], color=colores, height=0.65,
            error_kw=dict(ecolor=C.COLOR_TEXTO_2, lw=1, capsize=2))
    for i, (v, d) in enumerate(zip(top["media"], top["desv"])):
        ax.text(v + d + top["media"].max() * 0.02, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("Caída de AUPRC al desordenar la variable (media ± desv., 5 repeticiones)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Importancia por permutación: ¿qué mira el detector?", loc="left", pad=22)
    ax.text(0, 1.015, "Calculada en el período de prueba sobre la transacción cruda "
            "(tiempo_seg = hora, monto, V1..V28).", transform=ax.transAxes, fontsize=9,
            color=C.COLOR_TEXTO_2)
    guardar(fig, "ml_05_importancia_variables.png")


def main() -> None:
    print("=== MODELADO 3: Predicción ===")
    estilo()
    df = cargar_transacciones()
    entreno, prueba = dividir_temporal(df)
    Xtr, ytr = entreno[COLS_ENTRADA], entreno[OBJETIVO]
    Xte, yte = prueba[COLS_ENTRADA], prueba[OBJETIVO].to_numpy()

    if not C.OOF_PATH.exists():
        raise FileNotFoundError("Ejecuta antes la Parte 1 de modelado (python main.py ml1)")
    oof = np.load(C.OOF_PATH)
    elegido = str(oof["elegido"])
    puntaje_oof = oof[f"oof_{list(oof['nombres']).index(elegido)}"]
    umbrales = elegir_umbrales(oof["y"], puntaje_oof)
    print(f"  Modelo: {elegido} | umbrales {umbrales}")

    modelo = modelos_candidatos()[elegido].fit(Xtr, ytr)
    puntaje = modelo.predict_proba(Xte)[:, 1]
    montos = prueba["monto"].to_numpy()

    metricas = {"auprc": average_precision_score(yte, puntaje),
                "roc_auc": roc_auc_score(yte, puntaje)}
    por_umbral = pd.DataFrame({n: metricas_umbral(yte, puntaje, u, montos)
                               for n, u in [("revisar_o_mas", umbrales["revisar"]),
                                            ("bloquear", umbrales["bloquear"]),
                                            ("umbral_0.5", 0.5)]}).T
    decision = decidir(puntaje, umbrales)
    tabla_decision = (pd.crosstab(pd.Categorical(decision, DECISIONES),
                                  pd.Series(yte).map({0: "Normal", 1: "Fraudulenta"}))
                      .reindex(columns=["Normal", "Fraudulenta"], fill_value=0))
    tabla_decision.index.name, tabla_decision.columns.name = "decisión", "real"

    print("  Importancia por permutación…")
    rng = np.random.RandomState(SEMILLA)
    idx = np.concatenate([np.where(yte == 1)[0],
                          rng.choice(np.where(yte == 0)[0], 20_000, replace=False)])
    pi = permutation_importance(modelo, Xte.iloc[idx], yte[idx], scoring="average_precision",
                                n_repeats=5, random_state=SEMILLA, n_jobs=-1)
    imp = (pd.DataFrame({"media": pi.importances_mean, "desv": pi.importances_std},
                        index=["hora (tiempo_seg)" if c == "tiempo_seg" else c
                               for c in COLS_ENTRADA])
           .sort_values("media", ascending=False))

    vis_evaluacion(yte, puntaje, umbrales, tabla_decision)
    vis_importancia(imp)

    C.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    meta = {
        "modelo": elegido, "version_sklearn": sklearn.__version__,
        "entrenado": datetime.now().isoformat(timespec="seconds"),
        "columnas_entrada": COLS_ENTRADA, "umbrales": umbrales,
        "politica": {"BLOQUEAR": f"puntaje >= {umbrales['bloquear']:.4f} "
                                 f"(precisión OOF >= {PRECISION_BLOQUEO:.0%})",
                     "REVISAR": f"puntaje >= {umbrales['revisar']:.4f} "
                                f"(precisión OOF >= {PRECISION_REVISION:.0%})",
                     "APROBAR": "resto"},
        "entrenamiento": {"operaciones": len(entreno), "fraudes": int(ytr.sum()),
                          "horas": [round(entreno.tiempo_seg.min() / 3600, 2),
                                    round(entreno.tiempo_seg.max() / 3600, 2)]},
        "prueba": {"operaciones": len(prueba), "fraudes": int(yte.sum()),
                   "auprc": round(metricas["auprc"], 4), "roc_auc": round(metricas["roc_auc"], 4)},
    }
    joblib.dump({"pipeline": modelo, "umbrales": umbrales, "columnas": COLS_ENTRADA,
                 "referencia": referencia_legitimas(Xtr, ytr), "meta": meta},
                C.MODELO_PATH, compress=3)
    (C.DIR_MODELOS / "detector_fraude_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[modelo] Guardado en {C.MODELO_PATH.relative_to(C.RAIZ)} "
          f"({C.MODELO_PATH.stat().st_size / 2**20:.1f} MB)")

    r, b = por_umbral.loc["revisar_o_mas"], por_umbral.loc["bloquear"]
    seleccion = json.loads((C.DIR_REPORTES_ML / "seleccion_modelo.json").read_text(encoding="utf-8"))
    auprc_cv = next(m["auprc_mean"] for m in seleccion["metricas_cv"] if m["modelo"] == elegido)
    fraude_total = montos[yte == 1].sum()
    fmt_tabla = por_umbral.copy()
    for col in ("alertas", "vp", "fp", "fn", "vn"):
        fmt_tabla[col] = fmt_tabla[col].astype(int)
    for col in ("precision", "recall", "f1"):
        fmt_tabla[col] = fmt_tabla[col].astype(float).round(3)
    for col in ("monto_fraude_detectado", "monto_fraude_no_detectado"):
        fmt_tabla[col] = fmt_tabla[col].astype(float).round(2)

    texto = f"""MODELADO - PARTE 3: PREDICCIÓN DE FRAUDE EN OPERACIONES FUTURAS
{'=' * 70}

Adaptación: el enunciado pide predecir ventas futuras. En fraude
transaccional la predicción útil es: dada una operación NUEVA, ¿qué tan
probable es que sea fraude y qué hacemos con ella?

1. Nuevas características (dentro del pipeline)
-----------------------------------------------
* hora_sin, hora_cos: "época" del día como ciclo (equivale a la "época del
  año" del enunciado; aquí el patrón es diario).
* es_madrugada: franja 1-6 h, donde la tasa de fraude es ~5 veces mayor.
* monto_log: corrige la cola larga del monto.
* es_monto_cero: posibles operaciones de prueba.
* V1..V28 pasan sin cambios. Todo se estandariza con StandardScaler.
El pipeline recibe la transacción cruda (tiempo_seg, monto, V1..V28), así que
el mismo objeto sirve para entrenar y para puntuar en tiempo real.

2. Partición entrenamiento / prueba
-----------------------------------
Temporal, porque el modelo debe predecir el FUTURO:
* Entrenamiento: {len(entreno):,} operaciones ({int(ytr.sum())} fraudes), horas {entreno.tiempo_seg.min() / 3600:.1f}-{entreno.tiempo_seg.max() / 3600:.1f}
* Prueba:        {len(prueba):,} operaciones ({int(yte.sum())} fraudes), horas {prueba.tiempo_seg.min() / 3600:.1f}-{prueba.tiempo_seg.max() / 3600:.1f}

3. Algoritmo
------------
{elegido}, elegido en la Parte 1 por tener la mayor AUPRC en validación
cruzada frente a la regresión logística y el gradient boosting. Usa
class_weight="balanced_subsample" para compensar el desbalance (1:{int((len(ytr) - ytr.sum()) / ytr.sum())}).

4. Política de decisión (umbrales elegidos SIN ver la prueba)
--------------------------------------------------------------
Con las predicciones fuera de pliegue del entrenamiento:
* BLOQUEAR  si puntaje >= {umbrales['bloquear']:.3f}: menor umbral con precisión >= {PRECISION_BLOQUEO:.0%}
            (bloquear a un cliente legítimo es caro: debe haber mucha certeza).
* REVISAR   si puntaje >= {umbrales['revisar']:.3f}: menor umbral con precisión >= {PRECISION_REVISION:.0%}
            (el analista confirma al menos 1 de cada 2 alertas).
* APROBAR   en otro caso.

5. Evaluación en el período de prueba (operaciones futuras)
------------------------------------------------------------
AUPRC = {metricas['auprc']:.3f}   ROC-AUC = {metricas['roc_auc']:.3f}   (azar: AUPRC = {yte.mean():.4f})

Métricas por umbral (alerta = puntaje >= umbral):
{fmt_tabla.to_string()}

Decisión de la política vs. realidad:
{tabla_decision.to_string()}

Lectura:
* Con el umbral de REVISIÓN se detecta el {r['recall'] * 100:.0f}% de los fraudes futuros
  ({int(r['vp'])} de {int(yte.sum())}) y se generan {int(r['alertas'])} alertas entre {len(prueba):,} operaciones
  ({int(r['alertas']) / len(prueba) * 100:.2f}%); el {r['precision'] * 100:.0f}% de ellas es fraude real
  (el objetivo fijado con datos de entrenamiento era {PRECISION_REVISION:.0%}).
* Con el umbral de BLOQUEO se detiene automáticamente el {b['recall'] * 100:.0f}% de los fraudes
  con una precisión de {b['precision'] * 100:.0f}% ({int(b['fp'])} operación(es) legítima(s) bloqueada(s)).
* Monto: se detecta {r['monto_fraude_detectado']:,.2f} EUR de {fraude_total:,.2f} EUR defraudados en el período
  ({r['monto_fraude_detectado'] / fraude_total * 100:.0f}%).
* Los {int(r['fn'])} fraudes no detectados son los "camuflados" que el mapa t-SNE
  mostraba dentro de la nube legítima: hacen falta otras señales (historial
  de la cuenta, dispositivo, destino) que este set no incluye.
* Deriva temporal: la AUPRC pasa de {auprc_cv:.3f} (validación cruzada) a {metricas['auprc']:.3f}
  en el futuro, y la precisión del umbral de revisión baja de {PRECISION_REVISION:.0%} a
  {r['precision'] * 100:.0f}%. El bloqueo sigue siendo muy preciso; la franja de revisión es
  la más sensible a los cambios. Un sistema real debe recalibrar los umbrales
  con datos recientes y re-entrenar periódicamente.

6. ¿Qué mira el modelo? (ml_05_importancia_variables.png)
---------------------------------------------------------
Importancia por permutación (caída de AUPRC al desordenar cada variable):
{imp.head(10).round(4).to_string()}
Las tres más importantes ({', '.join(imp.index[:3])}) están entre las más discriminantes
del EDA. Otras que el EDA destacaba (V10, V12) bajan aquí porque, dentro de
los fraudes, llevan información redundante con V14: al desordenar una, el
bosque compensa con las demás. La hora y el monto aportan poco por sí solos
cuando ya están las componentes de comportamiento.

7. Artefacto para producción
----------------------------
{C.MODELO_PATH.relative_to(C.RAIZ)} contiene el pipeline, los umbrales, las columnas de
entrada y la referencia de operaciones normales para explicar alertas.
Metadatos legibles en modelos/detector_fraude_meta.json.
Uso en tiempo real: src/detector_tiempo_real.py
"""
    (C.DIR_REPORTES_ML / "parte3_prediccion.txt").write_text(texto, encoding="utf-8")
    print(f"[reportes] {(C.DIR_REPORTES_ML / 'parte3_prediccion.txt').relative_to(C.RAIZ)}")


if __name__ == "__main__":
    main()
