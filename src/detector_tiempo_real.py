"""
Detector de fraude en tiempo real.

Carga el modelo entrenado (modelos/detector_fraude.joblib) y, para cada
transacción que llega, devuelve:
    puntaje   probabilidad estimada de fraude (0-1)
    decisión  APROBAR / REVISAR / BLOQUEAR según la política de la Parte 3
    motivos   qué variables se alejan de una operación normal
    latencia  tiempo de respuesta en milisegundos

Uso:
    python src/detector_tiempo_real.py --casos                  # casos de prueba
    python src/detector_tiempo_real.py --simular                # repetición del período de prueba
    python src/detector_tiempo_real.py --simular --velocidad 600 # 1 h simulada = 6 s
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from ml_comun import (DECISIONES, cargar_transacciones, decidir,  # noqa: E402
                      derivar_variables, dividir_temporal)
from parte3_visualizacion import estilo, guardar  # noqa: E402

Z_MOTIVO = 4.0          # una variable es "anómala" si se aleja > 4 desv. de lo normal
VENTANA_SEG = 60        # micro-lotes de 1 minuto en la simulación


class TransaccionInvalida(ValueError):
    pass


@dataclass
class Resultado:
    id_transaccion: object
    puntaje: float
    decision: str
    motivos: list[str] = field(default_factory=list)
    latencia_ms: float = 0.0


class DetectorFraude:
    def __init__(self, ruta=C.MODELO_PATH):
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}. Entrena antes: python main.py ml1 ml3")
        art = joblib.load(ruta)
        self.pipeline = art["pipeline"]
        self.umbrales = art["umbrales"]
        self.columnas = art["columnas"]
        self.media = pd.Series(art["referencia"]["media"])
        self.desv = pd.Series(art["referencia"]["desv"])
        self.meta = art["meta"]
        # una transacción a la vez: paralelizar entre árboles sólo agrega latencia
        modelo = self.pipeline.named_steps["modelo"]
        if hasattr(modelo, "n_jobs"):
            modelo.n_jobs = 1

    # ---------------------------------------------------------------- entrada
    def _validar(self, X: pd.DataFrame) -> pd.DataFrame:
        faltan = [c for c in self.columnas if c not in X.columns]
        if faltan:
            raise TransaccionInvalida(f"Faltan campos: {', '.join(faltan)}")
        X = X[self.columnas].apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            malos = X.columns[X.isna().any()].tolist()
            raise TransaccionInvalida(f"Campos vacíos o no numéricos: {', '.join(malos)}")
        if (X["monto"] < 0).any():
            raise TransaccionInvalida("El monto no puede ser negativo")
        if (X["tiempo_seg"] < 0).any():
            raise TransaccionInvalida("tiempo_seg no puede ser negativo")
        return X

    # ---------------------------------------------------------------- motivos
    def motivos(self, X: pd.DataFrame) -> list[list[str]]:
        D = derivar_variables(X)
        Z = (D - self.media) / self.desv
        salida = []
        for i in range(len(D)):
            z = Z.iloc[i][[*C.COLS_PCA, "monto_log"]]
            fuertes = z[z.abs() >= Z_MOTIVO].sort_values(key=abs, ascending=False).head(3)
            m = [f"{'monto' if v == 'monto_log' else v} {'alto' if s > 0 else 'bajo'} (z={s:+.1f})"
                 for v, s in fuertes.items()]
            if D.iloc[i]["es_monto_cero"]:
                m.append("monto 0 (posible prueba)")
            if D.iloc[i]["es_madrugada"]:
                m.append("franja de madrugada")
            salida.append(m or ["combinación de variables (ninguna extrema por sí sola)"])
        return salida

    # ------------------------------------------------------------- evaluación
    def evaluar(self, transaccion: dict, id_transaccion=None) -> Resultado:
        t0 = time.perf_counter()
        X = self._validar(pd.DataFrame([transaccion]))
        puntaje = float(self.pipeline.predict_proba(X)[0, 1])
        decision = str(decidir(puntaje, self.umbrales)[0])
        motivos = self.motivos(X)[0] if decision != "APROBAR" else []
        return Resultado(id_transaccion, puntaje, decision, motivos,
                         (time.perf_counter() - t0) * 1000)

    def evaluar_lote(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._validar(df)
        puntaje = self.pipeline.predict_proba(X)[:, 1]
        out = pd.DataFrame({"puntaje": puntaje, "decision": decidir(puntaje, self.umbrales)},
                           index=df.index)
        alerta = out["decision"] != "APROBAR"
        out["motivos"] = ""
        if alerta.any():
            out.loc[alerta, "motivos"] = ["; ".join(m) for m in self.motivos(X[alerta])]
        return out


# ===================================================================== casos
def _fila_a_dict(fila: pd.Series, columnas) -> dict:
    return {c: float(fila[c]) for c in columnas}


def construir_casos(det: DetectorFraude, prueba: pd.DataFrame, entreno: pd.DataFrame) -> list[dict]:
    puntajes = det.evaluar_lote(prueba)["puntaje"]
    legit = prueba[prueba["es_fraude"] == 0]
    fraudes = prueba[prueba["es_fraude"] == 1]
    p_leg = puntajes.loc[legit.index]
    tipica = legit.loc[(p_leg - p_leg.median()).abs().idxmin()]
    fraude_claro = fraudes.loc[puntajes.loc[fraudes.index].idxmax()]
    fraude_camuflado = fraudes.loc[puntajes.loc[fraudes.index].idxmin()]

    perfil_ataque = entreno.loc[entreno["es_fraude"] == 1, ["V14", "V12", "V10", "V17", "V4"]].median()
    ataque = _fila_a_dict(tipica, det.columnas) | perfil_ataque.to_dict()
    prueba_monto_cero = _fila_a_dict(tipica, det.columnas) | {"monto": 0.0,
                                                              "tiempo_seg": 86_400 + 3 * 3600}
    invalida = _fila_a_dict(tipica, det.columnas)
    invalida.pop("V14")
    negativa = _fila_a_dict(tipica, det.columnas) | {"monto": -50.0}

    return [
        dict(caso="1. Operación normal típica (real, del período de prueba)",
             esperado="APROBAR", tx=_fila_a_dict(tipica, det.columnas), real="Normal",
             id=int(tipica["id_transaccion"])),
        dict(caso="2. Fraude real con perfil claro (real)", esperado="BLOQUEAR",
             tx=_fila_a_dict(fraude_claro, det.columnas), real="Fraudulenta",
             id=int(fraude_claro["id_transaccion"])),
        dict(caso="3. Fraude real 'camuflado' (real, el de menor puntaje)",
             esperado="APROBAR (limitación conocida)", tx=_fila_a_dict(fraude_camuflado, det.columnas),
             real="Fraudulenta", id=int(fraude_camuflado["id_transaccion"])),
        dict(caso="4. Ataque sintético: operación típica con V14/V12/V10/V17/V4 de fraude",
             esperado="REVISAR o BLOQUEAR", tx=ataque, real="Sintético", id="sint-1"),
        dict(caso="5. Monto 0 a las 3 h con comportamiento normal",
             esperado="APROBAR (una regla sola no basta)", tx=prueba_monto_cero,
             real="Sintético", id="sint-2"),
        dict(caso="6. Transacción sin el campo V14", esperado="ERROR DE DATOS", tx=invalida,
             real="Inválida", id="inv-1"),
        dict(caso="7. Transacción con monto negativo", esperado="ERROR DE DATOS", tx=negativa,
             real="Inválida", id="inv-2"),
    ]


def ejecutar_casos(det: DetectorFraude, prueba: pd.DataFrame, entreno: pd.DataFrame) -> str:
    lineas = ["CASOS DE PRUEBA DEL DETECTOR", "=" * 70,
              f"Umbrales: REVISAR >= {det.umbrales['revisar']:.3f} · "
              f"BLOQUEAR >= {det.umbrales['bloquear']:.3f}", ""]
    for c in construir_casos(det, prueba, entreno):
        try:
            r = det.evaluar(c["tx"], c["id"])
            obtenido = r.decision
            detalle = (f"puntaje {r.puntaje:.3f} · {r.latencia_ms:.1f} ms"
                       + (f"\n    motivos: {'; '.join(r.motivos)}" if r.motivos else ""))
        except TransaccionInvalida as e:
            obtenido, detalle = "ERROR DE DATOS", f"rechazada: {e}"
        ok = c["esperado"].split(" ")[0] in obtenido or obtenido in c["esperado"]
        lineas += [f"{c['caso']}",
                   f"    id={c['id']} · clase real: {c['real']} · esperado: {c['esperado']}",
                   f"    obtenido: {obtenido} [{'OK' if ok else 'DIFERENTE'}] · {detalle}", ""]
    texto = "\n".join(lineas)
    print(texto)
    return texto


# ================================================================ simulación
def medir_latencia(det: DetectorFraude, prueba: pd.DataFrame, n: int = 300) -> dict:
    muestra = prueba.sample(n, random_state=0)
    lat = [det.evaluar(_fila_a_dict(f, det.columnas)).latencia_ms for _, f in muestra.iterrows()]
    return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "max_ms": float(np.max(lat)), "n": n}


def simular(det: DetectorFraude, prueba: pd.DataFrame, velocidad: float = 0,
            max_alertas: int = 25) -> pd.DataFrame:
    """Reproduce el período de prueba en orden cronológico, en micro-lotes de
    1 minuto, como lo vería un motor de streaming. `velocidad` = segundos
    simulados por segundo real (0 = lo más rápido posible)."""
    ventanas = (prueba["tiempo_seg"] // VENTANA_SEG).astype(int)
    registros, impresas, hora_actual = [], 0, None
    cont = {d: 0 for d in DECISIONES} | {"fraudes": 0, "detectados": 0}
    print(f"\n▶ Simulación: {len(prueba):,} operaciones, horas "
          f"{prueba.tiempo_seg.min() / 3600:.1f}-{prueba.tiempo_seg.max() / 3600:.1f}\n")
    for ventana, lote in prueba.groupby(ventanas, sort=True):
        t0 = time.perf_counter()
        res = det.evaluar_lote(lote)
        lat_lote = (time.perf_counter() - t0) * 1000
        res = res.assign(id_transaccion=lote["id_transaccion"], tiempo_seg=lote["tiempo_seg"],
                         monto=lote["monto"], es_fraude=lote["es_fraude"], lote=ventana,
                         latencia_lote_ms=lat_lote, tamano_lote=len(lote))
        registros.append(res)
        for d, n in res["decision"].value_counts().items():
            cont[d] += n
        cont["fraudes"] += int(res["es_fraude"].sum())
        cont["detectados"] += int(((res["decision"] != "APROBAR") & (res["es_fraude"] == 1)).sum())

        for _, r in res[res["decision"] != "APROBAR"].iterrows():
            if impresas < max_alertas:
                s = int(r["tiempo_seg"])
                real = "FRAUDE ✔" if r["es_fraude"] else "normal ✘"
                print(f"  [t+{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}] {r['decision']:<8} "
                      f"id {int(r['id_transaccion']):>6} · {r['monto']:>8.2f} EUR · "
                      f"puntaje {r['puntaje']:.2f} · real: {real} · {r['motivos']}")
            impresas += 1
        hora = int(lote["tiempo_seg"].iloc[0] // 3600)
        if hora != hora_actual:
            hora_actual = hora
            print(f"── hora {hora:02d} · procesadas {sum(cont[d] for d in DECISIONES):,} · "
                  f"REVISAR {cont['REVISAR']} · BLOQUEAR {cont['BLOQUEAR']} · "
                  f"fraudes detectados {cont['detectados']}/{cont['fraudes']}")
        if velocidad:
            time.sleep(VENTANA_SEG / velocidad)
    if impresas > max_alertas:
        print(f"  … {impresas - max_alertas} alertas más (ver CSV)")
    return pd.concat(registros)


def vis_simulacion(log: pd.DataFrame) -> None:
    log = log.sort_values("tiempo_seg")
    h = log["tiempo_seg"] / 3600
    fraude = log["es_fraude"] == 1
    detectado = fraude & (log["decision"] != "APROBAR")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                   gridspec_kw={"hspace": 0.2, "height_ratios": [1.2, 1]})
    ax1.step(h[fraude], np.arange(1, fraude.sum() + 1), where="post", color=C.COLOR_FRAUDE,
             lw=2, label="Fraudes ocurridos")
    ax1.step(h[detectado], np.arange(1, detectado.sum() + 1), where="post",
             color=C.COLOR_LEGITIMA, lw=2, label="Fraudes detectados (REVISAR o BLOQUEAR)")
    for m in (fraude, detectado):
        ax1.text(h[m].iloc[-1] + 0.1, m.sum(), f"{m.sum()}", va="center", fontsize=9)
    ax1.set_ylabel("Acumulado")
    ax1.legend(loc="upper left")
    ax1.set_title("(a) Fraudes ocurridos vs. detectados a medida que llegan", loc="left",
                  fontsize=11)

    hora = (log["tiempo_seg"] // 3600).astype(int)
    alertas = (log[log["decision"] != "APROBAR"].groupby([hora, "decision"]).size()
               .unstack(fill_value=0).reindex(columns=["REVISAR", "BLOQUEAR"], fill_value=0)
               .reindex(range(hora.min(), hora.max() + 1), fill_value=0))
    base = np.zeros(len(alertas))
    for d, color in (("BLOQUEAR", C.COLOR_BLOQUEAR), ("REVISAR", C.COLOR_REVISAR)):
        ax2.bar(alertas.index + 0.5, alertas[d], bottom=base, width=0.85, color=color,
                label=d, edgecolor=C.COLOR_SUPERFICIE, linewidth=1)
        base += alertas[d].to_numpy()
    ax2.set_ylabel("Alertas por hora")
    ax2.set_xlabel("Hora desde el inicio de la captura")
    ax2.legend(loc="upper right")
    ax2.set_title("(b) Carga de trabajo: alertas emitidas por hora", loc="left", fontsize=11)
    fig.suptitle("Simulación en tiempo real sobre el período de prueba", x=0.07, ha="left",
                 fontsize=14, fontweight="bold", y=0.98)
    guardar(fig, "ml_06_simulacion_tiempo_real.png")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--casos", action="store_true", help="ejecuta los casos de prueba")
    ap.add_argument("--simular", action="store_true", help="simula el flujo del período de prueba")
    ap.add_argument("--velocidad", type=float, default=0,
                    help="segundos simulados por segundo real (0 = sin pausa)")
    ap.add_argument("--max-alertas", type=int, default=25, help="alertas a imprimir")
    args = ap.parse_args(argv)
    if not (args.casos or args.simular):
        args.casos = args.simular = True

    estilo()
    print("=== DETECTOR EN TIEMPO REAL ===")
    det = DetectorFraude()
    print(f"Modelo: {det.meta['modelo']} (entrenado {det.meta['entrenado']})")
    entreno, prueba = dividir_temporal(cargar_transacciones())
    C.DIR_REPORTES_ML.mkdir(parents=True, exist_ok=True)

    if args.casos:
        texto = ejecutar_casos(det, prueba, entreno)
        (C.DIR_REPORTES_ML / "casos_prueba.txt").write_text(texto, encoding="utf-8")

    if args.simular:
        lat = medir_latencia(det, prueba)
        log = simular(det, prueba, args.velocidad, args.max_alertas)
        alertas = log[log["decision"] != "APROBAR"]
        alertas.to_csv(C.DIR_REPORTES_ML / "simulacion_alertas.csv", index=False,
                       float_format="%.4f")
        vis_simulacion(log)
        fraude = log["es_fraude"] == 1
        det_ = fraude & (log["decision"] != "APROBAR")
        horas = (log.tiempo_seg.max() - log.tiempo_seg.min()) / 3600
        lotes = log.groupby("lote")[["latencia_lote_ms", "tamano_lote"]].first()
        resumen = f"""RESUMEN DE LA SIMULACIÓN EN TIEMPO REAL
{'=' * 70}
Período: horas {log.tiempo_seg.min() / 3600:.1f}-{log.tiempo_seg.max() / 3600:.1f} ({horas:.1f} h), {len(log):,} operaciones
en micro-lotes de {VENTANA_SEG} s ({len(lotes):,} lotes, mediana {lotes.tamano_lote.median():.0f} operaciones por lote).

Decisiones: {log['decision'].value_counts().reindex(DECISIONES, fill_value=0).to_dict()}
Alertas por hora (promedio): {len(alertas) / horas:.1f}
Fraudes: {int(fraude.sum())} ocurridos · {int(det_.sum())} detectados ({det_.sum() / fraude.sum() * 100:.0f}%) ·
         {int((fraude & (log.decision == 'BLOQUEAR')).sum())} bloqueados automáticamente
Precisión de las alertas: {alertas.es_fraude.mean() * 100:.0f}%
Monto defraudado detectado: {log.loc[det_, 'monto'].sum():,.2f} de {log.loc[fraude, 'monto'].sum():,.2f} EUR

Latencia por transacción individual ({lat['n']} llamadas a evaluar()):
  p50 = {lat['p50_ms']:.1f} ms · p95 = {lat['p95_ms']:.1f} ms · máx = {lat['max_ms']:.1f} ms
Latencia por micro-lote de 1 minuto: mediana {lotes.latencia_lote_ms.median():.0f} ms

Archivos: reportes/modelado/simulacion_alertas.csv (cada alerta con motivos),
          visualizaciones/ml_06_simulacion_tiempo_real.png
"""
        print("\n" + resumen)
        (C.DIR_REPORTES_ML / "simulacion_tiempo_real.txt").write_text(resumen, encoding="utf-8")


if __name__ == "__main__":
    main()
