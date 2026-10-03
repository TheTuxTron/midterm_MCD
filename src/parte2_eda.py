"""
Parte 2 - Análisis exploratorio (EDA).

* Lee la tabla limpia desde SQLite y la convierte en DataFrame de pandas.
* Describe cada campo (estadística + comparación fraude vs. legítima).
* Genera meta-datos a nivel dataset y a nivel campo.
* Escribe los resultados en `reportes/`:
    - descripcion_dataset.txt  ¿qué contiene y cómo se describe el set?
    - eda_campos.txt           hallazgos campo por campo
    - metadata.json            meta-datos legibles por máquina
    - metadata_campos.csv      meta-datos por campo en tabla
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import config as C

FUENTE = {
    "nombre": "Credit Card Fraud Detection",
    "kaggle": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    "autores": "Machine Learning Group - Université Libre de Bruxelles (ULB) y Worldline",
    "licencia": "Database Contents License (DbCL) v1.0",
    "cita": ("Dal Pozzolo, A., Caelen, O., Johnson, R. A. & Bontempi, G. (2015). "
             "Calibrating Probability with Undersampling for Unbalanced "
             "Classification. IEEE SSCI."),
    "periodo": "Dos días de septiembre de 2013, titulares europeos",
    "notas_documentacion": [
        "V1..V28 son componentes principales (PCA); las variables originales "
        "no se publican por confidencialidad.",
        "Time y Amount son las únicas variables sin transformar por PCA.",
        "Time = segundos transcurridos entre cada operación y la primera del set.",
        "Class = 1 en caso de fraude, 0 en otro caso.",
        "Por el desbalance se recomienda evaluar con AUPRC, no con exactitud.",
    ],
}


# --------------------------------------------------------------------------
# Lectura desde SQLite
# --------------------------------------------------------------------------
def leer_desde_sqlite() -> pd.DataFrame:
    with sqlite3.connect(C.DB_PATH) as con:
        df = pd.read_sql_query(f"SELECT * FROM {C.TABLA_TRANSACCIONES}", con)
    print(f"[sqlite] DataFrame leído: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


# --------------------------------------------------------------------------
# Perfil por campo
# --------------------------------------------------------------------------
def rol_del_campo(col: str) -> str:
    if col == "id_transaccion":
        return "identificador"
    if col == "es_fraude":
        return "objetivo"
    if col.startswith("es_"):
        return "bandera derivada"
    if col in ("dia", "hora_dia", "monto_log"):
        return "derivada"
    if col.startswith("V"):
        return "predictora (PCA anonimizada)"
    return "predictora (original)"


def perfil_campos(df: pd.DataFrame) -> pd.DataFrame:
    fraude = df["es_fraude"] == 1
    filas = []
    for col in df.columns:
        s = df[col]
        fila = dict(
            campo=col, rol=rol_del_campo(col), tipo=str(s.dtype),
            n=int(s.count()), nulos=int(s.isna().sum()),
            unicos=int(s.nunique()), min=s.min(), p25=s.quantile(.25),
            mediana=s.median(), p75=s.quantile(.75), max=s.max(),
            media=s.mean(), desv_std=s.std(), asimetria=s.skew(),
            curtosis=s.kurt(),
        )
        if col not in ("id_transaccion", "es_fraude"):
            a, b = s[fraude], s[~fraude]
            ks = stats.ks_2samp(a, b)
            sd_pool = np.sqrt((a.var() + b.var()) / 2)
            fila.update(
                media_fraude=a.mean(), media_legitima=b.mean(),
                mediana_fraude=a.median(), mediana_legitima=b.median(),
                ks_estadistico=ks.statistic, ks_pvalor=ks.pvalue,
                d_cohen=(a.mean() - b.mean()) / sd_pool if sd_pool else 0.0,
                corr_con_fraude=s.corr(df["es_fraude"]),
            )
        filas.append(fila)
    return pd.DataFrame(filas)


def fuerza(ks: float) -> str:
    if ks >= 0.6:
        return "muy alta"
    if ks >= 0.4:
        return "alta"
    if ks >= 0.2:
        return "moderada"
    return "baja"


def forma(asim: float) -> str:
    if abs(asim) < 0.5:
        return "aproximadamente simétrica"
    lado = "derecha" if asim > 0 else "izquierda"
    return ("fuertemente " if abs(asim) > 2 else "moderadamente ") + f"sesgada a la {lado}"


def interpretar(r: pd.Series) -> str:
    """Texto explicativo por campo, combinando reglas genéricas y conocimiento
    de la documentación del dataset."""
    c = r["campo"]
    if c == "id_transaccion":
        return ("Identificador secuencial creado en la limpieza; no es una variable "
                "de comportamiento y no debe usarse para modelar.")
    if c == "es_fraude":
        tasa = r["media"] * 100
        return (f"Variable objetivo binaria. Solo el {tasa:.3f}% de las operaciones "
                f"es fraude (≈1 de cada {1 / r['media']:,.0f}). El set está "
                "extremadamente desbalanceado: un modelo que siempre responda "
                "'legítima' tendría >99.8% de exactitud y no detectaría nada, por "
                "eso la documentación recomienda AUPRC.")
    base = (f"Distribución {forma(r['asimetria'])} (asimetría {r['asimetria']:.2f}, "
            f"curtosis {r['curtosis']:.1f}). Rango [{r['min']:.2f}, {r['max']:.2f}]. "
            f"Media fraude = {r['media_fraude']:.3f} vs. legítima = "
            f"{r['media_legitima']:.3f}; KS = {r['ks_estadistico']:.3f} → capacidad "
            f"de separación {fuerza(r['ks_estadistico'])}.")
    if c == "tiempo_seg":
        return (base + " No es una fecha: son segundos desde la primera operación "
                "(0 a ~172 800 s = 48 h). Su valor por sí mismo no generaliza; lo útil "
                "es el ciclo diario (ver hora_dia).")
    if c == "monto":
        return (base + " Montos en euros con cola larga: la mayoría son pequeños y "
                "pocos son muy grandes. Los fraudes tienen mediana "
                f"{r['mediana_fraude']:.2f} vs. {r['mediana_legitima']:.2f} en "
                "legítimas: el fraude no se concentra en montos grandes, sino que "
                "abunda en montos chicos (pruebas) y en algunos medianos.")
    if c == "monto_log":
        return base + " Transformación log que hace manejable la cola larga de monto."
    if c == "hora_dia":
        return (base + " Hora relativa al inicio de la captura (no necesariamente "
                "la hora del reloj). Las legítimas muestran un ciclo día/noche claro; "
                "el fraude es más uniforme, por lo que su TASA sube en horas de bajo "
                "volumen.")
    if c == "dia":
        return base + " Solo hay dos días; sirve para comparar la estabilidad del patrón."
    if c == "es_monto_cero":
        return (base + f" {int(r['media'] * r['n']):,} operaciones con monto 0 "
                "(verificaciones / pruebas de credenciales).")
    if c == "es_atipico_monto":
        return base + " Bandera IQR; ~11% de operaciones, reflejo de la cola larga."
    extra = ""
    if r["ks_estadistico"] >= 0.4:
        signo = "menores" if r["media_fraude"] < r["media_legitima"] else "mayores"
        extra = (f" En fraudes toma valores sistemáticamente {signo}: es una de las "
                 "variables más informativas del set.")
    return (base + " Componente PCA centrada en 0 (media global "
            f"{r['media']:.1e}); su significado de negocio es desconocido por "
            "confidencialidad." + extra)


# --------------------------------------------------------------------------
# Meta-datos
# --------------------------------------------------------------------------
def md5(path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def metadata_dataset(df: pd.DataFrame, perfil: pd.DataFrame) -> dict:
    with sqlite3.connect(C.DB_PATH) as con:
        log = pd.read_sql_query(f"SELECT * FROM {C.TABLA_LOG_LIMPIEZA}", con)
    n_fraude = int(df["es_fraude"].sum())
    top = (perfil.dropna(subset=["ks_estadistico"])
           .sort_values("ks_estadistico", ascending=False).head(8))
    return {
        "fuente": FUENTE,
        "generado": datetime.now().isoformat(timespec="seconds"),
        "archivo_crudo": {"ruta": str(C.CSV_CRUDO.relative_to(C.RAIZ)),
                          "bytes": C.CSV_CRUDO.stat().st_size if C.CSV_CRUDO.exists() else None,
                          "md5": md5(C.CSV_CRUDO)},
        "base_datos": {"ruta": str(C.DB_PATH.relative_to(C.RAIZ)),
                       "bytes": C.DB_PATH.stat().st_size,
                       "tablas": [C.TABLA_TRANSACCIONES, C.TABLA_DICCIONARIO,
                                  C.TABLA_LOG_LIMPIEZA]},
        "estructura": {"filas": len(df), "columnas": df.shape[1],
                       "memoria_mb": round(df.memory_usage(deep=True).sum() / 2**20, 1),
                       "nulos_totales": int(df.isna().sum().sum()),
                       "tipos": df.dtypes.astype(str).value_counts().to_dict()},
        "limpieza": {"filas_crudas": int(log["filas_antes"].iloc[0]),
                     "filas_limpias": int(log["filas_despues"].iloc[-1]),
                     "pasos": log["paso"].tolist()},
        "objetivo": {"campo": "es_fraude", "fraudes": n_fraude,
                     "legitimas": len(df) - n_fraude,
                     "tasa_fraude_pct": round(100 * n_fraude / len(df), 4),
                     "ratio_desbalance": f"1:{round((len(df) - n_fraude) / n_fraude)}"},
        "cobertura_temporal": {"segundos_min": float(df["tiempo_seg"].min()),
                               "segundos_max": float(df["tiempo_seg"].max()),
                               "horas": round(df["tiempo_seg"].max() / 3600, 1),
                               "dias": int(df["dia"].nunique())},
        "monto": {"total_eur": round(float(df["monto"].sum()), 2),
                  "total_fraude_eur": round(float(df.loc[df.es_fraude == 1, "monto"].sum()), 2),
                  "max_eur": float(df["monto"].max())},
        "variables_mas_discriminantes_ks": top[["campo", "ks_estadistico", "d_cohen"]]
            .round(3).to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# Reportes de texto
# --------------------------------------------------------------------------
def escribir_eda_campos(perfil: pd.DataFrame) -> None:
    lineas = ["DESCRIPCIÓN CAMPO POR CAMPO", "=" * 70,
              "KS = estadístico de Kolmogorov-Smirnov entre fraude y legítima "
              "(0 = distribuciones iguales, 1 = totalmente separadas).", ""]
    for _, r in perfil.iterrows():
        lineas += [f"■ {r['campo']}  [{r['rol']} · {r['tipo']}]",
                   f"  n={r['n']:,}  nulos={r['nulos']}  únicos={r['unicos']:,}  "
                   f"media={r['media']:.4g}  mediana={r['mediana']:.4g}  "
                   f"desv={r['desv_std']:.4g}",
                   "  " + interpretar(r), ""]
    (C.DIR_REPORTES / "eda_campos.txt").write_text("\n".join(lineas), encoding="utf-8")


def escribir_descripcion(df: pd.DataFrame, perfil: pd.DataFrame, meta: dict) -> None:
    o, t, m = meta["objetivo"], meta["cobertura_temporal"], meta["monto"]
    top = meta["variables_mas_discriminantes_ks"]
    top_txt = ", ".join(f"{d['campo']} (KS {d['ks_estadistico']:.2f})" for d in top[:6])
    debiles = perfil[(perfil["ks_estadistico"] < 0.1)
                     & perfil["campo"].str.match(r"V\d+$")]["campo"].tolist()
    por_hora = df.groupby("hora_dia")["es_fraude"].agg(["mean", "size"])
    h_max, h_min_vol = por_hora["mean"].idxmax(), por_hora["size"].idxmin()
    lim = meta["limpieza"]

    texto = f"""DESCRIPCIÓN DEL SET DE DATOS
{'=' * 70}

1. ¿Qué es?
-----------
"{FUENTE['nombre']}" ({FUENTE['autores']}). Son {lim['filas_crudas']:,} operaciones
reales con tarjeta realizadas por titulares europeos durante dos días de
septiembre de 2013. Tras la limpieza quedan {meta['estructura']['filas']:,} filas.

En este proyecto se usa como ANÁLOGO de un flujo de transferencias
interbancarias: cada fila es una operación con instante, monto, un vector de
atributos de comportamiento anonimizados y una etiqueta de fraude. Las
conclusiones metodológicas (desbalance, patrones horarios, montos de prueba,
variables latentes) se trasladan directamente a ese dominio.

2. ¿Qué información contiene?
-----------------------------
* tiempo_seg (Time): segundos desde la primera operación; cubre {t['horas']} h.
* monto (Amount): importe en euros, sin escalar. Total movido:
  {m['total_eur']:,.2f} EUR, de los cuales {m['total_fraude_eur']:,.2f} EUR son fraude.
* V1..V28: 28 componentes principales (PCA) de variables confidenciales. No
  sabemos qué representan (podrían ser país, canal, comercio, historial…),
  sólo que están centradas en 0, no correlacionadas entre sí y ordenadas por
  varianza explicada (V1 es la de mayor varianza).
* es_fraude (Class): {o['fraudes']:,} fraudes vs. {o['legitimas']:,} legítimas
  ({o['tasa_fraude_pct']}% → desbalance {o['ratio_desbalance']}).
* Derivadas en la limpieza: dia, hora_dia, monto_log, es_monto_cero,
  es_atipico_monto, id_transaccion.

3. ¿Cómo se puede describir?
----------------------------
* Es un set TABULAR, NUMÉRICO, SIN NULOS, de clasificación BINARIA
  EXTREMADAMENTE DESBALANCEADA y PRE-ANONIMIZADO. Casi todo el trabajo de
  ingeniería de atributos ya lo hizo el proveedor (PCA), lo que limita la
  interpretabilidad de negocio pero protege la privacidad.
* Contenía {lim['filas_crudas'] - lim['filas_limpias']:,} registros duplicados
  exactos que se eliminaron (posibles dobles registros).
* La señal de fraude se concentra en pocas componentes: {top_txt}.
  Componentes como {', '.join(debiles[:8])} apenas diferencian clases.
* El monto NO es un buen discriminador aislado: el fraude es frecuente en
  montos pequeños (pruebas) y la mediana de monto en fraudes es menor que en
  legítimas.
* El tiempo muestra un ciclo día/noche: el volumen legítimo cae de madrugada
  mientras que el fraude se mantiene, por lo que la tasa de fraude es máxima
  en la hora relativa {h_max} (la de menor volumen es la {h_min_vol}).
* Las componentes PCA tienen colas muy pesadas (curtosis alta): los valores
  extremos casi siempre corresponden a fraudes, lo que sugiere modelos
  robustos a atípicos (árboles) o escalado robusto.

4. Implicaciones para un sistema antifraude interbancario
---------------------------------------------------------
* Evaluar con precisión/recall y AUPRC, nunca con exactitud.
* Usar re-muestreo o pesos de clase y validación estratificada.
* Incluir atributos de hora/ventana (las operaciones fuera de horario son
  más riesgosas proporcionalmente).
* Vigilar operaciones de monto 0 o muy bajo como posibles pruebas.
* Al no tener identificador de cliente/cuenta, no es posible construir
  atributos de historial (velocidad, cambios de destino), que en banca son
  claves: es la principal limitación del análogo.
"""
    (C.DIR_REPORTES / "descripcion_dataset.txt").write_text(texto, encoding="utf-8")


def main() -> pd.DataFrame:
    print("=== PARTE 2: EDA ===")
    C.DIR_REPORTES.mkdir(parents=True, exist_ok=True)
    df = leer_desde_sqlite()
    print(df.head())
    perfil = perfil_campos(df)
    meta = metadata_dataset(df, perfil)

    perfil.to_csv(C.DIR_REPORTES / "metadata_campos.csv", index=False, float_format="%.6g")
    meta["campos"] = (perfil[["campo", "rol", "tipo", "n", "nulos", "unicos", "min",
                              "max", "media", "desv_std"]]
                      .round(6).to_dict(orient="records"))
    (C.DIR_REPORTES / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    escribir_eda_campos(perfil)
    escribir_descripcion(df, perfil, meta)
    print(f"[reportes] Escritos en {C.DIR_REPORTES.relative_to(C.RAIZ)}/")
    return df


if __name__ == "__main__":
    main()
