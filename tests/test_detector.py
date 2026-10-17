"""Pruebas automáticas del detector de fraude.

Ejecutar desde la raíz del proyecto (después de `python main.py`):
    python -m pytest -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config as C  # noqa: E402
from ml_comun import (COLS_ENTRADA, cargar_transacciones, decidir,  # noqa: E402
                      derivar_variables, dividir_temporal)

requiere_modelo = pytest.mark.skipif(
    not (C.MODELO_PATH.exists() and C.DB_PATH.exists()),
    reason="Falta el modelo o la base: ejecuta `python main.py` primero")


# ------------------------------------------------------------ sin modelo
def tx_base(**cambios) -> dict:
    tx = {c: 0.0 for c in COLS_ENTRADA} | {"tiempo_seg": 12 * 3600.0, "monto": 25.0}
    return tx | cambios


def test_hora_es_ciclica():
    X = pd.DataFrame([tx_base(tiempo_seg=23.5 * 3600), tx_base(tiempo_seg=0.5 * 3600),
                      tx_base(tiempo_seg=12 * 3600)])
    D = derivar_variables(X)
    pares = D[["hora_sin", "hora_cos"]].to_numpy()
    assert np.linalg.norm(pares[0] - pares[1]) < np.linalg.norm(pares[0] - pares[2])


def test_variables_derivadas():
    D = derivar_variables(pd.DataFrame([tx_base(monto=0.0, tiempo_seg=86_400 + 3 * 3600)]))
    assert D.loc[0, "es_monto_cero"] == 1
    assert D.loc[0, "es_madrugada"] == 1
    assert D.loc[0, "monto_log"] == 0
    assert list(D.columns[:28]) == C.COLS_PCA


def test_politica_de_decision():
    u = {"revisar": 0.1, "bloquear": 0.5}
    assert list(decidir([0.0, 0.0999, 0.1, 0.49, 0.5, 1.0], u)) == \
        ["APROBAR", "APROBAR", "REVISAR", "REVISAR", "BLOQUEAR", "BLOQUEAR"]


# ------------------------------------------------------------ con modelo
@pytest.fixture(scope="module")
def entorno():
    from detector_tiempo_real import DetectorFraude
    det = DetectorFraude()
    entreno, prueba = dividir_temporal(cargar_transacciones())
    puntajes = det.evaluar_lote(prueba)["puntaje"]
    return det, entreno, prueba, puntajes


def a_dict(fila, det):
    return {c: float(fila[c]) for c in det.columnas}


@requiere_modelo
def test_umbrales_coherentes(entorno):
    det, *_ = entorno
    assert 0 < det.umbrales["revisar"] < det.umbrales["bloquear"] < 1


@requiere_modelo
def test_operacion_tipica_se_aprueba(entorno):
    det, _, prueba, puntajes = entorno
    legit = prueba[prueba.es_fraude == 0]
    p = puntajes.loc[legit.index]
    tipica = legit.loc[(p - p.median()).abs().idxmin()]
    assert det.evaluar(a_dict(tipica, det)).decision == "APROBAR"


@requiere_modelo
def test_fraude_claro_se_bloquea_con_motivos(entorno):
    det, _, prueba, puntajes = entorno
    fraudes = prueba[prueba.es_fraude == 1]
    claro = fraudes.loc[puntajes.loc[fraudes.index].idxmax()]
    r = det.evaluar(a_dict(claro, det))
    assert r.decision == "BLOQUEAR"
    assert r.motivos


@requiere_modelo
def test_ataque_sintetico_genera_alerta(entorno):
    det, entreno, prueba, _ = entorno
    perfil = entreno.loc[entreno.es_fraude == 1, ["V14", "V12", "V10", "V17", "V4"]].median()
    normal = prueba[prueba.es_fraude == 0].iloc[0]
    r = det.evaluar(a_dict(normal, det) | perfil.to_dict())
    assert r.decision in ("REVISAR", "BLOQUEAR")


@pytest.mark.parametrize("cambio, mensaje", [
    ({"monto": -10.0}, "negativo"),
    ({"V14": "abc"}, "no numéricos"),
    ({"V14": None}, "no numéricos"),
])
@requiere_modelo
def test_entradas_invalidas_se_rechazan(entorno, cambio, mensaje):
    from detector_tiempo_real import TransaccionInvalida
    det, *_ = entorno
    with pytest.raises(TransaccionInvalida, match=mensaje):
        det.evaluar(tx_base(**cambio))


@requiere_modelo
def test_campo_faltante_se_rechaza(entorno):
    from detector_tiempo_real import TransaccionInvalida
    det, *_ = entorno
    tx = tx_base()
    tx.pop("V14")
    with pytest.raises(TransaccionInvalida, match="Faltan campos: V14"):
        det.evaluar(tx)


@requiere_modelo
def test_latencia_tiempo_real(entorno):
    det, _, prueba, _ = entorno
    lat = [det.evaluar(a_dict(f, det)).latencia_ms
           for _, f in prueba.sample(50, random_state=1).iterrows()]
    assert np.percentile(lat, 95) < 200


@requiere_modelo
def test_rendimiento_minimo_en_el_futuro(entorno):
    """Guarda contra regresiones: si un cambio empeora el modelo, la prueba falla."""
    det, _, prueba, puntajes = entorno
    y = prueba.es_fraude.to_numpy()
    alerta = puntajes.to_numpy() >= det.umbrales["revisar"]
    bloqueo = puntajes.to_numpy() >= det.umbrales["bloquear"]
    assert y[alerta].sum() / y.sum() >= 0.75        # recall de alertas
    assert y[bloqueo].mean() >= 0.90                # precisión del bloqueo
