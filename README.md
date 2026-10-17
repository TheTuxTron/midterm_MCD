# Detección de fraude en transacciones: limpieza, EDA, visualización y modelado

Este proyecto analiza el dataset **Credit Card Fraud Detection** (Kaggle
`mlg-ulb/creditcardfraud`, ULB / Worldline) como **análogo de un flujo de
transacciones interbancarias**. Las dos cosas comparten la misma estructura:
cada operación tiene un instante, un monto, atributos de comportamiento
(aquí anonimizados con PCA) y una etiqueta de fraude que aparece muy pocas veces.
Por eso los campos se renombran al vocabulario bancario
(`Time → tiempo_seg`, `Amount → monto`, `Class → es_fraude`).

## Estructura

```
├── main.py                              # corre el programa
├── requirements.txt
├── src/
│   ├── config.py                        # rutas, nombres, paleta
│   ├── parte1_adquisicion_limpieza.py   # lectura, limpieza, carga a SQLite
│   ├── parte2_eda.py                    # SQLite -> pandas, perfil por campo, metadatos
│   ├── parte3_visualizacion.py          # 3 visualizaciones + narrativa
│   ├── ml_comun.py                      # variables, partición temporal, pipelines
│   ├── ml_parte0_preparacion.py         # modelado 0: carga, verificación, tendencia
│   ├── ml_parte1_clasificacion.py       # modelado 1: Normal vs. Fraudulenta
│   ├── ml_parte2_segmentacion.py        # modelado 2: K-means vs. Mean Shift
│   ├── ml_parte3_prediccion.py          # modelado 3: predicción, umbrales, modelo final
│   └── detector_tiempo_real.py          # detector, casos de prueba y simulación
├── tests/test_detector.py               # pruebas automáticas (pytest)
├── modelos/
│   ├── detector_fraude.joblib           # pipeline + umbrales listos para usar
│   └── detector_fraude_meta.json
├── data/                                # creditcard.csv y transacciones.db
├── reportes/
│   ├── log_limpieza.txt                 # cada decisión de limpieza y su justificación
│   ├── descripcion_dataset.txt          # ¿qué contiene y cómo se describe el set?
│   ├── eda_campos.txt                   # hallazgos campo por campo
│   ├── metadata.json                    # metadatos del dataset (máquina)
│   ├── metadata_campos.csv              # metadatos por campo (tabla)
│   └── modelado/                        # un informe por parte + casos y simulación
└── visualizaciones/
    ├── 01_ridgeline_firma_fraude.png
    ├── 02_reloj_del_riesgo.png
    ├── 03_tsne_mapa_fraude.png
    ├── historia_visualizaciones.txt     # la historia que cuentan las gráficas
    └── ml_00 … ml_06_*.png              # gráficas del modelado
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py                      # todo (≈6 min)
python main.py ml1 ml3              # sólo re-entrenar el modelo
python main.py tr                   # casos de prueba + simulación en tiempo real
python src/detector_tiempo_real.py --simular --velocidad 600   # demo en vivo: 1 h = 6 s
python -m pytest -v                 # pruebas automáticas
```

**Adquisición de datos.** El script prueba tres fuentes en este orden:
1. `data/creditcard.csv`, si ya existe;
2. Kaggle con `kagglehub` (necesita credenciales en `~/.kaggle/kaggle.json`);
3. el espejo público del mismo archivo que usa TensorFlow
   (`storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`).


## Parte 1: Adquisición y limpieza

| # | Paso | Decisión | Por qué |
|---|------|----------|---------|
| 1 | Esquema | Se exigen las 31 columnas documentadas | Si el archivo cambia, el script falla al inicio en vez de seguir con datos incorrectos |
| 2 | Tipos | Conversión numérica forzada | Así un valor corrupto aparece como nulo |
| 3 | Nulos | Se eliminan (hay 0) | Ni la etiqueta ni las componentes PCA se pueden imputar sin introducir sesgo |
| 4 | Dominio | `Class∈{0,1}`, `Amount≥0`, `0≤Time≤48h` | Son los rangos que da la documentación |
| 5 | Duplicados | **Se eliminan 1,081** (19 fraudes) | Que 31 columnas continuas coincidan exactamente sólo pasa con un doble registro. Dejarlas infla los conteos y mezcla datos entre entrenamiento y prueba |
| 6 | Monto 0 | Se **marcan** (1,808) y no se eliminan | Suelen ser verificaciones o pruebas de credenciales, así que traen información sobre el fraude |
| 7 | Atípicos | Se **marcan** con IQR y no se eliminan | En detección de fraude, lo que buscamos son justamente las anomalías |
| 8 | Derivadas | `dia`, `hora_dia`, `monto_log`, `id_transaccion` | El tiempo sólo sirve como ciclo horario, y el monto tiene una cola muy larga |

La base `data/transacciones.db` tiene tres tablas: `transacciones` (con
índices por id, por fraude y por día/hora), `diccionario_datos` y
`log_limpieza`, donde queda la trazabilidad de la limpieza.

## Parte 2: EDA

* Lee la tabla con `pd.read_sql_query` y la convierte en un DataFrame.
* Por cada campo calcula su rol, tipo, conteos, cuantiles, asimetría, curtosis
  y una comparación entre fraude y legítima (medias, medianas, **KS**,
  **d de Cohen**, correlación). Con eso escribe una interpretación en texto.
* Genera metadatos del dataset: fuente, licencia, cita, hash MD5 del archivo
  crudo, tamaño, cobertura temporal, desbalance, montos totales y ranking de
  variables discriminantes. También genera metadatos por campo.

**Hallazgos principales:** 473 fraudes entre 283,726 operaciones (0.167 %,
proporción 1:599). La señal se concentra en V14, V10, V12, V4, V11 y V17
(KS 0.74–0.84). El monto por sí solo discrimina poco: la mediana del fraude
(9.82 €) es *menor* que la de las legítimas (22 €). De madrugada la tasa de
fraude se multiplica por 5.

## Parte 3: Visualización

1. **Ridgeline de densidades KDE.** Muestra la "firma" del fraude en las
   componentes PCA: un pico legítimo estrecho frente a fraudes desplazados y
   muy dispersos.
2. **Reloj del riesgo.** Combina gráficos polares de 24 h con un mapa de calor
   día × hora. Las legítimas siguen un ritmo humano y el fraude no, por lo que
   la tasa llega a 14.5 ‰ a la hora 2.
3. **Mapa t-SNE.** En la proyección no lineal, el 80 % de los fraudes vive en
   regiones donde predominan otros fraudes. El problema es en buena parte
   separable, aunque algunos fraudes quedan "camuflados".

La narrativa completa está en `visualizaciones/historia_visualizaciones.txt`.

---

# Modelado con scikit-learn

Todo el modelado usa scikit-learn: `Pipeline`, `FunctionTransformer`,
`StandardScaler`, `StratifiedKFold`, `RandomForestClassifier`,
`LogisticRegression`, `HistGradientBoostingClassifier`, `KMeans`, `MeanShift`
y `permutation_importance`. Se adaptó el enunciado (Online Retail II) a fraude
transaccional:

| Enunciado original | Adaptación a fraude |
|---|---|
| Parte 0: limpieza y EDA de ventas | Se reutiliza la limpieza y el EDA anteriores, y se agrega la tendencia temporal y la partición |
| Parte 1: clientes Normales vs. Premium | Transacción **Normal vs. Fraudulenta** |
| Parte 2: segmentar clientes | Segmentar **transacciones** (el set no tiene ID de cliente) |
| Parte 3: predecir ventas futuras | Predecir el fraude en **operaciones futuras** y operar un detector |

**Partición temporal:** se entrena con el 75 % más antiguo (212,794
operaciones, 380 fraudes) y se prueba con el 25 % más reciente (70,932
operaciones, 93 fraudes), como pasaría al poner el modelo en producción.

## Modelado 0: Preparación
Se carga la tabla limpia desde SQLite, se verifica (sin nulos, sin
duplicados, orden cronológico) y se completa el EDA. Los "productos más
vendidos" se adaptan a rangos de monto: el 36 % de los fraudes es de 0 a 1 €,
típicas pruebas de credenciales. Los "patrones estacionales" se adaptan al
ciclo diario: el fraude llega en ráfagas que no siguen el ritmo humano.

## Modelado 1: Clasificación Normal vs. Fraudulenta
Se agrega la columna `clase_transaccion` a partir de la investigación del
banco (`es_fraude`). Se comparan tres modelos con validación cruzada
estratificada de 5 pliegues sobre el entrenamiento, compensando el desbalance
con pesos de clase:

| Modelo | AUPRC (CV) |
|---|---|
| Línea base (siempre "Normal") | 0.002, con 99.8 % de exactitud y 0 fraudes detectados |
| Regresión logística | 0.708 |
| Gradient boosting (HGB) | 0.731 |
| **Random forest** | **0.848** ← elegido |

## Modelado 2: Segmentación (K-means vs. Mean Shift)
* **K-means** (k = 4, elegido por silhouette) aísla, **sin ver la etiqueta**,
  un segmento de 291 operaciones con 86 % de fraude que reúne el 53 % de
  todos los fraudes.
* **Mean Shift** encuentra un modo "normal" con el 96 % de las operaciones y
  41 clusters periféricos que concentran el 89 % del fraude. Se comporta como
  un detector de anomalías.
* La coincidencia entre ambas particiones es baja (ARI = 0.02). K-means sirve
  para perfilar y Mean Shift para separar lo atípico.

## Modelado 3: Predicción en operaciones futuras
Variables nuevas, calculadas dentro del pipeline: hora cíclica (seno y
coseno), franja de madrugada, `log(monto)` y monto cero. Los umbrales se
eligen con predicciones fuera de pliegue del entrenamiento, **sin mirar la
prueba**:

| Decisión | Regla |
|---|---|
| **BLOQUEAR** | puntaje ≥ 0.38 (precisión ≥ 90 % en entrenamiento) |
| **REVISAR** | puntaje ≥ 0.03 (precisión ≥ 50 % en entrenamiento) |
| APROBAR | resto |

**Resultados en el futuro (prueba):** AUPRC 0.809 · ROC-AUC 0.946.

| | Normal | Fraudulenta |
|---|---|---|
| APROBAR | 70,699 | 16 |
| REVISAR | 139 | 9 |
| BLOQUEAR | **1** | **68** |

* Se detecta el 83 % de los fraudes (77 de 93) y el 78 % del monto defraudado.
* El bloqueo automático tiene 99 % de precisión.
* La precisión de las alertas de revisión cae de 50 % a 35 % en el futuro
  (deriva temporal), así que los umbrales deben recalibrarse periódicamente.

## Detección en tiempo real
`src/detector_tiempo_real.py` carga `modelos/detector_fraude.joblib` y evalúa
cada transacción cruda (`tiempo_seg`, `monto`, `V1..V28`):

```python
import sys; sys.path.insert(0, "src")
from detector_tiempo_real import DetectorFraude

det = DetectorFraude()
tx = {"tiempo_seg": 140_000.0, "monto": 120.54, **{f"V{i}": 0.0 for i in range(1, 29)}}
r = det.evaluar(tx)
print(r.puntaje, r.decision, r.motivos, f"{r.latencia_ms:.1f} ms")
# 0.0 APROBAR [] 16.2 ms   (un fraude real devuelve p. ej. 0.73 BLOQUEAR
#                            ['V14 bajo (z=-8.9)', 'V17 bajo (z=-7.8)', ...])
```

* **Casos de prueba** (`reportes/modelado/casos_prueba.txt`): operación
  típica → APROBAR; fraude real → BLOQUEAR con motivos; fraude "camuflado" →
  APROBAR (limitación conocida); ataque sintético → REVISAR; datos
  inválidos → rechazo controlado.
* **Simulación:** reproduce las 9.3 h del período de prueba en micro-lotes de
  1 minuto e imprime cada alerta con sus motivos y si era fraude real. La
  latencia es de ~16 ms por transacción (p95 ≈ 25 ms).
* **Pruebas automáticas:** `python -m pytest -v` corre 13 pruebas (variables,
  política, casos, entradas inválidas, latencia y un rendimiento mínimo
  exigido).

