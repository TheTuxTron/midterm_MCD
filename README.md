# Detección de fraude en transacciones: limpieza, EDA y visualización

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
│   └── parte3_visualizacion.py          # 3 visualizaciones + narrativa
├── data/                                # creditcard.csv y transacciones.db
├── reportes/
│   ├── log_limpieza.txt                 # cada decisión de limpieza y su justificación
│   ├── descripcion_dataset.txt          # ¿qué contiene y cómo se describe el set?
│   ├── eda_campos.txt                   # hallazgos campo por campo
│   ├── metadata.json                    # metadatos del dataset (máquina)
│   └── metadata_campos.csv              # metadatos por campo (tabla)
└── visualizaciones/
    ├── 01_ridgeline_firma_fraude.png
    ├── 02_reloj_del_riesgo.png
    ├── 03_tsne_mapa_fraude.png
    └── historia_visualizaciones.txt     # la historia que cuentan las gráficas
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py          # todo (≈1 min)
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