from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIR_REPORTES = RAIZ / "reportes"
DIR_VIS = RAIZ / "visualizaciones"
DIR_MODELOS = RAIZ / "modelos"
DIR_REPORTES_ML = DIR_REPORTES / "modelado"

CSV_CRUDO = DIR_DATOS / "creditcard.csv"
DB_PATH = DIR_DATOS / "transacciones.db"
MODELO_PATH = DIR_MODELOS / "detector_fraude.joblib"
OOF_PATH = DIR_MODELOS / "oof_candidatos.npz"

TABLA_TRANSACCIONES = "transacciones"
TABLA_DICCIONARIO = "diccionario_datos"
TABLA_LOG_LIMPIEZA = "log_limpieza"

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
URL_ESPEJO = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# Columnas originales del dataset de Kaggle
COLS_PCA = [f"V{i}" for i in range(1, 29)]
COLS_ORIGINALES = ["Time", *COLS_PCA, "Amount", "Class"]

# Renombrado al dominio interbancario
RENOMBRAR = {"Time": "tiempo_seg", "Amount": "monto", "Class": "es_fraude"}

SEGUNDOS_DIA = 86_400

COLOR_LEGITIMA = "#2a78d6"
COLOR_FRAUDE = "#eb6834"
COLOR_SUPERFICIE = "#fcfcfb"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_2 = "#52514e"
COLOR_GRID = "#e4e3df"
SECUENCIAL_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SECUENCIAL_NARANJA = ["#fde3d6", "#f9bfa4", "#f39a72", "#eb6834", "#c94e1d", "#9e3a12", "#6f2709"]

# Colores de decisión del detector (siempre acompañados de etiqueta)
COLOR_APROBAR = "#0ca30c"
COLOR_REVISAR = "#fab219"
COLOR_BLOQUEAR = "#d03b3b"
COLOR_NEUTRO = "#898781"
DIVERGENTE = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f5b5b3", "#e34948", "#a32a29"]
