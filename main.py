import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import parte1_adquisicion_limpieza as p1  # noqa: E402
import parte2_eda as p2  # noqa: E402
import parte3_visualizacion as p3  # noqa: E402
import ml_parte0_preparacion as ml0  # noqa: E402
import ml_parte1_clasificacion as ml1  # noqa: E402
import ml_parte2_segmentacion as ml2  # noqa: E402
import ml_parte3_prediccion as ml3  # noqa: E402
import detector_tiempo_real as tr  # noqa: E402

PARTES = {"1": p1.main, "2": p2.main, "3": p3.main,
          "ml0": ml0.main, "ml1": ml1.main, "ml2": ml2.main, "ml3": ml3.main,
          "tr": lambda: tr.main([])}

if __name__ == "__main__":
    for parte in sys.argv[1:] or PARTES:
        PARTES[parte]()
        print()
