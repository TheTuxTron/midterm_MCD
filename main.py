import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import parte1_adquisicion_limpieza as p1  # noqa: E402
import parte2_eda as p2  # noqa: E402
import parte3_visualizacion as p3  # noqa: E402

PARTES = {"1": p1.main, "2": p2.main, "3": p3.main}

if __name__ == "__main__":
    for parte in sys.argv[1:] or PARTES:
        PARTES[parte]()
        print()
