import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from waste_detection.hf_app import demo


if __name__ == "__main__":
    demo.launch()
