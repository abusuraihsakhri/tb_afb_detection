from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "02_CODE" / "src"
sys.path.insert(0, str(SRC))
