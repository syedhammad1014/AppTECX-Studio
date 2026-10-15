import matplotlib
matplotlib.use("Agg")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def out_dir(week):
    p = ROOT / week / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p
