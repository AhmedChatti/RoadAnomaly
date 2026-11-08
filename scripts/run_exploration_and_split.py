"""Create the exploration figures and the train / validation / test set."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_anomaly import run

if __name__ == "__main__":
    summary = run(
        dataset_root=ROOT,
        output_dir=ROOT / "data" / "processed",
        figure_dir=ROOT / "reports" / "figures",
        table_dir=ROOT / "reports" / "tables",
    )
    printable = {key: value for key, value in summary.items() if key != "figures"}
    print(json.dumps(printable, indent=2))
    print("figures:")
    for name, path in summary["figures"].items():
        print(f"  {name}: {path}")
