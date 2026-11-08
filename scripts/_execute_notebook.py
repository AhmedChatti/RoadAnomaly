"""Execute the exploration notebook and save the outputs into the file."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks" / "01_exploration_and_preprocessing.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(root)}},
)
client.execute()
nbformat.write(notebook, path)
print("notebook executed", path)
