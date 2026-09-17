import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import nbformat
from nbclient import NotebookClient

nb_path = Path(r"D:\juandmm1233\NoElectivaIII\entrenamiento_plantas_medicinales.ipynb")
nb = nbformat.read(nb_path, as_version=4)

nb.setdefault("metadata", {})
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3.10 GPU (NoElectiva)",
    "language": "python",
    "name": "noelectiva-gpu",
}

print("Ejecutando notebook...", flush=True)
client = NotebookClient(
    nb,
    timeout=7200,
    kernel_name="noelectiva-gpu",
    allow_errors=False,
    resources={"metadata": {"path": str(nb_path.parent)}},
)
client.execute()
nbformat.write(nb, nb_path)
print("Notebook ejecutado con exito", flush=True)
