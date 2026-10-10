"""Write portable notebooks using the standard library (no authoring dependency)."""
import hashlib
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = '''
from pathlib import Path
import sys

# Locate the project whether Jupyter starts in the project or notebooks folder.
start = Path.cwd().resolve()
ROOT = next((p for p in (start, *start.parents) if (p / "course_utils.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Open this notebook from inside the OpenCV Practical project.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from course_utils import show_images, make_scene, load_image, save_image

rng = np.random.default_rng(42)
cv2.setRNGSeed(42)
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})
print(f"OpenCV {cv2.__version__} | NumPy {np.__version__}")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": dedent(text).strip() + "\n", "execution_count": None, "outputs": []}


def intro(title, level, prerequisites, objectives):
    if isinstance(prerequisites, (list, tuple)):
        prerequisites = "; ".join(prerequisites)
    if isinstance(objectives, str):
        objectives = [objectives]
    objective_text = "\n".join(f"- {item}" for item in objectives)
    return [md(f"# {title}\n\n**Level:** {level}  \n**Prerequisites:** {prerequisites}\n\n## What you will build and learn\n\n{objective_text}\n\n## How to use this lesson\n\nRun cells in order with **Shift+Enter**. Predict the result before running each example, then change one parameter and explain what changed. Every default example works offline after setup. Each notebook creates its own data, so previous notebooks do not need to have been run.\n\nTry the exercises before reading the worked solutions. Use **Kernel → Restart Kernel and Run All Cells** to check that your work is reproducible. Keep your own changes in a duplicate notebook if you want to retain the reference version."), md("## Setup\n\nThe cell below imports the libraries and the small display/file helpers. Algorithms are written in the lesson cells. Color arrays passed to `show_images` are assumed to be BGR (OpenCV's usual order); the helper converts them to RGB for Matplotlib. Grayscale arrays are displayed directly. `rng` is a repeatable random generator, so running the notebook again reproduces its sample data."), code(BOOTSTRAP)]


def save(filename, cells):
    path = ROOT / "notebooks" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{filename}:{index}".encode()).hexdigest()[:12]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (OpenCV Practical)", "language": "python", "name": "opencv-practical"}, "language_info": {"name": "python", "version": "3.13", "file_extension": ".py", "mimetype": "text/x-python", "pygments_lexer": "ipython3", "codemirror_mode": {"name": "ipython", "version": 3}}}, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path.name} ({len(cells)} cells)")
