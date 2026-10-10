"""Small display/file helpers. The computer-vision algorithms live in the lessons."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_images(images, titles=None, cols=3, cmap="gray", figsize=None):
    """Display BGR color images or grayscale arrays in a notebook; never opens a GUI."""
    if isinstance(images, np.ndarray):
        images = [images]
    images = list(images)
    if not images:
        raise ValueError("Pass at least one image.")
    titles = [""] * len(images) if titles is None else list(titles)
    if len(titles) != len(images):
        raise ValueError("Provide one title per image.")
    cols = min(max(int(cols), 1), len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (5 * cols, 3.6 * rows), squeeze=False)
    for ax, image, title in zip(axes.flat, images, titles):
        if image is None or image.size == 0:
            raise ValueError("Cannot display a missing or empty image.")
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
        elif image.ndim == 3 and image.shape[2] == 4:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        else:
            # Fix uint8 display range so brightness comparisons stay meaningful.
            bounds = {"vmin": 0, "vmax": 255} if image.dtype == np.uint8 else {}
            ax.imshow(image, cmap=cmap, **bounds)
        ax.set_title(title)
        ax.axis("off")
    for ax in list(axes.flat)[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def make_scene(width=640, height=420):
    """Create an original, deterministic BGR practice scene; no download required."""
    canvas = np.full((420, 640, 3), (235, 230, 220), np.uint8)
    ramp = np.linspace(0, 35, 640, dtype=np.uint8)
    canvas = cv2.subtract(canvas, np.repeat(ramp[None, :, None], 420, axis=0).repeat(3, axis=2))
    cv2.rectangle(canvas, (35, 85), (205, 260), (210, 90, 25), -1)
    cv2.circle(canvas, (335, 178), 80, (35, 55, 225), -1)
    triangle = np.array([[485, 80], [420, 265], [600, 265]], np.int32)
    cv2.fillPoly(canvas, [triangle], (45, 170, 55))
    cv2.rectangle(canvas, (35, 305), (605, 395), (245, 245, 245), -1)
    for x in range(50, 600, 22):
        cv2.line(canvas, (x, 318), (x, 382), (35, 35, 35), 2)
    cv2.putText(canvas, "OpenCV: learn by doing", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(canvas, (25, 285), (615, 285), (30, 30, 30), 3)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)


def load_image(path, flags=cv2.IMREAD_COLOR):
    """Read with Unicode-safe paths on Windows and a useful error for missing data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path.resolve()}")
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise ValueError(f"OpenCV cannot decode: {path}")
    return image


def save_image(path, image):
    """Encode an image and save using a Unicode-safe path; return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise IOError(f"Could not encode image: {path}")
    encoded.tofile(str(path))
    return path
