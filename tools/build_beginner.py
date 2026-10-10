"""Build the six introductory, executable OpenCV course notebooks.

Run from the project root with ``python -m tools.build_beginner``.
The lesson source lives here so corrections can be regenerated consistently.
"""
from tools.notebook_factory import md, code, intro, save


def lesson_one():
    cells = intro(
        "01 — Python, NumPy, and your first image", "Beginner · 60–90 minutes",
        ["No OpenCV experience required", "Know how to run a notebook cell with Shift+Enter"],
        ["Read a small Python program", "Explain shape, dtype, and image coordinates",
         "Create, inspect, and change an image using NumPy", "Avoid overflow and accidental edits"])
    cells += [
        md("""## Your first learning habit: predict → run → explain

Before running each code cell, predict one value or visible change. Run it, compare the
result, and write a sentence about why it happened. The setup cell imports `np` (NumPy),
`cv2` (OpenCV), and `plt` (Matplotlib). An **array** is a grid of numbers; an image is
one useful interpretation of that grid. The examples start with tiny grids you can inspect.

If a name is undefined, a previous cell has not run. Use **Restart Kernel and Run All**
after finishing a lesson to make sure your work does not depend on forgotten state.
"""),
        md("""## Step 1 — Read the Python building blocks

`=` stores a value under a name. `print` displays it. `#` begins a comment; text inside
quotes is a string. Python uses indentation to group instructions. Change `width` below
and predict the new pixel count before you run it.
"""),
        code("""width = 8
height = 5
number_of_pixels = width * height
label = "my first image"
print(label, "contains", number_of_pixels, "pixels")
print("A width is a", type(width).__name__)
"""),
        md("""A **list** stores several values. A `for` loop visits them. A **function** packages
reusable instructions; `return` sends a result back. This brightness check is deliberately
simple: a large average is brighter, but it does not tell us what objects an image contains.
"""),
        code("""sample_brightness = [20, 128, 240]

def describe_brightness(value):
    if value < 80:
        return "dark"
    elif value < 180:
        return "middle"
    return "bright"

for value in sample_brightness:
    print(value, "->", describe_brightness(value))
"""),
        md("""## Step 2 — Turn numbers into a grayscale image

`np.array` builds an array. `dtype=np.uint8` means unsigned 8-bit integers: whole numbers
from 0 to 255. For a grayscale image, 0 is black and 255 is white. `shape` reports
`(rows, columns)`; `ndim` counts axes; `size` counts stored values. A 3-channel image
has three stored values per pixel, so its `size` is not its pixel count.
"""),
        code("""tiny = np.array([
    [0, 40, 80, 120],
    [40, 80, 120, 180],
    [80, 120, 180, 255],
], dtype=np.uint8)
print(tiny)
print("shape:", tiny.shape, "axes:", tiny.ndim, "values:", tiny.size, "dtype:", tiny.dtype)
assert tiny.shape == (3, 4)
"""),
        code("""fig, ax = plt.subplots(figsize=(7, 3))
ax.imshow(tiny, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
for y in range(tiny.shape[0]):
    for x in range(tiny.shape[1]):
        ax.text(x, y, str(tiny[y, x]), ha="center", va="center",
                color="white" if tiny[y, x] < 128 else "black")
ax.set(xlabel="column x", ylabel="row y", title="Each square is one pixel")
plt.show()
"""),
        md("""The display is enlarged; the image still has only 12 pixels. Fixing `vmin=0` and
`vmax=255` prevents the plot from stretching every image to the same apparent contrast.
Image coordinates start at the **top left**: x increases rightward; y increases downward.

## Step 3 — Locate pixels and crop a region

Python counts from zero. `image[y, x]` addresses one pixel. A slice `start:stop` includes
`start` but excludes `stop`. Read the next slice as rows 0 and 1, columns 1 and 2.
"""),
        code("""print("top-left:", tiny[0, 0])
print("bottom-right:", tiny[-1, -1])
crop = tiny[0:2, 1:3]
print("crop:\\n", crop)
assert crop.shape == (2, 2)
show_images([tiny, crop], ["Whole 3×4 image", "Rows 0:2, columns 1:3"], cols=2)
"""),
        md("""A NumPy slice usually shares memory with its original: this is called a **view**.
Editing it changes the original. Use `.copy()` for an independent crop. The demonstration
uses a disposable array so that the earlier image stays unchanged.
"""),
        code("""original = tiny.copy()
view = original[0:2, 0:2]
view[:] = 255
independent_crop = original[0:2, 0:2].copy()
independent_crop[:] = 0
print("Original after editing its view:\\n", original)
print("Independent crop:\\n", independent_crop)
assert original[0, 0] == 255
assert not np.shares_memory(original, independent_crop)
"""),
        md("""## Step 4 — Calculate safely with pixels

An 8-bit number has limited storage. Ordinary NumPy `uint8` addition wraps around after
255; `cv2.add` saturates at 255. For general arithmetic, convert to a wider type,
calculate, **clip** to the allowed range, then convert back. `astype` changes the data
type; `np.clip(a, 0, 255)` bounds every value. Never rely on overflow to brighten a photo.
"""),
        code("""values = np.array([[10, 230, 250]], dtype=np.uint8)
increase = np.full_like(values, 30)
wrapped = values + increase
saturated = cv2.add(values, increase)
safe = np.clip(values.astype(np.int16) + 30, 0, 255).astype(np.uint8)
print("original:", values)
print("NumPy uint8 addition:", wrapped)
print("OpenCV saturated addition:", saturated)
print("wider arithmetic then clipping:", safe)
assert np.array_equal(safe, saturated)
"""),
        md("""**Vectorization** means applying an operation to a whole array at once. It is concise
and lets NumPy do the repeated work. `np.linspace` makes evenly spaced values, `np.tile`
repeats them into rows, and `np.mean` calculates their average. A Boolean comparison
such as `gradient > 180` creates a grid of `True`/`False` values.
"""),
        code("""gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (100, 1))
brighter = np.clip(gradient.astype(np.int16) + 45, 0, 255).astype(np.uint8)
bright_pixels = gradient > 180
print("Mean brightness:", float(gradient.mean()))
print("Fraction brighter than 180:", float(bright_pixels.mean()))
show_images([gradient, brighter, bright_pixels.astype(np.uint8) * 255],
            ["Original gradient", "Add 45 safely", "Pixels above 180"])
"""),
        md("""## Step 5 — Add color channels

OpenCV's usual 3-channel image order is **B, G, R**: blue, green, red. A color array
has shape `(height, width, 3)`. `np.zeros` creates a black canvas. Assigning a 3-number
tuple to a region gives every pixel in that region the same color.

Our `show_images` helper converts BGR to RGB for Matplotlib. Keep your working image
in BGR; convert only when passing it directly to `plt.imshow`.
"""),
        code("""color = np.zeros((140, 300, 3), dtype=np.uint8)
color[:, 0:100] = (255, 0, 0)       # Blue in BGR
color[:, 100:200] = (0, 255, 0)     # Green
color[:, 200:300] = (0, 0, 255)     # Red
print("shape:", color.shape, "pixels:", color.shape[0] * color.shape[1])
print("One red pixel, as BGR:", color[50, 250])
show_images([color], ["Blue | green | red"], cols=1)
"""),
        code("""blue_channel = color[:, :, 0]
green_channel = color[:, :, 1]
red_channel = color[:, :, 2]
show_images([blue_channel, green_channel, red_channel],
            ["Blue channel intensity", "Green channel intensity", "Red channel intensity"])
assert blue_channel.ndim == 2
"""),
        md("""**Expected observation:** in a channel image, white means that channel is strong.
A red object appears bright in the red channel even though this channel plot is grayscale.

### Parameter experiment

Change `brightness_offset` to -60, 0, and 90. Predict what will happen to the black
background and already-bright stripes. Clipping loses differences among pixels that
all become 0 or 255; it is not reversible.
"""),
        code("""brightness_offset = 35     # Experiment: -60, 0, 90
adjusted_color = np.clip(color.astype(np.int16) + brightness_offset, 0, 255).astype(np.uint8)
show_images([color, adjusted_color], ["Original", f"Offset {brightness_offset}"], cols=2)
print("Values at either endpoint:", np.mean((adjusted_color == 0) | (adjusted_color == 255)))
"""),
        md("""## Step 6 — Mini-project: an image inspection card

Create a reusable function that reports an image's dimensions and brightness. Convert
BGR to grayscale with `cv2.cvtColor` and `cv2.COLOR_BGR2GRAY`; this uses weighted color
contributions, so green and blue do not contribute equally. For this project, the function
accepts the grayscale or BGR `uint8` images used in the course.
"""),
        code("""def inspect_image(image):
    if image is None or image.size == 0:
        raise ValueError("Pass a non-empty image.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return {
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "channels": 1 if image.ndim == 2 else int(image.shape[2]),
        "dtype": str(image.dtype),
        "mean_gray": round(float(gray.mean()), 2),
        "min_gray": int(gray.min()),
        "max_gray": int(gray.max()),
    }

scene = make_scene()
report = inspect_image(scene)
print(report)
show_images([scene], ["A generated scene — no download needed"], cols=1)
assert report["width"] == scene.shape[1]
assert 0 <= report["mean_gray"] <= 255
"""),
        md("""### Try it yourself before opening the solutions

1. Make a 120-row, 200-column grayscale canvas filled with 60. Set its central
   rectangle (rows 30:90, columns 50:150) to 220. What fraction is bright?
2. Extract an independent copy of that central rectangle and make it black. Prove
   that the original central rectangle remains bright.
3. Write `safe_brighten(image, offset)` and test a negative offset.

Write your attempt in the next cell, or insert your own cells. The placeholder is valid
Python so **Run All** works before you solve the exercises.
"""),
        code("""# Your practice area. Example starting point:
practice_canvas = np.full((120, 200), 60, dtype=np.uint8)
# Add your own slicing and arithmetic here.
"""),
        md("""### Worked solutions — compare the reasoning, not just the picture

The rectangle covers `60 × 100` pixels out of `120 × 200`, so the bright fraction
is 0.25. Casting before addition makes both positive and negative offsets safe.
"""),
        code("""answer = np.full((120, 200), 60, dtype=np.uint8)
answer[30:90, 50:150] = 220
answer_crop = answer[30:90, 50:150].copy()
answer_crop[:] = 0

def safe_brighten(image, offset):
    return np.clip(image.astype(np.float32) + offset, 0, 255).astype(np.uint8)

assert np.isclose((answer > 180).mean(), 0.25)
assert np.all(answer[30:90, 50:150] == 220)
assert safe_brighten(answer, -100).min() == 0
show_images([answer, answer_crop, safe_brighten(answer, -100)],
            ["Exercise image", "Independent black crop", "Darker safely"])
"""),
        md("""### Reflection and debugging checklist

- Why does `(100, 200, 3)` describe 20,000 pixels rather than 60,000 pixels?
- Explain the difference between drawing coordinates `(x, y)` and array indexing `[y, x]`.
- Why can adding brightness make a `uint8` value darker with plain NumPy addition?
- When should you use `.copy()`? What evidence would show that two arrays share memory?

**You are ready for lesson 02 when** you can create a color image, crop it, print its
shape and dtype, and modify it safely without copying the solution.

Official reference: [OpenCV — basic image operations](https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html).
"""),
    ]
    save("01_numpy_and_images.ipynb", cells)


def lesson_two():
    cells = intro(
        "02 — Read, write, display, and inspect images", "Beginner · 60–90 minutes",
        ["Read lesson 01: arrays, indexing, uint8, and BGR order"],
        ["Use reliable paths and verify image loading", "Display color and grayscale correctly",
         "Compare lossless and lossy image formats", "Export an image report and thumbnail"])
    cells += [
        md("""## Step 1 — Give every file an explicit home

A path describes where a file lives. `Path` joins folders with `/` in Python on every
supported operating system. `ROOT` was found by the setup cell; it is the course folder.
This lesson generates its own input so you can work without downloads or photographs.
`mkdir(..., exist_ok=True)` creates a folder and permits rerunning the cell.
"""),
        code("""lesson_dir = ROOT / "outputs" / "02_read_write"
lesson_dir.mkdir(parents=True, exist_ok=True)
source = make_scene()
source_path = lesson_dir / "generated_scene.png"
save_image(source_path, source)
print("Course folder:", ROOT)
print("Generated file:", source_path)
print("Bytes on disk:", source_path.stat().st_size)
assert source_path.is_file()
"""),
        md("""## Step 2 — Read an image and check the result

`cv2.imread(str(path), flag)` returns an image array; ordinary read failures return
`None`. `IMREAD_COLOR` requests a 3-channel BGR result and drops alpha (transparency).
`IMREAD_GRAYSCALE` requests a single intensity channel. The `load_image` course helper
raises a readable error instead of letting a missing image fail much later.
"""),
        code("""loaded = load_image(source_path)
gray_from_file = load_image(source_path, cv2.IMREAD_GRAYSCALE)
print("Color:", loaded.shape, loaded.dtype)
print("Grayscale:", gray_from_file.shape, gray_from_file.dtype)
assert np.array_equal(source, loaded), "PNG should preserve these uint8 pixel values."
show_images([loaded, gray_from_file], ["BGR loaded from PNG", "Loaded as grayscale"], cols=2)
"""),
        code("""# Raw OpenCV API, used only after checking that the input path exists.
# The helper remains convenient for Unicode paths and readable failure messages.
raw_loaded = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
if raw_loaded is None:
    print("cv2.imread could not decode this path; load_image already succeeded above.")
else:
    print("cv2.imread returned:", raw_loaded.shape)
    assert raw_loaded.shape == loaded.shape
"""),
        md("""### Failure is information

Before debugging a color conversion, check that the input exists and was decoded.
The next cell intentionally catches one missing-file error; this keeps **Run All** working.
An exception's type and message are clues, not something to hide in a real pipeline.
"""),
        code("""missing_path = lesson_dir / "this_file_does_not_exist.png"
try:
    load_image(missing_path)
except (FileNotFoundError, ValueError, OSError) as error:
    print(type(error).__name__ + ":", error)
"""),
        md("""### Optional: use your own photograph

Keep `personal_path = None` for the generated example. To try a local photo, enter a
path such as `Path(r'C:\\Users\\you\\Pictures\\photo.jpg')`. The leading `r` makes a
raw string, so backslashes are not treated as special escapes. This cell changes only
`working`; the format comparison later still uses the known generated input.
"""),
        code(r"""personal_path = None  # Or Path(r"C:\Users\you\Pictures\photo.jpg")
working = loaded.copy() if personal_path is None else load_image(Path(personal_path))
print("Working image shape:", working.shape)
show_images([working], ["Your selected input"], cols=1)
"""),
        md("""## Step 3 — Display without swapping red and blue

Matplotlib expects RGB for a 3-channel color display. `cv2.cvtColor` rearranges or
converts channels according to its conversion flag. The first panel below deliberately
passes BGR straight to Matplotlib: use the red and blue regions to spot the mistake.
Display conversion does not change the BGR image held in `loaded`.
"""),
        code("""rgb_for_display = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(loaded)
axes[0].set_title("Incorrect: BGR passed as RGB")
axes[1].imshow(rgb_for_display)
axes[1].set_title("Correct: convert BGR → RGB")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
assert np.array_equal(cv2.cvtColor(rgb_for_display, cv2.COLOR_RGB2BGR), loaded)
"""),
        md("""Use `cmap='gray'` for a 2-dimensional array. Fix `vmin` and `vmax` when comparing
brightness across images; otherwise a plotting library may rescale each plot separately.
Here the darker image should actually look darker. This is a display decision, not an
image-processing operation.
"""),
        code("""darker = (gray_from_file.astype(np.float32) * 0.4).astype(np.uint8)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, image, title in zip(axes, [gray_from_file, darker], ["Original gray", "40% intensity"]):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""A PNG can contain a fourth channel: **alpha**, where 0 is transparent and 255 is
opaque. `IMREAD_UNCHANGED` preserves it. A 4-channel OpenCV image is commonly BGRA.
The example shows alpha separately, avoiding assumptions about the display helper.
"""),
        code("""bgra = cv2.cvtColor(loaded, cv2.COLOR_BGR2BGRA)
bgra[:, :, 3] = np.linspace(0, 255, loaded.shape[1], dtype=np.uint8)[None, :]
alpha_path = lesson_dir / "alpha_gradient.png"
save_image(alpha_path, bgra)
roundtrip_bgra = load_image(alpha_path, cv2.IMREAD_UNCHANGED)
print("Preserved channels:", roundtrip_bgra.shape)
assert roundtrip_bgra.shape[2] == 4
show_images([roundtrip_bgra[:, :, :3], roundtrip_bgra[:, :, 3]],
            ["Stored color", "Stored alpha: transparent → opaque"], cols=2)
"""),
        md("""## Step 4 — Make a thumbnail without stretching

`cv2.resize(image, (width, height), interpolation=...)` takes width **before** height,
unlike `image.shape`. Preserve the aspect ratio by multiplying both dimensions by
one scale. `INTER_AREA` is a useful starting point when shrinking. `INTER_NEAREST`
keeps existing values for masks; `INTER_LINEAR` blends neighbors for ordinary enlargement.
"""),
        code("""target_width = 240
h, w = working.shape[:2]
target_height = max(1, round(h * target_width / w))
thumbnail = cv2.resize(working, (target_width, target_height), interpolation=cv2.INTER_AREA)
print("Original width/height:", w / h)
print("Thumbnail width/height:", thumbnail.shape[1] / thumbnail.shape[0])
show_images([working, thumbnail], [f"Original {w}×{h}", f"Thumbnail {target_width}×{target_height}"], cols=2)
assert thumbnail.shape[:2] == (target_height, target_width)
"""),
        md("""### Parameter experiment: enlargement does not recover missing detail

The next example shrinks first, then enlarges two ways. Change `small_width` from
64 to 160. Compare letter edges at the same displayed size. Smooth edges can look
better while still failing to restore information that was removed.
"""),
        code("""small_width = 96  # Try 64 and 160
small_height = max(1, round(loaded.shape[0] * small_width / loaded.shape[1]))
small = cv2.resize(loaded, (small_width, small_height), interpolation=cv2.INTER_AREA)
destination_size = (loaded.shape[1], loaded.shape[0])
nearest = cv2.resize(small, destination_size, interpolation=cv2.INTER_NEAREST)
linear = cv2.resize(small, destination_size, interpolation=cv2.INTER_LINEAR)
show_images([loaded, nearest, linear], ["Original", "Shrink then nearest", "Shrink then linear"])
"""),
        md("""## Step 5 — Compare PNG and JPEG with evidence

PNG compression preserves the pixel values in this example. JPEG discards information
to reduce file size. `cv2.imencode` returns `(success, encoded_bytes)`; it selects the
format from the extension. JPEG quality ranges from 0 to 100. A higher setting usually
uses more bytes, but even quality 100 is not a promise of pixel equality.

We measure mean absolute error (MAE): average absolute channel difference from the
original. Cast before subtraction to avoid `uint8` wraparound. MAE alone does not fully
describe perceived quality, but it makes this comparison reproducible.
"""),
        code("""def encoded_roundtrip(image, extension, parameters=None):
    ok, buffer = cv2.imencode(extension, image, parameters or [])
    if not ok:
        raise RuntimeError(f"Encoding failed for {extension}")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Could not decode encoded bytes")
    mae = float(np.abs(image.astype(np.float32) - decoded.astype(np.float32)).mean())
    return decoded, int(buffer.nbytes), mae

png_result, png_bytes, png_mae = encoded_roundtrip(loaded, ".png")
print(f"PNG: {png_bytes:,} bytes, MAE={png_mae:.3f}")
assert png_mae == 0
"""),
        code("""jpeg_qualities = [20, 60, 95]  # Edit these and compare size versus distortion.
jpeg_results = []
for quality in jpeg_qualities:
    decoded, size, mae = encoded_roundtrip(loaded, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, quality])
    jpeg_results.append(decoded)
    print(f"JPEG quality {quality:3}: {size:>7,} bytes, MAE={mae:.3f}")
show_images(jpeg_results, [f"JPEG quality {q}" for q in jpeg_qualities])
"""),
        md("""Differences are often easiest to see around text and sharp color boundaries. A
multiplied difference image makes small changes visible; it is a diagnostic visualization,
not an honest display of how large those changes look in the original photograph.
"""),
        code("""difference = cv2.absdiff(loaded, jpeg_results[0])
amplified_difference = np.clip(difference.astype(np.float32) * 8, 0, 255).astype(np.uint8)
show_images([loaded, jpeg_results[0], amplified_difference],
            ["PNG reference", "Lowest configured JPEG quality", "Absolute error ×8"])
"""),
        md("""## Step 6 — Mini-project: export a thumbnail and an image report

A useful image pipeline saves both an artifact and the facts needed to interpret it.
The report records the input shape and thumbnail size. `json.dump` writes portable
structured text. `save_image` writes the BGR thumbnail and checks that encoding succeeds.
Rerunning this project replaces only these lesson output files.
"""),
        code("""import json

thumbnail_path = lesson_dir / "thumbnail.png"
save_image(thumbnail_path, thumbnail)
report = {
    "input": "generated_scene.png" if personal_path is None else str(personal_path),
    "input_shape_h_w_c": list(working.shape),
    "dtype": str(working.dtype),
    "thumbnail_shape_h_w_c": list(thumbnail.shape),
    "thumbnail_bytes": thumbnail_path.stat().st_size,
}
report_path = lesson_dir / "image_report.json"
with report_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)
print(json.dumps(report, indent=2))
print("Saved:", thumbnail_path, "and", report_path)
assert np.array_equal(load_image(thumbnail_path), thumbnail)
"""),
        md("""### Exercises — attempt first

1. Save a grayscale version of `working` as PNG, reload it as grayscale, and verify
   exact pixel equality.
2. Create a thumbnail no wider than 180 and no taller than 120. Preserve its aspect
   ratio, and do not enlarge already-small images. Hint: choose the smallest of
   three scale factors.
3. Explain why saving `rgb_for_display` with an OpenCV writer swaps its colors.
"""),
        code("""# Your practice area: use lesson_dir for any files you create.
# gray_practice = ...
"""),
        md("""### Worked solutions

The limiting dimension sets the scale; including 1.0 prevents enlargement. Rounding
down ensures the output fits within both size limits. Save BGR arrays with OpenCV;
RGB is used here only at the Matplotlib boundary.
"""),
        code("""gray_answer = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
gray_path = lesson_dir / "exercise_gray.png"
save_image(gray_path, gray_answer)
assert np.array_equal(load_image(gray_path, cv2.IMREAD_GRAYSCALE), gray_answer)

h, w = working.shape[:2]
scale = min(180 / w, 120 / h, 1.0)
fit_size = (max(1, int(w * scale)), max(1, int(h * scale)))
fit = cv2.resize(working, fit_size, interpolation=cv2.INTER_AREA)
assert fit.shape[1] <= 180 and fit.shape[0] <= 120
show_images([gray_answer, fit], ["Verified grayscale PNG", "Fits within 180×120"], cols=2)
"""),
        md("""### Reflect before continuing

- What would you check first if `cvtColor` complained about an empty source?
- Which arrays in this notebook are BGR, RGB, grayscale, and BGRA?
- Why do PNG and JPEG file sizes depend on image content as well as dimensions?
- Why is shrinking and re-enlarging an image different from keeping the original?

Official references: [Image loading and writing](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html),
[Image geometric transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).
"""),
    ]
    save("02_read_write_and_display.ipynb", cells)


def lesson_three():
    cells = intro(
        "03 — Pixels, drawing, regions, and masks", "Beginner · 60–90 minutes",
        ["Read lessons 01–02: BGR, slices, copying, image files"],
        ["Draw annotations with correct coordinates", "Build and combine binary masks",
         "Separate selection from the operation applied to it", "Create a reusable selective-edit pipeline"])
    cells += [
        md("""## Step 1 — Translate between coordinates and indexing

OpenCV drawing calls take `(x, y)`. NumPy reads pixels with `[y, x]`. Here a blank
canvas has height 300 and width 480. The selected pixel is 80 columns from the left
and 50 rows from the top. `cv2.circle` takes image, center, radius, BGR color, and
thickness. A thickness of -1 fills the shape.
"""),
        code("""canvas = np.full((300, 480, 3), 235, dtype=np.uint8)
x, y = 80, 50
cv2.circle(canvas, (x, y), 8, (0, 0, 255), thickness=-1)
print("Pixel at [y, x]:", canvas[y, x])
assert np.array_equal(canvas[50, 80], [0, 0, 255])
show_images([canvas], ["Red point at (x=80, y=50)"], cols=1)
"""),
        md("""## Step 2 — Build an annotated diagram

`rectangle` takes opposite corners; `line` takes endpoints; `arrowedLine` adds an
arrowhead. These functions edit the array **in place**, so start from `.copy()` when
you want to preserve the original. `LINE_AA` blends boundary pixels for smoother
annotations; use `LINE_8` when you need a mask containing only 0 and 255.
"""),
        code("""diagram = canvas.copy()
cv2.rectangle(diagram, (140, 55), (285, 150), (200, 80, 30), 3, lineType=cv2.LINE_AA)
cv2.circle(diagram, (375, 110), 48, (30, 180, 60), -1, lineType=cv2.LINE_AA)
cv2.line(diagram, (40, 235), (440, 235), (60, 60, 60), 2)
cv2.arrowedLine(diagram, (80, 70), (150, 190), (0, 0, 200), 2, tipLength=0.15)
show_images([canvas, diagram], ["Original canvas", "Drawing on a copy"], cols=2)
"""),
        md("""`putText` uses a **baseline origin**, not the top-left corner of the letters.
The font scale controls size, thickness controls stroke width, and `getTextSize`
estimates the width, height, and baseline needed for layout. OpenCV's built-in Hershey
fonts suit short ASCII labels; they are not a full multilingual text-layout system.
"""),
        code("""label = "Detection demo"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale, thickness = 0.8, 2
(text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
origin = (25, 285)
cv2.putText(diagram, label, origin, font, font_scale, (40, 40, 40), thickness, cv2.LINE_AA)
print("Text width, height, baseline:", text_width, text_height, baseline)
show_images([diagram], ["Text origin is on the baseline"], cols=1)
"""),
        md("""### Coordinate pitfall: inclusive drawing endpoints

NumPy slices exclude their stop index. A filled OpenCV rectangle includes its specified
endpoints. To draw exactly the slice `[y0:y1, x0:x1]`, use `(x1-1, y1-1)` as the bottom
right corner. This detail matters when testing areas or converting bounding boxes.
"""),
        code("""x0, y0, x1, y1 = 30, 20, 130, 80
slice_mask = np.zeros((100, 180), dtype=np.uint8)
slice_mask[y0:y1, x0:x1] = 255
drawn_mask = np.zeros_like(slice_mask)
cv2.rectangle(drawn_mask, (x0, y0), (x1 - 1, y1 - 1), 255, -1)
assert np.array_equal(slice_mask, drawn_mask)
print("Selected area:", cv2.countNonZero(slice_mask), "pixels")
assert cv2.countNonZero(slice_mask) == (x1 - x0) * (y1 - y0)
"""),
        md("""## Step 3 — Describe where an operation is allowed

A **mask** is a single-channel selection image matching the source height and width.
For the binary masks here, 0 means unselected and 255 means selected. A mask does
not itself change the source. Keeping selection separate lets you reuse it for edits,
measurements, and visualization.
"""),
        code("""scene = make_scene()
h, w = scene.shape[:2]
rectangle_mask = np.zeros((h, w), dtype=np.uint8)
rectangle_mask[h // 4:3 * h // 4, w // 6:4 * w // 6] = 255
circle_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(circle_mask, (w // 2, h // 2), min(h, w) // 3, 255, -1)
show_images([scene, rectangle_mask, circle_mask], ["Source", "Rectangle selection", "Circle selection"])
assert rectangle_mask.shape == scene.shape[:2]
assert set(np.unique(circle_mask)).issubset({0, 255})
"""),
        md("""`cv2.bitwise_and(source, source, mask=mask)` copies selected pixels to the
output and leaves the rest black. Supplying the same source twice is intentional:
the mask controls which results are retained. `countNonZero` measures selected area.
"""),
        code("""selected = cv2.bitwise_and(scene, scene, mask=circle_mask)
selection_fraction = cv2.countNonZero(circle_mask) / circle_mask.size
print(f"Selected {selection_fraction:.1%} of the image")
assert np.all(selected[circle_mask == 0] == 0)
assert np.array_equal(selected[circle_mask > 0], scene[circle_mask > 0])
show_images([scene, selected], ["Whole scene", "Inside circle only"], cols=2)
"""),
        md("""## Step 4 — Combine selections using logic

For binary masks, AND keeps the overlap, OR keeps either region, NOT inverts the
selection, and XOR keeps exactly one region. These meanings rely on using 0 and 255;
arbitrary grayscale values are bit patterns, not probabilities.
"""),
        code("""intersection = cv2.bitwise_and(rectangle_mask, circle_mask)
union = cv2.bitwise_or(rectangle_mask, circle_mask)
outside_circle = cv2.bitwise_not(circle_mask)
exclusive = cv2.bitwise_xor(rectangle_mask, circle_mask)
show_images([intersection, union, outside_circle, exclusive],
            ["AND: overlap", "OR: either", "NOT: outside circle", "XOR: exactly one"], cols=2)
assert cv2.countNonZero(intersection) <= cv2.countNonZero(union)
assert cv2.countNonZero(union) == (cv2.countNonZero(rectangle_mask)
    + cv2.countNonZero(circle_mask) - cv2.countNonZero(intersection))
"""),
        md("""You can also use a Boolean mask to assign a replacement color. In
`edited[mask > 0]`, NumPy selects all channels of the chosen pixels. This is useful
for marking regions and for writing pixels from another image of the same shape.
"""),
        code("""painted = scene.copy()
painted[intersection > 0] = (0, 200, 255)
mean_inside = cv2.mean(scene, mask=circle_mask)[:3]
print("Mean B, G, R inside the circle:", np.round(mean_inside, 1))
show_images([scene, painted], ["Source", "Paint overlap yellow"], cols=2)
"""),
        md("""### Parameter experiment: move the selection, keep the source fixed

Change the radius fraction and horizontal center. Predict the area change: a circle's
area grows approximately with the square of its radius until it meets an image edge.
Clipping at an edge is a useful reason why measured pixel area differs from a formula.
"""),
        code("""radius_fraction = 0.25   # Try 0.15, 0.35, and 0.55
center_x_fraction = 0.50 # Try 0.10 and 0.85
experiment_mask = np.zeros((h, w), dtype=np.uint8)
radius = max(1, round(h * radius_fraction))
center = (round(w * center_x_fraction), h // 2)
cv2.circle(experiment_mask, center, radius, 255, -1)
print("Raster area:", cv2.countNonZero(experiment_mask), "ideal circle area:", round(np.pi * radius ** 2))
show_images([experiment_mask, cv2.bitwise_and(scene, scene, mask=experiment_mask)],
            ["New selection", "Same source, new mask"], cols=2)
"""),
        md("""## Step 5 — Blend instead of replacing every pixel

`cv2.addWeighted(a, alpha, b, beta, gamma)` computes a weighted sum with an added
offset. Here `beta=1-alpha` and `gamma=0`, giving a transparent-looking overlay.
Increasing alpha gives the overlay more influence. It changes every pixel, so we copy
only the selected blended pixels back into the source.
"""),
        code("""alpha = 0.45  # Try 0.15 and 0.80; keep it between 0 and 1.
assert 0 <= alpha <= 1
overlay = np.full_like(scene, (0, 180, 255))
blended_everywhere = cv2.addWeighted(overlay, alpha, scene, 1 - alpha, 0)
highlighted = scene.copy()
highlighted[circle_mask > 0] = blended_everywhere[circle_mask > 0]
assert np.array_equal(highlighted[circle_mask == 0], scene[circle_mask == 0])
show_images([scene, highlighted], ["Source", "Transparent-looking circular highlight"], cols=2)
"""),
        md("""A hard binary boundary can look abrupt. A **soft mask** stores weights between
0 and 1. A Gaussian blur smooths the binary mask; `[..., None]` adds a channel axis
so the same weight applies to B, G, and R. Floating-point arithmetic keeps intermediate
fractions. This is compositing, not a transparent PNG alpha channel unless we save one.
"""),
        code("""soft = cv2.GaussianBlur(circle_mask, (41, 41), 0).astype(np.float32) / 255.0
weights = soft[..., None]
soft_highlight = np.clip(overlay.astype(np.float32) * weights * alpha
                        + scene.astype(np.float32) * (1 - weights * alpha), 0, 255).astype(np.uint8)
show_images([circle_mask, (soft * 255).astype(np.uint8), soft_highlight],
            ["Hard selection", "Smoothed selection weights", "Soft highlight"])
"""),
        md("""## Step 6 — Mini-project: a reusable selective blur editor

Given a rectangle `(x, y, width, height)`, blur only that region and label it. A Gaussian
kernel has positive odd dimensions; a larger kernel blends a wider neighborhood. This
example is an image-editing exercise: visible blurring is not a guarantee that sensitive
information cannot be recovered. Blur the region independently to avoid editing pixels
outside the box; the region edge may still be visible.
"""),
        code("""def blur_region(image, box, kernel_size=31):
    x, y, width, height = map(int, box)
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]
            and width > 0 and height > 0 and x + width <= image.shape[1]
            and y + height <= image.shape[0]):
        raise ValueError("Box must lie completely inside the image.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Use a positive odd kernel size.")
    result = image.copy()
    region = image[y:y + height, x:x + width]
    result[y:y + height, x:x + width] = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
    return result

box = (w // 4, h // 4, w // 2, h // 2)
blurred = blur_region(scene, box)
box_mask = np.zeros((h, w), dtype=np.uint8)
bx, by, bw, bh = box
box_mask[by:by + bh, bx:bx + bw] = 255
assert np.array_equal(scene[box_mask == 0], blurred[box_mask == 0])
annotated = blurred.copy()
cv2.rectangle(annotated, (bx, by), (bx + bw - 1, by + bh - 1), (0, 160, 255), 2)
cv2.putText(annotated, "Edited region", (bx, max(20, by - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 180), 2, cv2.LINE_AA)
show_images([scene, annotated], ["Before", "Selective blur + annotation"], cols=2)
"""),
        code("""lesson_dir = ROOT / "outputs" / "03_masks"
lesson_dir.mkdir(parents=True, exist_ok=True)
save_image(lesson_dir / "selective_blur.png", annotated)
save_image(lesson_dir / "selection_mask.png", box_mask)
print("Saved project and reusable selection to:", lesson_dir)
"""),
        md("""### Exercises — attempt first

1. Create a ring mask: retain a large disk but remove a smaller disk with the same center.
2. Change only the ring pixels to grayscale; keep all other pixels colored.
3. Measure the ring area and verify that it equals the outer disk area minus the inner
   disk area. Why should this identity hold for the rasterized masks too?
"""),
        code("""# Your practice area. Hint: inner must be completely contained inside outer.
# outer = np.zeros(scene.shape[:2], dtype=np.uint8)
"""),
        md("""### Worked solutions

Use AND with the inverted inner disk to remove its pixels. Convert the grayscale
result back to three channels before inserting it into a BGR image. Its channels then
have equal values, so it looks gray while matching the source array's shape.
"""),
        code("""outer = np.zeros((h, w), dtype=np.uint8)
inner = np.zeros((h, w), dtype=np.uint8)
cv2.circle(outer, (w // 2, h // 2), h // 3, 255, -1)
cv2.circle(inner, (w // 2, h // 2), h // 6, 255, -1)
ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
gray_bgr = cv2.cvtColor(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
ring_edit = scene.copy()
ring_edit[ring > 0] = gray_bgr[ring > 0]
assert cv2.countNonZero(ring) == cv2.countNonZero(outer) - cv2.countNonZero(inner)
assert np.array_equal(ring_edit[ring == 0], scene[ring == 0])
show_images([ring, ring_edit], ["Ring selection", "Grayscale ring only"], cols=2)
"""),
        md("""### Reflection

- How could an `(x, y)` / `[y, x]` mix-up appear to work on a square image?
- Why must a mask share the image's height and width, but not its channel count?
- Why do anti-aliased mask edges fail a strict assertion that values are only 0 or 255?
- Which is easier to reuse: a painted image, or an original image plus its mask?

Official references: [Drawing functions](https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html),
[Arithmetic and masking](https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html).
"""),
    ]
    save("03_pixels_drawing_and_masks.ipynb", cells)


def lesson_four():
    cells = intro(
        "04 — Color spaces, histograms, and color segmentation", "Beginner → intermediate · 75–105 minutes",
        ["Read lessons 01–03: BGR, masks, arithmetic, and plotting"],
        ["Choose useful representations of color", "Interpret intensity histograms",
         "Compare global and local contrast enhancement", "Build and evaluate a color-selection pipeline"])
    cells += [
        md("""## Step 1 — One image, several representations

A color space is a way to describe a color numerically. BGR is convenient for storing
images, HSV separates hue from saturation and brightness, and Lab separates a lightness
coordinate from two color coordinates. `cvtColor` performs these conversions; swapping
names or channel positions is not equivalent to converting color spaces.

Grayscale discards color information, which helps when shape or brightness matters.
Different colors can become similar gray values, so grayscale cannot solve every task.
"""),
        code("""scene = make_scene()
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB)
print("BGR:", scene.shape, "gray:", gray.shape, "HSV:", hsv.shape, "Lab:", lab.shape)
show_images([scene, gray], ["Color contains category clues", "Grayscale preserves intensity structure"], cols=2)
"""),
        md("""For OpenCV's ordinary **uint8 HSV** conversion, hue uses 0–179 and saturation
and value use 0–255. Hue describes a position around a color wheel. Saturation describes
colorfulness; value is the maximum BGR channel. Float conversions have different ranges,
so first check the dtype. We will use uint8 throughout this lesson.

Plot each channel separately. An HSV array passed directly to a BGR display function
will produce misleading colors because its three numbers have different meanings.
"""),
        code("""hue, saturation, value = cv2.split(hsv)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(hue, cmap="hsv", vmin=0, vmax=179)
axes[1].imshow(saturation, cmap="gray", vmin=0, vmax=255)
axes[2].imshow(value, cmap="gray", vmin=0, vmax=255)
for ax, title in zip(axes, ["Hue (color-coded)", "Saturation", "Value"]):
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
assert hue.max() <= 179
"""),
        md("""### A color wheel with a boundary

Red lies near both ends of the hue scale. That wraparound matters when selecting red
pixels. Hue also becomes unreliable as a color cue when saturation is near zero:
white, gray, and black do not have a useful visible hue.
"""),
        code("""hue_strip = np.zeros((60, 180, 3), dtype=np.uint8)
hue_strip[:, :, 0] = np.arange(180, dtype=np.uint8)
hue_strip[:, :, 1:] = 255
bgr_strip = cv2.cvtColor(hue_strip, cv2.COLOR_HSV2BGR)
show_images([bgr_strip], ["Hue 0 → 179: red appears at both ends"], cols=1)
print("BGR at hues 0, 60, 120, 179:", bgr_strip[0, [0, 60, 120, 179]].tolist())
"""),
        md("""## Step 2 — Select a color using ranges

`cv2.inRange(image, lower, upper)` returns a binary mask. A pixel is selected only
when **all** its channel values lie within the inclusive bounds. For red, combine a
low-hue range and a high-hue range with OR. A saturation floor excludes most gray
pixels; a value floor excludes very dark pixels.
"""),
        code("""red_low = cv2.inRange(hsv, (0, 90, 50), (10, 255, 255))
red_high = cv2.inRange(hsv, (170, 90, 50), (179, 255, 255))
red_mask = cv2.bitwise_or(red_low, red_high)
red_only = cv2.bitwise_and(scene, scene, mask=red_mask)
show_images([scene, red_mask, red_only], ["Input", "Red mask", "Selected pixels"])
assert red_mask.dtype == np.uint8 and red_mask.shape == scene.shape[:2]
print("Selected area:", cv2.countNonZero(red_mask), "pixels")
"""),
        md("""### Parameter experiment

Change the hue tolerance and saturation minimum below. A wider range admits more
orange/pink pixels; a lower saturation floor may admit pale backgrounds. Thresholds
that work on this scene are starting hypotheses for a new camera or lighting setup.
Inspect the mask, not only the attractive selected-color image.
"""),
        code("""hue_tolerance = 10   # Try 5 and 20; valid range here is 0..30.
minimum_saturation = 90  # Try 20 and 180.
minimum_value = 50
assert 0 <= hue_tolerance <= 30
assert 0 <= minimum_saturation <= 255 and 0 <= minimum_value <= 255
tuned_mask = cv2.inRange(hsv, (0, minimum_saturation, minimum_value), (hue_tolerance, 255, 255))
# Hue wraps every 180 units: distance 1 from hue 0 includes both 1 and 179.
# At tolerance 0 there is no wrapped interval; select hue 0 only.
if hue_tolerance > 0:
    wrapped_mask = cv2.inRange(hsv, (180 - hue_tolerance, minimum_saturation, minimum_value),
                               (179, 255, 255))
    tuned_mask = cv2.bitwise_or(tuned_mask, wrapped_mask)
show_images([tuned_mask, cv2.bitwise_and(scene, scene, mask=tuned_mask)],
            ["Tuned selection", "Selected source pixels"], cols=2)
"""),
        md("""## Step 3 — Read a histogram as a count of values

A histogram counts how many pixels fall in each intensity bin. It says nothing about
where those pixels are. `calcHist([image], [channel], mask, [bins], [range_start, range_end])`
uses a list of images, a channel index, an optional selection, and a half-open range.
For uint8 intensity, 256 bins over `[0, 256]` provide one bin per possible value.
"""),
        code("""histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
selected_histogram = cv2.calcHist([gray], [0], red_mask, [256], [0, 256]).ravel()
assert int(histogram.sum()) == gray.size
assert int(selected_histogram.sum()) == cv2.countNonZero(red_mask)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(histogram, label="All grayscale pixels")
ax.plot(selected_histogram, label="Only red-mask pixels")
ax.set(xlabel="Grayscale value", ylabel="Pixel count", xlim=(0, 255))
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""Two different arrangements can have exactly the same histogram. Flattening with
`ravel`, randomly permuting, and reshaping moves the pixels without changing their values.
This is why a histogram can help describe exposure or color distribution but cannot
uniquely identify the scene layout.
"""),
        code("""shuffle_rng = np.random.default_rng(42)
shuffled = shuffle_rng.permutation(gray.ravel()).reshape(gray.shape)
shuffled_hist = cv2.calcHist([shuffled], [0], None, [256], [0, 256]).ravel()
assert np.array_equal(histogram, shuffled_hist)
show_images([gray, shuffled], ["Original spatial arrangement", "Same histogram, scrambled layout"], cols=2)
"""),
        md("""## Step 4 — Improve contrast, then check what changed

Compress the grayscale values into a narrow band to simulate low contrast.
`equalizeHist` redistributes a single-channel uint8 image using its cumulative
histogram. It can reveal differences in a narrow range, but may also exaggerate noise
and change the intended appearance. More contrast is not automatically more information.
"""),
        code("""low_contrast = np.clip(gray.astype(np.float32) * 0.25 + 95, 0, 255).astype(np.uint8)
equalized = cv2.equalizeHist(low_contrast)
print("Low contrast min/max:", int(low_contrast.min()), int(low_contrast.max()))
print("Equalized min/max:", int(equalized.min()), int(equalized.max()))
show_images([gray, low_contrast, equalized], ["Original gray", "Compressed intensity range", "Global equalization"])
"""),
        code("""fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for image, label in [(low_contrast, "Low contrast"), (equalized, "Equalized")]:
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    axes[0].plot(hist, label=label)
    axes[1].plot(np.cumsum(hist) / hist.sum(), label=label)
axes[0].set(title="Histogram", xlabel="Intensity", ylabel="Pixel count", xlim=(0, 255))
axes[1].set(title="Cumulative fraction", xlabel="Intensity", ylabel="Fraction ≤ value", xlim=(0, 255))
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""CLAHE applies contrast enhancement in local tiles while limiting histogram peaks.
`createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` configures the operator; `apply` runs it.
The grid size is the **number of tiles**, not a kernel measured in pixels. The clip
limit controls contrast amplification. Compare details in bright and dark regions.
"""),
        code("""illumination = np.linspace(0.40, 1.0, gray.shape[1], dtype=np.float32)[None, :]
uneven_gray = np.clip(gray.astype(np.float32) * illumination, 0, 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
local_equalized = clahe.apply(uneven_gray)
show_images([uneven_gray, cv2.equalizeHist(uneven_gray), local_equalized],
            ["Uneven illumination", "Global equalization", "CLAHE"])
"""),
        md("""## Step 5 — Adjust lightness in a color image

Equalizing B, G, and R separately can change their relative proportions and shift colors.
An alternative is to convert to Lab, adjust its lightness channel, and convert back.
For uint8 Lab, OpenCV stores an encoded lightness in 0–255 and offsets its color
coordinates. Treat these stored values as encoded coordinates, not direct physical units.
Conversion, clipping, and gamut limits mean this is not perfect color preservation.
"""),
        code("""dim_color = np.clip(scene.astype(np.float32) * 0.50 + 25, 0, 255).astype(np.uint8)
dim_lab = cv2.cvtColor(dim_color, cv2.COLOR_BGR2LAB)
lightness, a_channel, b_channel = cv2.split(dim_lab)
enhanced_lightness = clahe.apply(lightness)
enhanced_lab = cv2.merge([enhanced_lightness, a_channel, b_channel])
enhanced_color = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
show_images([scene, dim_color, enhanced_color], ["Reference color", "Dimmed color", "CLAHE on Lab lightness"])
assert enhanced_color.shape == scene.shape and enhanced_color.dtype == np.uint8
"""),
        md("""### Contrast experiment

Try clip limits 1.0, 2.0, and 6.0. Record which setting improves small details without
making smooth regions distracting. A photograph with sensor noise is a harder test
than a clean generated image; later, compare both rather than trusting one example.
"""),
        code("""clip_limits = [1.0, 2.0, 6.0]
contrast_variants = [cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8)).apply(uneven_gray)
                     for limit in clip_limits]
show_images(contrast_variants, [f"CLAHE clipLimit={limit}" for limit in clip_limits])
"""),
        md("""## Step 6 — Mini-project: measure a red marker under changing brightness

Generate a known target and a green distractor. The target's ground-truth mask tells
us which pixels should be selected. Measure intersection over union (IoU): selected
overlap divided by all pixels selected by either mask. A perfect mask scores 1.
Brightness scaling is only one controlled stress test; it does not simulate every real
lighting change, reflection, shadow, or camera white-balance adjustment.
"""),
        code("""target = np.full((240, 360, 3), 190, dtype=np.uint8)
truth = np.zeros(target.shape[:2], dtype=np.uint8)
cv2.circle(target, (110, 120), 55, (25, 30, 220), -1)
cv2.circle(truth, (110, 120), 55, 255, -1)
cv2.rectangle(target, (240, 65), (320, 175), (30, 190, 35), -1)

def select_red(image, saturation_min=90, value_min=50):
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    first = cv2.inRange(converted, (0, saturation_min, value_min), (10, 255, 255))
    second = cv2.inRange(converted, (170, saturation_min, value_min), (179, 255, 255))
    return cv2.bitwise_or(first, second)

def mask_iou(predicted, expected):
    p, e = predicted > 0, expected > 0
    union_size = np.count_nonzero(p | e)
    return float(np.count_nonzero(p & e) / union_size) if union_size else 1.0

assert mask_iou(truth, truth) == 1.0
show_images([target, truth], ["Target plus distractor", "Known red target pixels"], cols=2)
"""),
        code("""brightness_factors = [1.0, 0.55, 0.15]
project_images, project_titles = [], []
for factor in brightness_factors:
    dimmed = np.clip(target.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    prediction = select_red(dimmed)
    score = mask_iou(prediction, truth)
    print(f"Brightness {factor:.2f}: selected {cv2.countNonZero(prediction):5} pixels, IoU={score:.3f}")
    project_images.extend([dimmed, prediction])
    project_titles.extend([f"Brightness ×{factor}", f"Prediction; IoU={score:.3f}"])
show_images(project_images, project_titles, cols=2)
"""),
        md("""**Expected observation:** the very dark marker is rejected by the fixed value
floor, even if its hue still looks red. Lowering the floor can recover it, but may admit
dark noise in real images. Evaluation makes this tradeoff explicit.

### Exercises — attempt first

1. Recover the darkest synthetic marker by changing only `value_min`. Report IoU.
2. Select the green distractor with a suitable HSV range and report its area.
3. Plot the B, G, and R histograms separately. Explain why a channel histogram alone
   cannot prove that a red object occupies a particular location.
"""),
        code("""# Your practice area. Example: inspect HSV values at the marker center.
print("Dark marker HSV:", cv2.cvtColor(dimmed, cv2.COLOR_BGR2HSV)[120, 110])
"""),
        md("""### Worked solutions

The selected value floor below is justified by inspecting this known synthetic target.
The green range is another dataset-specific choice. Inspecting representative pixels
is a better starting point than memorizing thresholds as universal constants.
"""),
        code("""dark_target = (target.astype(np.float32) * 0.15).astype(np.uint8)
recovered = select_red(dark_target, value_min=20)
print("Recovered red IoU:", mask_iou(recovered, truth))
assert mask_iou(recovered, truth) > 0.99
green_mask = cv2.inRange(cv2.cvtColor(target, cv2.COLOR_BGR2HSV), (40, 90, 50), (85, 255, 255))
print("Green area:", cv2.countNonZero(green_mask))
show_images([recovered, green_mask], ["Recovered dark red marker", "Green selection"], cols=2)

fig, ax = plt.subplots(figsize=(10, 3))
for index, color_name in enumerate(["blue", "green", "red"]):
    hist = cv2.calcHist([target], [index], None, [256], [0, 256]).ravel()
    ax.plot(hist, color=color_name, label=color_name)
ax.set(xlabel="Channel value", ylabel="Pixel count", xlim=(0, 255))
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""### Reflection

- Why does red need two hue intervals? Why does gray need a saturation check?
- Can a color histogram identify an object's position? Use the shuffle experiment as evidence.
- Which details did global equalization help, and which did CLAHE help?
- What additional photographs would test whether your color selector generalizes?

Official references: [Color spaces and inRange](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html),
[Histograms](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html),
[Histogram equalization and CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html).
"""),
    ]
    save("04_color_spaces_and_histograms.ipynb", cells)


def lesson_five():
    cells = intro(
        "05 — Filtering, noise, and measurable tradeoffs", "Beginner → intermediate · 75–105 minutes",
        ["Read lessons 01–04: dtype, grayscale, histograms, and masks"],
        ["Generate repeatable noise and retain a clean reference", "Compare mean, Gaussian, median, and bilateral filters",
         "Measure denoising error without overflow", "Choose parameters with evidence rather than appearance alone"])
    cells += [
        md("""## Step 1 — Build an image with known clean pixels

Denoising removes unwanted variation, but a filter cannot automatically know whether
a small feature is noise or useful detail. We will keep a clean reference and add
noise ourselves. This lets us measure error exactly in a controlled experiment.
The clean image contains flat regions, hard edges, a gradient, and thin text strokes.
"""),
        code("""clean = np.tile(np.linspace(35, 200, 420, dtype=np.uint8), (280, 1))
cv2.rectangle(clean, (30, 30), (160, 150), 210, -1)
cv2.circle(clean, (285, 105), 60, 65, -1)
cv2.putText(clean, "FILTER", (40, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 245, 2, cv2.LINE_AA)
show_images([clean], ["Ground truth: flat areas, edges, gradient, text"], cols=1)
assert clean.dtype == np.uint8
"""),
        md("""Gaussian noise adds a random offset to each pixel. `normal(0, sigma, shape)`
draws offsets with mean 0 and standard deviation `sigma`. Use float arithmetic before
clipping. A fixed random seed reproduces the same experiment, so changing a filter
does not secretly change its input too.
"""),
        code("""noise_sigma = 20.0  # Try 8, 20, and 40 later.
noise_rng = np.random.default_rng(42)
noise = noise_rng.normal(0, noise_sigma, clean.shape)
gaussian_noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
show_images([clean, gaussian_noisy], ["Clean reference", f"Gaussian noise σ={noise_sigma}"], cols=2)
print("Noise before clipping: mean", round(float(noise.mean()), 3),
      "standard deviation", round(float(noise.std()), 3))
"""),
        md("""Salt-and-pepper noise replaces some pixels with extreme dark or bright values.
`random(shape)` draws numbers from 0 up to, but excluding, 1. The same random field
decides which pixels become black and which become white. Compare the isolated outliers
with the fine-grained Gaussian noise. Noise type affects which filter works well.
"""),
        code("""impulse_fraction = 0.06
impulse_rng = np.random.default_rng(7)
random_field = impulse_rng.random(clean.shape)
impulse_noisy = clean.copy()
impulse_noisy[random_field < impulse_fraction / 2] = 0
impulse_noisy[random_field > 1 - impulse_fraction / 2] = 255
actual_changed_fraction = float(np.mean(impulse_noisy != clean))
print(f"Pixels changed: {actual_changed_fraction:.1%}")
show_images([gaussian_noisy, impulse_noisy], ["Gaussian noise", "Salt-and-pepper noise"], cols=2)
"""),
        md("""## Step 2 — Understand a neighborhood average

A **kernel** describes a neighborhood and its weights. `blur(image, (k, k))` replaces
each pixel with an average over a k-by-k neighborhood. It reduces random variation,
but also mixes both sides of an edge. We use odd sizes so there is a natural center.
`filter2D` can apply custom weights; `ddepth=-1` keeps the input's output depth.
"""),
        code("""kernel_size = 5
box_filtered = cv2.blur(gaussian_noisy, (kernel_size, kernel_size))
average_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size ** 2
custom_average = cv2.filter2D(gaussian_noisy, ddepth=-1, kernel=average_kernel)
show_images([gaussian_noisy, box_filtered, custom_average],
            ["Noisy input", "cv2.blur: local average", "filter2D: same weights"])
assert abs(float(average_kernel.sum()) - 1.0) < 1e-6
# Different optimized integer paths may round by one gray level.
assert np.abs(box_filtered.astype(np.int16) - custom_average.astype(np.int16)).max() <= 1
"""),
        md("""## Step 3 — Compare filters designed for different needs

`GaussianBlur(image, (k, k), sigmaX)` gives greater weight to nearby pixels. `sigmaX=0`
lets OpenCV infer a scale from the kernel. `medianBlur(image, k)` chooses the middle
value in each neighborhood rather than averaging; isolated extreme values then have
less influence. Here k must be an odd integer greater than 1.
"""),
        code("""gaussian_filtered = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0)
median_on_gaussian = cv2.medianBlur(gaussian_noisy, 5)
median_on_impulse = cv2.medianBlur(impulse_noisy, 5)
gaussian_on_impulse = cv2.GaussianBlur(impulse_noisy, (5, 5), 0)
show_images([gaussian_noisy, gaussian_filtered, median_on_gaussian,
             impulse_noisy, gaussian_on_impulse, median_on_impulse],
            ["Gaussian noise", "Gaussian filter", "Median filter",
             "Impulse noise", "Gaussian on impulses", "Median on impulses"], cols=3)
"""),
        md("""A bilateral filter uses both distance and intensity similarity: nearby pixels
with very different intensities receive less influence. In `bilateralFilter`, `d` is
the neighborhood diameter, `sigmaColor` sets the intensity-similarity scale, and
`sigmaSpace` sets the distance scale. It can preserve edges better, but costs more
computation and can flatten fine textures.
"""),
        code("""bilateral = cv2.bilateralFilter(gaussian_noisy, d=9, sigmaColor=40, sigmaSpace=40)
show_images([clean, gaussian_filtered, bilateral],
            ["Clean reference", "Gaussian smoothing", "Bilateral smoothing"])
# Inspect the same crop at the same display scale.
region = (slice(30, 165), slice(160, 360))
show_images([gaussian_noisy[region], gaussian_filtered[region], bilateral[region]],
            ["Noisy edge crop", "Gaussian edge crop", "Bilateral edge crop"])
"""),
        md("""## Step 4 — Measure the tradeoff instead of guessing

Mean squared error (MSE) averages squared differences from the clean reference; lower
is better for this metric. PSNR expresses that error on a logarithmic scale using the
maximum uint8 intensity 255; higher is better. An exact match gives infinite PSNR.
Neither score fully captures whether a face, text stroke, or important boundary survives.
"""),
        code("""def mse(reference, estimate):
    difference = reference.astype(np.float64) - estimate.astype(np.float64)
    return float(np.mean(difference ** 2))

def psnr(reference, estimate):
    error = mse(reference, estimate)
    return float("inf") if error == 0 else float(10 * np.log10(255.0 ** 2 / error))

assert mse(clean, clean) == 0
assert np.isinf(psnr(clean, clean))
gaussian_candidates = {
    "No filter": gaussian_noisy,
    "Box 5": box_filtered,
    "Gaussian 5": gaussian_filtered,
    "Median 5": median_on_gaussian,
    "Bilateral": bilateral,
}
for name, estimate in gaussian_candidates.items():
    print(f"{name:12} MSE={mse(clean, estimate):8.2f}   PSNR={psnr(clean, estimate):6.2f} dB")
"""),
        md("""A global average can hide a local problem. Compare a uniform patch inside the
rectangle with a strip crossing its edge. Less error in a smooth patch may accompany
more error at a boundary. We measure both using masks and display the actual sampling
locations so you can judge whether the comparison is fair.
"""),
        code("""flat_mask = np.zeros_like(clean)
flat_mask[60:120, 60:130] = 255
edge_mask = np.zeros_like(clean)
edge_mask[50:140, 156:166] = 255

def masked_mse(reference, estimate, mask):
    squared = (reference.astype(np.float64) - estimate.astype(np.float64)) ** 2
    return float(squared[mask > 0].mean())

for name, estimate in gaussian_candidates.items():
    print(f"{name:12} flat MSE={masked_mse(clean, estimate, flat_mask):8.2f}"
          f"  edge MSE={masked_mse(clean, estimate, edge_mask):8.2f}")
marked = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
marked[flat_mask > 0] = (0, 200, 0)
marked[edge_mask > 0] = (0, 0, 255)
show_images([marked], ["Green: flat patch; red: boundary strip"], cols=1)
"""),
        md("""### Parameter experiment: when does a larger kernel hurt?

Run a small controlled sweep over kernel sizes. Keep the noise fixed. The lowest MSE
in this list is the best of these candidates on this image; it is not proof of a universal
best setting. Look at text as well as the numerical score.
"""),
        code("""kernel_sizes = [3, 5, 9, 15]
sweep_images, sweep_scores = [], []
for k in kernel_sizes:
    filtered = cv2.GaussianBlur(gaussian_noisy, (k, k), 0)
    sweep_images.append(filtered)
    sweep_scores.append(mse(clean, filtered))
    print(f"k={k:2}: MSE={sweep_scores[-1]:.2f}")
show_images(sweep_images, [f"k={k}; MSE={score:.1f}" for k, score in zip(kernel_sizes, sweep_scores)], cols=2)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(kernel_sizes, sweep_scores, "o-")
ax.set(xlabel="Gaussian kernel width", ylabel="MSE against clean reference", xticks=kernel_sizes)
plt.tight_layout()
plt.show()
"""),
        md("""## Step 5 — Inspect border behavior and filter assumptions

At the image boundary, part of a kernel falls outside the array. A border rule supplies
those missing values. `BORDER_CONSTANT` uses zero here; `BORDER_REPLICATE` repeats
the nearest edge value; `BORDER_REFLECT_101` mirrors neighboring values without repeating
the border pixel. Border choice changes results near edges even on a uniform image.
"""),
        code("""uniform = np.full((50, 80), 180, dtype=np.uint8)
zero_border = cv2.GaussianBlur(uniform, (15, 15), 0, borderType=cv2.BORDER_CONSTANT)
replicate_border = cv2.GaussianBlur(uniform, (15, 15), 0, borderType=cv2.BORDER_REPLICATE)
reflect_border = cv2.GaussianBlur(uniform, (15, 15), 0, borderType=cv2.BORDER_REFLECT_101)
show_images([zero_border, replicate_border, reflect_border],
            ["Outside assumed black", "Replicated border", "Reflected border"])
assert zero_border[0, 0] < 180
assert np.all(replicate_border == 180) and np.all(reflect_border == 180)
"""),
        md("""A row profile makes edge spreading visible as numbers. The clean rectangle has
a sharp change; averaging spreads it across nearby pixels. Noise reduction and edge
preservation therefore compete even when a blurred output initially looks pleasant.
"""),
        code("""row = 90
fig, ax = plt.subplots(figsize=(10, 3))
for image, label in [(clean, "Clean"), (gaussian_noisy, "Noisy"),
                     (gaussian_filtered, "Gaussian"), (bilateral, "Bilateral")]:
    ax.plot(np.arange(135, 186), image[row, 135:186], label=label)
ax.set(xlabel="Column x across the rectangle boundary", ylabel="Intensity")
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""## Step 6 — Mini-project: a small denoising benchmark

Evaluate the same filter families on both noise types. Store results, choose the
lowest-MSE candidate for each synthetic case, and save those outputs. In a real
application the clean reference is usually unavailable; you would use representative
validation data and task-level measures such as text-reading or detection accuracy.
"""),
        code("""def filter_candidates(noisy):
    return {
        "unchanged": noisy,
        "gaussian5": cv2.GaussianBlur(noisy, (5, 5), 0),
        "median5": cv2.medianBlur(noisy, 5),
        "bilateral9": cv2.bilateralFilter(noisy, 9, 40, 40),
    }

lesson_dir = ROOT / "outputs" / "05_denoising"
lesson_dir.mkdir(parents=True, exist_ok=True)
benchmark = []
for noise_name, noisy in [("gaussian", gaussian_noisy), ("impulse", impulse_noisy)]:
    candidates = filter_candidates(noisy)
    best_name = min(candidates, key=lambda name: mse(clean, candidates[name]))
    for name, estimate in candidates.items():
        benchmark.append({"noise": noise_name, "filter": name,
                          "mse": mse(clean, estimate), "psnr": psnr(clean, estimate)})
    save_image(lesson_dir / f"{noise_name}_{best_name}.png", candidates[best_name])
    print(f"For {noise_name}, best tested MSE: {best_name}")
    show_images([noisy, candidates[best_name]], [noise_name + " input", best_name], cols=2)
"""),
        code("""import csv

benchmark_path = lesson_dir / "benchmark.csv"
with benchmark_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["noise", "filter", "mse", "psnr"])
    writer.writeheader()
    writer.writerows(benchmark)
print("Saved numerical comparison to:", benchmark_path)
"""),
        md("""### Exercises — attempt first

1. Compare median kernels 3, 5, and 7 on the impulse-noise image. Which minimizes MSE?
2. Repeat Gaussian filtering with sigma 1.0 and 3.0 using the same 15×15 kernel.
   Does a fixed kernel imply a fixed amount of smoothing?
3. Create an identity kernel for `filter2D` that leaves every pixel unchanged.
"""),
        code("""# Your practice area. Keep the clean reference and noisy inputs unchanged.
# median_scores = ...
"""),
        md("""### Worked solutions

Changing sigma changes the Gaussian weights even when kernel dimensions stay fixed.
An identity kernel has only its center equal to one; all neighbors have zero weight.
Use explicit comparisons to confirm your expectations.
"""),
        code("""median_scores = {}
for k in [3, 5, 7]:
    median_scores[k] = mse(clean, cv2.medianBlur(impulse_noisy, k))
print("Median MSE by size:", median_scores)
print("Best tested median size:", min(median_scores, key=median_scores.get))

sigma_one = cv2.GaussianBlur(gaussian_noisy, (15, 15), sigmaX=1.0)
sigma_three = cv2.GaussianBlur(gaussian_noisy, (15, 15), sigmaX=3.0)
show_images([sigma_one, sigma_three], ["15×15, sigma=1", "15×15, sigma=3"], cols=2)
identity_kernel = np.zeros((3, 3), dtype=np.float32)
identity_kernel[1, 1] = 1
identity_result = cv2.filter2D(clean, -1, identity_kernel)
assert np.array_equal(identity_result, clean)
"""),
        md("""### Reflection

- Why did we convert to a floating-point type before computing error?
- Which noise type was most suited to a median filter? What happened to thin strokes?
- Why can a filter win in a flat patch but lose near an edge?
- What does a good score on a generated image fail to establish about real photographs?

Official references: [Smoothing and filtering](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html),
[Filtering API and border modes](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html).
"""),
    ]
    save("05_filtering_and_noise.ipynb", cells)


def lesson_six():
    cells = intro(
        "06 — Thresholding, morphology, and clean segmentation masks", "Beginner → intermediate · 90–120 minutes",
        ["Read lessons 01–05: grayscale, masks, histograms, and filtering"],
        ["Choose global, Otsu, or adaptive thresholds", "Explain erosion, dilation, opening, and closing",
         "Measure segmentation quality against known labels", "Recover a document mask under uneven illumination"])
    cells += [
        md("""## Step 1 — Turn intensity into a yes/no decision

Thresholding assigns pixels to groups using their values. With `THRESH_BINARY`, values
strictly greater than the threshold become `maxval`; the rest become zero. With
`THRESH_BINARY_INV`, the assignment is reversed. Keep a clear convention: in this
lesson **white means foreground**, whether the original object was dark or bright.

`cv2.threshold` returns two values: the threshold actually used and the output image.
The underscore `_` is a conventional name for a result we do not need.
"""),
        code("""ramp = np.tile(np.arange(256, dtype=np.uint8), (80, 1))
threshold_value = 127
used_threshold, binary = cv2.threshold(ramp, threshold_value, 255, cv2.THRESH_BINARY)
_, inverse_binary = cv2.threshold(ramp, threshold_value, 255, cv2.THRESH_BINARY_INV)
show_images([ramp, binary, inverse_binary], ["Intensity ramp", "Bright foreground", "Dark foreground"])
assert np.all(binary[ramp <= threshold_value] == 0)
assert np.all(binary[ramp > threshold_value] == 255)
assert np.array_equal(cv2.bitwise_not(binary), inverse_binary)
print("Threshold used:", used_threshold)
"""),
        md("""A single global threshold works well when foreground and background intensities
are separated. The next scene has known object pixels and two brightness groups.
Gaussian noise makes the decision less trivial. The true mask will let us measure
performance instead of judging only by appearance.
"""),
        code("""truth_objects = np.zeros((260, 420), dtype=np.uint8)
cv2.circle(truth_objects, (100, 130), 55, 255, -1)
cv2.rectangle(truth_objects, (235, 60), (350, 200), 255, -1)
clean_objects = np.where(truth_objects > 0, 185, 45).astype(np.uint8)
threshold_rng = np.random.default_rng(42)
noisy_objects = np.clip(clean_objects.astype(np.float32)
                        + threshold_rng.normal(0, 24, clean_objects.shape), 0, 255).astype(np.uint8)
_, manual_mask = cv2.threshold(noisy_objects, 115, 255, cv2.THRESH_BINARY)
show_images([noisy_objects, truth_objects, manual_mask], ["Noisy grayscale", "Known object pixels", "Manual threshold 115"])
"""),
        md("""## Step 2 — Let Otsu choose one threshold

Otsu searches for a threshold that separates the intensity histogram into two groups
with low within-group variation. It chooses a number, not the semantic meaning of an
object. Use `THRESH_BINARY + THRESH_OTSU`; the supplied threshold argument is ignored.
Clear two-group histograms are a good starting case; shadows, tiny targets, or overlapping
intensities make the problem harder.
"""),
        code("""otsu_value, otsu_mask = cv2.threshold(noisy_objects, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
smoothed_objects = cv2.GaussianBlur(noisy_objects, (5, 5), 0)
smooth_otsu_value, smooth_otsu_mask = cv2.threshold(
    smoothed_objects, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu on noisy input:", otsu_value, "after Gaussian smoothing:", smooth_otsu_value)
show_images([manual_mask, otsu_mask, smooth_otsu_mask],
            ["Manual threshold", "Otsu", "Gaussian smoothing → Otsu"])
"""),
        code("""hist = cv2.calcHist([noisy_objects], [0], None, [256], [0, 256]).ravel()
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(hist)
ax.axvline(otsu_value, color="red", linestyle="--", label=f"Otsu = {otsu_value:.0f}")
ax.set(xlabel="Intensity", ylabel="Pixel count", title="Can one intensity value separate the groups?")
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""Use intersection over union (IoU) to compare binary masks. A correct rejection
of background does not inflate this score: only foreground intersection and union
enter the calculation. If both masks are empty, we define their IoU as 1 for this lesson.
"""),
        code("""def mask_iou(predicted, truth):
    p, t = predicted > 0, truth > 0
    union = np.count_nonzero(p | t)
    return float(np.count_nonzero(p & t) / union) if union else 1.0

for name, mask in [("Manual", manual_mask), ("Otsu", otsu_mask), ("Smoothed Otsu", smooth_otsu_mask)]:
    print(f"{name:15} IoU = {mask_iou(mask, truth_objects):.4f}")
assert mask_iou(truth_objects, truth_objects) == 1.0
"""),
        md("""## Step 3 — Handle uneven illumination with local decisions

Create a document with dark text on a background that becomes brighter from left to
right. The foreground labels come from the text drawing, before lighting or noise is
added. A global threshold may confuse shadowed paper with text or miss text in bright
regions because the intensity ranges overlap.
"""),
        code("""document_truth = np.zeros((300, 620), dtype=np.uint8)
for text, y in [("LEARN OPENCV", 80), ("TRY - OBSERVE", 155), ("CHANGE - EXPLAIN", 230)]:
    cv2.putText(document_truth, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, 255, 2, cv2.LINE_8)
paper = np.tile(np.linspace(95, 235, 620, dtype=np.float32), (300, 1))
document_clean = paper - (document_truth > 0).astype(np.float32) * 65
document_rng = np.random.default_rng(12)
document = np.clip(document_clean + document_rng.normal(0, 3, paper.shape), 0, 255).astype(np.uint8)
global_value, global_document = cv2.threshold(document, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
show_images([document, document_truth, global_document],
            ["Synthetic unevenly lit document", "Known text pixels", f"Global Otsu inverse ({global_value:.0f})"])
"""),
        md("""`adaptiveThreshold` estimates a local reference from a neighborhood, then subtracts
`C`. `blockSize` is the odd neighborhood width in pixels and must be greater than 1.
`ADAPTIVE_THRESH_GAUSSIAN_C` weights nearby pixels more strongly; `MEAN_C` uses a plain
average. With the inverse binary mode, sufficiently dark pixels become white foreground.
The function expects an 8-bit single-channel image.
"""),
        code("""block_size = 31
constant_c = 10
adaptive_document = cv2.adaptiveThreshold(document, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, block_size, constant_c)
mean_document = cv2.adaptiveThreshold(document, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY_INV, block_size, constant_c)
show_images([global_document, mean_document, adaptive_document],
            ["Global decision", "Local mean", "Local Gaussian"])
for name, mask in [("Global", global_document), ("Adaptive mean", mean_document),
                   ("Adaptive Gaussian", adaptive_document)]:
    print(f"{name:18} IoU = {mask_iou(mask, document_truth):.4f}")
"""),
        md("""### Parameter experiment

Try block sizes 11, 31, and 61, then change C from 10 to 3 or 18. Small neighborhoods
may follow the letters themselves; large neighborhoods may fail to follow illumination.
In inverse mode, increasing C generally demands a darker deviation from the local
reference, reducing selected pixels. Check both faint letters and background speckles.
"""),
        code("""experiment_c = 10
block_sizes = [11, 31, 61]
adaptive_variants = [cv2.adaptiveThreshold(document, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, b, experiment_c) for b in block_sizes]
show_images(adaptive_variants,
            [f"Block {b}, C={experiment_c}; IoU={mask_iou(mask, document_truth):.3f}"
             for b, mask in zip(block_sizes, adaptive_variants)])
"""),
        md("""## Step 4 — Use shape rules to refine masks

Morphology operates on neighborhoods defined by a **structuring element**. For white
foreground, erosion shrinks regions and dilation expands them. `getStructuringElement`
constructs a rectangle, ellipse, or cross; its shape influences which details survive.
`iterations=1` applies one pass. More iterations can erase or merge objects.
"""),
        code("""shape_mask = np.zeros((220, 420), dtype=np.uint8)
cv2.rectangle(shape_mask, (40, 50), (155, 165), 255, -1)
cv2.circle(shape_mask, (295, 110), 60, 255, -1)
cv2.line(shape_mask, (155, 110), (235, 110), 255, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
eroded = cv2.erode(shape_mask, kernel, iterations=1)
dilated = cv2.dilate(shape_mask, kernel, iterations=1)
show_images([shape_mask, eroded, dilated], ["Original white foreground", "Erosion: shrink", "Dilation: expand"])
assert cv2.countNonZero(eroded) <= cv2.countNonZero(shape_mask)
assert cv2.countNonZero(dilated) >= cv2.countNonZero(shape_mask)
print("Structuring element:\\n", kernel)
"""),
        md("""Opening is erosion followed by dilation; it removes small foreground protrusions
and isolated speckles. Closing is dilation followed by erosion; it can fill small
background gaps and holes. These operations do not understand which structures are
important. A thin legitimate line can disappear during opening, and two nearby objects
can merge during closing.
"""),
        code("""damaged = shape_mask.copy()
damage_rng = np.random.default_rng(23)
for x, y in damage_rng.integers([5, 5], [415, 215], size=(80, 2)):
    if damaged[y, x] == 0:
        cv2.circle(damaged, (int(x), int(y)), 1, 255, -1)
for center in [(80, 85), (115, 130), (280, 100), (315, 130)]:
    cv2.circle(damaged, center, 2, 0, -1)
opened = cv2.morphologyEx(damaged, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(damaged, cv2.MORPH_CLOSE, kernel)
open_then_close = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
show_images([damaged, opened, closed, open_then_close],
            ["Speckles and holes", "Opening", "Closing", "Opening → closing"], cols=2)
"""),
        md("""A morphological gradient is dilation minus erosion: it emphasizes the boundary
band. A top-hat is the original minus its opening, highlighting small bright structures.
A black-hat is the closing minus the original, highlighting small dark structures.
Their meaning depends on kernel size: 'small' is relative to the neighborhood.
"""),
        code("""morph_gradient = cv2.morphologyEx(shape_mask, cv2.MORPH_GRADIENT, kernel)
top_hat = cv2.morphologyEx(damaged, cv2.MORPH_TOPHAT, kernel)
black_hat = cv2.morphologyEx(damaged, cv2.MORPH_BLACKHAT, kernel)
show_images([morph_gradient, top_hat, black_hat],
            ["Boundary band", "Small bright structures", "Small dark structures"])
"""),
        md("""## Step 5 — Evaluate cleaning without hiding lost detail

Compare the cleaned mask to the undamaged reference. A mask can look cleaner while
losing its legitimate thin connector. Larger kernels impose a stronger shape assumption.
This sweep makes that failure visible instead of treating morphology as an automatic
'cleanup' button. The best choice depends on which features the task must preserve.
"""),
        code("""morph_sizes = [3, 5, 9]
cleaned_variants = []
for size in morph_sizes:
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    result = cv2.morphologyEx(damaged, cv2.MORPH_OPEN, element)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, element)
    cleaned_variants.append(result)
    print(f"Kernel {size}: IoU={mask_iou(result, shape_mask):.4f}, area={cv2.countNonZero(result)}")
show_images(cleaned_variants,
            [f"Open → close, kernel {size}" for size in morph_sizes])
"""),
        md("""## Step 6 — Mini-project: recover a document's text mask

Build a pipeline of optional smoothing, adaptive thresholding, and optional morphology.
Evaluate each stage. Text is thin, so even a 3×3 opening can remove useful strokes.
Keep the simpler candidate if a refinement reduces IoU. This miniature comparison
uses one generated image for learning; a deployment decision needs separate validation
images and more varied fonts, textures, and illumination.
"""),
        code("""lightly_smoothed = cv2.GaussianBlur(document, (3, 3), 0)
smooth_adaptive = cv2.adaptiveThreshold(lightly_smoothed, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
small_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
opened_text = cv2.morphologyEx(smooth_adaptive, cv2.MORPH_OPEN, small_cross)
text_candidates = {
    "adaptive_raw": adaptive_document,
    "adaptive_smoothed": smooth_adaptive,
    "adaptive_then_open": opened_text,
}
scores = {name: mask_iou(mask, document_truth) for name, mask in text_candidates.items()}
best_name = max(scores, key=scores.get)
best_mask = text_candidates[best_name]
for name, score in scores.items():
    print(f"{name:24} IoU={score:.4f}")
print("Best among these candidates:", best_name)
show_images(list(text_candidates.values()),
            [f"{name}\\nIoU={scores[name]:.3f}" for name in text_candidates])
"""),
        md("""An error overlay distinguishes two mistakes. **False positives** are predicted
foreground that is really background; **false negatives** are missed foreground.
Here green means correct text, red means extra pixels, and blue means missed text.
Black means correctly rejected background. This reveals where a single score hides
different kinds of failure.
"""),
        code("""prediction, expected = best_mask > 0, document_truth > 0
error_overlay = np.zeros((*best_mask.shape, 3), dtype=np.uint8)
error_overlay[prediction & expected] = (0, 180, 0)
error_overlay[prediction & ~expected] = (0, 0, 255)
error_overlay[~prediction & expected] = (255, 0, 0)
false_positives = int(np.count_nonzero(prediction & ~expected))
false_negatives = int(np.count_nonzero(~prediction & expected))
print("False positives:", false_positives, "false negatives:", false_negatives)
show_images([document, best_mask, error_overlay],
            ["Input", "Best tested text mask", "Green correct / red extra / blue missed"])
"""),
        code("""lesson_dir = ROOT / "outputs" / "06_thresholding"
lesson_dir.mkdir(parents=True, exist_ok=True)
save_image(lesson_dir / "document_input.png", document)
save_image(lesson_dir / "document_mask.png", best_mask)
save_image(lesson_dir / "document_errors.png", error_overlay)
assert set(np.unique(best_mask)).issubset({0, 255})
print("Saved input, selected mask, and error overlay to:", lesson_dir)
"""),
        md("""### Exercises — attempt first

1. Compare global thresholds 70, 115, and 160 on `noisy_objects`. Rank by IoU.
2. Construct a white rectangle containing a tiny black hole and show how closing
   fills it. Verify the center pixel becomes white.
3. On the synthetic document, compare C values 3, 10, and 18 with block size fixed at
   31. Explain the precision/recall tradeoff by counting extra and missed pixels.
"""),
        code("""# Your practice area: change one parameter at a time.
# ranked_thresholds = ...
"""),
        md("""### Worked solutions

A larger C in inverse adaptive thresholding generally selects fewer pixels, which
can reduce extra detections but miss faint text. The best balance depends on the task:
missing a character and adding a background speck may have different costs.
"""),
        code("""ranked_thresholds = []
for threshold in [70, 115, 160]:
    _, mask = cv2.threshold(noisy_objects, threshold, 255, cv2.THRESH_BINARY)
    ranked_thresholds.append((mask_iou(mask, truth_objects), threshold))
print("Global thresholds ranked by IoU:", sorted(ranked_thresholds, reverse=True))

hole_example = np.zeros((80, 100), dtype=np.uint8)
hole_example[15:65, 20:80] = 255
hole_example[39:42, 49:52] = 0
hole_filled = cv2.morphologyEx(hole_example, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
assert hole_filled[40, 50] == 255
show_images([hole_example, hole_filled], ["Small black hole", "Closing fills it"], cols=2)

for c_value in [3, 10, 18]:
    candidate = cv2.adaptiveThreshold(document, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 31, c_value) > 0
    expected = document_truth > 0
    extra = np.count_nonzero(candidate & ~expected)
    missed = np.count_nonzero(~candidate & expected)
    print(f"C={c_value:2}: extra={extra:5}, missed={missed:5}, IoU={mask_iou(candidate, expected):.4f}")
"""),
        md("""### Reflection and next step

- Why does Otsu's returned threshold differ between the noisy and smoothed inputs?
- Why is an inverse threshold convenient for dark ink on bright paper?
- Which legitimate structure did a large opening remove in the shape experiment?
- Why should you inspect false positives and false negatives separately?
- Can you explain the full pipeline: image → grayscale → threshold → optional shape
  refinement → evaluation → saved artifact?

You now have the foundations for contours, object measurements, geometric transforms,
and feature-based methods. Rebuild one mini-project using your own local image before
moving on, and record which assumptions stop working.

Official references: [Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html),
[Morphological transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html).
"""),
    ]
    save("06_thresholding_and_morphology.ipynb", cells)


def main():
    lesson_one()
    lesson_two()
    lesson_three()
    lesson_four()
    lesson_five()
    lesson_six()


if __name__ == "__main__":
    main()

