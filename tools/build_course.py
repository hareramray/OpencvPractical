"""Generate orientation and final capstones. Run with python -m tools.build_course."""
from tools.notebook_factory import code, intro, md, save


def orientation():
    cells = intro("00 — Start here: your practical OpenCV learning path", "Orientation", "No OpenCV experience; basic computer use", ["Learn how notebook cells and kernels work.", "Verify the project environment and run your first image pipeline.", "Choose a study rhythm and learn how to experiment deliberately."])
    cells += [md("""
    ## Step 1 — Understand your tools

    **Python** is the language. **NumPy** stores images as grids of numbers.
    **OpenCV** provides image/video algorithms. **Matplotlib** displays results inside
    the page. **JupyterLab** lets you mix explanations with executable Python cells.

    A **kernel** is the running Python process. Variables persist between cells until
    you restart it. A number such as `[4]` records execution order, not the position of
    a cell. Run from top to bottom so the code and your mental model agree.

    Click a code cell and press **Shift+Enter**. A `[*]` means it is still running.
    Use **Kernel → Interrupt Kernel** for an unexpectedly long calculation. Use
    **Kernel → Restart Kernel and Run All Cells** to reproduce the complete lesson.
    """), code("""
    import sys
    import platform
    import importlib.metadata as metadata

    print("Python executable:", sys.executable)
    print("Python version:", platform.python_version())
    for package in ["opencv-python-headless", "numpy", "matplotlib", "jupyterlab"]:
        print(f"{package}: {metadata.version(package)}")
    assert sys.version_info >= (3, 10), "Use a supported modern Python environment."
    """), md("""
    **Expected:** the interpreter path contains this project's `.venv`, and four package
    versions are printed. If imports fail, choose **Python (OpenCV Practical)** as your
    kernel. The README contains installation commands and troubleshooting.

    ## Step 2 — Your first image

    `make_scene()` draws an original sample with rectangles, a circle, text, and stripes.
    It returns a NumPy array. For a color image the shape is `(height, width, channels)`.
    Each `uint8` channel stores an integer from 0 to 255.
    """), code("""
    image = make_scene()
    print("Shape:", image.shape, "Data type:", image.dtype)
    print("A BGR pixel:", image[100, 100])  # row first, column second
    show_images([image], ["Your reusable practice scene"])
    assert image.shape == (420, 640, 3)
    """), md("""
    ## Step 3 — Build a tiny pipeline

    A **pipeline** applies several operations in a known order. Here we convert to
    grayscale, smooth small details, and detect edges. `cvtColor` changes the color
    representation. `GaussianBlur` averages nearby pixels with a center-weighted
    kernel. `Canny` finds strong boundaries and keeps connected weaker boundaries.

    You do not need to understand the full algorithms yet: later lessons unpack
    each step. First observe how the representation changes.
    """), code("""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)
    edges = cv2.Canny(blurred, threshold1=60, threshold2=150)
    show_images([gray, blurred, edges], ["1. Grayscale", "2. Blur", "3. Edges"])
    print("Fraction of edge pixels:", round(np.count_nonzero(edges) / edges.size, 4))
    assert gray.shape == edges.shape == image.shape[:2]
    """), md("""
    ## Step 4 — Make a prediction, change a parameter, measure

    A larger blur kernel suppresses fine detail. Predict which stripes or letters will
    disappear before running the next cell. Odd kernel sizes such as 3, 7, or 15 give
    the kernel a center pixel. Edge count is a diagnostic, not a quality score: fewer
    edges may mean less noise, or it may mean useful detail was lost.
    """), code("""
    experiments = []
    titles = []
    for kernel_size in [3, 7, 15]:
        smooth = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        result = cv2.Canny(smooth, 60, 150)
        experiments.append(result)
        titles.append(f"Kernel {kernel_size}: {np.count_nonzero(result)} edge pixels")
    show_images(experiments, titles)
    """), md("""
    ## Step 5 — Save and reload your result

    Paths are anchored at `ROOT`, so you do not have to guess the notebook's current
    folder. `outputs/` contains replaceable generated results. `save_image` creates
    parent directories and checks encoding. `load_image` gives a clear error if a file
    is missing or cannot be decoded. Notebook 02 teaches the underlying OpenCV APIs.
    """), code("""
    output_path = save_image(ROOT / "outputs" / "00_start" / "first_edges.png", edges)
    round_trip = load_image(output_path, cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(round_trip, edges), "PNG should preserve these pixels exactly."
    print("Saved and verified:", output_path)
    """), md("""
    ## Mini-project — Make a contact sheet

    A contact sheet puts multiple outputs beside one another. Here all images must
    have the same height and channel count before concatenation. `GRAY2BGR` repeats
    a gray value across three channels so the arrays have compatible shapes.
    """), code("""
    edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    contact_sheet = np.hstack([image, edge_color])
    show_images([contact_sheet], ["Input and its edge map"], figsize=(12, 4))
    save_image(ROOT / "outputs" / "00_start" / "contact_sheet.png", contact_sheet)
    """), md("""
    ## Exercise — Try first

    1. Build another edge map using thresholds 30 and 80. Does it keep more detail?
    2. Crop the upper-left quarter of `image`. What should its shape be?
    3. Save the crop to `outputs/00_start/my_crop.png`.

    Type your code below; the empty exercise cell does not prevent Run All. Write a
    sentence explaining your results before reading the solution.
    """), code("""
    # Your experiment goes here. Use new variable names to keep the reference intact.
    """), md("## Worked solution — compare after trying"), code("""
    extra_edges = cv2.Canny(blurred, 30, 80)
    h, w = image.shape[:2]
    crop = image[:h // 2, :w // 2].copy()
    assert crop.shape == (210, 320, 3)
    show_images([edges, extra_edges, crop], ["Thresholds 60/150", "Thresholds 30/80", "Upper-left quarter"])
    save_image(ROOT / "outputs" / "00_start" / "my_crop.png", crop)
    print("Edge counts:", np.count_nonzero(edges), np.count_nonzero(extra_edges))
    """), md("""
    ## Use your own images

    Put a photo in `data/personal/`, set `personal_path` below, and rerun this cell.
    Keep the generated examples as a baseline. A method that works on clean shapes
    can fail under shadows, reflections, motion blur, clutter, or new camera angles.
    """), code("""
    personal_path = None  # Example: ROOT / "data" / "personal" / "my_photo.jpg"
    if personal_path is not None:
        personal_image = load_image(personal_path)
        show_images([personal_image], ["My own input"])
    else:
        print("Optional personal image skipped. Every core lesson works without one.")
    """), md("""
    ## Your learning roadmap

    | Stage | Notebooks | Milestone |
    |---|---|---|
    | Foundations | 01–06 | Explain image arrays and clean up a foreground mask |
    | Intermediate | 07–12 | Measure objects, align images, and process video |
    | Advanced | 13–17 | Track motion, reason about cameras/depth, evaluate models |
    | Projects and engineering | 18–20 | Build, measure, and improve complete pipelines |

    Start with [01 — NumPy and images](01_numpy_and_images.ipynb). The root README
    links all 21 notebooks. No lesson relies on variables or output files from another.

    Suggested eight-week plan: 00–03, 04–06, 07–09, 10–12, 13–14, 15–17, 18–19,
    then 20 and an independent project. Slow down when exercises reveal gaps.

    ## Keep a small experiment journal

    Duplicate this Markdown cell and fill it out after each lesson:

    - **Question:** What am I trying to measure or detect?
    - **Prediction:** What do I think will happen, and why?
    - **Change:** Which single input or parameter did I change?
    - **Evidence:** Which image, count, or metric supports my observation?
    - **Failure:** When does the method stop working?
    - **Next experiment:** What would distinguish two possible explanations?

    ## Readiness check

    Can you explain why coordinates use row/column order, why the edge map has one
    channel, and why restarting a kernel is useful? If not, notebook 01 builds those
    foundations. Expertise grows from diagnosing failures on varied data, not from
    memorizing OpenCV function names.

    **Reference:** [Official OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html).
    """)]
    save("00_start_here.ipynb", cells)


def inspection():
    cells = intro("19 — Capstone: visual inspection with honest evaluation", "Advanced project", ["06 masks", "09 transforms", "10 registration", "16 evaluation"], ["Align a manufactured part to a reference and detect changes.", "Choose a threshold on validation data and evaluate once on held-out data.", "Distinguish pixel-level and image-level errors, and document failure cases."])
    cells += [md("""
    ## The task and its assumptions

    Our camera sees one flat part at nearly fixed scale and orientation. A good
    reference is available. We detect scratches or missing bright pads from image
    differences. This works because expected geometry is stable; it is not a general
    anomaly detector. All samples are generated here, with known defect masks.

    ## Step 1 — Create a reference and labeled samples

    `make_inspection_sample` creates a good or damaged part, changes global brightness,
    adds sensor-like noise, then translates it by a few pixels. The returned ground
    truth mask is in **reference coordinates**, matching our final aligned output.
    Each seed gives an independently generated sample.
    """), code("""
    reference = np.full((160, 224), 40, np.uint8)
    cv2.rectangle(reference, (20, 15), (204, 145), 145, -1)
    for center in [(65, 55), (155, 55), (65, 110), (155, 110)]:
        cv2.circle(reference, center, 14, 225, -1)
        cv2.circle(reference, center, 5, 70, -1)
    cv2.putText(reference, "A7", (95, 96), cv2.FONT_HERSHEY_SIMPLEX, .5, 65, 2)
    roi_mask = np.zeros_like(reference)
    roi_mask[23:138, 28:197] = 255

    def make_inspection_sample(seed, defective):
        sample_rng = np.random.default_rng(seed)
        image = reference.copy()
        truth = np.zeros_like(reference)
        if defective:
            if seed % 3 == 0:
                cv2.circle(image, (155, 55), 14, 145, -1)  # missing pad
                cv2.circle(truth, (155, 55), 14, 255, -1)
            else:
                y = int(sample_rng.integers(70, 105))
                cv2.line(image, (45, y), (175, y + 9), 35, 5)
                cv2.line(truth, (45, y), (175, y + 9), 255, 5)
        shift = sample_rng.integers(-4, 5, size=2).astype(float)
        matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32)
        shifted = cv2.warpAffine(image, matrix, (224, 160), borderMode=cv2.BORDER_REPLICATE)
        noise = sample_rng.normal(0, 2, shifted.shape)
        brightness = sample_rng.uniform(-12, 12)
        shifted = np.clip(shifted.astype(np.float32) + noise + brightness, 0, 255).astype(np.uint8)
        return {"image": shifted, "truth": truth, "defective": bool(defective), "shift": shift, "seed": seed}

    sample = make_inspection_sample(103, True)
    show_images([reference, sample["image"], sample["truth"]], ["Known good", "Observed part", "Truth in reference coordinates"])
    """), md("""
    ## Step 2 — Align before subtracting

    Even a good part shifted three pixels produces differences around every edge.
    `phaseCorrelate(reference, observed)` estimates translation `(dx, dy)` from the
    Fourier transforms. To align the observation, apply the **negative** displacement.
    The response is a diagnostic peak measure, not a calibrated probability.

    We assume translation only. Rotation, scale, strong perspective changes, repeated
    textures, or large defects can invalidate this estimate. A production system
    needs an alignment-quality gate and labeled examples of those failures.
    """), code("""
    def align_part(observed):
        window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(reference.astype(np.float32), observed.astype(np.float32), window)
        matrix = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]]], np.float32)
        aligned = cv2.warpAffine(observed, matrix, (reference.shape[1], reference.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        return aligned, np.array(shift), float(response)

    aligned, estimated_shift, response = align_part(sample["image"])
    print("True shift:", sample["shift"], "Estimated:", estimated_shift.round(2), "Response:", round(response, 3))
    assert np.linalg.norm(estimated_shift - sample["shift"]) < 1.5
    show_images([cv2.absdiff(reference, sample["image"]), cv2.absdiff(reference, aligned)], ["Difference before alignment", "Difference after alignment"])
    """), md("""
    ## Step 3 — Correct brightness and isolate changed regions

    Compute the median intensity difference inside the region of interest (ROI).
    If most ROI pixels are unchanged, this estimates a global brightness offset
    robustly. Subtract it in floating point to avoid unsigned-integer wraparound.

    Threshold the absolute difference, open with a 3×3 kernel to remove isolated noise,
    then keep connected components above a minimum area. A thin real scratch can be
    removed by this cleanup; minimum defect size is an explicit design choice.
    """), code("""
    def inspect_part(observed, difference_threshold=25, min_area=30):
        aligned, shift, response = align_part(observed)
        offset = float(np.median((aligned.astype(np.float32) - reference)[roi_mask > 0]))
        corrected = np.clip(aligned.astype(np.float32) - offset, 0, 255)
        difference = np.abs(corrected - reference.astype(np.float32))
        binary = np.where((difference > difference_threshold) & (roi_mask > 0), 255, 0).astype(np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(cleaned)
        prediction = np.zeros_like(cleaned)
        regions = []
        for label in range(1, count):  # zero is the background
            x, y, w, h, area = stats[label]
            if area >= min_area:
                prediction[labels == label] = 255
                regions.append({"box": [int(x), int(y), int(w), int(h)], "area": int(area)})
        return {"mask": prediction, "aligned": aligned, "difference": difference,
                "regions": regions, "defective": bool(regions), "shift": shift,
                "alignment_response": response, "brightness_offset": offset}

    result = inspect_part(sample["image"])
    show_images([result["aligned"], result["difference"], result["mask"]], ["Aligned", "Absolute difference (float scale)", "Detected changes"])
    print("Regions:", result["regions"])
    """), md("""
    ## Step 4 — Measure what matters

    Image-level **precision** asks: of flagged parts, how many were defective?
    **Recall** asks: of defective parts, how many were flagged? A false negative
    ships a defect; a false positive causes unnecessary review. The acceptable
    tradeoff depends on the application.

    Pixel intersection-over-union (IoU) measures localization. For a defective image,
    `IoU = intersection / union` of predicted and true masks. We report it on positive
    samples separately so many perfect empty masks do not inflate the score.
    """), code("""
    def evaluate(samples, threshold, min_area=30):
        tp = fp = fn = tn = 0
        positive_ious = []
        for item in samples:
            output = inspect_part(item["image"], threshold, min_area)
            predicted, actual = output["defective"], item["defective"]
            tp += int(predicted and actual)
            fp += int(predicted and not actual)
            fn += int(not predicted and actual)
            tn += int(not predicted and not actual)
            if actual:
                p, t = output["mask"] > 0, item["truth"] > 0
                positive_ious.append(float(np.count_nonzero(p & t) / max(1, np.count_nonzero(p | t))))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": precision,
                "recall": recall, "F1": 2 * precision * recall / max(1e-12, precision + recall),
                "positive_mean_IoU": float(np.mean(positive_ious))}

    validation = [make_inspection_sample(1000 + i, i % 2 == 1) for i in range(20)]
    candidates = [(threshold, evaluate(validation, threshold)) for threshold in [15, 25, 35, 45]]
    for threshold, metrics in candidates:
        print(f"Threshold {threshold}: {metrics}")
    # Select on validation only. Localization breaks ties in image-level F1.
    best_threshold, _ = max(candidates, key=lambda pair: (pair[1]["F1"], pair[1]["positive_mean_IoU"]))
    print("Chosen threshold:", best_threshold)
    """), md("""
    The generated parts are deliberately easy to inspect. High validation scores
    establish that the code handles these known conditions, not that it is ready
    for a factory. Test data below use different seeds but the same generator;
    they do not establish performance on new defect families or real cameras.

    ## Step 5 — Freeze settings, then evaluate held-out samples

    Once you inspect the test results, resist changing settings to improve this same
    test set. Further development needs a new final holdout. Here `min_area=30` was
    fixed before validation; threshold is the only tuned parameter.
    """), code("""
    held_out = [make_inspection_sample(2000 + i, i % 2 == 1) for i in range(24)]
    test_metrics = evaluate(held_out, best_threshold)
    print("Held-out results:", test_metrics)
    assert sum(test_metrics[key] for key in ["TP", "FP", "FN", "TN"]) == len(held_out)
    assert 0 <= test_metrics["positive_mean_IoU"] <= 1

    matrix = np.array([[test_metrics["TN"], test_metrics["FP"]], [test_metrics["FN"], test_metrics["TP"]]])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(matrix, cmap="Blues")
    for (r, c), value in np.ndenumerate(matrix):
        ax.text(c, r, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Good", "Defect"], yticklabels=["Good", "Defect"], xlabel="Predicted", ylabel="Actual", title="Held-out image decisions")
    plt.show()
    plt.close(fig)
    """), md("""
    ## Step 6 — Review overlays and export evidence

    An aggregate score can hide a systematic error. Inspect overlays on both good
    and defective samples. Red rectangles show accepted components. Saving the
    reference, settings, and metrics makes an experiment easier to reproduce.
    """), code("""
    import json

    overlays = []
    for item in held_out[:4]:
        output = inspect_part(item["image"], best_threshold)
        overlay = cv2.cvtColor(output["aligned"], cv2.COLOR_GRAY2BGR)
        for region in output["regions"]:
            x, y, w, h = region["box"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        overlays.append(overlay)
    show_images(overlays, [f"Actual defect: {item['defective']}" for item in held_out[:4]], cols=2)
    output_dir = ROOT / "outputs" / "19_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(output_dir / "reference.png", reference)
    save_image(output_dir / "example_overlay.png", overlays[1])
    report = {"threshold": int(best_threshold), "min_area": 30, "test_metrics": test_metrics,
              "test_seeds": [item["seed"] for item in held_out], "opencv": cv2.__version__,
              "scope": "Generated translation/noise/global-brightness samples only"}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved report:", output_dir / "report.json")
    """), md("""
    ## Stress test — Break an assumption

    Add a local shadow to a **good** part. Global median compensation cannot remove a
    local lighting change, so the inspector may call it a defect. A stronger system
    might use controlled lighting, local normalization, a better image representation,
    or learned segmentation. Each change still needs separate evaluation.
    """), code("""
    shadowed = reference.copy()
    shadowed[60:120, 85:135] = np.clip(shadowed[60:120, 85:135].astype(np.int16) - 55, 0, 255).astype(np.uint8)
    shadow_result = inspect_part(shadowed, best_threshold)
    show_images([shadowed, shadow_result["mask"]], ["Good part with local shadow", "False defect regions"])
    print("Flagged as defective:", shadow_result["defective"])
    """), md("""
    ## Exercises — Try first

    1. Measure validation performance for minimum areas 10, 30, and 100. Keep the test
       set closed while exploring. Which setting could miss small scratches?
    2. Calculate the number of parts sent to review per 100 observations on the holdout.
    3. Design a new data split that prevents near-duplicate photos of one physical part
       from appearing in both training and testing.
    4. Add rotation or a tiny two-pixel scratch to the generator and evaluate as a new
       stress set, without silently adding it to the original accuracy claim.
    """), code("# Your experiments and explanation go here."), md("## Worked solutions"), code("""
    for area in [10, 30, 100]:
        metrics = evaluate(validation, best_threshold, min_area=area)
        print(f"min_area={area}: F1={metrics['F1']:.3f}, positive IoU={metrics['positive_mean_IoU']:.3f}")
    review_rate = 100 * (test_metrics["TP"] + test_metrics["FP"]) / len(held_out)
    print(f"Holdout review rate: {review_rate:.1f} per 100 parts")
    """), md("""
    For exercise 3, split by physical part and capture session, not by individual image.
    Keep a separate camera/date if generalization across capture conditions matters.
    For exercise 4, preserve the baseline generator and report stress-set performance
    separately. If experiments guide design changes, reserve another final test set.

    ## Project extension and completion checklist

    Photograph a small collection of household parts on a fixed background. Define
    what counts as a defect before labeling. Record false positives and false negatives,
    then investigate the failures with overlays. Do not use this exercise for safety-
    critical quality decisions.

    You should be able to explain the order of alignment/normalization/difference,
    justify threshold and area settings, calculate precision/recall/IoU, and state at
    least three conditions where the pipeline fails.

    **References:** [Phase correlation](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html),
    [shape analysis and connected components](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).
    Next: [20 — Performance and reproducible pipelines](20_performance_and_reproducible_pipelines.ipynb).
    """)]
    save("19_visual_inspection_project.ipynb", cells)


def engineering():
    cells = intro("20 — Advanced practice: performance and reproducible pipelines", "Advanced engineering", ["01 NumPy", "06 masks", "08 components", "18–19 projects"], ["Turn exploratory cells into a reusable, validated function.", "Measure latency without assuming faster code is equally accurate.", "Save configuration, outputs, and environment information for reproducibility."])
    cells += [md("""
    ## Step 1 — Define an explicit input/output contract

    A reliable function states what it accepts and returns. Our example counts bright
    connected components on a dark background, assuming each one is a part. The
    generated parts are circular, but this algorithm does not check circularity.
    Inputs are non-empty `uint8` BGR arrays;
    outputs include a binary mask and component measurements. Configuration is
    separate from the algorithm, so experiments are easy to repeat.

    Save the implementation to an ordinary Python module, then import it as a real
    application would. The module is intentionally small and readable. `dataclass`
    bundles named settings; `frozen=True` prevents accidental changes after creation.
    """), code('''
    from textwrap import dedent
    import importlib.util

    output_dir = ROOT / "outputs" / "20_engineering"
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_source = dedent("""
    from dataclasses import dataclass
    import cv2
    import numpy as np

    @dataclass(frozen=True)
    class Config:
        threshold: int = 130
        blur_size: int = 5
        min_area: int = 80

    def analyze(image, config=Config()):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise TypeError("Expected a NumPy uint8 image.")
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("Expected a non-empty H x W x 3 BGR image.")
        for name in ("threshold", "blur_size", "min_area"):
            value = getattr(config, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer, excluding booleans.")
        if config.blur_size < 1 or config.blur_size % 2 != 1:
            raise ValueError("blur_size must be a positive odd integer.")
        if not 0 <= config.threshold <= 255 or config.min_area < 1:
            raise ValueError("threshold must be 0..255 and min_area positive.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (config.blur_size, config.blur_size), 0)
        _, mask = cv2.threshold(smooth, config.threshold, 255, cv2.THRESH_BINARY)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        components = []
        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            if area >= config.min_area:
                filtered[labels == label] = 255
                components.append({"box": [x, y, width, height], "area": area,
                                   "center": centers[label].tolist()})
        return {"mask": filtered, "components": components, "count": len(components)}
    """).strip() + "\\n"
    module_path = output_dir / "parts_pipeline.py"
    module_path.write_text(pipeline_source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("parts_pipeline", module_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline  # dataclasses look up their defining module
    spec.loader.exec_module(pipeline)
    config = pipeline.Config()
    print("Reusable module:", module_path)
    print("Settings:", config)
    '''), md("""
    ## Step 2 — Establish a correctness baseline

    Generate five separate bright circles. The known mask is our pixel-level ground
    truth. Count alone is insufficient: a badly segmented image can still have five
    components. Use both count and IoU, then inspect the visual difference.
    """), code("""
    truth = np.zeros((360, 540), np.uint8)
    for center, radius in [((70, 70), 25), ((220, 95), 34), ((410, 80), 18), ((150, 250), 43), ((400, 255), 30)]:
        cv2.circle(truth, center, radius, 255, -1)
    gray = np.where(truth > 0, 220, 35).astype(np.uint8)
    noise = np.random.default_rng(20).normal(0, 7, gray.shape)
    gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def mask_iou(a, b):
        a, b = a > 0, b > 0
        union = np.count_nonzero(a | b)
        return float(np.count_nonzero(a & b) / union) if union else 1.0

    baseline = pipeline.analyze(image, config)
    baseline_iou = mask_iou(baseline["mask"], truth)
    print("Count:", baseline["count"], "IoU:", round(baseline_iou, 4))
    assert baseline["count"] == 5
    assert baseline_iou > 0.95, "The generated baseline should be segmented accurately."
    show_images([image, baseline["mask"], cv2.absdiff(truth, baseline["mask"])], ["Input", "Prediction", "Pixel disagreements"])
    """), md("""
    ## Step 3 — Benchmark the actual work

    `perf_counter` measures elapsed wall time. Warmup calls reduce one-time setup
    effects. Report a median and a high percentile over multiple runs; one timing
    is noisy. These numbers describe this input and this computer. They do not
    measure camera capture, disk I/O, display, or end-to-end application latency.
    """), code("""
    from time import perf_counter

    def benchmark(function, repeats=15, warmup=3):
        for _ in range(warmup):
            function()
        durations = []
        for _ in range(repeats):
            start_time = perf_counter()
            function()
            durations.append((perf_counter() - start_time) * 1000)
        return {"median_ms": float(np.median(durations)), "p90_ms": float(np.percentile(durations, 90)), "repeats": repeats}

    baseline_timing = benchmark(lambda: pipeline.analyze(image, config))
    print(baseline_timing)
    print("Approximate sequential compute-only FPS:", round(1000 / baseline_timing["median_ms"], 1))
    """), md("""
    ## Experiment A — Python loops versus array operations

    All three implementations below add brightness with saturation at 255.
    Converting to a wider type before NumPy arithmetic prevents `uint8` overflow.
    Validate equal results **before** comparing speed. A faster answer with different
    numeric behavior is not a drop-in optimization.
    """), code("""
    small = gray[:120, :180].copy()
    def brighten_loop(source, amount=40):
        destination = np.empty_like(source)
        for row in range(source.shape[0]):
            for column in range(source.shape[1]):
                destination[row, column] = min(255, int(source[row, column]) + amount)
        return destination

    def brighten_numpy(source, amount=40):
        return np.clip(source.astype(np.int16) + amount, 0, 255).astype(np.uint8)

    loop_result = brighten_loop(small)
    numpy_result = brighten_numpy(small)
    opencv_result = cv2.add(small, 40)
    assert np.array_equal(loop_result, numpy_result)
    assert np.array_equal(loop_result, opencv_result)
    brightness_timings = {
        "Python loop": benchmark(lambda: brighten_loop(small), repeats=5),
        "NumPy": benchmark(lambda: brighten_numpy(small)),
        "OpenCV": benchmark(lambda: cv2.add(small, 40)),
    }
    for name, timing in brightness_timings.items():
        print(name, timing)
    """), md("""
    ## Experiment B — Resolution versus quality

    Smaller images have fewer pixels to process, but small objects and boundaries can
    disappear. If width/height scale by `s`, area scales by `s²`, so we adjust minimum
    component area accordingly. A fixed 5-pixel blur also covers more of the original
    scene when downsampled: this experiment compares the complete scaled pipeline,
    not an isolated pixel-count change.

    Resize masks back with nearest-neighbor interpolation; linear interpolation
    would introduce intermediate values into a categorical mask. Time resizing and
    processing together. Compare IoU with the same full-resolution ground truth.
    """), code("""
    def analyze_scaled(scale):
        reduced = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        scaled_config = pipeline.Config(threshold=config.threshold, blur_size=config.blur_size,
                                        min_area=max(1, round(config.min_area * scale * scale)))
        result = pipeline.analyze(reduced, scaled_config)
        restored = cv2.resize(result["mask"], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return restored, result["count"]

    scale_rows = []
    for scale in [1.0, 0.5, 0.25]:
        restored, count = analyze_scaled(scale)
        timing = benchmark(lambda: analyze_scaled(scale))
        scale_rows.append({"scale": scale, "count": count, "IoU": mask_iou(restored, truth), **timing})
    for row in scale_rows:
        print(row)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot([r["scale"] for r in scale_rows], [r["median_ms"] for r in scale_rows], "o-")
    axes[0].set(xlabel="Resolution scale", ylabel="Median milliseconds", title="Measured processing time")
    axes[1].plot([r["scale"] for r in scale_rows], [r["IoU"] for r in scale_rows], "o-")
    axes[1].set(xlabel="Resolution scale", ylabel="IoU", ylim=(0, 1.02), title="Segmentation quality")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    """), md("""
    ## Step 4 — Understand copies, views, and memory

    A NumPy slice usually shares memory with its parent. Editing the slice can change
    the original. `.copy()` creates independent pixels at a memory cost. `nbytes`
    reports the array's element storage; a view's `nbytes` is its logical size, not
    extra allocated memory. This distinction matters when processing large video.
    """), code("""
    scratch = image.copy()
    view = scratch[10:40, 10:40]
    independent = scratch[10:40, 10:40].copy()
    view[:] = (0, 0, 255)
    assert np.shares_memory(view, scratch)
    assert not np.shares_memory(independent, scratch)
    print("BGR uint8 storage MiB:", image.nbytes / 1024**2)
    print("BGR float32 storage MiB:", image.astype(np.float32).nbytes / 1024**2)
    show_images([scratch, independent], ["Parent changed through view", "Independent copy"])
    """), md("""
    ## Step 5 — Test useful behavior, including failures

    Empty inputs and invalid settings should fail clearly. A blank scene should
    return zero parts, and an analysis should not modify its input. These checks
    protect an API contract; they do not prove accuracy on real-world images.
    Expected errors are caught intentionally so Run All can finish.
    """), code("""
    before = image.copy()
    pipeline.analyze(image, config)
    assert np.array_equal(image, before), "The pipeline must not alter its input."
    blank = np.zeros_like(image)
    assert pipeline.analyze(blank, config)["count"] == 0
    for invalid_input in [np.zeros((0, 0, 3), np.uint8), gray, image.astype(np.float32)]:
        try:
            pipeline.analyze(invalid_input, config)
        except (TypeError, ValueError) as error:
            print("Expected rejection:", error)
        else:
            raise AssertionError("Invalid input was accepted.")
    for invalid_blur in [4, 5.0, True]:
        try:
            pipeline.analyze(image, pipeline.Config(blur_size=invalid_blur))
        except ValueError as error:
            print("Expected configuration rejection:", error)
        else:
            raise AssertionError("Invalid blur size was accepted.")
    """), md("""
    ## Step 6 — Save enough information to repeat an experiment

    Save settings, versions, an input checksum, output mask, and measurements. A
    checksum identifies bytes but does not preserve them, so also save the input.
    `sha256(image.tobytes())` hashes the array pixels in this example, not PNG bytes.
    Hardware and system load still affect timing; no fixed speed is guaranteed.
    """), code("""
    import hashlib
    import json
    import platform
    from dataclasses import asdict

    save_image(output_dir / "input.png", image)
    save_image(output_dir / "prediction.png", baseline["mask"])
    experiment = {
        "config": asdict(config), "opencv": cv2.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "opencv_threads": cv2.getNumThreads(), "input_shape": list(image.shape),
        "input_dtype": str(image.dtype), "input_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "count": baseline["count"], "IoU": baseline_iou, "timing": baseline_timing,
        "scale_experiments": scale_rows,
    }
    report_path = output_dir / "experiment.json"
    report_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    replay = pipeline.analyze(load_image(output_dir / "input.png"), pipeline.Config(**saved["config"]))
    assert np.array_equal(replay["mask"], baseline["mask"])
    print("Replay reproduced the mask. Report:", report_path)
    """), md("""
    ## Exercises — Try first

    1. Write a function that returns the largest component's area, or zero when no
       component is found. Check both cases.
    2. Choose a resolution from the measured rows subject to IoU ≥ 0.95 and correct
       object count. Explain why the choice may differ on another machine.
    3. Add a very small object and a touching pair. Which assumptions break, and which
       prior lesson suggests a remedy?
    4. Copy the exported module into your own application and write a batch loop over
       personal images. Save per-image errors instead of silently skipping failures.
    """), code("# Your experiments go here."), md("## Worked solutions and discussion"), code("""
    def largest_area(result):
        return max((component["area"] for component in result["components"]), default=0)

    assert largest_area(pipeline.analyze(blank, config)) == 0
    assert largest_area(baseline) > 0
    print("Largest component:", largest_area(baseline))
    eligible = [row for row in scale_rows if row["IoU"] >= 0.95 and row["count"] == 5]
    if eligible:
        choice = min(eligible, key=lambda row: row["median_ms"])
        print("Fastest eligible measured configuration:", choice)
    else:
        print("No measured setting meets the quality target. Improve the pipeline first.")
    """), md("""
    Minimum-area filtering can remove small true objects. Touching objects form one
    connected component; notebook 11's distance-transform/watershed approach may help.
    Neither change is automatically better: compare on representative labeled data.

    ## Independent final project

    Choose one: count tabletop objects; align repeated photos; track a colored marker;
    scan documents; or inspect a stable part. Define inputs, outputs, success criteria,
    and failure conditions before coding. Collect multiple lighting conditions and
    keep a final set untouched during tuning.

    Deliver a working notebook, a reusable processing function, a labeled evaluation
    set, quantitative results, failure-case images, and a short explanation of the
    quality/performance tradeoff. Add video capture only if the application needs it;
    measure capture and display time separately from processing.

    ## Completion checklist

    - Explain array layout, data types, color order, interpolation, and masks.
    - Choose an algorithm based on image formation and scene assumptions.
    - Evaluate on data that did not guide parameter selection.
    - Diagnose errors visually and with appropriate metrics.
    - Reproduce an experiment from saved input and settings.
    - Profile before optimizing, then recheck output quality.

    You now have a foundation for more specialized work: robust real-camera geometry,
    large-dataset evaluation, learned detection/segmentation, and application design.
    Further expertise comes from applying these habits to unfamiliar failure cases.

    **Reference:** [OpenCV performance measurement and improvement](https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html).
    """)]
    save("20_performance_and_reproducible_pipelines.ipynb", cells)


if __name__ == "__main__":
    orientation()
    inspection()
    engineering()
