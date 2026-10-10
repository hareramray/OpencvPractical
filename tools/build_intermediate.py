"""Rebuild the independent intermediate OpenCV learning notebooks (07–12)."""
from tools.notebook_factory import md, code, intro, save


def lesson(title, prerequisites, objectives):
    return intro(title, "Intermediate", prerequisites, objectives)


def add(cells, *pairs):
    for kind, text in pairs:
        cells.append(md(text) if kind == "m" else code(text))


def build_edges():
    cells = lesson("07 · Gradients, edges, and Hough lines", "Notebooks 01–06; grayscale images, blur, and binary masks.",
                   ["Distinguish gradients from binary edges.", "Tune Canny using visible evidence.", "Detect and filter line segments with the Hough transform.", "Build a simple document-border inspector."])
    add(cells,
    ("m", """## Step 1 — Construct a scene with known boundaries
An edge is a rapid change of intensity; it is not automatically an object boundary. We will generate a clean scene, add repeatable Gaussian noise, and inspect both. Predict which image will produce more false edges before you run the next cells."""),
    ("c", """clean = np.full((360, 560), 35, np.uint8)
cv2.rectangle(clean, (60, 55), (475, 285), 205, -1)
cv2.circle(clean, (165, 160), 58, 75, -1)
cv2.line(clean, (310, 85), (435, 245), 60, 6)
cv2.putText(clean, 'EDGE', (235, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 95, 2)
noise = rng.normal(0, 12, clean.shape)
noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
show_images([clean, noisy], ['Known boundaries', 'Noise adds unwanted changes'], cols=2)"""),
    ("c", """profile_y = 110
plt.figure(figsize=(10, 3))
plt.plot(clean[profile_y], label='clean')
plt.plot(noisy[profile_y], alpha=.55, label='noisy')
plt.title(f'Intensity along row {profile_y}')
plt.xlabel('x pixel'); plt.ylabel('intensity'); plt.legend(); plt.show()"""),
    ("m", """## Step 2 — Measure signed gradients
`Sobel(image, CV_32F, dx, dy, ksize=3)` differentiates along x or y. A negative derivative is useful information: an unsigned 8-bit output would clip it. Compute in floating point; use absolute values only for display. `ksize` must be an appropriate odd kernel size."""),
    ("c", """blurred = cv2.GaussianBlur(noisy, (5, 5), 1.2)
gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
magnitude = cv2.magnitude(gx, gy)
show_images([blurred, np.abs(gx), np.abs(gy), magnitude],
            ['Blur first', '|horizontal derivative|', '|vertical derivative|', 'gradient magnitude'], cols=2)
print('Signed x-gradient range:', float(gx.min()), float(gx.max()))
assert gx.min() < 0 < gx.max(), 'This scene contains rising and falling edges.'"""),
    ("c", """bad_unsigned = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
good_display = cv2.convertScaleAbs(gx)
show_images([bad_unsigned, good_display], ['CV_8U loses negative derivatives', 'Absolute float derivative'], cols=2)
print('Negative gradient pixels:', int(np.count_nonzero(gx < 0)))"""),
    ("m", """The Laplacian is a second derivative. It responds strongly to noise and changes around edges; it is not a drop-in replacement for Canny. Compare the noisy and smoothed versions."""),
    ("c", """lap_raw = cv2.Laplacian(noisy, cv2.CV_32F, ksize=3)
lap_blur = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
show_images([np.abs(lap_raw), np.abs(lap_blur)], ['Laplacian before blur', 'Laplacian after blur'], cols=2)
print('Mean absolute response:', round(float(np.abs(lap_raw).mean()), 2),
      '->', round(float(np.abs(lap_blur).mean()), 2))"""),
    ("m", """## Step 3 — Turn gradients into thin edges with Canny
Canny applies gradient estimation, thinning, and hysteresis. Pixels above the high threshold start strong edges; eligible pixels above the low threshold can survive when connected to them. These are gradient thresholds, not input intensity thresholds. `L2gradient=True` uses Euclidean gradient magnitude."""),
    ("c", """settings = [(25, 75), (60, 140), (200, 400)]
edge_maps = [cv2.Canny(blurred, low, high, L2gradient=True) for low, high in settings]
show_images(edge_maps, [f'low={a}, high={b}' for a, b in settings])
for setting, edges in zip(settings, edge_maps):
    print(setting, 'edge pixels:', np.count_nonzero(edges))
assert all(set(np.unique(e)).issubset({0, 255}) for e in edge_maps)"""),
    ("c", """blur_sizes = [1, 5, 11]
blur_edges = []
for k in blur_sizes:
    candidate = noisy if k == 1 else cv2.GaussianBlur(noisy, (k, k), 0)
    blur_edges.append(cv2.Canny(candidate, 60, 140, L2gradient=True))
show_images(blur_edges, [f'Blur kernel {k} x {k}' for k in blur_sizes])"""),
    ("m", """**Experiment:** change noise standard deviation from 12 to 25, then rerun from Step 1. Does raising thresholds or increasing blur preserve the small text better? A visually cleaner edge map may have discarded useful fine structure. Record both edge count and one detail you wanted to retain."""),
    ("c", """edges = edge_maps[1]
overlay = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
overlay[edges > 0] = (0, 0, 255)
show_images([noisy, edges, overlay], ['Input', 'Canny mask', 'Edges overlaid in red'])"""),
    ("m", """## Step 4 — Detect line segments with HoughLinesP
The probabilistic Hough transform votes for straight lines from edge pixels. `rho=1` is a one-pixel distance bin, `theta=np.pi/180` is a one-degree angle bin, and `threshold` is a vote cutoff. `minLineLength` rejects short segments; `maxLineGap` joins nearby collinear evidence. One physical border may generate several detections."""),
    ("c", """def detect_lines(edge_image, min_length=90, max_gap=12, votes=55):
    found = cv2.HoughLinesP(edge_image, 1, np.pi / 180, threshold=votes,
                            minLineLength=min_length, maxLineGap=max_gap)
    return np.empty((0, 4), dtype=np.int32) if found is None else found.reshape(-1, 4)

segments = detect_lines(edges)
line_view = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
for x1, y1, x2, y2 in segments:
    cv2.line(line_view, (x1, y1), (x2, y2), (0, 220, 0), 2)
show_images([edges, line_view], ['Evidence', f'{len(segments)} segments'], cols=2)
assert len(segments) > 0, 'Expected long rectangle edges in the generated image.'"""),
    ("c", """lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
angles = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0]))
for length, angle in zip(lengths, angles):
    print(f'length={length:6.1f} px, angle={angle:6.1f} degrees')
print('Empty-image handling:', detect_lines(np.zeros_like(edges)).shape)"""),
    ("c", """hough_views = []
for min_length in [30, 120, 240]:
    found = detect_lines(edges, min_length=min_length)
    canvas = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in found:
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    hough_views.append(canvas)
show_images(hough_views, ['minLineLength=30', 'minLineLength=120', 'minLineLength=240'])"""),
    ("m", """## Step 5 — Mini-project: a document-border inspector
Keep long, nearly horizontal or vertical lines. This is a deliberately limited detector: it is suitable for a roughly aligned document, and it will miss a strongly rotated page. Use orientation tolerance rather than exact angle equality. The result is a border *candidate* overlay, not proof that a document exists."""),
    ("c", """axis_angle = np.abs(angles) % 180
horizontal = np.minimum(axis_angle, 180 - axis_angle) < 8
vertical = np.abs(axis_angle - 90) < 8
keep = (lengths > 120) & (horizontal | vertical)
border_view = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
for x1, y1, x2, y2 in segments[keep]:
    cv2.line(border_view, (x1, y1), (x2, y2), (255, 180, 0), 3)
show_images([border_view], [f'{np.count_nonzero(keep)} border candidates'])"""),
    ("c", """out_dir = ROOT / 'outputs' / '07_edges'
out_dir.mkdir(parents=True, exist_ok=True)
save_image(out_dir / 'document_border_candidates.png', border_view)
print('Saved:', out_dir / 'document_border_candidates.png')"""),
    ("m", """### Pitfalls and debugging
- White Canny pixels are edge evidence; filled object masks require a different pipeline.
- A blank result can mean thresholds are too high, blur is too strong, or the input has little contrast.
- Hough returns `None` when no segments are detected. Handle it before reshaping.
- Lines do not directly give enclosed areas. Contours in Notebook 08 serve that purpose.

### Practice before opening the solutions
1. Count the fraction of Canny pixels for each threshold setting.
2. Write a detector that returns only diagonal segments between 25° and 65° from horizontal, regardless of endpoint order.
3. Compare Canny on the noisy input and the blurred input using the same thresholds. Explain why count alone does not determine quality."""),
    ("m", """## Worked solutions — run after your own attempt
Use modulo 180 because reversing a segment changes its direction by 180° but leaves the geometric line unchanged."""),
    ("c", """for (low, high), edge_image in zip(settings, edge_maps):
    print(f'{low:3}/{high:3}: {np.count_nonzero(edge_image) / edge_image.size:.2%} edge pixels')

all_segments = detect_lines(edges, min_length=35, votes=30)
all_angles = np.degrees(np.arctan2(all_segments[:, 3] - all_segments[:, 1],
                                 all_segments[:, 2] - all_segments[:, 0])) % 180
acute = np.minimum(all_angles, 180 - all_angles)
diagonals = all_segments[(acute >= 25) & (acute <= 65)]
print('Diagonal segments:', len(diagonals))"""),
    ("c", """without_blur = cv2.Canny(noisy, 60, 140, L2gradient=True)
with_blur = cv2.Canny(blurred, 60, 140, L2gradient=True)
show_images([without_blur, with_blur], ['Same thresholds, no blur', 'Same thresholds, with blur'], cols=2)
print('Counts:', np.count_nonzero(without_blur), np.count_nonzero(with_blur))
# Look for continuous real borders and fragmented noise, not only the smaller count."""),
    ("m", """### Reflection and next step
Explain why storing the derivative in uint8 can erase one side of an object. Which change removed false edges without losing the feature you wanted? Next, use closed boundaries to measure shapes.

Official references: [Sobel and Laplacian](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html), [Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html), [Hough lines](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)."""))
    save("07_gradients_edges_and_hough.ipynb", cells)


def build_contours():
    cells = lesson("08 · Contours and shape analysis", "Notebooks 01–07; masks, morphology, and edges.",
                   ["Extract and visualize contour hierarchy.", "Measure area, perimeter, centroid, and circularity.", "Compare contour geometry with pixel counts.", "Create an explainable synthetic part inspector."])
    add(cells,
    ("m", """## Step 1 — Generate an inspectable binary scene
Contours describe boundaries of nonzero regions. Start from a clean binary mask so you can separate geometry mistakes from thresholding mistakes. Coordinates use `(x, y)`; NumPy indexing uses `[y, x]`."""),
    ("c", """mask = np.zeros((420, 640), np.uint8)
cv2.circle(mask, (110, 115), 55, 255, -1)
cv2.rectangle(mask, (220, 65), (350, 175), 255, -1)
triangle = np.array([[465, 55], [405, 180], [565, 180]], np.int32)
cv2.fillPoly(mask, [triangle], 255)
cv2.circle(mask, (160, 310), 65, 255, -1)
cv2.circle(mask, (160, 310), 28, 0, -1)
cv2.ellipse(mask, (420, 310), (95, 40), -20, 0, 360, 255, -1)
show_images([mask], ['Five separate parts, one has a hole'])"""),
    ("c", """contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)
show_images([canvas], [f'{len(contours)} external contours'])
assert len(contours) == 5, 'The generated parts should be disconnected.'"""),
    ("m", """## Step 2 — Measure shapes and guard degenerate cases
`contourArea` integrates the polygon boundary; `arcLength(..., True)` measures a closed perimeter. Moments give centroid `(m10/m00, m01/m00)` when `m00 != 0`. A line or point can have zero area, so guard division. Pixel area and polygon area are related but not identical."""),
    ("c", """def shape_record(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    moments = cv2.moments(contour)
    center = None if moments['m00'] == 0 else (moments['m10'] / moments['m00'],
                                               moments['m01'] / moments['m00'])
    x, y, w, h = cv2.boundingRect(contour)
    circularity = 4 * np.pi * area / perimeter**2 if perimeter > 0 else 0.0
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0.0
    return dict(area=area, perimeter=perimeter, center=center, box=(x, y, w, h),
                circularity=circularity, solidity=solidity, aspect=w / h)

records = [shape_record(c) for c in contours]
for i, record in enumerate(records):
    print(i, {k: round(v, 3) if isinstance(v, float) else v for k, v in record.items()})"""),
    ("c", """measure_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
for i, record in enumerate(records):
    x, y, w, h = record['box']
    cv2.rectangle(measure_view, (x, y), (x + w - 1, y + h - 1), (255, 150, 0), 2)
    if record['center'] is not None:
        cx, cy = map(round, record['center'])
        cv2.circle(measure_view, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(measure_view, str(i), (cx + 7, cy), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
show_images([measure_view], ['Bounding boxes and moment centroids'])"""),
    ("c", """circle_mask = np.zeros((130, 130), np.uint8)
cv2.circle(circle_mask, (65, 65), 45, 255, -1)
circle_contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('Nonzero pixel count:', cv2.countNonZero(circle_mask))
print('Polygon area:', cv2.contourArea(circle_contours[0]))
print('Ideal continuous circle area:', round(np.pi * 45**2, 2))
degenerate = np.array([[[0, 0]], [[10, 0]]], dtype=np.int32)
assert shape_record(degenerate)['center'] is None"""),
    ("m", """## Step 3 — Understand holes with contour hierarchy
`RETR_EXTERNAL` ignores hole boundaries. `RETR_TREE` returns every level. Each hierarchy row contains `[next, previous, first_child, parent]`; `-1` means absent. To measure a ring, subtract the hole contour area from its outer contour area. For arbitrarily nested shapes, alternate signs by depth."""),
    ("c", """tree_contours, tree_hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
tree_hierarchy = tree_hierarchy[0]
print('index: [next, previous, first_child, parent]')
for i, h in enumerate(tree_hierarchy):
    print(i, ':', h.tolist())
hole_ids = [i for i, h in enumerate(tree_hierarchy) if h[3] >= 0]
print('Hole contour IDs for this simple scene:', hole_ids)
assert len(hole_ids) == 1"""),
    ("c", """tree_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
for i, contour in enumerate(tree_contours):
    color = (0, 0, 255) if tree_hierarchy[i, 3] >= 0 else (0, 200, 0)
    cv2.drawContours(tree_view, [contour], -1, color, 2)
hole_id = hole_ids[0]
parent_id = int(tree_hierarchy[hole_id, 3])
ring_area = cv2.contourArea(tree_contours[parent_id]) - cv2.contourArea(tree_contours[hole_id])
print('Ring area estimated from boundary polygons:', ring_area)
show_images([tree_view], ['Green: outer boundaries; red: hole boundary'])"""),
    ("m", """## Step 4 — Approximate geometry and inspect tolerance
`approxPolyDP` simplifies a contour. Set epsilon as a fraction of perimeter so it scales with object size. A small epsilon preserves detail; a large one can turn a circle into a few-sided polygon. Vertex count is therefore a feature with a tolerance, not an infallible class label."""),
    ("c", """approx_views = []
for fraction in [.005, .02, .08]:
    view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, fraction * cv2.arcLength(contour, True), True)
        cv2.polylines(view, [approx], True, (0, 0, 255), 2)
        for point in approx[:, 0]:
            cv2.circle(view, tuple(point), 4, (255, 0, 0), -1)
    approx_views.append(view)
show_images(approx_views, ['epsilon=0.005 x P', 'epsilon=0.02 x P', 'epsilon=0.08 x P'])"""),
    ("c", """rotated_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
for contour in contours:
    rotated_rect = cv2.minAreaRect(contour)
    corners = np.rint(cv2.boxPoints(rotated_rect)).astype(np.int32)
    cv2.polylines(rotated_view, [corners], True, (0, 150, 255), 2)
show_images([rotated_view], ['Minimum-area rotated boxes'])"""),
    ("m", """## Step 5 — Mini-project: inspect synthetic parts
Use several interpretable features: vertex count for triangles/quadrilaterals, circularity for round shapes, and contour hierarchy for holes. Circularity of a raster circle is usually below 1. The following thresholds are tuned to our generated scene; real parts need representative validation data."""),
    ("c", """def classify_outer(contour, has_hole=False):
    record = shape_record(contour)
    approximation = cv2.approxPolyDP(contour, .02 * record['perimeter'], True)
    if has_hole:
        return 'ring'
    if len(approximation) == 3:
        return 'triangle'
    if len(approximation) == 4:
        return 'quadrilateral'
    if record['circularity'] > .82:
        return 'round'
    return 'elongated / other'

inspection = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
labels = []
for i, contour in enumerate(tree_contours):
    if tree_hierarchy[i, 3] != -1:
        continue
    label = classify_outer(contour, has_hole=tree_hierarchy[i, 2] != -1)
    labels.append(label)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.putText(inspection, label, (max(0, x - 10), max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 180, 255), 1, cv2.LINE_AA)
show_images([inspection], ['Explainable shape labels'])
assert 'ring' in labels and 'triangle' in labels"""),
    ("c", """out_dir = ROOT / 'outputs' / '08_contours'
out_dir.mkdir(parents=True, exist_ok=True)
save_image(out_dir / 'part_inspection.png', inspection)
print('Saved inspection to', out_dir)"""),
    ("m", """### Stress test: add small specks
Noise can create many valid but irrelevant contours. A minimum area removes tiny components. This filter is stated in pixel²; it must change if image scale changes."""),
    ("c", """speckled = mask.copy()
for x, y in rng.integers([0, 0], [640, 420], size=(90, 2)):
    cv2.circle(speckled, (int(x), int(y)), 1, 255, -1)
noisy_contours, _ = cv2.findContours(speckled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
large_contours = [c for c in noisy_contours if cv2.contourArea(c) > 500]
print('All contours:', len(noisy_contours), '| area > 500:', len(large_contours))
show_images([mask, speckled], ['Original', 'Speckled'], cols=2)"""),
    ("m", """### Pitfalls and practice
- Contour ordering is not a stable identity. Sort or match by geometry when needed.
- Centroids can fall outside a concave shape or inside its hole.
- An external contour's area includes its enclosed holes unless you account for hierarchy.

1. Return all parts whose axis-aligned aspect ratio exceeds 1.5.
2. Compute net polygon area across the entire hierarchy using depth parity.
3. Double the scene size with nearest-neighbor interpolation. Compare contour area scaling to perimeter scaling."""),
    ("m", """## Worked solutions
Use depth parity for nested contours: outer boundary (+), hole (−), island inside a hole (+)."""),
    ("c", """wide_ids = [i for i, r in enumerate(records) if r['aspect'] > 1.5]
print('Wide external contour IDs:', wide_ids)

net_area = 0.0
for i, contour in enumerate(tree_contours):
    depth, parent = 0, int(tree_hierarchy[i, 3])
    while parent != -1:
        depth += 1
        parent = int(tree_hierarchy[parent, 3])
    net_area += (-1)**depth * cv2.contourArea(contour)
print('Net polygon area:', net_area)
print('Foreground pixel area:', cv2.countNonZero(mask))"""),
    ("c", """large_mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
large_cs, _ = cv2.findContours(large_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
large_cs = sorted(large_cs, key=cv2.contourArea, reverse=True)
print('Largest-part area ratio:', cv2.contourArea(large_cs[0]) / cv2.contourArea(contours[0]))
print('Largest-part perimeter ratio:', cv2.arcLength(large_cs[0], True) / cv2.arcLength(contours[0], True))
print('Expect approximately 4x area and 2x perimeter; raster boundaries cause small differences.')"""),
    ("m", """### Reflection
Which feature changes if you rotate an elongated object: axis-aligned aspect ratio, circularity, or both? What evidence would you collect before trusting these rules on photographed parts?

Official references: [Contour features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html), [Contour properties](https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html), [Hierarchy](https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html)."""))
    save("08_contours_and_shape_analysis.ipynb", cells)


def build_geometry():
    cells = lesson("09 · Geometric transforms and document rectification", "Notebooks 01–08; coordinates, arrays, and interpolation.",
                   ["Apply translation, rotation, and perspective transforms.", "Keep source-to-destination coordinates consistent.", "Transform points and images with the same matrix.", "Rectify a synthetic photographed document and measure alignment."])
    add(cells,
    ("m", """## Step 1 — Build a reference document
We know the original geometry, so we can check a warp quantitatively. A transformation maps coordinates; interpolation determines sampled pixel values. OpenCV output sizes are `(width, height)`, whereas array shapes begin `(height, width)`."""),
    ("c", """doc = np.full((300, 440, 3), 245, np.uint8)
cv2.rectangle(doc, (8, 8), (431, 291), (30, 30, 30), 3)
cv2.putText(doc, 'OPENCV LAB', (35, 57), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 80, 170), 2)
for y in range(90, 255, 27):
    cv2.line(doc, (35, y), (395, y), (110, 110, 110), 2)
cv2.rectangle(doc, (310, 210), (395, 260), (190, 120, 40), -1)
h, w = doc.shape[:2]
show_images([doc], ['Reference document'])"""),
    ("c", """small = cv2.resize(doc, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
nearest = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
linear = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
show_images([doc, nearest, linear], ['Original', 'Down then up: nearest', 'Down then up: linear'])
print('Resizing discards information; enlargement cannot recover it.')"""),
    ("m", """## Step 2 — Translate and rotate using 2×3 affine matrices
An affine transform preserves straight lines and parallelism. Translation uses `[[1,0,tx],[0,1,ty]]`. Positive x moves right; positive y moves down. `warpAffine` internally samples the source for each destination pixel; pass the forward source-to-destination matrix unless using `WARP_INVERSE_MAP`."""),
    ("c", """translation = np.float32([[1, 0, 45], [0, 1, 25]])
shifted = cv2.warpAffine(doc, translation, (w, h), borderValue=(25, 25, 25))
rotation = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 22, 1.0)
rotated = cv2.warpAffine(doc, rotation, (w, h), borderValue=(25, 25, 25))
show_images([doc, shifted, rotated], ['Original', 'Shift right 45, down 25', 'Rotate 22 degrees'])"""),
    ("c", """point = np.float32([[[35, 57]]])
shifted_point = cv2.transform(point, translation)[0, 0]
rotated_point = cv2.transform(point, rotation)[0, 0]
print('Original point:', point[0, 0])
print('Translated point:', shifted_point)
print('Rotated point:', rotated_point)
assert np.allclose(shifted_point, [80, 82])"""),
    ("m", """A fixed output canvas crops rotated corners. Enlarge the canvas and adjust the translation terms to keep the rotated image centered. Use image dimensions for the canvas bounds; minor one-pixel border differences come from pixel-center conventions."""),
    ("c", """cosine, sine = abs(rotation[0, 0]), abs(rotation[0, 1])
bound_w = int(np.ceil(w * cosine + h * sine))
bound_h = int(np.ceil(h * cosine + w * sine))
expanded_rotation = rotation.copy()
expanded_rotation[0, 2] += (bound_w - w) / 2
expanded_rotation[1, 2] += (bound_h - h) / 2
uncropped = cv2.warpAffine(doc, expanded_rotation, (bound_w, bound_h), borderValue=(25, 25, 25))
show_images([rotated, uncropped], ['Original canvas clips corners', 'Expanded canvas'], cols=2)
assert uncropped.shape[:2] == (bound_h, bound_w)"""),
    ("m", """## Step 3 — Compose transforms before resampling
Use 3×3 homogeneous matrices to compose affine transforms. For column-vector coordinates, `B @ A` applies A first and B second. A single warp usually retains more detail than repeatedly resampling the image."""),
    ("c", """T = np.vstack([translation, [0, 0, 1]])
R = np.vstack([rotation, [0, 0, 1]])
combined = T @ R  # rotation, then translation
one_pass = cv2.warpAffine(doc, combined[:2], (w, h), borderValue=(25, 25, 25))
two_pass = cv2.warpAffine(rotated, translation, (w, h), borderValue=(25, 25, 25))
show_images([one_pass, two_pass], ['Compose, then warp once', 'Warp twice (also clips intermediate pixels)'], cols=2)
p_h = np.array([35, 57, 1.0])
assert np.allclose(combined @ p_h, T @ (R @ p_h))"""),
    ("c", """source_triangle = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
target_triangle = np.float32([[30, 15], [w - 45, 35], [65, h - 20]])
affine = cv2.getAffineTransform(source_triangle, target_triangle)
sheared = cv2.warpAffine(doc, affine, (w, h), borderValue=(35, 35, 35))
show_images([sheared], ['Three non-collinear point pairs determine an affine transform'])
assert np.allclose(cv2.transform(source_triangle[None], affine)[0], target_triangle, atol=1e-3)"""),
    ("m", """## Step 4 — Perspective warp using four ordered corners
A homography models a planar surface under perspective. It has eight independent degrees of freedom after scale normalization. Four point pairs must be in matching order and must not be degenerate. We use top-left, top-right, bottom-right, bottom-left for both arrays."""),
    ("c", """source_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
photo_corners = np.float32([[120, 65], [555, 100], [515, 415], [70, 370]])
H_forward = cv2.getPerspectiveTransform(source_corners, photo_corners)
photo = cv2.warpPerspective(doc, H_forward, (640, 480), borderValue=(55, 55, 55))
corner_view = photo.copy()
for i, p in enumerate(photo_corners.astype(int)):
    cv2.circle(corner_view, tuple(p), 6, (0, 0, 255), -1)
    cv2.putText(corner_view, str(i), tuple(p + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
show_images([corner_view], ['Matched order: 0 TL, 1 TR, 2 BR, 3 BL'])"""),
    ("c", """mapped_corners = cv2.perspectiveTransform(source_corners[None], H_forward)[0]
corner_error = np.linalg.norm(mapped_corners - photo_corners, axis=1)
print('Forward corner errors (pixels):', corner_error)
assert corner_error.max() < .001
print('H matrix:'); print(H_forward)"""),
    ("c", """H_inverse = cv2.getPerspectiveTransform(photo_corners, source_corners)
rectified = cv2.warpPerspective(photo, H_inverse, (w, h))
show_images([doc, photo, rectified], ['Reference', 'Simulated photograph', 'Rectified'])
inverse_check = H_inverse @ H_forward
inverse_check /= inverse_check[2, 2]
assert np.allclose(inverse_check, np.eye(3), atol=1e-5)"""),
    ("m", """## Step 5 — Mini-project: quantify rectification quality
Correct point geometry does not imply identical pixels: two interpolation passes blur edges. Compare an interior region to avoid border artifacts. Then perturb the measured corners and observe how geometric annotation errors affect the result."""),
    ("c", """interior = np.s_[15:-15, 15:-15]
mae = np.mean(np.abs(rectified[interior].astype(np.float32) - doc[interior].astype(np.float32)))
difference = cv2.absdiff(doc, rectified)
print(f'Interior mean absolute pixel error: {mae:.3f} / 255')
show_images([rectified, difference], ['Rectified', 'Absolute difference (unscaled)'], cols=2)"""),
    ("c", """errors = []
views = []
for sigma in [0, 3, 12]:
    measured = photo_corners + rng.normal(0, sigma, photo_corners.shape).astype(np.float32)
    estimate = cv2.getPerspectiveTransform(measured, source_corners)
    recovered = cv2.warpPerspective(photo, estimate, (w, h))
    errors.append(np.mean(np.abs(recovered[interior].astype(float) - doc[interior].astype(float))))
    views.append(recovered)
show_images(views, [f'Corner noise sigma={s}px' for s in [0, 3, 12]])
print('Interior MAE:', np.round(errors, 2))"""),
    ("c", """out_dir = ROOT / 'outputs' / '09_geometry'
out_dir.mkdir(parents=True, exist_ok=True)
save_image(out_dir / 'perspective_photo.png', photo)
save_image(out_dir / 'rectified_document.png', rectified)
np.save(out_dir / 'photo_to_document_homography.npy', H_inverse)
print('Saved images and a reusable transformation matrix to', out_dir)"""),
    ("m", """### Pitfalls and practice
- Use nearest-neighbor interpolation for class-label masks. Linear interpolation invents intermediate labels.
- Four arbitrary points do not guarantee a useful homography; they must span a plane and correspond correctly.
- A homography cannot model general depth changes, curved paper, or lens distortion.

1. Translate the document left 30 pixels and down 40 pixels; verify one point numerically.
2. Warp a binary mask using both nearest and linear interpolation. Inspect unique values.
3. Verify the inverse by mapping the four photographed corners back to the document."""),
    ("m", """## Worked solutions
Keep the same output coordinate convention for points and pixels."""),
    ("c", """exercise_translation = np.float32([[1, 0, -30], [0, 1, 40]])
exercise_image = cv2.warpAffine(doc, exercise_translation, (w, h))
exercise_point = cv2.transform(np.float32([[[100, 100]]]), exercise_translation)[0, 0]
assert np.allclose(exercise_point, [70, 140])
show_images([exercise_image], ['Left 30, down 40'])"""),
    ("c", """binary = np.zeros((h, w), np.uint8)
cv2.rectangle(binary, (60, 50), (300, 230), 255, -1)
mask_nearest = cv2.warpPerspective(binary, H_forward, (640, 480), flags=cv2.INTER_NEAREST)
mask_linear = cv2.warpPerspective(binary, H_forward, (640, 480), flags=cv2.INTER_LINEAR)
print('Nearest values:', np.unique(mask_nearest))
print('Linear has', len(np.unique(mask_linear)), 'distinct values')
assert set(np.unique(mask_nearest)).issubset({0, 255})
recovered_corners = cv2.perspectiveTransform(photo_corners[None], H_inverse)[0]
assert np.allclose(recovered_corners, source_corners, atol=.001)
show_images([mask_nearest, mask_linear], ['Nearest preserves labels', 'Linear creates partial values'], cols=2)"""),
    ("m", """### Reflection
Explain why the inverse matrix can be numerically correct while the reconstructed pixels differ. How would you collect corner correspondences for a real document? Notebook 10 estimates correspondences from visual features.

Official references: [Geometric transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html), [Transform API and interpolation](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)."""))
    save("09_geometric_transforms.ipynb", cells)


def build_features():
    cells = lesson("10 · Features, matching, and image registration", "Notebooks 01–09; grayscale, coordinates, and homographies.",
                   ["Detect ORB keypoints and describe local appearance.", "Match binary descriptors with Hamming distance and a ratio test.", "Reject outliers with RANSAC.", "Evaluate an estimated homography against known geometry."])
    add(cells,
    ("m", """## Step 1 — Create a textured planar target
Feature registration needs distinctive local texture. Blank walls and repeated tiles are difficult. We generate many reproducible marks, then apply a known homography. This gives ground truth without internet images. A homography is appropriate because our target is a single plane."""),
    ("c", """texture = np.full((440, 600, 3), 235, np.uint8)
for i in range(220):
    x, y = rng.integers([18, 18], [582, 422])
    color = tuple(int(v) for v in rng.integers(15, 210, 3))
    radius = int(rng.integers(3, 12))
    if i % 2:
        cv2.circle(texture, (int(x), int(y)), radius, color, -1)
    else:
        cv2.rectangle(texture, (int(x), int(y)), (int(x + radius), int(y + radius)), color, -1)
for i, label in enumerate(['OPENCV', 'MATCH 42', 'PLANAR', 'FEATURES']):
    cv2.putText(texture, label, (35, 80 + i * 95), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (15, 15, 15), 2)
h, w = texture.shape[:2]
corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
target_corners = np.float32([[55, 30], [615, 55], [595, 450], [30, 425]])
H_true = cv2.getPerspectiveTransform(corners, target_corners)
target = cv2.warpPerspective(texture, H_true, (660, 480), borderValue=(55, 55, 55))
show_images([texture, target], ['Reference texture', 'Known perspective view'], cols=2)"""),
    ("c", """gray1 = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
print('Reference dimensions:', gray1.shape, '| target dimensions:', gray2.shape)
assert np.linalg.det(H_true) != 0"""),
    ("m", """## Step 2 — Detect keypoints and compute ORB descriptors
A keypoint stores position, scale, orientation, and response. A descriptor summarizes its local appearance. ORB descriptors are binary; use Hamming distance. `nfeatures` caps the retained features, while `fastThreshold` changes corner sensitivity. More keypoints can increase cost and include weak evidence."""),
    ("c", """orb = cv2.ORB_create(nfeatures=1800, fastThreshold=12)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
if des1 is None or des2 is None:
    raise RuntimeError('No descriptors. Add texture or reduce the detector threshold.')
print('Keypoints:', len(kp1), len(kp2))
print('Descriptor arrays:', des1.shape, des2.shape, des1.dtype)
keypoint_view = cv2.drawKeypoints(texture, kp1, None, color=(0, 130, 255),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_images([keypoint_view], ['ORB positions, scale, and orientation'])"""),
    ("c", """for threshold in [5, 20, 60]:
    probe = cv2.ORB_create(nfeatures=1800, fastThreshold=threshold)
    keypoints, descriptors = probe.detectAndCompute(gray1, None)
    print(f'fastThreshold={threshold:2}: {len(keypoints)} keypoints')
blank_kp, blank_des = orb.detectAndCompute(np.full_like(gray1, 150), None)
print('Uniform-image keypoints:', len(blank_kp), '| descriptors:', blank_des)
assert blank_des is None, 'A blank image should not provide useful ORB features.'"""),
    ("m", """## Step 3 — Match descriptors and reject ambiguity
For every reference descriptor, `knnMatch(k=2)` finds the two closest candidates in the target. Keep the best only when its distance is less than `ratio × second_best`. A smaller ratio is stricter. This is an ambiguity filter, not a probability. Some queries can have fewer than two neighbors: guard that case."""),
    ("c", """matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
neighbors = matcher.knnMatch(des1, des2, k=2)

def ratio_filter(knn_matches, ratio=.75):
    return [pair[0] for pair in knn_matches
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]

good = ratio_filter(neighbors, .75)
print('Queries:', len(neighbors), '| after ratio test:', len(good))
match_view = cv2.drawMatches(texture, kp1, target, kp2,
                            sorted(good, key=lambda m: m.distance)[:70], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show_images([match_view], ['Best ratio-test matches (not yet geometric inliers)'])"""),
    ("c", """for ratio in [.55, .7, .8, .9]:
    candidates = ratio_filter(neighbors, ratio)
    print(f'ratio={ratio:.2f}: {len(candidates)} candidate matches')
assert len(ratio_filter(neighbors, .55)) <= len(ratio_filter(neighbors, .9))"""),
    ("m", """## Step 4 — Estimate a homography with RANSAC
At least four suitable point pairs are required, but more distributed matches are safer. RANSAC repeatedly fits small samples and seeks a transformation supported by many pairs. `ransacReprojThreshold=3` allows about 3 target-image pixels of reprojection error. The returned inlier mask aligns with the input match list."""),
    ("c", """def estimate_homography(matches, threshold=3.0):
    if len(matches) < 4:
        return None, None
    source_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(source_points, target_points, cv2.RANSAC, threshold)
    if H is None or inliers is None or not np.isfinite(H).all():
        return None, None
    return H, inliers.ravel().astype(bool)

cv2.setRNGSeed(42)
H_est, inlier_mask = estimate_homography(good)
if H_est is None:
    raise RuntimeError('Homography estimation failed. Inspect texture, matches, and point distribution.')
print(f'RANSAC inliers: {inlier_mask.sum()} / {len(good)} ({inlier_mask.mean():.1%})')
assert inlier_mask.sum() >= 4
assert estimate_homography(good[:3]) == (None, None)"""),
    ("c", """inlier_matches = [m for m, keep in zip(good, inlier_mask) if keep]
verified_view = cv2.drawMatches(texture, kp1, target, kp2, inlier_matches[:65], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show_images([verified_view], ['Matches consistent with one planar transformation'])"""),
    ("c", """estimated_corners = cv2.perspectiveTransform(corners[None], H_est)[0]
true_corners = cv2.perspectiveTransform(corners[None], H_true)[0]
errors = np.linalg.norm(estimated_corners - true_corners, axis=1)
outline = target.copy()
cv2.polylines(outline, [np.rint(true_corners).astype(np.int32)], True, (0, 200, 0), 4)
cv2.polylines(outline, [np.rint(estimated_corners).astype(np.int32)], True, (0, 0, 255), 2)
show_images([outline], ['Green: ground truth; red: estimated outline'])
print('Corner errors in pixels:', np.round(errors, 3))
assert errors.mean() < 8, 'Large synthetic corner error: inspect match consistency.'"""),
    ("m", """Compare mapped points, not raw matrix entries: homographies have an arbitrary multiplicative scale. Also inspect spatial coverage. Twenty inliers in one small corner can extrapolate poorly to the rest of the document."""),
    ("c", """src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
projected = cv2.perspectiveTransform(src, H_est)
residuals = np.linalg.norm(projected - dst, axis=2).ravel()
plt.figure(figsize=(9, 3))
plt.hist(residuals[inlier_mask], bins=20, label='inliers')
plt.xlabel('Reprojection error (target pixels)'); plt.ylabel('Match count'); plt.legend(); plt.show()
print('Median inlier residual:', float(np.median(residuals[inlier_mask])))"""),
    ("m", """## Step 5 — Mini-project: align a photographed label
Warp the target back into reference coordinates using the inverse estimated homography. Evaluate within the target's valid image footprint. Synthetic ground truth enables a real error measurement; in real data you would hold out verified correspondences or inspect downstream task performance."""),
    ("c", """aligned = cv2.warpPerspective(target, np.linalg.inv(H_est), (w, h))
valid_target = np.full(gray2.shape, 255, np.uint8)
valid = cv2.warpPerspective(valid_target, np.linalg.inv(H_est), (w, h), flags=cv2.INTER_NEAREST) > 0
valid[:12] = False; valid[-12:] = False; valid[:, :12] = False; valid[:, -12:] = False
diff = cv2.absdiff(texture, aligned)
print('Valid-region MAE:', float(diff[valid].mean()))
show_images([texture, aligned, diff], ['Reference', 'Aligned target', 'Absolute difference'])"""),
    ("c", """out_dir = ROOT / 'outputs' / '10_registration'
out_dir.mkdir(parents=True, exist_ok=True)
save_image(out_dir / 'aligned_label.png', aligned)
save_image(out_dir / 'estimated_outline.png', outline)
np.save(out_dir / 'reference_to_target_H.npy', H_est)
print('Saved registration artifacts to', out_dir)"""),
    ("m", """### Pitfalls and practice
- ORB uses Hamming; standard float descriptors such as SIFT normally use L2. Mixing them gives meaningless comparisons.
- `queryIdx` indexes the reference descriptors and `trainIdx` indexes the target descriptors in this call.
- A large match count does not guarantee correct registration; inspect inlier rate, coverage, and geometric error.
- Parallax, repetitive textures, strong blur, and non-planar objects can defeat a single homography.

1. Try ratio thresholds .6, .75, and .9. Report matches, inliers, and mean corner error.
2. Compare RANSAC thresholds 1, 3, and 8 pixels with the same matches.
3. Explain why an empty descriptor set must be handled before calling the matcher."""),
    ("m", """## Worked solutions
The loosest threshold may produce more inliers while accepting less accurate correspondences. Evaluate the desired geometry as well as counts."""),
    ("c", """for ratio in [.6, .75, .9]:
    candidate_matches = ratio_filter(neighbors, ratio)
    H_candidate, candidate_inliers = estimate_homography(candidate_matches)
    if H_candidate is None:
        print(ratio, 'insufficient or degenerate matches')
        continue
    candidate_corners = cv2.perspectiveTransform(corners[None], H_candidate)[0]
    corner_mae = np.linalg.norm(candidate_corners - true_corners, axis=1).mean()
    print(f'ratio={ratio:.2f}, matches={len(candidate_matches)}, inliers={candidate_inliers.sum()}, corner error={corner_mae:.3f}px')"""),
    ("c", """for threshold in [1.0, 3.0, 8.0]:
    H_candidate, candidate_inliers = estimate_homography(good, threshold)
    if H_candidate is None:
        print(threshold, 'estimation failed')
        continue
    candidate_corners = cv2.perspectiveTransform(corners[None], H_candidate)[0]
    error = np.linalg.norm(candidate_corners - true_corners, axis=1).mean()
    print(f'RANSAC threshold={threshold:.1f}px, inliers={candidate_inliers.sum()}, corner error={error:.3f}px')
print('None descriptors mean there are no descriptor rows to compare; return a clear no-match result.')"""),
    ("m", """### Reflection
Explain why descriptor matching and geometric verification solve different problems. What would be a trustworthy failure signal for a real label scanner?

Official references: [ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html), [Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html), [Feature matching and homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)."""))
    save("10_features_matching_and_registration.ipynb", cells)


def build_segmentation():
    cells = lesson("11 · Watershed and GrabCut segmentation", "Notebooks 01–10; binary masks, morphology, and connected components.",
                   ["Separate touching objects using distance-transform seeds.", "Understand every watershed marker value.", "Use all four GrabCut mask states correctly.", "Measure segmentation with intersection over union."])
    add(cells,
    ("m", """## Step 1 — Create touching objects and inspect the limitation of components
A binary connected-component counter merges objects that touch. Watershed can separate them when we can provide one reliable seed per object. It cannot invent a correct object definition from a binary mask alone."""),
    ("c", """touching = np.zeros((300, 520), np.uint8)
centers = [(125, 150), (230, 150), (365, 150)]
for center in centers:
    cv2.circle(touching, center, 62, 255, -1)
object_image = np.full((300, 520, 3), (45, 50, 55), np.uint8)
object_image[touching > 0] = (175, 205, 225)
component_count, binary_labels = cv2.connectedComponents(touching)
print('Connected foreground regions:', component_count - 1, '| intended objects:', len(centers))
show_images([object_image, touching], ['Two disks touch; the third is separate', 'Binary foreground'], cols=2)"""),
    ("c", """distance = cv2.distanceTransform(touching, cv2.DIST_L2, 5)
sure_foreground = np.uint8(distance > .62 * distance.max()) * 255
show_images([distance, sure_foreground], ['Distance to nearest zero pixel', 'Interior seeds'], cols=2)
seed_count, seed_labels = cv2.connectedComponents(sure_foreground)
print('Foreground seed components:', seed_count - 1)
assert seed_count - 1 == 3, 'The selected seed threshold should separate the three centers.'"""),
    ("m", """## Step 2 — Build watershed markers deliberately
Markers must be int32, one channel, and the same height/width as the BGR image. Positive labels identify known regions. Reserve 1 for known background and labels 2+ for object seeds. Set uncertain pixels to 0. After watershed, boundaries are −1. Never pass a 0/255 mask directly and assume it encodes separate objects."""),
    ("c", """kernel = np.ones((3, 3), np.uint8)
background_envelope = cv2.dilate(touching, kernel, iterations=3)
unknown = cv2.subtract(background_envelope, sure_foreground)
markers = seed_labels.astype(np.int32) + 1
markers[unknown > 0] = 0
before_markers = markers.copy()
print('Marker values before:', np.unique(markers))
assert markers.dtype == np.int32
assert set(np.unique(markers)) == {0, 1, 2, 3, 4}
show_images([background_envelope, unknown, markers], ['Envelope around possible objects', 'Unknown pixels', 'Marker labels'], cmap='nipy_spectral')"""),
    ("c", """cv2.watershed(object_image.copy(), markers)  # modifies markers in place
watershed_view = object_image.copy()
watershed_view[markers == -1] = (0, 0, 255)
object_labels = [int(label) for label in np.unique(markers) if label > 1]
print('Marker values after:', np.unique(markers))
print('Separated object IDs:', object_labels)
show_images([before_markers, markers, watershed_view], ['Seeds', 'Expanded region labels', 'Red watershed boundaries'], cmap='nipy_spectral')
assert len(object_labels) == 3 and np.any(markers == -1)"""),
    ("c", """for label in object_labels:
    ys, xs = np.where(markers == label)
    print(f'Object {label}: area={len(xs)} pixels, center=({xs.mean():.1f}, {ys.mean():.1f})')
predicted_foreground = markers > 1
ground_truth = touching > 0
intersection = np.count_nonzero(predicted_foreground & ground_truth)
union = np.count_nonzero(predicted_foreground | ground_truth)
print('Combined foreground IoU:', intersection / union)"""),
    ("m", """## Step 3 — Experiment with seed thresholds
Too low a distance threshold merges seeds; too high may remove smaller objects. The threshold is relative to the largest distance in this image, which is not scale-aware for objects of very different sizes. Count seeds before trusting the final object count."""),
    ("c", """seed_views = []
for fraction in [.25, .62, .9]:
    seed_mask = np.uint8(distance > fraction * distance.max()) * 255
    n, _ = cv2.connectedComponents(seed_mask)
    seed_views.append(seed_mask)
    print(f'fraction={fraction}: {n - 1} foreground seeds')
show_images(seed_views, ['25% of max distance', '62% of max distance', '90% of max distance'])"""),
    ("m", """## Step 4 — Extract a foreground object with GrabCut
GrabCut uses color models plus spatial smoothness, initialized by a rectangle or a mask. It does not understand semantic concepts such as “person.” We create a colored ellipse on a textured background and keep the exact ellipse mask as ground truth."""),
    ("c", """hh, ww = 300, 440
background_noise = rng.normal(0, 5, (hh, ww, 3))
grab_image = np.clip(np.full((hh, ww, 3), (165, 125, 85), np.float32) + background_noise, 0, 255).astype(np.uint8)
true_mask = np.zeros((hh, ww), np.uint8)
cv2.ellipse(true_mask, (225, 150), (83, 103), -12, 0, 360, 255, -1)
object_noise = rng.normal(0, 5, (hh, ww, 3))
object_color = np.clip(np.full((hh, ww, 3), (40, 165, 225), np.float32) + object_noise, 0, 255).astype(np.uint8)
grab_image[true_mask > 0] = object_color[true_mask > 0]
cv2.line(grab_image, (200, 100), (250, 210), (35, 120, 185), 12)
show_images([grab_image, true_mask], ['Synthetic foreground and background', 'Known object mask'], cols=2)"""),
    ("m", """GrabCut masks have four states, **not** ordinary grayscale intensity:

| Constant | Value | Meaning |
|---|---:|---|
| `GC_BGD` | 0 | definite background |
| `GC_FGD` | 1 | definite foreground |
| `GC_PR_BGD` | 2 | probable background |
| `GC_PR_FGD` | 3 | probable foreground |

The final binary foreground includes states 1 **and** 3. The model arrays have shape `(1, 65)` and dtype float64. A rectangle is `(x, y, width, height)`, not two corner coordinates."""),
    ("c", """grab_mask = np.zeros((hh, ww), np.uint8)
bg_model = np.zeros((1, 65), np.float64)
fg_model = np.zeros((1, 65), np.float64)
rect = (110, 25, 230, 255)
cv2.setRNGSeed(42)
cv2.grabCut(grab_image, grab_mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
grab_binary = np.isin(grab_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255
cutout = cv2.bitwise_and(grab_image, grab_image, mask=grab_binary)
print('Resulting GrabCut states:', np.unique(grab_mask))
show_images([grab_mask.astype(np.int32), grab_binary, cutout], ['Four-state labels (display rescaled)', 'States 1 or 3 are foreground', 'Extracted object'])
assert set(np.unique(grab_mask)).issubset({0, 1, 2, 3})"""),
    ("c", """def mask_iou(prediction, truth):
    p, t = np.asarray(prediction) > 0, np.asarray(truth) > 0
    union = np.count_nonzero(p | t)
    return np.count_nonzero(p & t) / union if union else 1.0

print('GrabCut IoU:', mask_iou(grab_binary, true_mask))
assert mask_iou(grab_binary, true_mask) > .9, 'The generated colors should separate cleanly.'
assert mask_iou(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0"""),
    ("m", """A rectangle alone can exclude part of the object permanently because outside pixels are definite background. To refine a result, add confident foreground/background strokes to the four-state mask and call `GC_INIT_WITH_MASK`. Keep model arrays when continuing the same image; create fresh arrays for a new image."""),
    ("c", """refined_mask = grab_mask.copy()
cv2.circle(refined_mask, (225, 150), 22, cv2.GC_FGD, -1)
cv2.line(refined_mask, (25, 20), (410, 20), cv2.GC_BGD, 8)
cv2.grabCut(grab_image, refined_mask, None, bg_model, fg_model, 3, cv2.GC_INIT_WITH_MASK)
refined_binary = np.isin(refined_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255
print('Refined IoU:', mask_iou(refined_binary, true_mask))
show_images([refined_mask.astype(np.int32), refined_binary], ['Mask labels (display rescaled)', 'Refined extraction'], cols=2)"""),
    ("m", """## Step 5 — Mini-project: export a foreground cutout with transparency
Use the binary mask as the alpha channel. OpenCV stores this array as BGRA, so convert it to RGBA when displaying with Matplotlib directly. Our normal image helper is intended primarily for BGR/gray images."""),
    ("c", """bgra = cv2.cvtColor(grab_image, cv2.COLOR_BGR2BGRA)
bgra[:, :, 3] = refined_binary
out_dir = ROOT / 'outputs' / '11_segmentation'
out_dir.mkdir(parents=True, exist_ok=True)
save_image(out_dir / 'foreground_cutout.png', bgra)
save_image(out_dir / 'watershed_boundaries.png', watershed_view)
plt.figure(figsize=(7, 4))
plt.imshow(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)); plt.axis('off'); plt.title('Transparent foreground cutout'); plt.show()
assert np.array_equal(bgra[:, :, 3], refined_binary)
print('Saved:', out_dir)"""),
    ("m", """### Pitfalls and practice
- Watershed modifies markers and labels the outer image border as boundary too. A red border is expected.
- The dilated mask often called `sure_bg` is an envelope: pixels *outside* it are known background.
- GrabCut label 2 is probable background, not 50% transparency.
- IoU evaluates pixel overlap, not whether touching instances were separated correctly.

1. Count the pixels in each GrabCut state.
2. Show the disagreement mask between the GrabCut result and the known foreground.
3. Use a too-tight initialization rectangle and measure the resulting IoU. Explain the lost foreground."""),
    ("m", """## Worked solutions
Run a fresh model for the rectangle comparison so previous model state does not confound the experiment."""),
    ("c", """state_names = {0: 'definite background', 1: 'definite foreground',
               2: 'probable background', 3: 'probable foreground'}
for state, name in state_names.items():
    print(name, ':', np.count_nonzero(refined_mask == state))
disagreement = np.uint8((refined_binary > 0) != (true_mask > 0)) * 255
show_images([disagreement], ['White pixels disagree with ground truth'])"""),
    ("c", """tight_mask = np.zeros((hh, ww), np.uint8)
tight_bg, tight_fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(grab_image, tight_mask, (175, 75, 105, 150), tight_bg, tight_fg, 4, cv2.GC_INIT_WITH_RECT)
tight_binary = np.isin(tight_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255
print('Too-tight rectangle IoU:', mask_iou(tight_binary, true_mask))
show_images([true_mask, tight_binary], ['Complete object', 'Outside-rectangle pixels cannot become foreground'], cols=2)"""),
    ("m", """### Reflection
When would you choose watershed, and when would you choose GrabCut? What seed or annotation error would be most damaging in each method?

Official references: [Watershed tutorial](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html), [GrabCut tutorial](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html), [Segmentation API](https://docs.opencv.org/4.x/d3/d47/group__imgproc__segmentation.html)."""))
    save("11_segmentation_watershed_grabcut.ipynb", cells)


def build_video():
    cells = lesson("12 · Video I/O and background subtraction", "Notebooks 01–11; masks, morphology, and contour measurements.",
                   ["Generate and save a video without a camera.", "Read frames safely and release device/file resources.", "Build a simple moving-object detector with MOG2.", "Distinguish video frame rate from processing throughput."])
    add(cells,
    ("m", """## Step 1 — Generate a short reproducible video
Video is a time-ordered sequence of images. Every frame written to one video must have the same dimensions and channel layout. We synthesize a moving rectangle over a static background, including initial empty frames to learn the background. No camera or download is needed."""),
    ("c", """width, height, fps, frame_count = 320, 220, 20.0, 90
background = np.full((height, width, 3), (60, 70, 80), np.uint8)
for x in range(0, width, 40):
    cv2.line(background, (x, 0), (x, height - 1), (67, 77, 87), 1)
frames, truth_masks = [], []
for index in range(frame_count):
    frame = background.copy()
    truth = np.zeros((height, width), np.uint8)
    if index >= 20:
        x = 10 + (index - 20) * 3
        y = 90 + int(20 * np.sin(index / 9))
        cv2.rectangle(frame, (x, y), (x + 44, y + 34), (30, 180, 235), -1)
        cv2.rectangle(truth, (x, y), (x + 44, y + 34), 255, -1)
    frames.append(frame)
    truth_masks.append(truth)
show_images([frames[i] for i in [0, 25, 55, 85]], [f'Frame {i}' for i in [0, 25, 55, 85]], cols=2)
assert all(f.shape == (height, width, 3) and f.dtype == np.uint8 for f in frames)"""),
    ("c", """print('Nominal duration:', frame_count / fps, 'seconds')
print('Raw frame storage:', round(sum(f.nbytes for f in frames) / 1024**2, 2), 'MiB')
print('Frame 40 nominal time:', 40 / fps, 'seconds')"""),
    ("m", """## Step 2 — Write video with codec fallbacks and guaranteed release
`VideoWriter` takes a filename, FourCC codec, frames per second, and `(width, height)`. Codec support depends on your OpenCV build. Try MJPG/AVI first and mp4v/MP4 second. Always inspect `isOpened()` and release the writer in `finally`, even after an exception. Video compression is usually lossy, so decoded pixels may differ."""),
    ("c", """out_dir = ROOT / 'outputs' / '12_video'
out_dir.mkdir(parents=True, exist_ok=True)
video_path = None
writer = None
attempts = [('MJPG', '.avi'), ('mp4v', '.mp4')]
for codec, suffix in attempts:
    candidate_path = out_dir / f'synthetic_motion{suffix}'
    candidate_writer = cv2.VideoWriter(str(candidate_path), cv2.VideoWriter_fourcc(*codec),
                                        fps, (width, height))
    if candidate_writer.isOpened():
        writer, video_path = candidate_writer, candidate_path
        print('Using codec:', codec)
        break
    candidate_writer.release()

if writer is None:
    raise RuntimeError('No tested video codec is available. Check your OpenCV wheel/backend installation.')
try:
    for frame in frames:
        writer.write(frame)
finally:
    writer.release()
assert video_path.exists() and video_path.stat().st_size > 0
print('Saved:', video_path, '| bytes:', video_path.stat().st_size)"""),
    ("c", """print('Video I/O backends reported by this OpenCV build:')
build_lines = cv2.getBuildInformation().splitlines()
start = next((i for i, line in enumerate(build_lines) if 'Video I/O:' in line), None)
if start is not None:
    for line in build_lines[start:start + 10]:
        print(line)"""),
    ("m", """## Step 3 — Read until the stream ends
Check the Boolean `ret` before using a frame. A failed read may mean end of file, unavailable camera, or decoder failure. Metadata such as frame count can be approximate; the actual decoded count is useful evidence. Keep the read loop finite and release the capture in `finally`."""),
    ("c", """capture = cv2.VideoCapture(str(video_path))
decoded = []
try:
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open saved video: {video_path}')
    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Reported fps:', reported_fps, '| reported frames:', reported_count)
    for _ in range(frame_count + 5):
        ret, frame = capture.read()
        if not ret:
            break
        decoded.append(frame)
finally:
    capture.release()
print('Actually decoded:', len(decoded))
assert len(decoded) == frame_count
assert decoded[0].shape == frames[0].shape"""),
    ("c", """sample_index = 45
codec_difference = cv2.absdiff(frames[sample_index], decoded[sample_index])
show_images([frames[sample_index], decoded[sample_index], codec_difference],
            ['Before encoding', 'Decoded frame', 'Absolute codec difference'])
print('Encoding/decoding MAE:', float(codec_difference.mean()))"""),
    ("m", """Seeking is convenient but can depend on keyframes and decoder behavior. For exact sequential processing, read frames in order. Here, compare a seeked frame with our already decoded list and report any difference."""),
    ("c", """seek_capture = cv2.VideoCapture(str(video_path))
try:
    if not seek_capture.isOpened():
        raise RuntimeError('Could not reopen the generated video.')
    seek_ok = seek_capture.set(cv2.CAP_PROP_POS_FRAMES, sample_index)
    read_ok, seek_frame = seek_capture.read()
    if seek_ok and read_ok:
        print('Seek-vs-sequential MAE:', float(cv2.absdiff(seek_frame, decoded[sample_index]).mean()))
    else:
        print('This backend did not support the requested seek.')
finally:
    seek_capture.release()"""),
    ("m", """## Step 4 — Learn a background model with MOG2
MOG2 models each pixel's background over time. `history` controls its adaptation horizon; `varThreshold` controls how easily a pixel matches a background component. `learningRate=0` freezes the model. We explicitly learn the first 20 empty frames, then freeze it so a stationary foreground object will remain foreground in this exercise."""),
    ("c", """subtractor = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=25, detectShadows=False)
foreground_masks = []
for index, frame in enumerate(decoded):
    learning_rate = .15 if index < 20 else 0.0
    foreground_masks.append(subtractor.apply(frame, learningRate=learning_rate))
show_images([foreground_masks[i] for i in [0, 19, 45, 80]],
            ['First frame: untrained model', 'Background learned', 'Moving foreground', 'Later foreground'], cols=2)
print('Mask values without shadow detection:', np.unique(foreground_masks[45]))"""),
    ("c", """shadow_model = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=25, detectShadows=True)
for index, frame in enumerate(decoded[:46]):
    shadow_mask = shadow_model.apply(frame, learningRate=.15 if index < 20 else 0)
strict_foreground = np.uint8(shadow_mask == 255) * 255
print('Shadow-enabled values:', np.unique(shadow_mask))
print('Default shadow label:', shadow_model.getShadowValue())
show_images([shadow_mask, strict_foreground], ['Shadow-aware mask (may include 127)', 'Keep only definite foreground 255'], cols=2)"""),
    ("m", """Shadows may receive 127 rather than 255 when enabled. `mask > 0` would include them. For this lesson, treat only 255 as foreground. Real cameras also introduce flicker, autofocus changes, camera motion, and illumination changes that this static synthetic setup avoids."""),
    ("c", """cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean_masks = []
for mask in foreground_masks:
    binary = np.uint8(mask == 255) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cleanup_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cleanup_kernel)
    clean_masks.append(binary)
show_images([foreground_masks[45], clean_masks[45]], ['Raw foreground', 'After opening and closing'], cols=2)"""),
    ("m", """## Step 5 — Mini-project: annotate moving-object boxes and evaluate masks
Extract contours above a minimum pixel area and draw boxes. This is frame-by-frame detection, not multi-object tracking: we do not assign persistent identities. Evaluate masks after the warmup frames against the generated ground truth."""),
    ("c", """annotated = []
detection_counts, ious = [], []
for index, (frame, mask, truth) in enumerate(zip(decoded, clean_masks, truth_masks)):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = [c for c in cs if cv2.contourArea(c) >= 200] if index >= 20 else []
    view = frame.copy()
    for contour in objects:
        x, y, bw, bh = cv2.boundingRect(contour)
        cv2.rectangle(view, (x, y), (x + bw - 1, y + bh - 1), (0, 0, 255), 2)
    cv2.putText(view, f'frame {index:02} | objects {len(objects)}', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    annotated.append(view)
    detection_counts.append(len(objects))
    if index >= 20:
        p, t = mask > 0, truth > 0
        ious.append(np.count_nonzero(p & t) / max(1, np.count_nonzero(p | t)))
show_images([annotated[i] for i in [25, 50, 85]], ['Detected object at t=1.25s', 'At t=2.5s', 'At t=4.25s'])
print('Mean foreground IoU after warmup:', float(np.mean(ious)))
assert np.mean(ious) > .8, 'The static synthetic background should permit accurate detection.'"""),
    ("c", """plt.figure(figsize=(10, 3))
plt.plot(np.arange(20, frame_count) / fps, ious)
plt.ylim(0, 1.05); plt.xlabel('Nominal video time (seconds)'); plt.ylabel('Foreground IoU')
plt.title('Detection quality over time'); plt.show()
save_image(out_dir / 'motion_detection_sample.png', annotated[50])"""),
    ("m", """### Timing experiment: source frame rate versus processing throughput
The saved video's 20 FPS specifies playback timing. Processing FPS depends on hardware, image size, and algorithm. Measure only the operation you intend to compare; this example excludes file I/O and rendering. A single tiny benchmark is illustrative, not a hardware performance claim."""),
    ("c", """from time import perf_counter
benchmark_model = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=25, detectShadows=False)
start_time = perf_counter()
for frame in decoded:
    benchmark_model.apply(frame)
elapsed = perf_counter() - start_time
print(f'Processing: {len(decoded) / max(elapsed, 1e-9):.1f} FPS over {elapsed:.3f}s')
print(f'Playback: {fps:.1f} FPS, duration {len(decoded) / fps:.2f}s')"""),
    ("m", """### Optional webcam experiment — OFF by default
Only change `ENABLE_WEBCAM` to `True` when you explicitly want to access your camera. The block captures at most 30 frames and releases the camera even if reading fails. It displays one frame inline and never opens a GUI window. Some backends can block while opening unavailable hardware; leave this disabled for normal Run All."""),
    ("c", """ENABLE_WEBCAM = False
if ENABLE_WEBCAM:
    webcam = cv2.VideoCapture(0)
    latest_frame = None
    try:
        if not webcam.isOpened():
            raise RuntimeError('Camera unavailable. Check permissions or device index.')
        for _ in range(30):
            ok, frame = webcam.read()
            if not ok:
                break
            latest_frame = frame
    finally:
        webcam.release()
    if latest_frame is not None:
        show_images([latest_frame], ['Optional camera sample'])
else:
    print('Webcam disabled. All core lessons use the generated video.')"""),
    ("m", """### Practice
1. Compute the mean detected-object count after warmup and the fraction of frames with exactly one detection.
2. Let MOG2 adapt while an object remains stationary for many frames. Compare it with a frozen model.
3. Write an annotated result video using the selected codec/container. Check opening and release it reliably.

## Worked solutions
For the adaptation experiment, repeat the same object frame. This isolates learning behavior from motion and compression changes."""),
    ("c", """counts = np.asarray(detection_counts[20:])
print('Mean detections:', counts.mean())
print('Exactly one detection:', np.mean(counts == 1))
adapt_model = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=25, detectShadows=False)
freeze_model = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=25, detectShadows=False)
for _ in range(25):
    adapt_model.apply(background, learningRate=.2)
    freeze_model.apply(background, learningRate=.2)
stationary = frames[45]
for _ in range(100):
    adaptive_mask = adapt_model.apply(stationary, learningRate=.05)
    frozen_mask = freeze_model.apply(stationary, learningRate=0)
show_images([stationary, adaptive_mask, frozen_mask], ['Stationary object', 'Adaptation absorbs object', 'Frozen background retains object'])
assert np.count_nonzero(adaptive_mask) < np.count_nonzero(frozen_mask)"""),
    ("c", """annotated_path = out_dir / f'annotated_motion{video_path.suffix}'
selected_codec = 'MJPG' if video_path.suffix == '.avi' else 'mp4v'
output_writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*selected_codec),
                                fps, (width, height))
try:
    if not output_writer.isOpened():
        raise RuntimeError('Unable to open annotated output writer.')
    for frame in annotated:
        output_writer.write(frame)
finally:
    output_writer.release()
print('Annotated video:', annotated_path)
assert annotated_path.exists() and annotated_path.stat().st_size > 0"""),
    ("m", """### Reflection
Why does the first foreground mask often mark nearly the whole frame? What happens when the camera moves? When would you freeze a background model, and when would adaptation be essential?

Official references: [Video capture tutorial](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html), [Background subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html), [VideoWriter API](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html), [MOG2 API](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)."""))
    save("12_video_and_background_subtraction.ipynb", cells)


def main():
    build_edges()
    build_contours()
    build_geometry()
    build_features()
    build_segmentation()
    build_video()


if __name__ == '__main__':
    main()
