"""Build independent, offline advanced OpenCV practical notebooks."""
from tools.notebook_factory import md, code, intro, save


def lesson(filename, title, prerequisites, objectives, sections):
    cells = intro(title, "Advanced / guided expert practice", prerequisites, objectives)
    for kind, content in sections:
        cells.append(md(content) if kind == "m" else code(content))
    save(filename, cells)


def build():
    lesson("13_optical_flow_and_tracking.ipynb", "13 · Optical flow and tracking", 
           ["NumPy indexing", "Corners, gradients, and video frames"],
           ["Track sparse corners and reject inconsistent tracks", "Measure dense optical flow against known motion", "Understand drift, occlusion, and the aperture problem"], [
    ("m", """## Step 1 — Create motion with ground truth
Optical flow estimates apparent motion between frames. Under brightness constancy, a moving patch keeps its appearance: `I(x, y, t) ≈ I(x+u, y+v, t+1)`. A local linear approximation gives `Ix*u + Iy*v + It = 0`. One equation cannot identify both motion components at a single pixel; a textured neighborhood supplies more constraints.

We translate a textured image by **(5, 3) pixels**. This deliberately controlled case lets us measure error. A camera, external video, or internet connection is unnecessary."""),
    ("c", r"""flow_rng = np.random.default_rng(130)
height, width = 240, 360
texture = flow_rng.integers(0, 256, (height, width), dtype=np.uint8)
frame0 = cv2.GaussianBlur(texture, (5, 5), 0.8)
cv2.putText(frame0, 'TRACK', (55, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 240, 3)
truth = np.array([5.0, 3.0], dtype=np.float32)
translation = np.array([[1, 0, truth[0]], [0, 1, truth[1]]], dtype=np.float32)
frame1 = cv2.warpAffine(frame0, translation, (width, height), borderMode=cv2.BORDER_CONSTANT)
show_images([frame0, frame1], ['Frame t', 'Frame t+1: +5 x, +3 y'])"""),
    ("m", """Before executing the next cell, predict the sign of the x and y flow. Image y increases downward. The padded border has no valid predecessor, so validation must exclude it."""),
    ("c", r"""points0 = cv2.goodFeaturesToTrack(frame0, maxCorners=150, qualityLevel=0.03,
                                 minDistance=10, blockSize=7)
assert points0 is not None and len(points0) > 20
interior = ((points0[:, 0, 0] > 20) & (points0[:, 0, 0] < width-25) &
            (points0[:, 0, 1] > 20) & (points0[:, 0, 1] < height-25))
points0 = points0[interior]
marked = cv2.cvtColor(frame0, cv2.COLOR_GRAY2BGR)
for x, y in points0[:, 0]:
    cv2.circle(marked, (int(x), int(y)), 3, (0, 0, 255), -1)
show_images([marked], [f'{len(points0)} well-spaced interior corners'])"""),
    ("m", """## Step 2 — Sparse Lucas–Kanade tracking
`calcOpticalFlowPyrLK` follows selected points through an image pyramid. Coarse levels make motion smaller; finer levels refine it. `status` tells us whether a track was found, but **does not prove it is correct**. Its `err` value is also not ground-truth endpoint error."""),
    ("c", r"""lk_parameters = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
points1, status, lk_error = cv2.calcOpticalFlowPyrLK(frame0, frame1, points0, None, **lk_parameters)
found = status.ravel().astype(bool)
displacements = points1[:, 0] - points0[:, 0]
endpoint_error = np.linalg.norm(displacements - truth, axis=1)
print('Found tracks:', found.sum(), '/', len(found))
print('Median measured [dx, dy]:', np.median(displacements[found], axis=0).round(3))
print('Median endpoint error (pixels):', round(float(np.median(endpoint_error[found])), 4))
assert found.sum() > 30 and np.median(endpoint_error[found]) < 0.3"""),
    ("c", r"""track_view = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
for old, new in zip(points0[found, 0], points1[found, 0]):
    cv2.arrowedLine(track_view, tuple(np.round(old).astype(int)),
                   tuple(np.round(new).astype(int)), (0, 0, 255), 1, tipLength=0.35)
show_images([track_view], ['Sparse motion arrows'])
plt.figure(figsize=(6, 3))
plt.hist(endpoint_error[found], bins=20)
plt.xlabel('Endpoint error (pixels)'); plt.ylabel('Track count'); plt.show()"""),
    ("m", """## Step 3 — Forward–backward checks and occlusion
Track forward, then backward. If the returned point is far from the original point, reject the track. This is a consistency check, not a guarantee: a repeated pattern can produce a consistent but incorrect match in both directions."""),
    ("c", r"""def track_with_consistency(previous, current, points, threshold=0.8):
    forward, s1, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **lk_parameters)
    if forward is None:
        return None, np.zeros(len(points), dtype=bool), np.full(len(points), np.inf)
    backward, s2, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **lk_parameters)
    if backward is None:
        return forward, np.zeros(len(points), dtype=bool), np.full(len(points), np.inf)
    fb_error = np.linalg.norm(backward[:, 0] - points[:, 0], axis=1)
    inside = ((forward[:, 0, 0] >= 0) & (forward[:, 0, 0] < current.shape[1]) &
              (forward[:, 0, 1] >= 0) & (forward[:, 0, 1] < current.shape[0]))
    good = s1.ravel().astype(bool) & s2.ravel().astype(bool) & inside & (fb_error < threshold)
    return forward, good, fb_error

occluded = frame1.copy()
cv2.rectangle(occluded, (100, 60), (220, 180), 20, -1)
occluded_tracks, accepted, fb_error = track_with_consistency(frame0, occluded, points0)
print('Accepted after occlusion:', accepted.sum(), '/', len(accepted))
show_images([occluded], ['An object blocks the tracked texture'])"""),
    ("c", r"""occlusion_view = cv2.cvtColor(occluded, cv2.COLOR_GRAY2BGR)
for point, keep in zip(points0[:, 0], accepted):
    cv2.circle(occlusion_view, tuple(np.round(point).astype(int)), 4,
               (0, 200, 0) if keep else (0, 0, 255), -1)
show_images([occlusion_view], ['Original locations: green accepted / red rejected'])
for cutoff in [0.2, 0.8, 2.0]:
    _, mask, _ = track_with_consistency(frame0, occluded, points0, cutoff)
    print(f'FB threshold {cutoff:.1f} px -> {mask.sum()} accepted')"""),
    ("m", """## Step 4 — Dense Farnebäck flow
Dense flow assigns a vector to every pixel, including pixels where motion is ambiguous. Exclude borders and report a distribution of errors rather than trusting an attractive color plot. Hue below encodes direction; value encodes speed relative to this frame pair."""),
    ("c", r"""dense = cv2.calcOpticalFlowFarneback(frame0, frame1, None, 0.5, 3, 21, 5, 7, 1.5, 0)
assert dense.shape == (height, width, 2)
valid_region = np.zeros((height, width), dtype=bool)
valid_region[25:-25, 25:-25] = True
dense_error = np.linalg.norm(dense - truth, axis=2)
print('Dense EPE median / 90th percentile:', np.percentile(dense_error[valid_region], [50, 90]))
assert np.median(dense_error[valid_region]) < 0.5
magnitude, angle = cv2.cartToPolar(dense[..., 0], dense[..., 1])
hsv = np.zeros((height, width, 3), dtype=np.uint8)
hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
hsv[..., 1] = 255
hsv[..., 2] = np.clip(magnitude / 8 * 255, 0, 255).astype(np.uint8)
show_images([cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), dense_error],
            ['Dense flow: hue=direction, brightness=speed', 'Endpoint error (pixels)'])"""),
    ("c", r"""window_scores = []
for window in [9, 15, 25, 35]:
    estimated = cv2.calcOpticalFlowFarneback(frame0, frame1, None, 0.5, 3, window, 4, 7, 1.5, 0)
    error = np.linalg.norm(estimated - truth, axis=2)
    window_scores.append(float(np.median(error[valid_region])))
plt.plot([9, 15, 25, 35], window_scores, marker='o')
plt.xlabel('Averaging window'); plt.ylabel('Median endpoint error (pixels)'); plt.show()"""),
    ("m", """A larger window gathers more evidence but can mix two objects moving differently. The optimum on this single translating plane is not a universal setting. Try a foreground rectangle that moves in the opposite direction."""),
    ("m", """## Step 5 — Multi-frame tracking and drift
Each estimate depends on the previous estimate. Errors accumulate, tracks disappear, and an application must decide when to detect fresh features. A low track count is an observable reason to reinitialize; silently pretending a lost point still exists is not."""),
    ("c", r"""sequence = [cv2.warpAffine(frame0, np.float32([[1, 0, i*2], [0, 1, i]]),
                           (width, height)) for i in range(12)]
current_points = points0.copy()
identities = np.arange(len(points0))
counts, median_drift = [len(points0)], [0.0]
for index in range(1, len(sequence)):
    moved, good, _ = track_with_consistency(sequence[index-1], sequence[index], current_points)
    if moved is None or not good.any():
        print('No surviving tracks; redetect features before continuing.')
        break
    identities = identities[good]
    current_points = moved[good]
    expected = points0[identities, 0] + np.array([index*2, index])
    counts.append(len(current_points))
    median_drift.append(float(np.median(np.linalg.norm(current_points[:, 0] - expected, axis=1))))
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(counts, marker='o'); axes[0].set(xlabel='Frame', ylabel='Surviving tracks')
axes[1].plot(median_drift, marker='o'); axes[1].set(xlabel='Frame', ylabel='Median drift (pixels)')
plt.tight_layout(); plt.show()"""),
    ("m", """## Step 6 — Investigate an unobservable case
A completely uniform image is unchanged by a translation away from its border. No image-based method can recover the motion there. A motion estimate of zero can mean **no evidence**, rather than a stationary object."""),
    ("c", r"""flat0 = np.full_like(frame0, 128)
flat1 = cv2.warpAffine(flat0, translation, (width, height), borderMode=cv2.BORDER_REFLECT)
flat_corners = cv2.goodFeaturesToTrack(flat0, 100, 0.01, 5)
flat_flow = cv2.calcOpticalFlowFarneback(flat0, flat1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
print('Detected flat-image corners:', flat_corners)
print('Median estimated flow:', np.median(flat_flow[valid_region], axis=0))
print('True translation remains:', truth, '— it cannot be inferred from these observations.')
assert np.array_equal(flat0, flat1)"""),
    ("m", """### Practice before opening the solutions
1. Change the translation to `(12, -4)` and compare pyramidal LK with `maxLevel=0`.
2. Measure the percentage of surviving tracks whose true error exceeds one pixel after occlusion.
3. Create a spatial mask for texture-rich pixels, and compare dense error inside and outside it. Explain why ground-truth-free consistency is different from accuracy.

Write your prediction and result in a new Markdown cell. Keep validation regions inside the overlapping field of view."""),
    ("c", r"""# Your experiment: change one parameter, rerun, and record a numerical result.
experiment_translation = (12, -4)
experiment_frame = cv2.warpAffine(frame0, np.float32([[1, 0, experiment_translation[0]],
                                                  [0, 1, experiment_translation[1]]]), (width, height))
show_images([experiment_frame], ['Your larger-motion test input'])"""),
    ("m", """### Worked solutions
The exercise compares the same points under two pyramid settings, so the metric isolates that choice. Pyramids help large motions but cannot solve complete occlusion or missing texture."""),
    ("c", r"""for levels in [0, 3]:
    params = {**lk_parameters, 'maxLevel': levels}
    p, s, _ = cv2.calcOpticalFlowPyrLK(frame0, experiment_frame, points0, None, **params)
    keep = s.ravel().astype(bool)
    epe = np.linalg.norm(p[keep, 0] - points0[keep, 0] - experiment_translation, axis=1)
    print(f'Pyramid maxLevel={levels}: tracks={keep.sum()}, median EPE={np.median(epe):.3f} px')
if accepted.any():
    error = np.linalg.norm(occluded_tracks[accepted, 0] - points0[accepted, 0] - truth, axis=1)
    print('Accepted occluded-scene tracks with EPE > 1px:', f'{100*np.mean(error > 1):.1f}%')
gx = cv2.Sobel(frame0, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(frame0, cv2.CV_32F, 0, 1)
energy = cv2.GaussianBlur(gx*gx + gy*gy, (11, 11), 0)
rich = energy > np.median(energy[valid_region])
for label, mask in [('more gradient energy', rich), ('less gradient energy', ~rich)]:
    print(label, ': median dense EPE=', round(float(np.median(dense_error[mask & valid_region])), 4))"""),
    ("m", """### Checkpoint and references
You can now distinguish tracking status, consistency, and actual error. Real scenes add changing exposure, motion blur, rolling shutter, independently moving objects, and camera motion. Test these separately before calling a tracker reliable.

- [OpenCV optical flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [OpenCV tracking API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)

Next: recover camera geometry and pose instead of only image motion.""")])

    lesson("14_camera_calibration_and_pose.ipynb", "14 · Camera calibration and pose", 
           ["Image coordinates", "Perspective transformations", "NumPy arrays"],
           ["Estimate intrinsics from several known planar targets", "Separate reprojection error from parameter accuracy", "Estimate pose and validate on a held-out view"], [
    ("m", """## Step 1 — Define a camera and a measured target
A pinhole camera projects 3D camera coordinates with `u = fx*X/Z + cx`, `v = fy*Y/Z + cy`. The intrinsic matrix K contains focal lengths **in pixels** and the principal point. Distortion bends these ideal projections. Extrinsics rotate and translate world points into the camera frame.

Our board contains **9 × 6 inner corners**, spaced **0.025 metres** apart. Those units determine the units of estimated translation. We know the synthetic camera, so we can check estimated parameters as well as image error."""),
    ("c", r"""calibration_rng = np.random.default_rng(140)
image_size = (640, 480)  # OpenCV uses (width, height)
board_size = (9, 6)     # INNER corner columns, rows
square_m = 0.025
object_points = np.zeros((board_size[0]*board_size[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
object_points[:, :2] *= square_m
object_points[:, :2] -= object_points[:, :2].mean(axis=0)
K_true = np.array([[800., 0., 320.], [0., 815., 240.], [0., 0., 1.]])
dist_true = np.array([-0.12, 0.04, 0., 0., 0.])  # k1, k2, p1, p2, k3
print('Known K:\n', K_true)
print('Board extent (m):', np.ptp(object_points[:, :2], axis=0))"""),
    ("m", """Calibration needs varied views: tilt the target and move it across the image. A pile of nearly identical frontal views adds little geometric information. Here we generate corner observations directly; this isolates calibration from corner-detection mistakes."""),
    ("c", r"""observations, rotations_true, translations_true = [], [], []
for i in range(24):
    rotation = calibration_rng.uniform([-0.45, -0.5, -0.2], [0.45, 0.5, 0.2])
    translation = calibration_rng.uniform([-0.065, -0.045, 0.48], [0.065, 0.045, 0.78])
    projected, _ = cv2.projectPoints(object_points, rotation, translation, K_true, dist_true)
    observed = projected + calibration_rng.normal(0, 0.12, projected.shape)
    observations.append(observed.astype(np.float32))
    rotations_true.append(rotation)
    translations_true.append(translation)
fig, ax = plt.subplots(figsize=(7, 5))
for i, corners in enumerate(observations):
    ax.scatter(corners[:, 0, 0], corners[:, 0, 1], s=3, alpha=0.5)
ax.set(xlim=(0, image_size[0]), ylim=(image_size[1], 0), xlabel='u (pixels)', ylabel='v (pixels)',
       title='24 varied synthetic board views'); ax.set_aspect('equal'); plt.show()"""),
    ("m", """## Step 2 — Detect and refine real image corners
The next tiny rendered board demonstrates the detector API. A 9×6 inner-corner board has 10×7 squares. This frontal image alone is **not** the calibration dataset. Real capture should contain sharp, fully visible, tilted boards distributed across the frame."""),
    ("c", r"""cell_px = 40
board = np.full((7*cell_px + 60, 10*cell_px + 60), 200, np.uint8)
for row in range(7):
    for col in range(10):
        color = 255 if (row+col) % 2 else 0
        cv2.rectangle(board, (30+col*cell_px, 30+row*cell_px),
                       (30+(col+1)*cell_px-1, 30+(row+1)*cell_px-1), color, -1)
detected, corners = cv2.findChessboardCorners(board, board_size)
assert detected, 'The generated board should be detectable.'
refined = cv2.cornerSubPix(board, corners, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
board_view = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
cv2.drawChessboardCorners(board_view, board_size, refined, detected)
show_images([board_view], [f'Detected {len(refined)} inner corners'])"""),
    ("m", """Subpixel refinement estimates the corner location between pixel centres. It does not rescue a wrong corner ordering, an incorrect board size, or a target that is bent rather than planar."""),
    ("m", """## Step 3 — Calibrate and assess identifiable parameters
`calibrateCamera` jointly fits K, distortion, and one pose per image. We fix k3 and tangential distortion to their known zero values to keep this controlled model appropriately small. On real data, justify flags from the camera model and validation, not from a desire for a lower training error."""),
    ("c", r"""calibration_flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
def calibrate(corner_sets):
    return cv2.calibrateCamera([object_points.copy() for _ in corner_sets], corner_sets,
                               image_size, None, None, flags=calibration_flags)
rms, K_est, dist_est, rvecs, tvecs = calibrate(observations)
print('Calibration RMS (pixels):', rms)
print('Estimated K:\n', K_est.round(3))
print('Estimated distortion:', dist_est.ravel().round(5))
focal_relative_error = np.abs(np.diag(K_est)[:2] / np.diag(K_true)[:2] - 1)
print('Focal relative error:', focal_relative_error)
print('Principal point error (pixels):', np.linalg.norm(K_est[:2, 2] - K_true[:2, 2]))
assert rms < 0.5 and np.max(focal_relative_error) < 0.05"""),
    ("c", r"""view_errors = []
for detected_points, rvec, tvec in zip(observations, rvecs, tvecs):
    reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, K_est, dist_est)
    residual = detected_points[:, 0] - reprojected[:, 0]
    view_errors.append(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
plt.bar(np.arange(len(view_errors)), view_errors)
plt.xlabel('Calibration view'); plt.ylabel('RMS reprojection error (pixels)'); plt.show()
assert np.isclose(np.sqrt(np.mean(np.array(view_errors)**2)), rms, atol=1e-3)"""),
    ("m", """A low reprojection error can coexist with poor focal or distortion estimates if views are uninformative. Since synthetic ground truth is available, inspect both. High-order distortion is particularly difficult to identify without points near the image boundaries."""),
    ("c", r"""sample_counts, focal_errors, fitted_rms = [3, 6, 12, 24], [], []
for count in sample_counts:
    subset_rms, subset_K, _, _, _ = calibrate(observations[:count])
    fitted_rms.append(subset_rms)
    focal_errors.append(100*np.mean(np.abs(np.diag(subset_K)[:2]/np.diag(K_true)[:2]-1)))
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(sample_counts, focal_errors, 'o-'); axes[0].set(xlabel='Views', ylabel='Mean focal error (%)')
axes[1].plot(sample_counts, fitted_rms, 'o-'); axes[1].set(xlabel='Views', ylabel='Training reprojection RMS (px)')
plt.tight_layout(); plt.show()"""),
    ("m", """## Step 4 — Estimate pose on an unseen view
`solvePnP` uses known 3D-to-2D correspondences and a calibrated camera. Its returned translation is the target origin expressed in **camera coordinates**, not the camera's position in the world. OpenCV's usual camera axes are x right, y down, z forward."""),
    ("c", r"""test_rvec = np.array([0.27, -0.22, 0.12])
test_tvec = np.array([0.015, -0.02, 0.63])
test_image_points, _ = cv2.projectPoints(object_points, test_rvec, test_tvec, K_true, dist_true)
test_observed = (test_image_points + calibration_rng.normal(0, 0.12, test_image_points.shape)).astype(np.float32)
pose_ok, pose_rvec, pose_tvec = cv2.solvePnP(object_points, test_observed, K_est, dist_est,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
assert pose_ok
R_est, _ = cv2.Rodrigues(pose_rvec)
R_true, _ = cv2.Rodrigues(test_rvec)
angle_error = np.degrees(np.arccos(np.clip((np.trace(R_est @ R_true.T)-1)/2, -1, 1)))
translation_error = np.linalg.norm(pose_tvec.ravel() - test_tvec)
heldout_reprojection, _ = cv2.projectPoints(object_points, pose_rvec, pose_tvec, K_est, dist_est)
heldout_rms = np.sqrt(np.mean(np.sum((heldout_reprojection-test_observed)**2, axis=2)))
print(f'Held-out RMS: {heldout_rms:.3f} px; rotation error: {angle_error:.3f} degrees')
print(f'Translation error: {translation_error*1000:.2f} mm')
print('Camera centre in board coordinates (m):', (-R_est.T @ pose_tvec).ravel())
assert heldout_rms < 1 and translation_error < 0.03"""),
    ("c", r"""pose_canvas = np.full((image_size[1], image_size[0], 3), 245, np.uint8)
for p in test_observed[:, 0]:
    cv2.circle(pose_canvas, tuple(np.round(p).astype(int)), 3, (30, 30, 30), -1)
cv2.drawFrameAxes(pose_canvas, K_est, dist_est, pose_rvec, pose_tvec, 0.07, 3)
show_images([pose_canvas], ['Board pose axes: x red, y green, z blue'])"""),
    ("m", """## Step 5 — Correct points and save reusable calibration
`undistortPoints(..., P=K)` returns corrected pixel coordinates. Without `P`, it returns normalized coordinates. Mixing those coordinate systems is a common source of plausible-looking geometry errors. Camera calibration belongs to a particular resolution and optical configuration."""),
    ("c", r"""corrected_points = cv2.undistortPoints(test_observed, K_est, dist_est, P=K_est)
ideal_points, _ = cv2.projectPoints(object_points, test_rvec, test_tvec, K_true, np.zeros(5))
print('Mean correction magnitude (pixels):',
      np.mean(np.linalg.norm(corrected_points-test_observed, axis=2)))
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(test_observed[:, 0, 0], test_observed[:, 0, 1], label='Distorted observations', s=12)
ax.scatter(corrected_points[:, 0, 0], corrected_points[:, 0, 1], label='Undistorted', s=12)
ax.invert_yaxis(); ax.set_aspect('equal'); ax.legend(); plt.show()
calibration_dir = ROOT / 'outputs' / '14_calibration'
calibration_dir.mkdir(parents=True, exist_ok=True)
np.savez(calibration_dir / 'synthetic_calibration.npz', K=K_est, distortion=dist_est,
         image_size=image_size, square_size_m=square_m)
print('Saved:', calibration_dir / 'synthetic_calibration.npz')"""),
    ("m", """## Step 6 — Expose a scale ambiguity
If every target dimension is doubled while the same image points are used, estimated translation doubles. Image reprojection alone cannot reveal that the ruler was wrong. This is why recording physical square size is essential."""),
    ("c", r"""ok_scaled, r_scaled, t_scaled = cv2.solvePnP(object_points*2, test_observed, K_est, dist_est)
assert ok_scaled
print('Original translation:', pose_tvec.ravel())
print('Wrong square-size translation:', t_scaled.ravel())
print('Ratio:', (t_scaled / pose_tvec).ravel())
np.testing.assert_allclose(t_scaled, 2*pose_tvec, rtol=1e-3, atol=1e-5)"""),
    ("m", """### Practice
1. Add one pixel of measurement noise to every corner and recalibrate. Compare parameter error and reprojection RMS.
2. Resize an image by 0.5. Derive the new K: focal lengths and principal point both scale, but distortion coefficients do not.
3. Explain why the 24 generated views validate geometry but do not test real corner detection, lens asymmetry, board warping, or rolling shutter."""),
    ("c", r"""# Your experiment area: increase this noise standard deviation.
extra_noise_px = 1.0
noisy_observations = [(p + calibration_rng.normal(0, extra_noise_px, p.shape)).astype(np.float32)
                      for p in observations]
print('Created', len(noisy_observations), 'noisier views; calibrate them next.')"""),
    ("m", """### Worked solutions
Error need not improve monotonically with more views: view geometry, coverage, and measurement quality matter. Evaluate a fixed held-out view and inspect residual patterns when comparing capture strategies."""),
    ("c", r"""noisy_rms, noisy_K, _, _, _ = calibrate(noisy_observations)
print('Noisy RMS:', noisy_rms)
print('Noisy focal error (%):', 100*np.abs(np.diag(noisy_K)[:2]/np.diag(K_true)[:2]-1))
scale = 0.5
resized_K = K_est.copy()
resized_K[:2, :] *= scale
projected_full, _ = cv2.projectPoints(object_points, pose_rvec, pose_tvec, K_est, dist_est)
projected_half, _ = cv2.projectPoints(object_points, pose_rvec, pose_tvec, resized_K, dist_est)
np.testing.assert_allclose(projected_half, projected_full*scale, atol=1e-4)
print('Half-resolution K:\n', resized_K)
print('Resized-coordinate projection check passed.')"""),
    ("m", """### References and next challenge
- [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV perspective-n-point pose documentation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [Calibration and 3D reconstruction API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

For a real calibration, capture a measured, rigid board at varied poses; reject blurred detections; check unseen images; record resolution, square units, camera, focus, and calibration date. Next we combine two cameras to estimate depth.""")])

    lesson("15_stereo_depth_and_3d.ipynb", "15 · Stereo depth and 3D", 
           ["Camera intrinsics and extrinsics", "Grayscale images", "Array masks"],
           ["Estimate disparity on a synthetic rectified stereo pair", "Convert disparity to metric depth with explicit units", "Handle invalid estimates and triangulate 3D points"], [
    ("m", """## Step 1 — Construct a rectified stereo pair
For rectified cameras with equal intrinsics, corresponding points lie on the same image row. Left disparity is `d = x_left - x_right`. With focal length `f` in pixels and baseline `B` in metres, depth is `Z = f*B/d` in metres.

We synthesize a textured fronto-parallel plane with disparity **12 pixels**, focal length **240 pixels**, and baseline **0.10 metres**. Its true depth is **2 metres**. The data is already rectified; arbitrary side-by-side photographs cannot use this formula directly."""),
    ("c", r"""stereo_rng = np.random.default_rng(150)
h, w = 240, 384
left = stereo_rng.integers(0, 256, (h, w), dtype=np.uint8)
left = cv2.GaussianBlur(left, (3, 3), 0.6)
disparity_true = 12
focal_px, baseline_m = 240.0, 0.10
depth_true_m = focal_px * baseline_m / disparity_true
right = np.zeros_like(left)
right[:, :-disparity_true] = left[:, disparity_true:]
right[:, -disparity_true:] = stereo_rng.integers(0, 256, (h, disparity_true), dtype=np.uint8)
show_images([left, right], ['Left', 'Right: corresponding texture is shifted left'])
print('Known plane depth:', depth_true_m, 'metres')"""),
    ("c", r"""row = 100
plt.figure(figsize=(10, 3))
plt.plot(np.arange(60, 160), left[row, 60:160], label='Left intensity')
plt.plot(np.arange(60, 160)+disparity_true, right[row, 60:160], '--', label='Right moved +12 for comparison')
plt.xlabel('Left-image x coordinate'); plt.ylabel('Intensity'); plt.legend(); plt.show()
assert np.array_equal(left[:, disparity_true:], right[:, :-disparity_true])"""),
    ("m", """## Step 2 — Match scanlines with StereoSGBM
Semi-global block matching balances local appearance with smoothness along several directions. `numDisparities` must be divisible by 16; `blockSize` is an odd matching window. The penalties P1 and P2 discourage sudden disparity changes, with P2 larger than P1.

OpenCV returns fixed-point disparity scaled by **16**, so divide by 16 before any depth calculation. Invalid results must be masked, not converted into plausible distances."""),
    ("c", r"""def compute_disparity(first, second, block_size=5, num_disparities=64):
    assert block_size % 2 == 1 and num_disparities % 16 == 0
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparities,
        blockSize=block_size, P1=8*block_size**2, P2=32*block_size**2,
        disp12MaxDiff=1, uniquenessRatio=8, speckleWindowSize=0,
        speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    raw = matcher.compute(first, second)
    return raw, raw.astype(np.float32)/16.0

raw_disparity, disparity = compute_disparity(left, right)
print('Raw dtype:', raw_disparity.dtype, '| disparity units: pixels')
valid = np.isfinite(disparity) & (disparity > 0)
# Keep evaluation away from image borders and the matcher search margin.
evaluation_region = np.zeros_like(valid)
evaluation_region[15:-15, 80:-20] = True
evaluate = valid & evaluation_region
assert evaluate.any()
print('Valid fraction in evaluation region:', evaluate.sum()/evaluation_region.sum())
print('Median disparity:', np.median(disparity[evaluate]))
assert np.median(np.abs(disparity[evaluate]-disparity_true)) < 0.5"""),
    ("c", r"""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
im = axes[0].imshow(np.ma.masked_where(~valid, disparity), cmap='turbo', vmin=0, vmax=24)
axes[0].set_title('Disparity (pixels); blank is invalid'); fig.colorbar(im, ax=axes[0])
axes[1].imshow(valid, cmap='gray'); axes[1].set_title('Validity mask')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()"""),
    ("m", """## Step 3 — Convert to metric depth safely
Depth grows nonlinearly as disparity approaches zero. Fill invalid depths with `NaN`; do not divide by zero and then hide infinities. A far-depth cutoff is an application decision and should be recorded explicitly."""),
    ("c", r"""depth = np.full(disparity.shape, np.nan, np.float32)
np.divide(focal_px*baseline_m, disparity, out=depth, where=valid)
assert np.isfinite(depth[valid]).all() and np.isnan(depth[~valid]).all()
depth_error = np.abs(depth[evaluate]-depth_true_m)
print('Median depth (m):', np.median(depth[evaluate]))
print('Depth absolute error median / 95th percentile (m):', np.percentile(depth_error, [50, 95]))
assert np.median(depth_error) < 0.1
plt.figure(figsize=(7, 4))
plt.imshow(depth, cmap='viridis', vmin=1.5, vmax=2.5)
plt.colorbar(label='Depth (metres)'); plt.title('Reconstructed plane'); plt.axis('off'); plt.show()"""),
    ("c", r"""sample_disparities = np.linspace(1, 48, 250)
sample_depths = focal_px*baseline_m/sample_disparities
sigma_disparity_px = 0.5
# First-order propagation: sigma_Z ≈ f*B/d² * sigma_d.
sigma_depth_m = focal_px*baseline_m/sample_disparities**2 * sigma_disparity_px
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].plot(sample_disparities, sample_depths); axes[0].set(xlabel='Disparity (px)', ylabel='Depth (m)')
axes[1].plot(sample_depths, sigma_depth_m); axes[1].set(xlabel='Depth (m)', ylabel='Approx. depth uncertainty (m)')
plt.tight_layout(); plt.show()
print('At d=12 px, 0.5 px error implies about', focal_px*baseline_m/12**2*0.5, 'm depth uncertainty.')"""),
    ("m", """## Step 4 — Build a small 3D point cloud
For a rectified camera, `X=(u-cx)*Z/f`, `Y=(v-cy)*Z/f`. The result below uses metres, with y downward and z forward. Subsampling makes plotting fast; it does not change the reconstruction equations."""),
    ("c", r"""cx, cy = w/2, h/2
v, u = np.indices(depth.shape)
X = (u-cx)*depth/focal_px
Y = (v-cy)*depth/focal_px
cloud_mask = evaluate & ((u % 8) == 0) & ((v % 8) == 0)
cloud = np.column_stack([X[cloud_mask], Y[cloud_mask], depth[cloud_mask]])
assert len(cloud) > 100 and np.isfinite(cloud).all()
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=left[cloud_mask], cmap='gray', s=5)
ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Synthetic plane point cloud')
ax.set_zlim(1.5, 2.5); plt.show()"""),
    ("c", r"""stereo_dir = ROOT / 'outputs' / '15_stereo'
stereo_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(stereo_dir / 'synthetic_depth.npz', disparity=disparity, depth_m=depth,
                    valid=valid, focal_px=focal_px, baseline_m=baseline_m)
np.savetxt(stereo_dir / 'plane_xyz_metres.csv', cloud, delimiter=',',
           header='X_m,Y_m,Z_m', comments='')
print('Saved arrays and point cloud to', stereo_dir)"""),
    ("m", """## Step 5 — Test failure modes and parameters
Stereo matching assumes synchronized views, matching appearance, and rectification. Textureless areas and repeated patterns are ambiguous. Invalid-output masks are useful but cannot identify every confidently wrong match."""),
    ("c", r"""right_misaligned = cv2.warpAffine(right, np.float32([[1, 0, 0], [0, 1, 3]]), (w, h))
_, bad_disparity = compute_disparity(left, right_misaligned)
bad_valid = (bad_disparity > 0) & evaluation_region
for name, values, mask in [('Rectified', disparity, evaluate),
                            ('Vertical offset 3px', bad_disparity, bad_valid)]:
    errors = np.abs(values[mask]-disparity_true)
    print(name, '| valid coverage:', round(float(mask.sum()/evaluation_region.sum()), 3),
          '| median error:', round(float(np.median(errors)), 3) if errors.size else 'no matches',
          '| >1px errors:', round(float(np.mean(errors > 1)), 3) if errors.size else 'n/a')
show_images([disparity, bad_disparity], ['Rectified disparity', 'Broken row alignment'])"""),
    ("c", r"""results = []
for window in [3, 5, 9, 15]:
    _, candidate = compute_disparity(left, right, block_size=window)
    mask = (candidate > 0) & evaluation_region
    results.append((window, mask.sum()/evaluation_region.sum(),
                    float(np.median(np.abs(candidate[mask]-disparity_true)))))
print('Block size | valid coverage | median disparity error')
for row_result in results:
    print(f'{row_result[0]:10} | {row_result[1]:14.3f} | {row_result[2]:.4f} px')"""),
    ("m", """## Step 6 — Verify triangulation independently
`triangulatePoints` reconstructs homogeneous coordinates from corresponding pixels and two projection matrices. A homogeneous point `(Xh,Yh,Zh,Wh)` becomes Euclidean after division by `Wh`. This is a separate geometric check using perfect correspondences, not a claim that stereo matching always finds them."""),
    ("c", r"""K = np.array([[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1.]])
P_left = K @ np.column_stack([np.eye(3), np.zeros(3)])
P_right = K @ np.column_stack([np.eye(3), [-baseline_m, 0, 0]])
known_xyz = np.array([[-0.2, -0.1, 2.], [0.1, 0.05, 1.5], [0.3, 0.12, 3.]], dtype=np.float64)
homogeneous_xyz = np.column_stack([known_xyz, np.ones(len(known_xyz))]).T
pixels_left = P_left @ homogeneous_xyz; pixels_left = pixels_left[:2]/pixels_left[2]
pixels_right = P_right @ homogeneous_xyz; pixels_right = pixels_right[:2]/pixels_right[2]
reconstructed_h = cv2.triangulatePoints(P_left, P_right, pixels_left, pixels_right)
assert np.all(np.abs(reconstructed_h[3]) > 1e-10)
reconstructed = (reconstructed_h[:3]/reconstructed_h[3]).T
np.testing.assert_allclose(reconstructed, known_xyz, atol=1e-6)
print('Reconstructed XYZ (m):\n', reconstructed.round(4))"""),
    ("m", """### Practice
1. Double the baseline and keep the plane at 2 m. What disparity should the new synthetic pair have?
2. Forget the `/16` conversion deliberately in a separate variable. How wrong is the resulting depth?
3. Replace a central image patch with a uniform value in both images. Measure valid coverage and accuracy there. Explain why valid coverage alone is insufficient.

Do not change `disparity` in place; preserve a trustworthy reference result."""),
    ("c", r"""# Your experiment workspace.
new_baseline_m = 0.20
predicted_new_disparity_px = focal_px * new_baseline_m / depth_true_m
print('Prediction for the doubled baseline:', predicted_new_disparity_px, 'pixels')"""),
    ("m", """### Worked solutions
The increased baseline improves geometric depth sensitivity in this ideal case, while real wide-baseline cameras also face larger occlusions and appearance differences."""),
    ("c", r"""wrong_depth = focal_px*baseline_m/raw_disparity[evaluate].astype(float)
print('Correct median depth:', np.median(depth[evaluate]), 'm')
print('Wrong unscaled-disparity depth:', np.median(wrong_depth), 'm (16 times too small)')
flat_left, flat_right = left.copy(), right.copy()
flat_left[70:170, 150:270] = 128
flat_right[70:170, 150-disparity_true:270-disparity_true] = 128
_, flat_disp = compute_disparity(flat_left, flat_right)
patch = np.zeros_like(valid); patch[85:155, 170:250] = True
patch_valid = patch & (flat_disp > 0)
print('Textureless patch valid coverage:', patch_valid.sum()/patch.sum())
if patch_valid.any():
    print('Textureless patch median error:', np.median(np.abs(flat_disp[patch_valid]-disparity_true)))
show_images([flat_left, flat_disp], ['Uniform patch', 'Smoothness can fill an ambiguous region'])"""),
    ("m", """### References and real-world extension
- [OpenCV StereoSGBM API](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)
- [OpenCV stereo depth tutorial](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [Calibration, stereoRectify, and triangulation API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

A real stereo pipeline adds intrinsic calibration, `stereoCalibrate`, `stereoRectify`, and `initUndistortRectifyMap`/`remap` before matching. Validate known distances and left–right consistency, and record baseline units.""")])

    build_learning_and_project()


def build_learning_and_project():
    lesson("16_hog_and_machine_learning.ipynb", "16 · HOG features and classical machine learning",
           ["Gradients and histograms", "NumPy arrays", "Basic train/test vocabulary"],
           ["Build a labeled synthetic dataset", "Select an SVM using validation data", "Evaluate held-out data and distribution shift"], [
    ("m", """## Step 1 — Define the problem and split first
Classify circle, square, and triangle images. HOG summarizes local edge directions; an SVM learns boundaries between feature vectors. This deliberately small synthetic task is not evidence of real-world recognition ability.

Use independent random streams for training, validation, and testing. Choose hyperparameters with validation accuracy. Inspect test predictions only after choosing the model. Never scatter augmented versions of one source image across splits."""),
    ("c", r"""class_names = ['circle', 'square', 'triangle']
def make_shape(label, random, offset_x=0):
    image = np.full((64, 64), random.integers(10, 35), np.uint8)
    center = np.array([32 + offset_x, 32]) + random.integers(-5, 6, 2)
    radius = int(random.integers(14, 22))
    shade = int(random.integers(180, 256))
    angle = random.uniform(-np.pi/5, np.pi/5)
    if label == 0:
        cv2.circle(image, tuple(center), radius, shade, -1, cv2.LINE_AA)
    else:
        count = 4 if label == 1 else 3
        base_angle = np.pi/4 if label == 1 else -np.pi/2
        angles = np.arange(count)*2*np.pi/count + base_angle + angle
        vertices = center + radius*np.column_stack([np.cos(angles), np.sin(angles)])
        cv2.fillConvexPoly(image, vertices.round().astype(np.int32), shade, cv2.LINE_AA)
    return np.clip(image.astype(np.float32)+random.normal(0, 5, image.shape), 0, 255).astype(np.uint8)

def make_dataset(samples_per_class, seed, offset_x=0):
    random = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), samples_per_class).astype(np.int32)
    images = np.stack([make_shape(int(label), random, offset_x) for label in labels])
    order = random.permutation(len(labels))
    return images[order], labels[order]

train_images, y_train = make_dataset(90, 1601)
val_images, y_val = make_dataset(30, 1602)
test_images, y_test = make_dataset(40, 1603)
print('Train / validation / test:', len(y_train), len(y_val), len(y_test))"""),
    ("c", r"""show_images(list(train_images[:12]), [class_names[i] for i in y_train[:12]], cols=4, figsize=(10, 7))
for split_name, labels in [('train', y_train), ('validation', y_val), ('test', y_test)]:
    print(split_name, 'class counts:', np.bincount(labels, minlength=3))
train_hashes = {image.tobytes() for image in train_images}
assert not any(image.tobytes() in train_hashes for image in val_images)
assert not any(image.tobytes() in train_hashes for image in test_images)"""),
    ("m", """## Step 2 — Extract HOG
Each 8×8 cell votes gradient strength into nine orientation bins. Overlapping 16×16 blocks normalize contrast. A 64×64 image has 7×7 blocks, each with 2×2 cells: 7×7×2×2×9 = **1764 features**.

HOG depends on position. Centering, scale, and crop policy are part of the model contract. This baseline uses HOG's local block normalization without an additional learned transform."""),
    ("c", r"""hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
def extract_hog(images):
    return np.stack([hog.compute(np.ascontiguousarray(image)).ravel() for image in images]).astype(np.float32)

X_train, X_val, X_test = map(extract_hog, [train_images, val_images, test_images])
assert X_train.shape == (270, 1764) and np.isfinite(X_train).all()
print('Training features:', X_train.shape, X_train.dtype)
plt.figure(figsize=(10, 2))
plt.plot(X_train[0]); plt.xlabel('HOG index'); plt.ylabel('Normalized vote'); plt.show()"""),
    ("c", r"""example = train_images[0]
dx = cv2.Sobel(example, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(example, cv2.CV_32F, 0, 1)
strength, direction = cv2.cartToPolar(dx, dy, angleInDegrees=True)
show_images([example, strength, direction % 180], ['Input', 'Gradient magnitude', 'Unsigned orientation'])
plt.hist((direction % 180).ravel(), bins=np.linspace(0, 180, 10), weights=strength.ravel())
plt.xlabel('Direction (degrees)'); plt.ylabel('Magnitude votes'); plt.show()"""),
    ("m", """The global histogram above illustrates voting; it is not the full HOG descriptor. HOG also preserves spatial cells and normalizes blocks. Removing spatial layout loses useful shape information."""),
    ("m", """## Step 3 — Choose an SVM using validation
A linear SVM seeks a large-margin separating boundary. C controls the penalty for training errors: larger C prioritizes fitting training examples. OpenCV handles multiclass prediction internally through binary classifiers. Labels are integers and features are float32."""),
    ("c", r"""def train_svm(features, labels, C):
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setC(float(C))
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 3000, 1e-6))
    assert model.train(features, cv2.ml.ROW_SAMPLE, labels)
    return model

c_values = [0.01, 0.1, 1.0, 10.0]
candidate_models, training_scores, validation_scores = [], [], []
for C in c_values:
    candidate = train_svm(X_train, y_train, C)
    candidate_models.append(candidate)
    training_scores.append(float(np.mean(candidate.predict(X_train)[1].ravel() == y_train)))
    validation_scores.append(float(np.mean(candidate.predict(X_val)[1].ravel() == y_val)))
best_index = int(np.argmax(validation_scores))  # first maximum breaks ties toward smaller C
best_model = candidate_models[best_index]
for C, training, validation in zip(c_values, training_scores, validation_scores):
    print(f'C={C:5g}: train={training:.3f}, validation={validation:.3f}')
print('Chosen C:', c_values[best_index])"""),
    ("c", r"""plt.semilogx(c_values, training_scores, 'o-', label='Training')
plt.semilogx(c_values, validation_scores, 's-', label='Validation')
plt.xlabel('C'); plt.ylabel('Accuracy'); plt.ylim(0, 1.05); plt.legend(); plt.show()
print('Majority validation baseline:', np.bincount(y_val).max()/len(y_val))"""),
    ("m", """## Step 4 — Evaluate the held-out test set
Model selection is complete. Accuracy summarizes overall performance; confusion matrices reveal class-specific failures. Rows below are true classes, columns are predictions. Repeatedly tuning after seeing this result converts the test set into another validation set."""),
    ("c", r"""test_predictions = best_model.predict(X_test)[1].ravel().astype(int)
test_accuracy = float(np.mean(test_predictions == y_test))
confusion = np.zeros((3, 3), dtype=int)
np.add.at(confusion, (y_test, test_predictions), 1)
assert confusion.sum() == len(y_test) and test_accuracy > 0.75
print(f'Held-out accuracy: {test_accuracy:.1%} ({len(y_test)} samples)')
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(confusion, cmap='Blues'); fig.colorbar(im, ax=ax)
ax.set(xticks=range(3), yticks=range(3), xticklabels=class_names, yticklabels=class_names,
       xlabel='Predicted', ylabel='True')
for row in range(3):
    for col in range(3): ax.text(col, row, str(confusion[row, col]), ha='center', va='center')
plt.tight_layout(); plt.show()"""),
    ("c", r"""tp = np.diag(confusion).astype(float)
precision = np.divide(tp, confusion.sum(axis=0), out=np.zeros(3), where=confusion.sum(axis=0)>0)
recall = np.divide(tp, confusion.sum(axis=1), out=np.zeros(3), where=confusion.sum(axis=1)>0)
for name, p, r in zip(class_names, precision, recall):
    print(f'{name:8s}: precision={p:.3f}, recall={r:.3f}')
wrong = np.flatnonzero(test_predictions != y_test)
examples = wrong[:8] if len(wrong) else np.arange(6)
show_images(list(test_images[examples]),
            [f'True {class_names[y_test[i]]}\nPred {class_names[test_predictions[i]]}' for i in examples], cols=4)
print('Incorrect held-out examples:', len(wrong))"""),
    ("m", """## Step 5 — Test a distribution change
Shift shapes to the right while retaining labels. Some shapes now touch or cross the crop boundary. This robustness set answers a different question from IID test accuracy. Report it separately."""),
    ("c", r"""shifted_images, shifted_labels = make_dataset(40, 1604, offset_x=14)
shifted_prediction = best_model.predict(extract_hog(shifted_images))[1].ravel().astype(int)
shifted_accuracy = np.mean(shifted_prediction == shifted_labels)
print(f'Original test: {test_accuracy:.1%}; shifted/cropped diagnostic: {shifted_accuracy:.1%}')
show_images(list(shifted_images[:8]),
            [f'{class_names[t]} -> {class_names[p]}' for t, p in zip(shifted_labels[:8], shifted_prediction[:8])], cols=4)"""),
    ("m", """Possible improvements include detecting and centering the shape or augmenting training positions. Choose changes with a new validation protocol, then evaluate fresh test seeds. Do not improve a reported score by editing its test set."""),
    ("m", """## Step 6 — Save a reproducible predictor
Record label order, dimensions, grayscale policy, HOG geometry, and split seeds alongside the model. A reloaded predictor should reproduce its predictions."""),
    ("c", r"""import json
ml_dir = ROOT / 'outputs' / '16_hog_svm'; ml_dir.mkdir(parents=True, exist_ok=True)
model_path = ml_dir / 'shape_svm.xml'
best_model.save(str(model_path))
metadata = dict(classes=class_names, image_size=[64, 64], input='uint8 grayscale',
                hog=dict(block=[16, 16], stride=[8, 8], cell=[8, 8], bins=9),
                C=c_values[best_index], train_seed=1601, validation_seed=1602, test_seed=1603,
                description='Synthetic shapes; not a general object classifier')
(ml_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
loaded = cv2.ml.SVM_load(str(model_path))
np.testing.assert_array_equal(loaded.predict(X_test)[1].ravel().astype(int), test_predictions)
print('Reloaded predictions match. Saved to:', ml_dir)"""),
    ("m", """### Practice
1. Add a cross as a fourth class, updating generation, label names, and matrix dimensions.
2. Explain how a majority predictor can achieve high accuracy while ignoring minority classes.
3. Fit a nearest-class-centroid baseline using training HOG features and compare validation accuracy.

If adding scaling or PCA, fit it only on training features, then apply the saved transform to other splits."""),
    ("c", r"""# Experiment workspace: compute class prototypes from training data only.
centroids = np.stack([X_train[y_train == label].mean(axis=0) for label in range(3)])
print('Centroid matrix:', centroids.shape)"""),
    ("m", """### Worked solutions
Nearest-centroid classification chooses the closest class mean. It learns three prototypes and has a different bias from the SVM. The comparison below remains on validation data."""),
    ("c", r"""distances = ((X_val[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
centroid_predictions = distances.argmin(axis=1)
print('Nearest-centroid validation accuracy:', np.mean(centroid_predictions == y_val))
imbalanced = np.array([0]*90 + [1]*5 + [2]*5)
print('Imbalanced majority accuracy:', np.mean(np.zeros_like(imbalanced) == imbalanced))
print('Minority recall is zero; balanced accuracy is (1+0+0)/3 =', 1/3)"""),
    ("m", """### References
- [OpenCV HOGDescriptor](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html)
- [OpenCV SVM introduction](https://docs.opencv.org/4.x/d1/d73/tutorial_introduction_to_svm.html)
- [OpenCV SVM API](https://docs.opencv.org/4.x/d1/d2d/classcv_1_1ml_1_1SVM.html)

Next: distinguish training a classical model from running a neural network with OpenCV DNN.""")])

    build_dnn()


def build_dnn():
    lesson("17_deep_learning_with_opencv_dnn.ipynb", "17 · Deep learning inference with OpenCV DNN",
           ["Array shapes and channels", "Train/validation/test separation", "Resizing"],
           ["Inspect NCHW preprocessing", "Run an actual offline DNN forward pass", "Understand output contracts and NMS"], [
    ("m", """## Step 1 — Distinguish training and inference
OpenCV's DNN module loads and executes neural-network graphs. It is primarily an inference interface, not a substitute for a training framework. A graph without trained weights cannot acquire semantic recognition.

This notebook runs a real forward pass through a hand-built **ReLU → global average pooling** network. It has no learned weights and is **not an image classifier**. Its NumPy equivalent lets us verify preprocessing and inference exactly, offline. An optional branch later shows where a real ONNX classifier fits."""),
    ("c", r"""dnn_image = make_scene(width=320, height=240)
show_images([dnn_image], ['Input in OpenCV BGR order'])
print('Input shape / dtype:', dnn_image.shape, dnn_image.dtype)
dnn_dir = ROOT / 'outputs' / '17_dnn'; dnn_dir.mkdir(parents=True, exist_ok=True)"""),
    ("m", """## Step 2 — Inspect the preprocessing contract
blobFromImage can resize, subtract a mean, scale, swap B/R, and return NCHW: batch, channels, height, width. These choices must match training.

Our toy contract uses 64×64 RGB and (pixel−127.5)/127.5, approximately [-1,1]. Equal means avoid mean-order ambiguity here; for unequal means, RGB/BGR ordering matters. Stretch resize changes aspect ratio and is not universally appropriate."""),
    ("c", r"""input_size = (64, 64)  # width, height
blob = cv2.dnn.blobFromImage(dnn_image, scalefactor=1/127.5, size=input_size,
                             mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
manual_rgb = cv2.cvtColor(cv2.resize(dnn_image, input_size), cv2.COLOR_BGR2RGB)
manual_blob = ((manual_rgb.astype(np.float32)-127.5)/127.5).transpose(2, 0, 1)[None]
assert blob.shape == (1, 3, 64, 64) and blob.dtype == np.float32
np.testing.assert_allclose(blob, manual_blob, atol=2e-7)
print('Blob shape/range:', blob.shape, float(blob.min()), float(blob.max()))
show_images([blob[0, c] for c in range(3)], ['R normalized', 'G normalized', 'B normalized'])"""),
    ("c", r"""red_bgr = np.zeros((8, 8, 3), np.uint8); red_bgr[..., 2] = 255
red_rgb_blob = cv2.dnn.blobFromImage(red_bgr, 1/255, (8, 8), swapRB=True)
red_bgr_blob = cv2.dnn.blobFromImage(red_bgr, 1/255, (8, 8), swapRB=False)
print('Correct RGB channel means:', red_rgb_blob.mean(axis=(0, 2, 3)))
print('Wrong BGR channel means:  ', red_bgr_blob.mean(axis=(0, 2, 3)))
np.testing.assert_allclose(red_rgb_blob[0, :, 0, 0], [1, 0, 0])"""),
    ("m", """## Step 3 — Build and execute a genuine graph
ReLU computes max(x,0); global average pooling averages each channel over space. The three outputs are **positive-channel averages**, not class scores or probabilities.

The normal path adds layers directly. A text-only Caffe graph provides the same operations for bindings without layer creation. No weights or network downloads are required."""),
    ("c", r'''def make_toy_network():
    network = cv2.dnn.Net()
    if hasattr(network, 'addLayer'):
        network.addLayer('relu', 'ReLU', cv2.CV_32F, {})
        network.connect('', 'relu')
        network.addLayer('positive_channel_mean', 'Pooling', cv2.CV_32F,
                         {'pool': 'AVE', 'global_pooling': True})
        network.connect('relu', 'positive_channel_mean')
    else:
        graph = """name: "toy_relu_average"
input: "data"
input_shape { dim: 1 dim: 3 dim: 64 dim: 64 }
layer { name: "relu" type: "ReLU" bottom: "data" top: "relu" }
layer { name: "positive_channel_mean" type: "Pooling" bottom: "relu" top: "means"
        pooling_param { pool: AVE global_pooling: true } }
"""
        graph_path = dnn_dir / 'toy_network.prototxt'
        graph_path.write_text(graph, encoding='utf-8')
        network = cv2.dnn.readNetFromCaffe(str(graph_path))
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return network

net = make_toy_network()
net.setInput(blob)
output = net.forward()
expected_output = np.maximum(blob, 0).mean(axis=(2, 3), keepdims=True)
print('Layers:', net.getLayerNames())
print('Output shape:', output.shape)
print('DNN / NumPy:', output.ravel(), expected_output.ravel())
np.testing.assert_allclose(output.reshape(1, 3), expected_output.reshape(1, 3), atol=2e-5)
print('Actual DNN inference matches an independent NumPy reference.')'''),
    ("c", r"""batch = np.concatenate([blob, np.zeros_like(blob), np.ones_like(blob)], axis=0)
net.setInput(batch)
batch_output = net.forward().reshape(3, 3)
np.testing.assert_allclose(batch_output, np.maximum(batch, 0).mean(axis=(2, 3)), atol=2e-5)
print('Batch rows = scene, zeros, ones:\n', batch_output)
plt.bar(['R', 'G', 'B'], output.ravel(), color=['red', 'green', 'blue'])
plt.ylabel('Average positive normalized activation'); plt.show()"""),
    ("m", """Batches can improve throughput while increasing memory and latency. A particular graph may require fixed shapes. This small graph accepts different batch sizes; do not assume the same for a downloaded model."""),
    ("c", r"""from time import perf_counter
net.setInput(blob); net.forward()  # warm up
timings_ms = []
for _ in range(15):
    start = perf_counter()
    net.setInput(blob); net.forward()
    timings_ms.append((perf_counter()-start)*1000)
print('Toy CPU latency median / 90th percentile (ms):', np.percentile(timings_ms, [50, 90]))
print('Timing includes setInput + forward, excludes capture, preprocessing, and plotting.')"""),
    ("m", """## Step 4 — Interpret outputs only with their model contract
A single-label classifier may output logits. Softmax normalizes them with exp(zᵢ)/sum(exp(zⱼ)); subtracting the maximum prevents overflow. Normalized values are not automatically calibrated confidence.

Do not softmax our channel means and call them predictions. Do not apply softmax again to probabilities. Multilabel networks often use independent sigmoids instead."""),
    ("c", r"""def stable_softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=np.float64)
    centered = logits-logits.max(axis=axis, keepdims=True)
    exponentials = np.exp(centered)
    return exponentials/exponentials.sum(axis=axis, keepdims=True)

illustrative_logits = np.array([[1001., 1002., 999.]])  # invented arithmetic, not model predictions
illustrative_probabilities = stable_softmax(illustrative_logits)
print('Arithmetic-only example:', illustrative_probabilities)
np.testing.assert_allclose(illustrative_probabilities.sum(axis=1), 1)
np.testing.assert_allclose(stable_softmax(illustrative_logits+5000), illustrative_probabilities)"""),
    ("m", """Detectors have different contracts: boxes, scores, classes, sometimes masks. Coordinates may be normalized, xyxy, centre-width-height, or xywh. Decode the specific architecture first. The following boxes are explicitly invented to practice postprocessing."""),
    ("m", """## Step 5 — Non-maximum suppression
NMS retains high-score boxes and suppresses overlapping lower-score boxes. IoU is intersection area divided by union area. OpenCV NMSBoxes expects [x,y,width,height]. Apply NMS per class to avoid suppressing distinct categories."""),
    ("c", r"""boxes = [[30, 35, 100, 80], [38, 40, 100, 80], [195, 60, 65, 90], [40, 150, 70, 40]]
scores = [0.95, 0.85, 0.90, 0.20]
classes = [0, 0, 1, 0]
def class_aware_nms(boxes, scores, classes, score_threshold=0.4, iou_threshold=0.4):
    kept = []
    for class_id in sorted(set(classes)):
        members = [i for i, label in enumerate(classes) if label == class_id]
        local = cv2.dnn.NMSBoxes([boxes[i] for i in members], [scores[i] for i in members],
                                 score_threshold, iou_threshold)
        kept.extend(members[int(i)] for i in np.asarray(local).reshape(-1))
    return sorted(kept)

kept = class_aware_nms(boxes, scores, classes)
assert kept == [0, 2]
canvas = np.full((240, 320, 3), 245, np.uint8)
for i, (x, y, bw, bh) in enumerate(boxes):
    color = (0, 140, 0) if i in kept else (0, 0, 220)
    cv2.rectangle(canvas, (x, y), (x+bw, y+bh), color, 2)
    label_y = y+18+(18 if i == 1 else 0)  # separate overlapping boxes' labels
    cv2.putText(canvas, f'{i}: {scores[i]:.2f}', (x+4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
show_images([canvas], ['Invented boxes: green kept, red removed'])"""),
    ("c", r"""def box_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    intersection = max(0, min(ax+aw, bx+bw)-max(ax, bx))*max(0, min(ay+ah, by+bh)-max(ay, by))
    union = aw*ah+bw*bh-intersection
    return intersection/union if union > 0 else 0.0

print('First two boxes IoU:', box_iou(boxes[0], boxes[1]))
for threshold in [0.1, 0.4, 0.8]:
    print('IoU threshold', threshold, 'keeps', class_aware_nms(boxes, scores, classes, iou_threshold=threshold))"""),
    ("m", """## Step 6 — Optional real ONNX classification
The default notebook completes offline. To enable this branch, supply a trusted classification ONNX model, license/model card, ordered labels, shape, channel order, normalization, resize policy, and output description.

This sample supports only one RGB/BGR NCHW input, stretch resize, and a [1,number_of_classes] logits/probability output. Fill every contract field from the model documentation; incompatible architectures need a different decoder."""),
    ("c", r"""RUN_ONNX_CLASSIFIER = False
MODEL_PATH = ROOT / 'models' / 'personal' / 'your_classifier.onnx'
LABELS_PATH = ROOT / 'models' / 'personal' / 'your_labels.txt'
MODEL_SIZE = None    # verified (width, height)
MODEL_RGB = None     # True for RGB, False for BGR
MODEL_MEAN = None    # three channel means in 0..1 units
MODEL_STD = None     # three positive channel standard deviations
MODEL_OUTPUT = None # 'logits' or 'probabilities'
MODEL_RESIZE = None # sample supports only 'stretch'

if RUN_ONNX_CLASSIFIER:
    assert MODEL_PATH.is_file() and LABELS_PATH.is_file(), 'Provide model and labels.'
    assert MODEL_SIZE is not None and MODEL_RGB is not None
    assert MODEL_MEAN is not None and MODEL_STD is not None
    assert MODEL_OUTPUT in ('logits', 'probabilities') and MODEL_RESIZE == 'stretch'
    mean = np.asarray(MODEL_MEAN, np.float32).reshape(1, 1, 3)
    std = np.asarray(MODEL_STD, np.float32).reshape(1, 1, 3)
    assert np.all(std > 0)
    resized = cv2.resize(dnn_image, tuple(MODEL_SIZE))
    if MODEL_RGB:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (resized.astype(np.float32)/255.0-mean)/std
    model_blob = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
    labels = LABELS_PATH.read_text(encoding='utf-8').splitlines()
    classifier = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    classifier.setInput(model_blob)
    raw_scores = classifier.forward()
    assert raw_scores.shape == (1, len(labels)), 'Output decoder mismatch.'
    assert np.isfinite(raw_scores).all()
    probabilities = stable_softmax(raw_scores)[0] if MODEL_OUTPUT == 'logits' else raw_scores[0]
    assert (probabilities >= 0).all() and np.isclose(probabilities.sum(), 1, atol=1e-3)
    for index in np.argsort(probabilities)[-5:][::-1]:
        print(labels[index], float(probabilities[index]))
else:
    print('Optional ONNX classifier disabled. Actual toy DNN inference already ran above.')"""),
    ("m", """### Practice
1. Predict the toy output for an all-negative blob, then execute.
2. Move a fixed-size colored patch across an input. Explain global pooling's response.
3. Put the second overlapping box in another class. Should it survive?
4. List the preprocessing assumptions you must verify before integrating a learned model."""),
    ("c", r"""negative_input = -np.ones((1, 3, 64, 64), dtype=np.float32)
net.setInput(negative_input)
negative_output = net.forward()
print('All-negative input output:', negative_output.ravel())"""),
    ("m", """### Worked solutions
ReLU removes negative activations. Global pooling discards spatial position, so different spatial arrangements can yield identical outputs. This is another reason the untrained toy graph cannot recognize semantic content."""),
    ("c", r"""np.testing.assert_allclose(negative_output, 0, atol=1e-7)
patch_a = np.zeros((1, 3, 64, 64), np.float32); patch_a[0, 0, 5:15, 5:15] = 1
patch_b = np.zeros_like(patch_a); patch_b[0, 0, 35:45, 40:50] = 1
net.setInput(patch_a); result_a = net.forward().copy()
net.setInput(patch_b); result_b = net.forward().copy()
np.testing.assert_allclose(result_a, result_b, atol=1e-7)
print('Translated patch outputs:', result_a.ravel())
assert class_aware_nms(boxes, scores, [0, 1, 1, 0]) == [0, 1, 2]
print('Different classes preserve both overlapping boxes.')"""),
    ("m", """### References
- [OpenCV DNN API](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)
- [OpenCV Net graph/inference API](https://docs.opencv.org/4.x/db/d30/classcv_1_1dnn_1_1Net.html)
- [OpenCV classification sample](https://github.com/opencv/opencv/blob/4.x/samples/dnn/classification.py)

For learned recognition, verify preprocessing against the training framework and compare outputs on known images before evaluating a held-out dataset. Never substitute invented detections for a failed model run.""")])


if __name__ == '__main__':
    build()
