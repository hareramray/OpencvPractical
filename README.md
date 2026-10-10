# OpenCV Practical — learn by doing

21 Jupyter notebooks take you from basic Python arrays to advanced computer vision projects. Each lesson has small runnable steps, visual results, parameter experiments, exercises, worked solutions, and common mistakes to investigate. Core lessons generate their own data and work offline after installation; no camera, GPU, paid service, or external dataset is required.

This is a path toward expert practice, not a promise of expertise from completing a checklist. The final lessons emphasize assumptions, measurement, failure analysis, and applying a working prototype to new data.

## Start on this Windows computer

The project uses its own `.venv`; your system Python packages are separate.

From PowerShell in this folder:

```powershell
.\setup.ps1
.\start-learning.ps1
```

Setup is already performed on the computer where this course was created. On first use elsewhere, setup needs internet access to install the packages. It expects Python 3.13 available through `py -3.13`. If you use a different supported Python installation, create `.venv` with that interpreter first.

If PowerShell blocks scripts, the equivalent commands work without changing execution policy:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name opencv-practical --display-name "Python (OpenCV Practical)"
.\.venv\Scripts\python.exe -m jupyterlab notebooks/00_start_here.ipynb
```

Select **Python (OpenCV Practical)** as the notebook kernel. Use **Shift+Enter** to run one cell. Work through cells from top to bottom; **Kernel → Restart Kernel and Run All Cells** catches hidden dependencies. In VS Code, open this folder, install its Jupyter extension if needed, and select `.venv\Scripts\python.exe` as the kernel interpreter.

On macOS/Linux, use `python3 -m venv .venv`, replace `.venv\Scripts\python.exe` with `.venv/bin/python`, and run the same module commands. Windows/Python 3.13 is the environment used for this course's execution checks.

## Learning order

Allow roughly 1–2 hours for an early lesson and 2–4 hours for an advanced lesson, including exercises. Spend more time reproducing results and testing your own data than reading solutions.

| Notebook | Level | Practical outcome |
|---|---|---|
| [00 Start here](notebooks/00_start_here.ipynb) | Orientation | Check setup, run your first pipeline, and plan your study |
| [01 NumPy and images](notebooks/01_numpy_and_images.ipynb) | Beginner | Understand arrays, pixels, coordinates, and data types |
| [02 Read, write, and display](notebooks/02_read_write_and_display.ipynb) | Beginner | Load, inspect, resize, and save images safely |
| [03 Pixels, drawing, and masks](notebooks/03_pixels_drawing_and_masks.ipynb) | Beginner | Annotate images and process selected regions |
| [04 Color and histograms](notebooks/04_color_spaces_and_histograms.ipynb) | Beginner | Segment colors and inspect contrast |
| [05 Filtering and noise](notebooks/05_filtering_and_noise.ipynb) | Beginner | Compare blur, denoising, and sharpening |
| [06 Thresholding and morphology](notebooks/06_thresholding_and_morphology.ipynb) | Beginner | Build and clean foreground masks |
| [07 Edges and Hough](notebooks/07_gradients_edges_and_hough.ipynb) | Intermediate | Detect boundaries and straight lines |
| [08 Contours and shapes](notebooks/08_contours_and_shape_analysis.ipynb) | Intermediate | Count, describe, and measure objects |
| [09 Geometry](notebooks/09_geometric_transforms.ipynb) | Intermediate | Resize, rotate, and rectify perspective |
| [10 Features and registration](notebooks/10_features_matching_and_registration.ipynb) | Intermediate | Match keypoints and align images with RANSAC |
| [11 Segmentation](notebooks/11_segmentation_watershed_grabcut.ipynb) | Intermediate | Separate regions using watershed and GrabCut |
| [12 Video](notebooks/12_video_and_background_subtraction.ipynb) | Intermediate | Write/read video and detect moving foreground |
| [13 Optical flow and tracking](notebooks/13_optical_flow_and_tracking.ipynb) | Advanced | Estimate motion and track features over time |
| [14 Calibration and pose](notebooks/14_camera_calibration_and_pose.ipynb) | Advanced | Estimate camera parameters and object pose |
| [15 Stereo and depth](notebooks/15_stereo_depth_and_3d.ipynb) | Advanced | Turn disparity into depth with validity checks |
| [16 HOG and machine learning](notebooks/16_hog_and_machine_learning.ipynb) | Advanced | Train and evaluate a classical image classifier |
| [17 OpenCV DNN](notebooks/17_deep_learning_with_opencv_dnn.ipynb) | Advanced | Prepare tensors, run an offline toy network, and study model integration |
| [18 Document scanner](notebooks/18_document_scanner_project.ipynb) | Project | Detect a page, rectify it, and improve readability |
| [19 Visual inspection](notebooks/19_visual_inspection_project.ipynb) | Project | Build and evaluate a defect-detection pipeline |
| [20 Engineering practice](notebooks/20_performance_and_reproducible_pipelines.ipynb) | Advanced practice | Profile, package, validate, and save a reproducible pipeline |

## How to study

1. Predict what a cell will show, then run it and explain the result in your own words.
2. Change one parameter, record the effect, and restore the baseline before changing another.
3. Attempt the exercises before scrolling to the worked solutions. Exercises are intentionally non-blocking: default Run All still works.
4. Rebuild the mini-project in a fresh notebook without copying, then compare with the reference.
5. Replace generated inputs with your own images and record failure cases. Good results on synthetic data do not establish real-world accuracy.

Suggested eight-week rhythm: orientation/01–03; 04–06; 07–09; 10–12; 13–14; 15–17; 18–19; 20 plus an independent project. Adjust the pace to your background.

## Files and data

- `notebooks/`: the lessons; each can start in a fresh kernel.
- `course_utils.py`: small display, sample-scene, and image-file helpers. Vision algorithms remain visible in notebook cells.
- `data/personal/`: an optional place for your photos. Optional camera/model sections are disabled by default.
- `models/personal/`: optional model files you choose to use.
- `outputs/`: generated images, videos, metrics, executed copies, and validation reports. Lessons may replace their own generated outputs when rerun.
- `tools/build_*.py`: course source generators. Rebuilding replaces the reference notebooks, so keep your experiments in separate copies.
- `requirements.txt`: supported dependency ranges; `requirements-tested.txt`: exact versions from the verified environment.

No third-party photos or pretrained weights are bundled. The DNN lesson's small offline network demonstrates real computation, not a pretrained object recognizer. Synthetic calibration and stereo examples establish concepts; real cameras need appropriate observations, calibration, and evaluation.

## Troubleshooting

- **`python` opens the Microsoft Store:** use `py -3.13` or the explicit `.venv\Scripts\python.exe` path above.
- **`No module named cv2`:** select the project kernel; do not install into an unrelated Python interpreter.
- **Import errors after package changes:** restart the kernel. Only one OpenCV package should be installed in an environment; this course uses `opencv-python-headless`.
- **Colors look wrong:** OpenCV images are usually BGR; Matplotlib expects RGB. The helper converts for display.
- **No desktop image window:** expected. Results render inside Jupyter; lessons do not use `cv2.imshow`.
- **A file cannot be read:** inspect its resolved path, confirm it exists, and check the notebook's input settings.
- **Video codec unavailable:** notebook 12 reports the attempted codecs and checks whether reading/writing succeeded.

All 21 notebooks passed fresh-kernel execution on Python 3.13 with OpenCV 4.14.0,
with example outputs saved in the lesson files. The combined results are in
`outputs/validation_report.json`. Optional webcam and user-supplied photo/model
branches were disabled during these checks.

To independently rerun all checks (each notebook gets a fresh kernel):

```powershell
.\.venv\Scripts\python.exe -m tools.validate_notebooks
```

This saves executed copies in `outputs/executed/` and a JSON report. Add `--pattern '14_*.ipynb'` for one lesson. `--in-place` stores successful output in the source notebook; use that option only on copies or unmodified reference notebooks.

## Further reading

Lessons link to relevant sections of the [official OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html). The [OpenCV DNN reference](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html) documents inference APIs. Packaging guidance is available on the [OpenCV headless package page](https://pypi.org/project/opencv-python-headless/).
