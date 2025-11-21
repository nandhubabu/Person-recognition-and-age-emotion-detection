# Face detection and age/emotion scan

Small demo that runs YOLOv8 for face detection and uses DeepFace to analyze age and emotion from camera frames.

Project layout
- `camera_test.py` - main demo script. Reads from webcam, detects faces with Ultralytics YOLO model, crops faces, runs DeepFace analysis (age, emotion), and overlays results on the frame.
- `yolov8n.pt` - YOLOv8 model file (already included in repository root).
- `camenv/` - (optional) bundled virtual environment used by the author. You can use it or create your own venv.

Quick start
1. (Recommended) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Or use the included virtualenv (Linux):

```bash
source ./camenv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the model file `yolov8n.pt` is in the project root (it already is).

4. Run the demo:

```bash
python camera_test.py
```

Press `q` to quit the camera window.

Notes and caveats
- This demo sets `CUDA_VISIBLE_DEVICES=-1` in the script to force CPU-only for DeepFace. If you want GPU acceleration, remove or modify that line and ensure compatible CUDA, PyTorch and/or TensorFlow installations.
- Real-time face analysis can be CPU-heavy. Prefer running on a machine with a GPU or accept lower FPS.
- DeepFace may internally require TensorFlow (or other backends) depending on versions. If you want a reproducible environment, use the included `camenv` or create a venv and pin exact library versions.
