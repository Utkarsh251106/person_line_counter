# Person Segmentation + Directional Line Counter

A production-oriented baseline for:

1. Reading a webcam/video stream.
2. Segmenting people with an Ultralytics YOLO segmentation model.
3. Using the segmentation mask itself to calculate a person centroid.
4. Tracking people across frames without using bounding boxes in the application logic.
5. Detecting crossings of a vertical line.
6. Incrementing for left -> right and decrementing for right -> left.
7. Drawing masks, track IDs, trajectory points, line, and counters.
8. Saving the processed video and CSV events.

## Architecture

Camera / Video
    |
    v
YOLO Instance Segmentation
    |
    +--> person masks
    |
    v
Mask centroid
    |
    v
Centroid-based multi-object tracker
    |
    v
Per-track state machine
    |
    +--> LEFT -> RIGHT : counter += 1
    |
    +--> RIGHT -> LEFT : counter -= 1
    |
    v
OpenCV visualization + CSV event log

Important: YOLO's segmentation implementation may internally use box information, but this project does not read or use `result.boxes` for tracking or counting. The application logic operates on masks, centroids and track history.

## 1. Environment

Recommended:
- Python 3.10 or 3.11
- NVIDIA GPU + CUDA-enabled PyTorch for real-time performance when available
- CPU works for testing but may be slower

Windows:

    py -3.11 -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Linux/macOS:

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Ultralytics installs the YOLO package. The first run downloads the pretrained segmentation checkpoint automatically.

## 2. Put your input video in data/

Example:

    data/input.mp4

Or use a webcam:

    python -m src.main --source 0

## 3. Run

Video file:

    python -m src.main --source data/input.mp4

Webcam:

    python -m src.main --source 0

Custom line position:

    python -m src.main --source data/input.mp4 --line-x 640

Use a fraction of the frame width instead:

    python -m src.main --source data/input.mp4 --line-ratio 0.50

Useful tuning:

    python -m src.main --source data/input.mp4 \
        --conf 0.45 \
        --mask-threshold 0.50 \
        --max-distance 100 \
        --max-missed 20 \
        --buffer-half-width 25

Press `q` to stop.

## 4. Outputs

The application writes:

    outputs/processed.mp4
    outputs/events.csv

The CSV contains:
- timestamp
- frame number
- track_id
- direction
- signed counter after event

## 5. How the counting logic works

For every person mask:

    mask -> centroid -> track_id -> trajectory -> line state

A vertical line at x = LINE_X divides the image into left and right regions.

A small buffer is used around the line:

    x < LINE_X - buffer  => LEFT
    x > LINE_X + buffer  => RIGHT
    otherwise             => BUFFER

Only a confirmed transition is counted:

    LEFT -> BUFFER -> RIGHT = +1
    RIGHT -> BUFFER -> LEFT = -1

This prevents a person jittering around the line from generating many false counts.

## 6. Why no bounding boxes?

The application never reads `result.boxes`.

The segmentation mask is converted into a centroid using image moments. The centroid is the point used for tracking and line crossing.

This is a valid design when:
- the segmentation model gives reliable instance masks,
- people are reasonably separated,
- the camera view is stable,
- a centroid is sufficient to represent the person's movement.

For very crowded scenes, centroid-only tracking can become fragile. In that case, use a stronger tracker and/or mask-IoU matching.

## 7. Replace the pretrained model

The default model is:

    yolo26n-seg.pt

For your own trained segmentation model:

    python -m src.main --source data/input.mp4 --model models/best.pt

Your model should contain a `person` class. If person is not class 0, pass:

    --person-class-id <id>

## 8. Production improvements

Before production deployment:
- benchmark FPS and latency on the target hardware
- validate segmentation precision/recall on your actual camera
- test occlusion and crowded scenes
- tune confidence, mask threshold and tracking distance
- add camera reconnect handling
- expose health/metrics
- containerize the application
- add structured logs
- version the model
- add automated tests
- pin dependency versions after validation
- consider ONNX/TensorRT export for edge deployment

## 9. Docker

Build:

    docker build -t person-line-counter .

Run:

    docker run --rm -it person-line-counter

For GPU deployment, use an NVIDIA CUDA base image and the NVIDIA Container Toolkit rather than the simple CPU Dockerfile included here.

## 10. Important implementation note

This is a strong baseline, not a claim that centroid-only tracking is the best tracker for every production camera. The correct tracker depends on camera angle, crowd density, FPS, occlusion, lighting and target hardware.
