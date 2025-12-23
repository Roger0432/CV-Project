# Intersection Traffic Analysis System

This project implements a complete computer vision system for traffic analysis at intersections using state-of-the-art models.

##  Architecture

- **Detection**: YOLOv8 (detects vehicles and pedestrians).
- **Tracking**: ByteTrack (associates detections temporally).
- **Lane Assignment**: Virtual lanes defined by polygons and geometry-based assignment.
- **Anomaly Detection**:
    - Speeding.
    - Pedestrians on the road.
    - Forbidden zones.
    - Wrong-way driving.

##  Installation

1.  Clone the repository:
    ```console
    git clone https://github.com/Roger0432/CV-Project.git
    cd CV-Project
    ```

2.  Create necessary directories (not included in the repository):
    ```console
    mkdir data
    mkdir results
    ```

3.  Create a virtual environment (optional but recommended):
    ```console
    python -m venv venv
    ```

    **Windows:**
    ```powershell
    .\venv\Scripts\activate
    ```

    **Linux/Mac:**
    ```bash
    source venv/bin/activate
    ```

4.  Install dependencies:
    ```console
    pip install -r requirements.txt
    ```

5.  Download a video from the UA-DETRAC dataset (or use your own) and save it to the `data/` folder.

##  Configuration

You can adjust the system parameters in `utils/config.py`:
- **Paths**: Set `VIDEO_PATH` for input and `GROUND_TRUTH_PATH` for XML annotations (UA-DETRAC format).
- **Calibration**: 
    - `HOMOGRAPHY_MATRIX`: 3x3 matrix for accurate Pixel -> World mapping (Recommended).
    - `CAMERA_CALIBRATION_FACTOR`: Simple meters/pixel scale (Fallback).
- **Lanes**: `LANE_POLYGONS` defines the geometry of the intersection lanes.
- **Anomalies**: 
    - `SPEED_THRESHOLD`: Absolute limit (default 50 km/h).
    - `FORBIDDEN_ZONES`: Polygons for restricted areas.

## ▶️ Execution

Follow these steps to run the complete workflow:

### 1. Prepare Video
Convert the sequence of images from the dataset into a video file:
```bash
python tools/convert_images_to_video.py ./data/ua-detrac-orig/DETRAC-Images/DETRAC-Images/MVI_20061 ./data/input_video.mp4 --fps 25
```

### 2. Define Lanes
Draw polygons on the video to define the traffic lanes manually (this helps update coordinates for `config.py`):
```bash
python tools/draw_lanes.py --video data/input_video.mp4
```

### 3. Define Forbidden Zones
Interactively draw polygons for forbidden areas (e.g., sidewalks, islands) and get the code snippet:
```bash
python tools/draw_zones.py --video data/input_video.mp4
```
Copy the output into `utils/config.py`.

### 4. Calibrate Camera
Compute the pixel-to-meter scale for accurate speed estimation:
```bash
python tools/calibrate_camera.py --video data/input_video.mp4
```

### 5. Run Traffic Analysis
Execute the main system to perform detection, tracking, lane assignment, and anomaly detection:
```bash
python src/main.py
```
This script performs:
- **Stabilization**: Fixes small camera movements.
- **Detection & Tracking**: Identifies and follows vehicles.
- **Lane Assignment**: Maps vehicles to the defined lanes.
- **Anomaly Detection**: Flags speeding, wrong-way driving, and forbidden zone entries.
- **Evaluation**: Compares results with Ground Truth (if available).

##  Results

Results will be saved to the `results/` folder:
- **`output_video.mp4`**: Processed video with visualizations.
- **`tracking_results.json`**: Full history of tracked objects and their positions.
- **`anomaly_detection.csv`**: List of all detected anomalies with timestamps and values.
- **`lane_accuracy.csv`**: Evaluation metrics for lane assignment (if GT is available).

##  Directory Structure

```
CV-Project/
├── data/           # Input videos
├── results/        # Generated outputs
├── src/            # Source code modules
├── utils/          # Utilities and configuration
├── requirements.txt
└── README.md
```
