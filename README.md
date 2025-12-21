# Intersection Traffic Analysis System

This project implements a complete computer vision system for traffic analysis at intersections using state-of-the-art models.

## 🏗️ Architecture

- **Detection**: YOLOv8 (detects vehicles and pedestrians).
- **Tracking**: ByteTrack (associates detections temporally).
- **Lane Assignment**: Virtual lanes defined by polygons and geometry-based assignment.
- **Anomaly Detection**:
    - Speeding.
    - Unusual trajectories (clustering).
    - Pedestrians on the road.

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Roger0432/CV-Project.git
    cd CV-Project
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/Mac
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Download a video from the UA-DETRAC dataset (or use your own) and save it to the `data/` folder.

## ⚙️ Configuration

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

To run the complete analysis pipeline:

```bash
# Ensure you are in the project root directory
python src/main.py
```

The script will:
1. Detect and track vehicles using YOLOv8 + ByteTrack.
2. Assign vehicles to lanes based on geometry.
3. Estimate speeds (using Homography or Scale).
4. Detect anomalies (Speeding, Wrong Direction, Forbidden Zones).
5. Compare with Ground Truth (if configured).

## 📊 Results

Results will be saved to the `results/` folder:
- **`output_video_2.mp4`**: Processed video with visualizations.
- **`tracking_results.json`**: Full history of tracked objects and their positions.
- **`anomaly_detection.csv`**: List of all detected anomalies with timestamps and values.
- **`lane_accuracy.csv`**: Evaluation metrics for lane assignment (if GT is available).

## 🛠️ Directory Structure

```
CV-Project/
├── data/           # Input videos
├── results/        # Generated outputs
├── src/            # Source code modules
├── utils/          # Utilities and configuration
├── requirements.txt
└── README.md
```
