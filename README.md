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
- `VIDEO_PATH`: Path to the input video.
- `CAMERA_CALIBRATION_FACTOR`: Meters per pixel (calibrate according to the camera).
- `LANE_POLYGONS`: Coordinates of the virtual lane polygons.
- `SPEED_THRESHOLD`: Limit for detecting speeding (km/h).

## ▶️ Execution

To run the complete analysis pipeline:

```bash
python src/main.py
```

## 📊 Results

Results will be saved to:
- `results/output_video.mp4`: Processed video with visualizations.
- `results/tracking_data.json`: Structured trajectory data.
- `results/anomalies.csv`: Record of detected anomalies.

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
