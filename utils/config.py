import numpy as np
import os

# --- PATHS ---
DATA_DIR = "data"
RESULTS_DIR = "results"
VIDEO_FILENAME = "input_video.mp4" # Default filename, can be overridden
VIDEO_PATH = os.path.join(DATA_DIR, VIDEO_FILENAME)
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "output_video_2.mp4")
TRACKING_RESULTS_PATH = os.path.join(RESULTS_DIR, "tracking_results.json")
ANOMALY_RESULTS_PATH = os.path.join(RESULTS_DIR, "anomaly_detection.csv")
LANE_ACCURACY_PATH = os.path.join(RESULTS_DIR, "lane_accuracy.csv")

# Path to the UA-DETRAC XML Ground Truth for the current video
# Note: Adjust path if folder structure differs
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ua-detrac-orig", "DETRAC-Train-Annotations-XML", "DETRAC-Train-Annotations-XML", "MVI_40171.xml")
# --- CAMERA & REAL WORLD ---
# Factor to convert pixel distance to meters.
# This must be calibrated for the specific camera view.
# Example: 0.05 meters per pixel.
CAMERA_CALIBRATION_FACTOR = 0.05 
FPS = 25 # Default assumption, usually read from video

# --- DETECTION (YOLO) ---
MODEL_WEIGHTS = "yolov8n.pt" # Nano model for speed
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
TARGET_CLASSES = [2, 3, 5, 7] # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck (and maybe 0=person)
PEDESTRIAN_CLASS_ID = 0

# --- TRACKING (ByteTrack) ---
TRACKER_THRESH = 0.25 # high_thresh
TRACKER_MATCH_THRESH = 0.8
TRACK_BUFFER = 30 # Number of frames to keep lost tracks

# --- LANE ASSIGNMENT ---
# Virtual lanes defined as polygons (List of [x, y] points).
# coordinates must be adjusted to the video resolution (e.g., 1920x1080).
# Example polygons (dummy values, user needs to update):
LANE_POLYGONS = {
    1: np.array([[315, 539], [537, 539], [539, 20], [530, 20], [315, 536]]),
    2: np.array([[543, 537], [771, 537], [555, 23], [540, 23], [543, 537]]),
    3: np.array([[779, 537], [957, 537], [959, 484], [574, 22], [558, 24], [779, 537]]),
    4: np.array([[957, 420], [578, 21], [585, 23], [958, 288], [957, 421]]),
    5: np.array([[958, 287], [959, 224], [597, 21], [587, 22], [958, 287]]),
    6: np.array([[957, 219], [957, 174], [616, 18], [603, 21], [958, 224]]),
}

LANE_NAMES = {
    1: "Lane 1",
    2: "Lane 2",
    3: "Lane 3",
    4: "Lane 4",
    5: "Lane 5",
    6: "Lane 6",
}

# --- ANOMALY DETECTION ---
SPEED_THRESHOLD = 50.0 # km/h
SPEED_HISTORY_WINDOW = 15 # Number of frames to average speed
TRAJECTORY_DEVIATION_SIGMA = 2.0 # Standard deviations for clustering outlier detection

# Polygons where vehicles should NOT be (e.g. sidewalks, central islands)
# Similar format to LANE_POLYGONS
FORBIDDEN_ZONES = {
    # 1: np.array([[x1, y1], [x2, y2], ...]),
}

# Homography Matrix (3x3) for Pixel -> Real World mapping
# If None, uses calibration factor
HOMOGRAPHY_MATRIX = np.array([
    [0.234230834218914, 0.08802305234978285, -125.02667542891139],
    [-0.006040920941760074, 0.3342642921107224, -100.0074461908375],
    [-0.0004620286965838227, 0.024722955400739113, 1.0],
])

# --- VISUALIZATION ---
DRAW_TRAJECTORIES = True
DRAW_LANES = True
DRAW_SPEED = True
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)
]
