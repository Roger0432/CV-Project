import numpy as np
import os

# --- PATHS ---
DATA_DIR = "data"
RESULTS_DIR = "results"
VIDEO_FILENAME = "input_video.mp4" # Default filename, can be overridden
VIDEO_PATH = os.path.join(DATA_DIR, VIDEO_FILENAME)
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "output_video.mp4")
TRACKING_RESULTS_PATH = os.path.join(RESULTS_DIR, "tracking_results.json")
ANOMALY_RESULTS_PATH = os.path.join(RESULTS_DIR, "anomaly_detection.csv")
LANE_ACCURACY_PATH = os.path.join(RESULTS_DIR, "lane_accuracy.csv")

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
    1: np.array([[100, 1000], [100, 800], [400, 800], [400, 1000]]), # Entry Lane 1
    2: np.array([[500, 1000], [500, 800], [800, 800], [800, 1000]]), # Entry Lane 2
    3: np.array([[900, 1000], [900, 800], [1200, 800], [1200, 1000]]), # Exit Lane 3
    4: np.array([[1300, 1000], [1300, 800], [1600, 800], [1600, 1000]]) # Exit Lane 4
}
LANE_NAMES = {
    1: "Entry Lane 1",
    2: "Entry Lane 2",
    3: "Exit Lane 3",
    4: "Exit Lane 4"
}

# --- ANOMALY DETECTION ---
SPEED_THRESHOLD = 50.0 # km/h
SPEED_HISTORY_WINDOW = 5 # Number of frames to average speed
TRAJECTORY_DEVIATION_SIGMA = 2.0 # Standard deviations for clustering outlier detection

# Polygons where vehicles should NOT be (e.g. sidewalks, central islands)
# Similar format to LANE_POLYGONS
FORBIDDEN_ZONES = {
    # 1: np.array([[x1, y1], [x2, y2], ...]),
}

# Homography Matrix (3x3) for Pixel -> Real World mapping
# If None, uses calibration factor
HOMOGRAPHY_MATRIX = None 
# HOMOGRAPHY_MATRIX = np.array([...])

# --- VISUALIZATION ---
DRAW_TRAJECTORIES = True
DRAW_LANES = True
DRAW_SPEED = True
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)
]
