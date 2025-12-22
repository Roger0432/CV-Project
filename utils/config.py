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
LANE_POLYGONS = {
    1: np.array([[522, 532], [528, 28], [541, 27], [750, 531], [523, 532]]),
    2: np.array([[761, 531], [543, 27], [558, 26], [957, 487], [956, 532], [763, 535]]),
    3: np.array([[291, 537], [508, 532], [525, 27], [516, 28], [292, 536]]),
}

LANE_NAMES = {
    1: "Lane 1",
    2: "Lane 2",
    3: "Lane 3",
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
HOMOGRAPHY_MATRIX = np.array([
    [0.06864633854339794, 0.015986133633394056, -31.925484316890362],
    [-0.0016044642291373936, 0.10001160361623052, -38.404054672575235],
    [0.0003325954439259714, 0.004053316242000917, 0.9999999999999999],
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
