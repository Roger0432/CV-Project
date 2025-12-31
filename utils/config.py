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
    1: np.array([[4, 537], [89, 537], [367, 191], [247, 201], [0, 389], [6, 535]]),
    2: np.array([[445, 89], [368, 191], [247, 198], [420, 74], [444, 87]]),
    3: np.array([[291, 537], [519, 30], [528, 31], [515, 535], [292, 537]]),
    4: np.array([[766, 536], [956, 537], [958, 477], [561, 24], [549, 27], [559, 54], [768, 535]]),
    5: np.array([[767, 534], [519, 534], [529, 31], [548, 30], [767, 531]]),
    6: np.array([[464, 155], [89, 162], [70, 195], [441, 193], [463, 160]]),
    7: np.array([[959, 282], [955, 214], [600, 24], [589, 28], [958, 284]]),
    8: np.array([[957, 420], [957, 291], [586, 29], [577, 28], [578, 33], [957, 420]]),
    9: np.array([[959, 134], [888, 120], [693, 48], [699, 42], [957, 116], [957, 135]]),
    10: np.array([[957, 213], [957, 165], [624, 26], [605, 25], [956, 213]]),
    11: np.array([[864, 123], [885, 118], [924, 134], [909, 147], [865, 127]]),
}

LANE_NAMES = {
    1: "Lane 1",
    2: "Lane 2",
    3: "Lane 3",
    4: "Lane 4",
    5: "Lane 5",
    6: "Lane 6",
    7: "Lane 7",
    8: "Lane 8",
    9: "Lane 9",
    10: "Lane 10",
    11: "Lane 11",
}
# --- ANOMALY DETECTION ---
SPEED_THRESHOLD = 50.0 # km/h
SPEED_HISTORY_WINDOW = 15 # Number of frames to average speed
TRAJECTORY_DEVIATION_SIGMA = 2.0 # Standard deviations for clustering outlier detection

# Polygons where vehicles should NOT be (e.g. sidewalks, central islands)
# Similar format to LANE_POLYGONS
FORBIDDEN_ZONES = {
    1: np.array([[282, 539], [438, 195], [428, 193], [243, 536], [279, 539]]),
}
# Homography Matrix (3x3) for Pixel -> Real World mapping
HOMOGRAPHY_MATRIX = np.array([
    [0.13086671649275125, 0.048030869351062995, -66.5728732192407],
    [-0.001126423188883012, 0.19261836529899712, -57.915611468015136],
    [-0.0005714525377025321, 0.013517905837176747, 0.9999999999999999],
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
