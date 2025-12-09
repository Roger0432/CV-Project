import os
import cv2
import numpy as np
from tqdm import tqdm
import json
import pandas as pd

from utils import config
from utils import visualization
from src.detection import VehicleDetector
from src.tracking import TrafficTracker
from src.lane_assignment import LaneAssigner
from src.anomaly_detection import AnomalyDetector
from src.evaluation import Evaluator

def main():
    print("🚦 Starting Traffic Analysis System...")
    
    # 1. Setup & Initialization
    if not os.path.exists(config.VIDEO_PATH):
        print(f"❌ Error: Video file not found at {config.VIDEO_PATH}")
        print("Please place a video file in the 'data' directory and update config.py if necessary.")
        # Create data dir if not exists (although I created it earlier)
        os.makedirs(config.DATA_DIR, exist_ok=True)
        return

    # Cap video to get info
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"ℹ️ Video Info: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Update config FPS if needed
    config.FPS = fps

    # Initialize Modules
    print("▶️ Initializing modules...")
    detector = VehicleDetector(config.MODEL_WEIGHTS)
    tracker = TrafficTracker()
    lane_assigner = LaneAssigner(config.LANE_POLYGONS)
    anomaly_detector = AnomalyDetector()
    evaluator = Evaluator() # If ground truth is available
    
    video_writer = visualization.setup_video_writer(config.OUTPUT_VIDEO_PATH, width, height, int(fps))

    all_tracks_data = {} # {track_id: {frames: [], ...}}
    detected_anomalies = []

    # 2. Main Processing Loop
    print("🔄 Processing frames...")
    pbar = tqdm(total=total_frames)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A. Detection
        detections = detector.detect(frame)

        # B. Tracking
        tracked_detections = tracker.update(detections)       

        # C. Lane Assignment
        lane_assignments = lane_assigner.assign(tracked_detections)

        # D. Anomaly Detection
        frame_anomalies = anomaly_detector.analyze(tracked_detections, lane_assignments)
        detected_anomalies.extend(frame_anomalies)
        
        # Update Evaluation Stats
        evaluator.update(tracked_detections, frame_anomalies)

        # E. Visualization
        annotated_frame = visualization.draw_frame(frame, tracked_detections, lane_assignments, frame_anomalies)
        video_writer.write(annotated_frame)

        # F. Data Collection (for evaluation/export)
        # Store basic info if needed
        if tracked_detections.tracker_id is not None:
             for tid in tracked_detections.tracker_id:
                  all_tracks_data[int(tid)] = all_tracks_data.get(int(tid), 0) + 1 # Just counting frames for now

        pbar.update(1)
        frame_idx += 1
        # if frame_idx > 100: # Limit for testing purposes, remove later
        #     break

    cap.release()
    video_writer.release()
    pbar.close()

    # 3. Post-Processing & Evaluation
    print("📊 Generating reports...")
    evaluator.generate_report(all_tracks_data)
    
    # 4. Save Results
    print(f"💾 Saving results to {config.RESULTS_DIR}...")
    with open(config.TRACKING_RESULTS_PATH, 'w') as f:
        json.dump(all_tracks_data, f, indent=4)
    
    pd.DataFrame(detected_anomalies).to_csv(config.ANOMALY_RESULTS_PATH, index=False)
    
    print(f"✅ Analysis Complete! Video saved to {config.OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()
