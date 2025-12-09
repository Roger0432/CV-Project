import numpy as np
from collections import deque
from utils import config

class AnomalyDetector:
    def __init__(self):
        """
        Initializes the Anomaly Detector.
        """
        # Store recent positions to calculate speed: {track_id: deque([(x, y), ...])}
        self.track_history = {} 

    def analyze(self, detections, lane_assignments):
        """
        Detects anomalies in the current frame.
        Args:
            detections (sv.Detections): Current tracked detections.
            lane_assignments (dict): Current lane assignments.
        Returns:
            list: List of anomalies [{'type': 'SPEEDING', 'id': int, 'value': float, 'bbox': list}, ...]
        """
        anomalies = []
        
        if detections.tracker_id is None:
            return anomalies

        for i, tracker_id in enumerate(detections.tracker_id):
            tid = int(tracker_id)
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]
            
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_pos = (center_x, center_y)

            # --- Update History ---
            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=config.SPEED_HISTORY_WINDOW)
            self.track_history[tid].append(current_pos)

            # --- 1. Speed Detection ---
            speed_kmh = self._calculate_speed(tid)
            if speed_kmh > config.SPEED_THRESHOLD:
                anomalies.append({
                    'type': 'SPEEDING',
                    'id': tid,
                    'value': round(speed_kmh, 2),
                    'bbox': bbox
                })

            # --- 2. Pedestrian in Road ---
            if class_id == config.PEDESTRIAN_CLASS_ID:
                # Check if the pedestrian is inside any defined lane
                # We can check if they have been assigned a lane
                assignment = lane_assignments.get(tid)
                if assignment and (assignment['entry_lane'] is not None or assignment['exit_lane'] is not None):
                     anomalies.append({
                        'type': 'PEDESTRIAN_IN_ROAD',
                        'id': tid,
                        'value': None,
                        'bbox': bbox
                    })

            # --- 3. Unusual Trajectory (Simplified) ---
            # Ideally requires full trajectory analysis post-processing or a trained model.
            # Here we could check for driving wrong way if we defined lane directions.
            
        return anomalies

    def _calculate_speed(self, tid):
        """
        Calculates speed in km/h based on pixel displacement.
        """
        history = self.track_history[tid]
        if len(history) < 2:
            return 0.0
        
        # Calculate distance between first and last point in the window
        # (Using simple Euclidean distance, better would be sum of segments if window is large)
        start_pos = np.array(history[0])
        end_pos = np.array(history[-1])
        
        pixel_dist = np.linalg.norm(end_pos - start_pos)
        
        # Convert to real world distance
        meters_dist = pixel_dist * config.CAMERA_CALIBRATION_FACTOR
        
        # Time elapsed
        frames_elapsed = len(history) - 1
        if frames_elapsed == 0:
            return 0.0
        
        time_seconds = frames_elapsed / config.FPS
        
        speed_mps = meters_dist / time_seconds
        speed_kmh = speed_mps * 3.6
        
        return speed_kmh
