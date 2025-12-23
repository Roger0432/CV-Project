import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from collections import deque
from utils import config

class AnomalyDetector:
    def __init__(self):
        """
        Initializes the Anomaly Detector.
        """
        # Store recent positions to calculate speed: {track_id: deque([(x, y), ...])}
        self.track_history = {} 
        
        self.forbidden_zones = {}
        for zone_id, poly_coords in config.FORBIDDEN_ZONES.items():
            self.forbidden_zones[zone_id] = Polygon(poly_coords) 

        # Stats for dynamic thresholds: { lane_id: {'speeds': [], 'vectors': []} }
        self.lane_stats = {}


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

            # --- 1. Speed Detection (Absolute & Relative) ---
            strength = 0.0
            is_speeding = False
            speed_kmh = self._calculate_speed(tid)
            
            # A) Absolute Threshold
            if speed_kmh > config.SPEED_THRESHOLD:
                is_speeding = True
                strength = speed_kmh
            
            # B) Relative Threshold (> 1.3x Lane Average)
            current_lane = lane_assignments.get(tid, {}).get('current_lane')
            avg_lane_speed = self._get_lane_avg_speed(current_lane)
            
            # Only apply relative check if the car is moving significantly (e.g. > 30km/h)
            # This prevents flagging slow cars just because the average is also very slow.
            if avg_lane_speed > 0 and speed_kmh > 30 and speed_kmh > 1.3 * avg_lane_speed:
                is_speeding = True
                strength = max(strength, speed_kmh) # Keep the speed value
            
            if is_speeding:
                anomalies.append({
                    'type': 'SPEEDING',
                    'id': tid,
                    'value': round(strength, 2),
                    'bbox': bbox
                })

            # --- Update Lane Stats (Speed) ---
            if current_lane is not None and speed_kmh > 5: # Only count moving cars for stats
                self._update_lane_stats(current_lane, speed=speed_kmh)


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

            # --- 3. Forbidden Zones ---
            # Check if vehicle center is in a forbidden zone
            # (pedestrians or cars)
            point = Point(center_x, y2) # Use bottom center (feet)
            for zone_id, poly in self.forbidden_zones.items():
                if poly.contains(point):
                     anomalies.append({
                        'type': 'FORBIDDEN_ZONE',
                        'id': tid,
                        'value': f"Zone {zone_id}",
                        'bbox': bbox
                    })

            # --- 4. Wrong Direction ---
            # Check if trajectory opposes dominant lane flow
            if current_lane is not None:
                motion_vector = self._get_motion_vector(tid)
                if motion_vector is not None:
                    # Update stats first (assuming most cars are correct)
                    # To avoid outliers polluting, we could weight it, but simple avg is compliant with "dominant flow"
                    self._update_lane_stats(current_lane, vector=motion_vector)
                    
                    # Check against dominant flow
                    # Warm-up: Only check if we have enough samples to be sure of the direction
                    if len(self.lane_stats[current_lane]['vectors']) > 20: 
                        dominant_vector = self._get_lane_dominant_vector(current_lane)
                        if dominant_vector is not None:
                            # Cosine similarity
                            dot_prod = np.dot(motion_vector, dominant_vector)
                            # cos(150 deg) approx -0.866
                            if dot_prod < -0.86: 
                                anomalies.append({
                                    'type': 'WRONG_DIRECTION',
                                    'id': tid,
                                    'value': f"Lane {current_lane}",
                                    'bbox': bbox
                                })

        
        current_speeds = {tid: self._calculate_speed(tid) for tid in detections.tracker_id} if detections.tracker_id is not None else {}
        return anomalies, current_speeds

    def _calculate_speed(self, tid):
        """
        Calculates speed in km/h based on regression over history.
        """
        history = self.track_history[tid]
        if len(history) < config.SPEED_HISTORY_WINDOW // 2: # Wait for at least half window
            return 0.0
        
        # Convert history dequqe to list of points
        points = np.array(history)
        
        # Map to World Coordinates
        if config.HOMOGRAPHY_MATRIX is not None:
            world_points = np.array([self._apply_homography(p, config.HOMOGRAPHY_MATRIX) for p in points])
        else:
            # Simple scaling assumption (less accurate)
            # Center everything at 0,0 to avoid huge numbers
            world_points = points * config.CAMERA_CALIBRATION_FACTOR
            
        # We need to regress Distance vs Time.
        # Since vehicles move in 2D, we can approximate "distance traveled" 
        # as distance from the FIRST point in the window (assuming relatively straight motion in short window)
        # OR proper cumulative distance.
        # "Displacement from start of window" is robust enough for 0.5s windows.
        
        start_point = world_points[0]
        distances = np.linalg.norm(world_points - start_point, axis=1) # [0, d1, d2, ...]
        
        # Time points (seconds)
        # 0, 1/FPS, 2/FPS...
        num_points = len(distances)
        times = np.arange(num_points) / config.FPS
        
        # Linear Regression: Distance = Speed * Time + c
        # Use np.polyfit(times, distances, 1) -> Returns [slope, intercept]
        # slope is Speed in meters/second
        
        if num_points < 2:
            return 0.0
            
        slope, intercept = np.polyfit(times, distances, 1)
        
        speed_mps = abs(slope) # Speed is magnitude
        speed_kmh = speed_mps * 3.6
        
        return speed_kmh

    def _apply_homography(self, point, H):
        """
        Applies homography matrix to a 2D point (x, y).
        """
        # Convert to homogenous coordinate [x, y, 1]
        p = np.array([point[0], point[1], 1]).reshape(3, 1)
        # Apply matrix
        new_p = np.dot(H, p)
        # Normalize by z (scale)
        if new_p[2] != 0:
            x = new_p[0] / new_p[2]
            y = new_p[1] / new_p[2]
            return np.array([x, y]).flatten()
        return np.array([0, 0])

    def _get_motion_vector(self, tid):
        """Returns normalized motion vector (dx, dy) based on history or None if not moving."""
        history = self.track_history[tid]
        if len(history) < config.SPEED_HISTORY_WINDOW // 4:
            return None
            
        # Use first and last point of window
        p_start = np.array(history[0])
        p_end = np.array(history[-1])
        
        # Determine vector in World Coordinates if possible
        if config.HOMOGRAPHY_MATRIX is not None:
            p_start = self._apply_homography(p_start, config.HOMOGRAPHY_MATRIX)
            p_end = self._apply_homography(p_end, config.HOMOGRAPHY_MATRIX)
            
        vec = p_end - p_start
        norm = np.linalg.norm(vec)
        
        if norm < 0.5: # Effectively stationary (0.5 meters movement)
            return None
            
        return vec / norm

    def _update_lane_stats(self, lane_id, speed=None, vector=None):
        if lane_id not in self.lane_stats:
            self.lane_stats[lane_id] = {'speeds': deque(maxlen=100), 'vectors': deque(maxlen=100)}
            
        if speed is not None:
            self.lane_stats[lane_id]['speeds'].append(speed)
            
        if vector is not None:
            self.lane_stats[lane_id]['vectors'].append(vector)

    def _get_lane_avg_speed(self, lane_id):
        if lane_id not in self.lane_stats or not self.lane_stats[lane_id]['speeds']:
            return 0.0
        return np.mean(self.lane_stats[lane_id]['speeds'])

    def _get_lane_dominant_vector(self, lane_id):
        if lane_id not in self.lane_stats or not self.lane_stats[lane_id]['vectors']:
            return None
        # Average all unit vectors
        all_vecs = np.array(self.lane_stats[lane_id]['vectors'])
        avg_vec = np.mean(all_vecs, axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm == 0: return None
        return avg_vec / norm
