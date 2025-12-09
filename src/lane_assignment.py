import cv2
import numpy as np
from utils import config

class LaneAssigner:
    def __init__(self, lane_polygons=config.LANE_POLYGONS):
        """
        Initializes the Lane Assigner with lane polygon definitions.
        Args:
            lane_polygons (dict): Dictionary mapping lane_id to polygon coordinates (numpy array).
        """
        self.lane_polygons = lane_polygons
        # Dictionary to store lane assignments for each track_id
        # Structure: {track_id: {'entry_lane': id, 'exit_lane': id}}
        self.assignments = {} 

    def assign(self, detections):
        """
        Updates lane assignments for the given detections.
        Args:
            detections (sv.Detections): Detections object with tracker_id.
        Returns:
            dict: Updated assignments dictionary.
        """
        if detections.tracker_id is None:
            return self.assignments

        # Iterate through detections
        for i, tracker_id in enumerate(detections.tracker_id):
            tid = int(tracker_id)
            
            # Initialize if new track
            if tid not in self.assignments:
                self.assignments[tid] = {'entry_lane': None, 'exit_lane': None}

            # Calculate center point
            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            point = (center_x, center_y)

            # Check inclusion in each lane polygon
            for lane_id, poly in self.lane_polygons.items():
                # Ensure polygon is int32 for OpenCV
                contour = poly.astype(np.int32)
                
                # Check if point is inside polygon (measureDist=False returns +1, -1, 0)
                result = cv2.pointPolygonTest(contour, point, False)
                
                if result >= 0: # Inside or on edge
                    # If entry lane is not set, this is the first lane seen -> Entry Lane (Simplified logic)
                    # Ideally, entry lane is the lane at the first frame of the track.
                    if self.assignments[tid]['entry_lane'] is None:
                         self.assignments[tid]['entry_lane'] = lane_id
                    
                    # Always update exit lane to the current lane
                    self.assignments[tid]['exit_lane'] = lane_id
                    
                    # Assuming a vehicle is in only one lane at a time, break after match
                    break 
        
        return self.assignments
