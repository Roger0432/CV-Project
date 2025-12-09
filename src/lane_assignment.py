import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from utils import config

class LaneAssigner:
    def __init__(self, lane_polygons=config.LANE_POLYGONS):
        """
        Initializes the Lane Assigner with lane polygon definitions.
        Args:
            lane_polygons (dict): Dictionary mapping lane_id to polygon coordinates (numpy array).
        """
        self.polygons = {}
        for lane_id, cols in lane_polygons.items():
            # Convert numpy array to Shapely Polygon
            # cols is expected to be [[x,y], [x,y], ...]
            self.polygons[lane_id] = Polygon(cols)
            
        # Dictionary to store lane assignments for each track_id
        # Structure: {track_id: {'entry_lane': id, 'exit_lane': id, 'history': []}}
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
                self.assignments[tid] = {
                    'entry_lane': None, 
                    'exit_lane': None,
                    'current_lane': None, # current frame status
                    'history': []
                }

            # Calculate "feet" point (bottom center)
            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            bottom_y = y2
            point = Point(center_x, bottom_y)

            current_lane = None
            
            # Check inclusion in each lane polygon
            for lane_id, poly in self.polygons.items():
                if poly.contains(point):
                    current_lane = lane_id
                    break
            
            track_data = self.assignments[tid]
            
            # Update history
            track_data['history'].append(current_lane)
            track_data['current_lane'] = current_lane
            
            if current_lane is not None:
                # If entry lane is not set, this is the first lane seen -> Entry Lane
                if track_data['entry_lane'] is None:
                    track_data['entry_lane'] = current_lane
                
                # Update exit lane to the current lane (assuming last seen lane is exit)
                track_data['exit_lane'] = current_lane
                        
        return self.assignments
