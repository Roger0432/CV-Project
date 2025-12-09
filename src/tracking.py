import supervision as sv
from utils import config

class TrafficTracker:
    def __init__(self):
        """
        Initialize ByteTrack tracker.
        """
        print("Initializing ByteTrack...")
        # Note: Parameters might need streaming depending on supervision version, 
        # but these are standard ByteTrack params.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.TRACKER_THRESH,
            lost_track_buffer=config.TRACK_BUFFER,
            minimum_matching_threshold=config.TRACKER_MATCH_THRESH,
            frame_rate=config.FPS
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the tracker with new detections.
        Args:
            detections (sv.Detections): Detections from the detector.
        Returns:
            sv.Detections: Detections with assigned tracker_id.
        """
        # supervision's update_with_detections returns the detections that are currently tracked
        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections
