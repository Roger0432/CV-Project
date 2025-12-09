import json
import logging

class Evaluator:
    def __init__(self):
        self.stats = {
            "total_frames": 0,
            "total_tracks": 0,
            "total_anomalies": 0,
            "anomalies_by_type": {}
        }

    def update(self, tracks, anomalies):
        """
        Update stats with current frame data.
        """
        self.stats["total_frames"] += 1
        self.stats["total_anomalies"] += len(anomalies)
        
        for anomaly in anomalies:
            a_type = anomaly['type']
            self.stats["anomalies_by_type"][a_type] = self.stats["anomalies_by_type"].get(a_type, 0) + 1

    def generate_report(self, all_tracks_data):
        """
        Generates a summary report.
        Args:
            all_tracks_data (dict): Dictionary of all tracks.
        """
        self.stats["total_tracks"] = len(all_tracks_data)
        
        print("\n--- EVALUATION REPORT ---")
        print(f"Total Frames Processed: {self.stats['total_frames']}")
        print(f"Total Unique Tracks: {self.stats['total_tracks']}")
        print(f"Total Anomalies Detected: {self.stats['total_anomalies']}")
        for k, v in self.stats["anomalies_by_type"].items():
            print(f"  - {k}: {v}")
        print("-------------------------")
        
        # In a real scenario, here we would compare against ground truth 
        # using metrics like MOTA/MOTP if ground truth labels were loaded.
        
        return self.stats
