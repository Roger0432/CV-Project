import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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

    def evaluate_lane_assignment(self, predictions, ground_truth):
        """
        Calculates Lane Assignment Accuracy (LAA).
        Args:
            predictions (dict): {track_id: {'entry': int, 'exit': int}}
            ground_truth (dict): {track_id: {'entry': int, 'exit': int}}
        """
        correct = 0
        total = 0
        y_true_entry, y_pred_entry = [], []
        
        for tid, gt_data in ground_truth.items():
            if tid in predictions:
                total += 1
                pred_data = predictions[tid]
                
                # Check match (None is handling missing lanes)
                entry_match = (str(pred_data.get('entry_lane')) == str(gt_data.get('entry')))
                exit_match = (str(pred_data.get('exit_lane')) == str(gt_data.get('exit')))
                
                if entry_match and exit_match:
                    correct += 1
                    
                y_true_entry.append(str(gt_data.get('entry')))
                y_pred_entry.append(str(pred_data.get('entry_lane')))
        
        accuracy = correct / total if total > 0 else 0
        print(f"Lane Assignment Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
        return accuracy

    def evaluate_speed(self, predictions, ground_truth):
        """
        Calculates MAE for speed.
        Args:
            predictions (dict): {track_id: speed_float}
            ground_truth (dict): {track_id: speed_float}
        """
        errors = []
        for tid, gt_speed in ground_truth.items():
            if tid in predictions:
                pred_speed = predictions[tid]
                errors.append(abs(pred_speed - gt_speed))
        
        mae = np.mean(errors) if errors else 0
        print(f"Speed Estimation MAE: {mae:.2f} km/h")
        return mae

    def evaluate_anomalies(self, predicted_anomalies, gt_anomalies):
        """
        Calculates Precision, Recall, F1 for anomalies.
        Args:
            predicted_anomalies (list): List of dicts or set of (frame, id, type)
            gt_anomalies (list): List of dicts or set of (frame, id, type)
            # Simplified: checking existence of anomaly per track ID
        """
        # Set of track IDs flagged specific anomaly types
        pred_set = set((d['id'], d['type']) for d in predicted_anomalies)
        gt_set = set((d['id'], d['type']) for d in gt_anomalies)
        
        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Anomaly Detection Rules:")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1: {f1:.2f}")
        return precision, recall, f1
