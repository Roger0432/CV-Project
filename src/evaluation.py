import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import xml.etree.ElementTree as ET
import os
import sys
import collections
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.config as config

class Evaluator:
    def __init__(self):
        self.total_frames = 0
        self.total_tracks = set()
        self.anomalies_counts = collections.defaultdict(int)
        self.unique_anomalies = set() # Store (id, type) tuples
        
        # Ground Truth Data: {frame_idx: {gt_id: {'bbox': [x1, y1, x2, y2], 'speed': float, 'center': (x, y)}}}
        self.ground_truth = {}
        self.speed_errors = [] # List of absolute errors
        self.centroid_errors = [] # List of distances
        
        # Load GT if configured
        if hasattr(config, 'GROUND_TRUTH_PATH') and config.GROUND_TRUTH_PATH and os.path.exists(config.GROUND_TRUTH_PATH):
            self.load_ground_truth(config.GROUND_TRUTH_PATH)
        else:
            print(f"⚠️ Warning: Ground Truth file not found or not configured.")

    def load_ground_truth(self, xml_path):
        """Parses UA-DETRAC XML file."""
        print(f"Loading Ground Truth from {xml_path}...")
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for frame in root.findall('frame'):
                frame_num = int(frame.get('num'))
                frame_idx = frame_num - 1 # 0-indexed
                
                self.ground_truth[frame_idx] = {}
                
                target_list = frame.find('target_list')
                if target_list is not None:
                    for target in target_list.findall('target'):
                        tid = int(target.get('id'))
                        box = target.find('box')
                        attr = target.find('attribute')
                        
                        if box is not None and attr is not None:
                            left = float(box.get('left'))
                            top = float(box.get('top'))
                            width = float(box.get('width'))
                            height = float(box.get('height'))
                            try:
                                speed = float(attr.get('speed', 0))
                            except:
                                speed = 0.0

                            x1, y1 = left, top
                            x2, y2 = left + width, top + height
                            center_x = left + width / 2
                            center_y = top + height / 2
                            
                            self.ground_truth[frame_idx][tid] = {
                                'bbox': [x1, y1, x2, y2],
                                'speed': speed, 
                                'center': (center_x, center_y)
                            }
            print(f"✅ Ground Truth loaded: {len(self.ground_truth)} frames.")
            
        except Exception as e:
            print(f"❌ Error loading XML: {e}")

    def update(self, detections, frame_anomalies, frame_idx=None, current_speeds=None):
        """
        Updates evaluation statistics for a frame.
        """
        self.total_frames += 1
        
        if detections.tracker_id is not None:
            for tid in detections.tracker_id:
                self.total_tracks.add(tid)
        
        if self.unique_anomalies is None:
            self.unique_anomalies = set()

        for anomaly in frame_anomalies:
            unique_key = (anomaly['id'], anomaly['type'])
            if unique_key not in self.unique_anomalies:
                self.unique_anomalies.add(unique_key)
                self.anomalies_counts[anomaly['type']] += 1

        # Compare with Ground Truth if available and frame_idx provided
        if self.ground_truth and frame_idx is not None:
            self._evaluate_frame(detections, frame_idx, current_speeds)

    def _evaluate_frame(self, detections, frame_idx, current_speeds):
        if frame_idx not in self.ground_truth:
            return

        gt_targets = self.ground_truth[frame_idx]
        if not gt_targets:
            return

        if detections.tracker_id is None:
            return
            
        for i, tid in enumerate(detections.tracker_id):
            x1, y1, x2, y2 = detections.xyxy[i]
            pred_center = ((x1 + x2)/2, (y1 + y2)/2)
            
            min_dist = float('inf')
            closest_gt_id = None
            
            for gt_id, gt_data in gt_targets.items():
                gt_center = gt_data['center']
                dist = np.linalg.norm(np.array(pred_center) - np.array(gt_center))
                if dist < min_dist:
                    min_dist = dist
                    closest_gt_id = gt_id
            
            if min_dist < 100: # Reasonable range in pixels
                self.centroid_errors.append(min_dist)
                
                # Check speed error if we matched a GT vehicle and have a speed estimate
                if closest_gt_id is not None and current_speeds and tid in current_speeds:
                    pred_speed = current_speeds[tid]
                    gt_speed = gt_targets[closest_gt_id]['speed']
                    
                    # Ensure positive speeds
                    error = abs(pred_speed - gt_speed)
                    self.speed_errors.append(error)

    def generate_report(self, all_tracks_data):
        """
        Generates a summary report.
        Args:
            all_tracks_data (dict): Dictionary of all tracks.
        """
        self.total_tracks = len(all_tracks_data)
        
        print("\n--- EVALUATION REPORT ---")
        print(f"Total Frames Processed: {self.total_frames}")
        print(f"Total Unique Tracks: {self.total_tracks}")
        print(f"Total Anomalies Detected: {sum(self.anomalies_counts.values())}")
        print(f"Total Anomalies Detected: {sum(self.anomalies_counts.values())}")
        
        known_anomalies = ['SPEEDING', 'WRONG_DIRECTION', 'FORBIDDEN_ZONE', 'PEDESTRIAN_IN_ROAD']
        for k in known_anomalies:
            v = self.anomalies_counts.get(k, 0)
            print(f"  - {k}: {v}")
            
        for k, v in self.anomalies_counts.items():
            if k not in known_anomalies:
                print(f"  - {k}: {v}")
            
        if self.centroid_errors:
            avg_centroid_error = np.mean(self.centroid_errors)
            print(f"Mean Centroid Position Error (vs GT): {avg_centroid_error:.2f} pixels")
            
            if self.speed_errors:
                mae_speed = np.mean(self.speed_errors)
                print(f"Mean Absolute Speed Error (vs GT): {mae_speed:.2f} km/h")
        else:
            print("No Ground Truth comparison performed (or no matches found).")
            
        print("-------------------------")
        
        return self.anomalies_counts

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
