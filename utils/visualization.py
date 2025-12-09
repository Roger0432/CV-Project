import cv2
import numpy as np
import supervision as sv
from utils import config

# Initialize Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
# trace_annotator = sv.TraceAnnotator() # Optional: Draw historical trails

def setup_video_writer(output_path, width, height, fps):
    """
    Creates and returns a cv2.VideoWriter object.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def draw_frame(frame, detections, lane_assignments, anomalies):
    """
    Draws bounding boxes, lanes, labels, and anomalies on the frame.
    """
    # 1. Draw Lanes
    frame = draw_lanes(frame)

    # 2. Draw Detections & Tracks
    if detections.tracker_id is not None:
        # Prepare labels: "ID: LaneEntry->LaneExit"
        labels = []
        for i, tracker_id in enumerate(detections.tracker_id):
            tid = int(tracker_id)
            assignment = lane_assignments.get(tid, {})
            entry = assignment.get('entry_lane', '?')
            exit_ = assignment.get('exit_lane', '?')
            label = f"#{tid} {entry}->{exit_}"
            labels.append(label)

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # 3. Draw Anomalies
    for anomaly in anomalies:
        bbox = anomaly.get('bbox')
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            text = f"ALERT: {anomaly['type']}"
            if anomaly.get('value'):
                text += f" {anomaly['value']}"
            
            # Red box and text for anomaly
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

def draw_lanes(frame):
    overlay = frame.copy()
    alpha = 0.3
    
    for lane_id, poly in config.LANE_POLYGONS.items():
        pts = poly.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw filled polygon for visualization
        color = config.COLOR_PALETTE[lane_id % len(config.COLOR_PALETTE)]
        cv2.fillPoly(overlay, [pts], color)
        
        # Draw border
        cv2.polylines(frame, [pts], True, color, 2)
        
        # Draw Label
        center = np.mean(poly, axis=0).astype(int)
        cv2.putText(frame, config.LANE_NAMES.get(lane_id, f"Lane {lane_id}"), 
                    tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
    # Blend overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
