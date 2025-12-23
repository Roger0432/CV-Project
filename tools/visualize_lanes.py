import cv2
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from utils import config

def draw_lanes_on_frame():
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video")
        return

    # Draw Lanes
    overlay = frame.copy()
    for lane_id, poly in config.LANE_POLYGONS.items():
        pts = poly.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], config.COLOR_PALETTE[(lane_id - 1) % len(config.COLOR_PALETTE)])
        
        # Draw label
        centroid = np.mean(poly, axis=0)
        cv2.putText(frame, f"Lane {lane_id}", (int(centroid[0]), int(centroid[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    output_path = "results/lanes_visualization.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    draw_lanes_on_frame()
