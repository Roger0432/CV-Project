import cv2
import numpy as np
import argparse
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

def draw_zones(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    cap.release()

    # Resize for easier viewing if 4K
    target_width = 1280
    h, w = frame.shape[:2]
    scale = target_width / w
    target_height = int(h * scale)
    frame = cv2.resize(frame, (target_width, target_height))

    print(f"Frame resized to {target_width}x{target_height} for annotation.")
    print("Instructions:")
    print("  - Left Click: Add point to current polygon")
    print("  - Right Click: Close current polygon (minimum 3 points)")
    print("  - 'n': Start new zone/polygon")
    print("  - 'z': Undo last point")
    print("  - 'q': Quit and print coordinates")

    current_polygon = []
    all_polygons = {}
    zone_count = 1

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_polygon) >= 3:
                # Store normalized coordinates (or scaled back to original resolution)
                # Here we store scaled back to original resolution
                original_scale_poly = [(int(px / scale), int(py / scale)) for px, py in current_polygon]
                all_polygons[zone_count] = np.array(original_scale_poly)
                print(f"Zone {zone_count} saved.")
                
                # Prepare for next
                # zone_count += 1
                current_polygon = []
            else:
                print("Need at least 3 points for a polygon.")

    cv2.namedWindow("Draw Forbidden Zones")
    cv2.setMouseCallback("Draw Forbidden Zones", mouse_callback)

    while True:
        display_frame = frame.copy()

        # Draw existing polygons
        for idx, poly_coords in all_polygons.items():
            # Scale to display
            display_poly = (poly_coords * scale).astype(int)
            cv2.polylines(display_frame, [display_poly], True, (0, 0, 255), 2)
            # Label
            M = cv2.moments(display_poly)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(display_frame, f"Zone {idx}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw current polygon being drawn
        if len(current_polygon) > 0:
            pts = np.array(current_polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], False, (0, 255, 255), 2)
            for pt in current_polygon:
                cv2.circle(display_frame, pt, 3, (0, 255, 255), -1)

        cv2.imshow("Draw Forbidden Zones", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n'):
            if len(current_polygon) == 0:
                zone_count += 1
                print(f"Switched to Zone ID {zone_count}")
            else:
                print("Finish current polygon first (Right Click) or Undo (z)")
        elif key == ord('z'):
            if current_polygon:
                current_polygon.pop()

    cv2.destroyAllWindows()

    print("\n--- CONFIGURATION SNIPPET ---")
    print("Copy this into utils/config.py inside FORBIDDEN_ZONES = { ... }:\n")
    print("FORBIDDEN_ZONES = {")
    for idx, poly in all_polygons.items():
        # Format as list of lists
        poly_list = poly.tolist()
        print(f"    {idx}: np.array({poly_list}),")
    print("}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw forbidden zones on a video frame.")
    parser.add_argument("--video", type=str, default="data/input_video.mp4", help="Path to input video")
    args = parser.parse_args()
    
    draw_zones(args.video)
