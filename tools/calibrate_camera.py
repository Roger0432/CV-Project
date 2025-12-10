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

def calibrate_camera(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    cap.release()

    # Resize for display (consistent with draw_lanes)
    target_width = 1280
    h, w = frame.shape[:2]
    scale = target_width / w
    target_height = int(h * scale)
    frame = cv2.resize(frame, (target_width, target_height))

    print(f"Frame resized to {target_width}x{target_height} for calibration.")
    print("INSTRUCTIONS:")
    print("1. Click 4 points forming a RECTANGLE in the real world.")
    print("   Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    print("2. After 4 clicks, the script will ask for real-world dimensions.")
    print("   (e.g., width=3.5 meters, length=10 meters)")
    
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                print(f"Point {len(points)}: ({x}, {y})")

    cv2.namedWindow("Calibrate Camera")
    cv2.setMouseCallback("Calibrate Camera", mouse_callback)

    while True:
        display_frame = frame.copy()

        # Draw points
        for i, pt in enumerate(points):
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw lines connecting points
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(display_frame, points[i], points[i+1], (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(display_frame, points[3], points[0], (0, 255, 0), 2)

        cv2.imshow("Calibrate Camera", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        if len(points) == 4:
            print("\n4 Points selected. converting to original scale...")
            # Scale points back to original resolution
            orig_points = [(p[0]/scale, p[1]/scale) for p in points]
            src_pts = np.float32(orig_points)
            
            # Ask for user input on dimensions
            print("\nDefine the Real-World dimensions of this rectangle:")
            try:
                rw_width = float(input("Enter Width (meters) [e.g. lane width ~3.5]: "))
                rw_length = float(input("Enter Length (meters) [distance along road]: "))
            except ValueError:
                print("Invalid number. Using default 3.5m x 10m")
                rw_width = 3.5
                rw_length = 10.0

            # Destination points (top-down view)
            # We map the source polygon to a rectangle of (rw_width * factor, rw_length * factor)
            # Factor is purely for coordinate scale, let's say 1 meter = 1 unit
            dst_pts = np.float32([
                [0, 0],
                [rw_width, 0],
                [rw_width, rw_length],
                [0, rw_length]
            ])
            
            # Calculate Homography
            H, mask = cv2.findHomography(src_pts, dst_pts)
            
            print("\n--- COPY TO config.py ---")
            print("HOMOGRAPHY_MATRIX = np.array([")
            for row in H:
                print(f"    [{row[0]}, {row[1]}, {row[2]}],")
            print("])")
            print("-------------------------")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera homography.")
    parser.add_argument("--video", type=str, default="data/input_video.mp4", help="Path to input video")
    args = parser.parse_args()
    
    calibrate_camera(args.video)
