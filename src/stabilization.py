import cv2
import numpy as np

class VideoStabilizer:
    def __init__(self):
        """
        Initializes the Video Stabilizer.
        It stores the first frame (gray) and its keypoints as reference.
        """
        self.prev_gray = None
        self.ref_gray = None # Reference frame (usually the first one)
        self.kp_ref = None   # Keypoints from the reference frame
        self.initialized = False

    def stabilize(self, frame):
        """
        Stabilizes the given frame by aligning it to the reference frame.
        Args:
            frame: Input video frame (BGR).
        Returns:
            stabilized_frame: The warped frame aligned to the reference.
        """
        if frame is None:
            return None

        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Initialization (First Frame)
        if not self.initialized:
            self.ref_gray = curr_gray
            # Detect features (corners) to track in the reference frame
            self.kp_ref = cv2.goodFeaturesToTrack(self.ref_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            self.initialized = True
            return frame

        # 2. Calculate Optical Flow (Lucas-Kanade) from Reference to Current
        # Note: We track features from the REFERENCE frame to the CURRENT frame
        # directly to avoid drift accumulation over time (Global Stabilization).
        if self.kp_ref is None or len(self.kp_ref) == 0:
            # If tracking lost, reset reference (not ideal, but fallback)
             self.ref_gray = curr_gray
             self.kp_ref = cv2.goodFeaturesToTrack(self.ref_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
             return frame

        kp_curr, status, err = cv2.calcOpticalFlowPyrLK(self.ref_gray, curr_gray, self.kp_ref, None)

        # 3. Filter valid points
        # status == 1 means flow was found
        if kp_curr is None:
             return frame
             
        valid_ref = self.kp_ref[status == 1]
        valid_curr = kp_curr[status == 1]

        if len(valid_ref) < 4:
            # Not enough points to estimate transform
            return frame

        # 4. Estimate Affine Transformation (Translation + Rotation + Scale)
        # We want Matrix M that maps Current (src) to Reference (dst).
        
        # transform_matrix, inliers = cv2.estimateAffinePartial2D(valid_curr, valid_ref) # src=curr, dst=ref
        transform_matrix, inliers = cv2.estimateAffinePartial2D(valid_curr, valid_ref)

        if transform_matrix is None:
            return frame

        # 5. Warp the current frame
        h, w = self.ref_gray.shape
        stabilized_frame = cv2.warpAffine(frame, transform_matrix, (w, h))

        return stabilized_frame
