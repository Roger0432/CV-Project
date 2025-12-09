from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from utils import config

class VehicleDetector:
    def __init__(self, model_weights=config.MODEL_WEIGHTS):
        """
        Initialize the Vehicle Detector model.
        """
        print(f"Loading YOLOv8 model: {model_weights}...")
        self.model = YOLO(model_weights)
        self.tracker_id = None # Used if we were doing internal tracking, but we use external ByteTrack

    def detect(self, frame):
        """
        Detects vehicles and pedestrians in a frame.
        Returns:
            sv.Detections: Detections object containing bounding boxes, confidence, class_id.
        """
        # Inference with YOLOv8
        # verbose=False to reduce clutter
        results = self.model(frame, verbose=False)[0]
        
        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter by Confidence
        detections = detections[detections.confidence > config.CONFIDENCE_THRESHOLD]
        
        # Filter by Class ID (Car, Truck, Bus, Motorcycle, Person)
        # Note: We must ensure class_id is in config.TARGET_CLASSES
        # sv.Detections.class_id is a numpy array
        
        mask = np.isin(detections.class_id, config.TARGET_CLASSES)
        detections = detections[mask]
        
        return detections
