"""
Object Detector using YOLOv8
Original code by: [Team Member]
Restructured for: Professional architecture compliance
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)




# What is ultralytics?

# ultralytics is the official Python package for YOLOv8.

# YOLOv8 is a state-of-the-art object detection model.

# Ultralytics provides:

# Pre-trained YOLOv8 models (small, medium, large)





# Refines Object Detection

# YOLO only gives rough bounding boxes.

# SAM produces exact shapes, which is critical for realistic AI styling.


class ObjectDetector:
    """
    YOLOv8-based object detection for room furniture
    
    Detects: bed, sofa, chair, dining table, window, door, tv, plants, etc.
    Output: List of detections with bounding boxes and confidence scores
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence: Minimum confidence threshold (0.0-1.0)
        """
        logger.info(f"Initializing YOLOv8 detector - Model: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            logger.info("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        self.confidence = confidence
        
        # Target classes for room furniture (as per original code)
        self.target_classes = [
            'bed', 'sofa', 'chair', 'dining table', 'window', 
            'door', 'tvmonitor', 'pottedplant', 'couch', 'toilet',
            'tv', 'laptop', 'sink', 'refrigerator', 'oven',
            'microwave', 'clock', 'vase', 'book', 'bottle'
        ]
        
        # Cache directory for detection visualizations
        self.cache_dir = Path("data/cache/detections")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def detect(self, image: np.ndarray, save_visualization: bool = True) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect objects in image (original logic preserved)
        
        Args:
            image: Input image as numpy array (H, W, 3)
            save_visualization: Whether to save annotated image
        
        Returns:
            Tuple of:
            - List of detections: [{"class": str, "bbox": [x1,y1,x2,y2], "conf": float}, ...]
            - Annotated image with bounding boxes
        """
        # Run YOLO inference
        results = self.model(image, conf=self.confidence)[0]
        
        detections = []
        
        # Extract detections (original logic)
        for box in results.boxes:
            label = self.model.names[int(box.cls[0])]
            
            # Filter only relevant room objects
            if label in self.target_classes:
                detections.append({
                    "class": label,
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "conf": float(box.conf[0])
                })
        
        # Get annotated image
        annotated_image = results.plot()
        
        logger.info(f"Detected {len(detections)} objects")
        
        return detections, annotated_image
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Tuple[List[Dict], np.ndarray]]:
        """
        Detect objects in multiple images
        
        Args:
            images: List of images
        
        Returns:
            List of (detections, annotated_image) tuples
        """
        results = []
        
        for idx, image in enumerate(images):
            logger.info(f"Detecting objects in image {idx+1}/{len(images)}")
            detection_result = self.detect(image, save_visualization=False)
            results.append(detection_result)
        
        return results
    
    def get_detected_classes(self, detections: List[Dict]) -> List[str]:
        """
        Extract list of detected object classes
        
        Args:
            detections: List of detection dicts
        
        Returns:
            List of class names
        """
        return [d['class'] for d in detections]


# Testing
if __name__ == "__main__":
    print("="*80)
    print("STAGE 2 - OBJECT DETECTOR TEST")
    print("="*80)
    
    detector = ObjectDetector()
    
    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    print("\nRunning detection on test image...")
    detections, annotated = detector.detect(test_image)
    
    print(f"\nResults:")
    print(f"  Detections: {len(detections)}")
    print(f"  Annotated image shape: {annotated.shape}")
    print(f"  ✅ Detection working!")