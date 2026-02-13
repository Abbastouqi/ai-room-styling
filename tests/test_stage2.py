"""
Unit Tests for Stage 2: Object Detection & Segmentation
"""

import pytest
import numpy as np
import json
from pathlib import Path

from src.stage2_detection import ObjectDetector, InventoryManager
# Note: ObjectSegmenter test requires SAM model download


class TestObjectDetector:
    """Test YOLO detector"""
    
    @pytest.fixture
    def detector(self):
        return ObjectDetector()
    
    @pytest.fixture
    def test_image(self):
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        """Test if detector initializes"""
        assert detector.model is not None
        assert len(detector.target_classes) > 0
    
    def test_detect_returns_correct_format(self, detector, test_image):
        """Test detection output format"""
        detections, annotated = detector.detect(test_image)
        
        assert isinstance(detections, list)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == test_image.shape
    
    def test_get_detected_classes(self, detector):
        """Test class extraction"""
        test_detections = [
            {"class": "chair", "bbox": [0,0,10,10], "conf": 0.9},
            {"class": "bed", "bbox": [0,0,10,10], "conf": 0.8}
        ]
        
        classes = detector.get_detected_classes(test_detections)
        assert classes == ["chair", "bed"]


class TestInventoryManager:
    """Test inventory manager"""
    
    @pytest.fixture
    def manager(self, tmp_path):
        return InventoryManager(output_dir=str(tmp_path))
    
    def test_classify_bedroom(self, manager):
        """Test bedroom classification"""
        detections = [{"class": "bed", "bbox": [], "conf": 0.9}]
        room_type = manager.classify_room(detections)
        assert room_type == "Bedroom"
    
    def test_classify_living_room(self, manager):
        """Test living room classification"""
        detections = [{"class": "sofa", "bbox": [], "conf": 0.9}]
        room_type = manager.classify_room(detections)
        assert room_type == "Living Room"
    
    def test_create_inventory(self, manager):
        """Test inventory creation"""
        detections = [
            {"class": "bed", "bbox": [], "conf": 0.9},
            {"class": "chair", "bbox": [], "conf": 0.8}
        ]
        
        inventory = manager.create_inventory(detections, save_json=False)
        
        assert inventory['room_type'] == "Bedroom"
        assert inventory['total_items'] == 2
        assert "bed" in inventory['detected_objects']
    
    def test_get_object_counts(self, manager):
        """Test object counting"""
        detections = [
            {"class": "chair", "bbox": [], "conf": 0.9},
            {"class": "chair", "bbox": [], "conf": 0.8},
            {"class": "table", "bbox": [], "conf": 0.7}
        ]
        
        counts = manager.get_object_counts(detections)
        assert counts['chair'] == 2
        assert counts['table'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])