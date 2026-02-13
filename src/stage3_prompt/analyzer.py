"""
Scene Analyzer - Production Version
Integrates with Stage 1 (MiDaS) and Stage 2 (YOLO + Inventory)
"""

import numpy as np
from typing import Dict, List, Optional


class SceneAnalyzer:
    """
    Analyzes room scene from pipeline data
    
    Inputs:
    - Objects from Stage 2 YOLO detection
    - Image from Stage 1 for brightness
    - Depth map from Stage 1 for size
    - Room type hint from Stage 2 inventory (optional)
    
    Outputs:
    - Room type classification
    - Brightness level
    - Room size estimation
    - Object analysis
    """
    
    ROOM_TYPE_INDICATORS = {
        "bedroom": ["bed", "nightstand", "dresser"],
        "living_room": ["sofa", "couch", "tv", "tvmonitor"],
        "kitchen": ["refrigerator", "oven", "microwave", "sink"],
        "bathroom": ["toilet", "sink"],
        "dining_room": ["dining table"],
    }
    
    def __init__(self):
        self.scene_data = {}
    
    def analyze(self, 
                detected_objects: List[str],
                image_array: Optional[np.ndarray] = None,
                depth_map: Optional[np.ndarray] = None,
                room_type_hint: Optional[str] = None) -> Dict:
        """
        Main analysis function
        
        Args:
            detected_objects: Object names from YOLO (e.g., ["bed", "chair"])
            image_array: RGB image from Stage 1
            depth_map: Depth map from MiDaS (Stage 1)
            room_type_hint: Room type from Stage 2 inventory (e.g., "Bedroom")
        
        Returns:
            Analysis dictionary with room_type, brightness, size, objects
        """
        analysis = {}
        
        # Room type: Use Stage 2 hint if available, otherwise classify
        if room_type_hint:
            analysis['room_type'] = self._map_inventory_room_type(room_type_hint)
        else:
            analysis['room_type'] = self._classify_room_type(detected_objects)
        
        analysis['room_type_confidence'] = self._get_classification_confidence(
            detected_objects, analysis['room_type']
        )
        
        # Brightness analysis
        if image_array is not None:
            analysis['brightness'] = self._analyze_brightness(image_array)
            analysis['brightness_value'] = self._get_brightness_value(image_array)
        else:
            analysis['brightness'] = 'medium'
            analysis['brightness_value'] = 128
        
        # Room size estimation
        analysis['room_size'] = self._analyze_room_size(depth_map, len(detected_objects))
        
        # Object information
        analysis['objects'] = detected_objects
        analysis['object_count'] = len(detected_objects)
        analysis['is_cluttered'] = len(detected_objects) > 8
        
        self.scene_data = analysis
        return analysis
    
    def _map_inventory_room_type(self, inventory_type: str) -> str:
        """
        Convert Stage 2 inventory room type to internal format
        
        Stage 2 format: "Bedroom", "Living Room", etc.
        Internal format: "bedroom", "living_room", etc.
        """
        mapping = {
            "Bedroom": "bedroom",
            "Living Room": "living_room",
            "Dining Room": "dining_room",
            "Kitchen": "kitchen",
            "Bathroom": "bathroom",
            "General Indoor Space": "room"
        }
        return mapping.get(inventory_type, "room")
    
    def _classify_room_type(self, objects: List[str]) -> str:
        """Classify room type from detected objects"""
        objects_lower = [obj.lower() for obj in objects]
        scores = {}
        
        for room_type, indicators in self.ROOM_TYPE_INDICATORS.items():
            score = sum(1 for indicator in indicators if indicator in objects_lower)
            scores[room_type] = score
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "room"
    
    def _get_classification_confidence(self, objects: List[str], room_type: str) -> float:
        """Calculate room type classification confidence"""
        if room_type == "room":
            return 0.5
        
        objects_lower = [obj.lower() for obj in objects]
        indicators = self.ROOM_TYPE_INDICATORS.get(room_type, [])
        matches = sum(1 for indicator in indicators if indicator in objects_lower)
        
        confidence = min(matches / max(len(indicators), 1), 1.0)
        return round(confidence, 2)
    
    def _analyze_brightness(self, image_array: np.ndarray) -> str:
        """Analyze image brightness level"""
        brightness_value = self._get_brightness_value(image_array)
        
        if brightness_value < 85:
            return "dark"
        elif brightness_value < 170:
            return "medium"
        else:
            return "bright"
    
    def _get_brightness_value(self, image_array: np.ndarray) -> float:
        """Calculate average brightness (0-255)"""
        if len(image_array.shape) == 3:
            gray = np.dot(image_array[...,:3], [0.299, 0.587, 0.114])
        else:
            gray = image_array
        
        return float(np.mean(gray))
    
    def _analyze_room_size(self, depth_map: Optional[np.ndarray], object_count: int) -> str:
        """
        Estimate room size from depth map or object count
        
        Method 1 (preferred): Depth variance
        Method 2 (fallback): Object count
        """
        if depth_map is not None:
            depth_variance = float(np.var(depth_map))
            
            if depth_variance < 100:
                return "small"
            elif depth_variance < 500:
                return "medium"
            else:
                return "large"
        else:
            # Fallback: More objects typically = larger room
            if object_count < 4:
                return "small"
            elif object_count < 8:
                return "medium"
            else:
                return "large"
    
    def get_summary(self) -> str:
        """Human-readable summary of analysis"""
        if not self.scene_data:
            return "No analysis performed yet"
        
        return f"""Scene Analysis:
- Room: {self.scene_data['room_type']} (confidence: {self.scene_data['room_type_confidence']})
- Brightness: {self.scene_data['brightness']} ({self.scene_data['brightness_value']:.1f}/255)
- Size: {self.scene_data['room_size']}
- Objects: {self.scene_data['object_count']} items
- Cluttered: {'Yes' if self.scene_data['is_cluttered'] else 'No'}
- Items: {', '.join(self.scene_data['objects'])}"""