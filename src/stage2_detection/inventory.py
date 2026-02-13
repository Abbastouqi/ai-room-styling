"""
Inventory Manager & Room Classifier
Original code by: [Team Member]
Restructured for: Professional architecture compliance
"""

import json
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class InventoryManager:
    """
    Manages object inventory and classifies room type
    
    Based on detected furniture, determines:
    - Room type (Bedroom, Living Room, etc.)
    - Total item count
    - Object list
    """
    
    def __init__(self, output_dir: str = "data/cache/detections"):
        """
        Initialize inventory manager
        
        Args:
            output_dir: Directory to save inventory JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"InventoryManager initialized - Output: {output_dir}")
    
    def classify_room(self, detections: List[Dict]) -> str:
        """
        Classify room type based on detected furniture (original logic)
        
        Args:
            detections: List of detection dicts
        
        Returns:
            Room type string
        """
        # Extract object classes
        items = [d['class'] for d in detections]
        
        # Classification logic (original)
        if "bed" in items:
            return "Bedroom"
        elif any(x in items for x in ["sofa", "tvmonitor", "couch", "tv"]):
            return "Living Room"
        elif "dining table" in items:
            return "Dining Room"
        elif any(x in items for x in ["toilet", "sink"]):
            return "Bathroom"
        elif any(x in items for x in ["refrigerator", "oven", "microwave"]):
            return "Kitchen"
        else:
            return "General Indoor Space"
    
    def create_inventory(self, detections: List[Dict], save_json: bool = True) -> Dict:
        """
        Create inventory data structure (original logic)
        
        Args:
            detections: List of detections
            save_json: Whether to save as JSON file
        
        Returns:
            Inventory dictionary
        """
        # Classify room
        room_type = self.classify_room(detections)
        
        # Extract items
        items = [d['class'] for d in detections]
        
        # Build inventory data (original structure)
        inventory_data = {
            "room_type": room_type,
            "total_items": len(items),
            "detected_objects": items,
            "status": "Ready for Styling"
        }
        
        # Save JSON file (original logic)
        if save_json:
            file_path = self.output_dir / "inventory.json"
            with open(file_path, "w") as f:
                json.dump(inventory_data, f, indent=4)
            
            logger.info(f"Inventory saved: {file_path}")
        
        return inventory_data
    
    def get_object_counts(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count occurrences of each object type
        
        Args:
            detections: List of detections
        
        Returns:
            Dictionary: {class_name: count}
        """
        counts = {}
        for det in detections:
            class_name = det['class']
            counts[class_name] = counts.get(class_name, 0) + 1
        
        return counts


# Testing
if __name__ == "__main__":
    print("="*80)
    print("STAGE 2 - INVENTORY MANAGER TEST")
    print("="*80)
    
    manager = InventoryManager()
    
    # Test with bedroom data
    test_detections = [
        {"class": "bed", "bbox": [100, 100, 300, 300], "conf": 0.9},
        {"class": "chair", "bbox": [350, 200, 400, 350], "conf": 0.85},
        {"class": "pottedplant", "bbox": [50, 400, 100, 500], "conf": 0.75}
    ]
    
    inventory = manager.create_inventory(test_detections, save_json=False)
    
    print("\nInventory Results:")
    print(json.dumps(inventory, indent=2))
    print(f"\n✅ Room classified as: {inventory['room_type']}")