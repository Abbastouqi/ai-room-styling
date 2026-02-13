"""
SAM-based Segmentation Engine
Original code by: [Team Member]
Restructured for: Professional architecture compliance
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    logger.error("segment_anything not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git")
    raise

logger = logging.getLogger(__name__)


class ObjectSegmenter:
    """
    SAM (Segment Anything Model) for precise object masking
    
    Input: Image + YOLO bounding boxes
    Output: Binary masks for each detected object
    """
    
    def __init__(self, model_path: str = "models/sam_vit_b.pth"):
        """
        Initialize SAM segmentation model
        
        Args:
            model_path: Path to SAM checkpoint file
        """
        self.model_type = "vit_b"  # CPU-optimized variant
        self.model_path = Path(model_path)
        
        logger.info(f"Loading SAM model from: {model_path}")
        
        try:
            # Load SAM model (original logic)
            self.sam = sam_model_registry[self.model_type](checkpoint=str(self.model_path))
            self.sam.to(device="cpu")  # CPU only as per requirements
            self.predictor = SamPredictor(self.sam)
            logger.info("✅ SAM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            logger.info("Download SAM checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            raise
        
        # Cache directory for masks
        self.cache_dir = Path("data/cache/masks")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_masks(self, image: np.ndarray, detections: List[Dict], 
                      save_to_cache: bool = True) -> List[Dict]:
        """
        Generate segmentation masks for detected objects (original logic preserved)
        
        Args:
            image: RGB image array (H, W, 3)
            detections: List of YOLO detections with bboxes
            save_to_cache: Whether to save masks to disk
        
        Returns:
            List of mask results: [{"class": str, "mask": np.ndarray, "mask_path": str}, ...]
        """
        # Ensure image is RGB
        if image.shape[-1] == 3 and image.dtype == np.uint8:
            image_rgb = image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM predictor
        self.predictor.set_image(image_rgb)
        
        mask_results = []
        
        for i, det in enumerate(detections):
            # Convert YOLO bbox to SAM input format (original logic)
            input_box = np.array(det['bbox'])
            
            # Generate mask
            masks, scores, _ = self.predictor.predict(
                box=input_box,
                multimask_output=False  # Single best mask
            )
            
            mask = masks[0]  # Get first (best) mask
            
            mask_data = {
                "class": det['class'],
                "mask": mask,
                "confidence": det['conf']
            }
            
            # Save to cache if requested (original logic)
            if save_to_cache:
                mask_name = f"mask_{det['class']}_{i}.png"
                mask_path = self.cache_dir / mask_name
                
                # Save binary mask (0 or 255)
                cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
                mask_data["mask_path"] = str(mask_path)
                
                logger.info(f"Mask saved: {det['class']} → {mask_path}")
            
            mask_results.append(mask_data)
        
        logger.info(f"Generated {len(mask_results)} segmentation masks")
        
        return mask_results
    
    def generate_masks_batch(self, images: List[np.ndarray], 
                            detections_list: List[List[Dict]]) -> List[List[Dict]]:
        """
        Generate masks for multiple images
        
        Args:
            images: List of images
            detections_list: List of detection lists (one per image)
        
        Returns:
            List of mask results (one list per image)
        """
        all_masks = []
        
        for idx, (image, detections) in enumerate(zip(images, detections_list)):
            logger.info(f"Generating masks for image {idx+1}/{len(images)}")
            masks = self.generate_masks(image, detections, save_to_cache=False)
            all_masks.append(masks)
        
        return all_masks


# Testing
if __name__ == "__main__":
    print("="*80)
    print("STAGE 2 - SEGMENTATION ENGINE TEST")
    print("="*80)
    
    # Check if SAM model exists
    model_path = Path("models/sam_vit_b.pth")
    
    if not model_path.exists():
        print(f"\n⚠️  SAM model not found at: {model_path}")
        print("\nDownload from:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print(f"\nSave as: {model_path}")
    else:
        segmenter = ObjectSegmenter(str(model_path))
        
        # Test with dummy data
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_detections = [
            {"class": "chair", "bbox": [100, 100, 200, 200], "conf": 0.9}
        ]
        
        print("\nGenerating test mask...")
        masks = segmenter.generate_masks(test_image, test_detections)
        
        print(f"\nResults:")
        print(f"  Masks generated: {len(masks)}")
        print(f"  ✅ Segmentation working!")