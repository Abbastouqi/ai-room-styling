"""
Depth Estimator using MiDaS
Generates depth maps for room geometry understanding
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    MiDaS-based depth estimation for room scenes
    
    Depth maps help:
    - Understand room geometry
    - Preserve structure in redesign (via ControlNet)
    - Estimate room size
    
    Model: MiDaS v3.0 (DPT_Large or DPT_Hybrid for CPU)
    """
    
    def __init__(self, model_type: str = "DPT_Hybrid"):
        """
        Initialize MiDaS depth estimator
        
        Args:
            model_type: "DPT_Large" (better quality) or "DPT_Hybrid" (faster on CPU)
        """
        self.model_type = model_type
        self.device = torch.device("cpu")  # CPU only as per requirements
        
        logger.info(f"Initializing MiDaS depth estimator - Model: {model_type}")


        # torch.hub.load() SERIOULSY AMAZING POINT AS IT DOWNLOADS DIRECTLY FROM THE GITHUB NO MANUAL DOWNLOADING IS NEEDED NOW (WAQI MAI CHASSS AA GAI HAIIII)

        
        # Load MiDaS model
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if model_type == "DPT_Large":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            logger.info("✅ MiDaS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise
        
        # Cache directory for depth maps
        self.cache_dir = Path("data/cache/depth")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map for single image
        
        Args:
            image: RGB image array (H, W, 3)
        
        Returns:
            Depth map array (H, W) - normalized to 0-255
        
        Process:
        1. Transform image for MiDaS
        2. Run inference
        3. Normalize depth values
        4. Return as uint8 array
        """
        # Convert to format MiDaS expects (RGB, 0-255)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply MiDaS transform
        input_batch = self.transform(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-255 range
        depth_map = self._normalize_depth(depth_map)
        
        return depth_map
    
    def estimate_batch(self, images: List[np.ndarray], save_to_cache: bool = True) -> List[np.ndarray]:
        """
        Generate depth maps for multiple frames
        
        Args:
            images: List of RGB images
            save_to_cache: Whether to save depth maps to cache
        
        Returns:
            List of depth maps
        """
        depth_maps = []
        
        for idx, image in enumerate(images):
            logger.info(f"Generating depth map {idx+1}/{len(images)}")
            
            depth_map = self.estimate(image)
            depth_maps.append(depth_map)
            
            # Save to cache
            if save_to_cache:
                cache_path = self.cache_dir / f"depth_{idx:04d}.png"
                cv2.imwrite(str(cache_path), depth_map)
        
        logger.info(f"✅ Generated {len(depth_maps)} depth maps")
        
        return depth_maps
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize depth values to 0-255 range
        
        Why? Makes it easier to:
        - Visualize depth maps
        - Use in ControlNet (expects 0-255 images)
        - Analyze room size (variance calculation)
        """
        # Remove invalid values (inf, nan)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to 0-1
        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)
        
        # Scale to 0-255
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        return depth_uint8   
    
                       #Every image has a data type.

                     # Common image types:

                      # uint8 → values from 0 to 255

                      # float32 → values from 0.0 to 1.0
    
    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create colorized depth map for visualization
        
        Args:
            depth_map: Grayscale depth map (0-255)
        
        Returns:
            Colorized depth map (H, W, 3) using COLORMAP_INFERNO
        """
        colorized = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        
        return colorized
    
                                      # If you process 4 images together:

# Batch size = 4

                                   # What is Channels?

# Channels = color components.

# RGB image:

# Red

# Green

# Blue

# So channels = 3

                                          # FINAL
    # (1,3,512,512)
 



# Testing
if __name__ == "__main__":
    print("="*80)
    print("STAGE 1 - DEPTH ESTIMATOR TEST")
    print("="*80)
    
    print("\nInitializing depth estimator (this may take a moment)...")
    
    try:
        estimator = DepthEstimator(model_type="DPT_Hybrid")  # Faster for CPU
        
        print("✅ Model loaded successfully")
        
        # Create test image
        print("\nGenerating test depth map...")
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        depth_map = estimator.estimate(test_image)
        
        print(f"\nResult:")
        print(f"  Depth map shape: {depth_map.shape}")
        print(f"  Value range: {depth_map.min()} - {depth_map.max()}")
        print(f"  Expected: (512, 512), 0-255")
        print(f"  ✅ Success!" if depth_map.shape == (512, 512) else "❌ Failed")
        
        # Save example
        output_path = Path("data/cache/depth/test_depth.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), depth_map)
        print(f"\n  Saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: First run downloads MiDaS model (~400MB)")
        print("Ensure internet connection is available")
# ```

# **Explanation:**

# **Key Concepts:**
# 1. **torch.hub.load:** Downloads MiDaS from PyTorch Hub automatically
# 2. **No gradient:** `with torch.no_grad()` - faster inference, less memory
# 3. **Normalization:** Converts raw depth → 0-255 for compatibility
# 4. **Batch processing:** Handles multiple frames efficiently

# **MiDaS Logic:**
# ```
# Input Image (512×512×3)
#   → Transform (resize, normalize)
#   → MiDaS Model
#   → Raw depth prediction (float values)
#   → Normalize to 0-255
#   → Output depth map (512×512)