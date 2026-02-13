"""
Input Processor
Handles: Video frame extraction, Image resizing, File validation
Requirements: Extract 1 frame/second from video, resize to 512×512
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputProcessor:
    """
    Processes room images and videos for the pipeline
    
    Responsibilities:
    1. Validate input (image or video)
    2. Extract frames from video (1 fps)
    3. Resize all images to standard resolution
    4. Save processed frames to cache
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize processor
        
        Args:
            target_size: Output image dimensions (width, height)
        """
        self.target_size = target_size
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
        # Cache directory for extracted frames
        self.cache_dir = Path("data/cache/frames")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"InputProcessor initialized - Target size: {target_size}")
    
    def process(self, input_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Main processing function - handles both images and videos
        
        Args:
            input_path: Path to image or video file
        
        Returns:
            List of processed frames as numpy arrays (H, W, 3)
        
        Example:
            >>> processor = InputProcessor()
            >>> frames = processor.process("room_video.mp4")
            >>> len(frames)  # Number of extracted frames
            15
        """
        input_path = Path(input_path)
        
        # Validate file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Route to appropriate handler
        if self._is_video(input_path):
            logger.info(f"Processing VIDEO: {input_path.name}")
            return self._process_video(input_path)
        elif self._is_image(input_path):
            logger.info(f"Processing IMAGE: {input_path.name}")
            return self._process_image(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    def _is_video(self, path: Path) -> bool:
        """Check if file is a supported video format"""
        return path.suffix.lower() in self.supported_video_formats
    
    def _is_image(self, path: Path) -> bool:
        """Check if file is a supported image format"""
        return path.suffix.lower() in self.supported_image_formats
    
    def _process_image(self, image_path: Path) -> List[np.ndarray]:
        """
        Process single image
        
        Steps:
        1. Read image
        2. Resize to target size
        3. Return as single-item list
        """
        # Read image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        resized = cv2.resize(image, self.target_size)
        
        logger.info(f"Image processed - Shape: {resized.shape}")
        
        return [resized]
    
    def _process_video(self, video_path: Path) -> List[np.ndarray]:
        """
        Process video - extract 1 frame per second
        
        Requirements (from PDF):
        - Extract 1 frame per second
        - 10-20 second video → 10-20 frames
        
        Steps:
        1. Open video
        2. Get FPS (frames per second)
        3. Extract every Nth frame (where N = FPS)
        4. Resize each frame
        5. Save to cache
        """
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video info - FPS: {fps}, Duration: {duration:.1f}s, Total frames: {total_frames}")
        
        # Calculate frame extraction interval
        # Extract 1 frame per second = every 'fps' frames
        frame_interval = fps
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame every 'frame_interval' frames (1 per second)
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                resized = cv2.resize(frame_rgb, self.target_size)
                
                frames.append(resized)
                
                # Save to cache (optional - for debugging)
                cache_path = self.cache_dir / f"frame_{extracted_count:04d}.jpg"
                cv2.imwrite(str(cache_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
                
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video (1 fps)")
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        return frames
    
    def process_multiple_images(self, image_paths: List[Union[str, Path]]) -> List[np.ndarray]:
        """
        Process multiple images (3-6 images as per requirements)
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            List of processed frames
        """
        if not (3 <= len(image_paths) <= 6):
            logger.warning(f"Recommended 3-6 images, got {len(image_paths)}")
        
        frames = []
        for img_path in image_paths:
            frame = self._process_image(Path(img_path))[0]
            frames.append(frame)
        
        logger.info(f"Processed {len(frames)} images")
        
        return frames
    
    def get_info(self) -> dict:
        """Get processor configuration"""
        return {
            'target_size': self.target_size,
            'supported_image_formats': self.supported_image_formats,
            'supported_video_formats': self.supported_video_formats,
            'cache_directory': str(self.cache_dir)
        }


# Testing
if __name__ == "__main__":
    print("="*80)
    print("STAGE 1 - INPUT PROCESSOR TEST")
    print("="*80)
    
    processor = InputProcessor()
    
    # Print config
    print("\nConfiguration:")
    for key, value in processor.get_info().items():
        print(f"  {key}: {value}")
    
    # Test with simulated image (create dummy)
    print("\n" + "="*80)
    print("TEST: Simulated Image Processing")
    print("="*80)
    
    # Create a test image
    test_image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
    test_path = Path("data/cache/test_input.jpg")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    
    # Process it
    result = processor.process(test_path)
    
    print(f"\nResult:")
    print(f"  Number of frames: {len(result)}")
    print(f"  Frame shape: {result[0].shape}")
    print(f"  Expected shape: (512, 512, 3)")
    print(f"  ✅ Success!" if result[0].shape == (512, 512, 3) else "❌ Failed")
    
    # Cleanup
    test_path.unlink()

# *Explanation:**

# **Key Concepts:**
# 1. **Single Responsibility:** Only handles input processing
# 2. **Type Hints:** `-> List[np.ndarray]` tells what function returns
# 3. **Logging:** Professional error tracking
# 4. **Validation:** Checks file exists before processing
# 5. **Flexibility:** Works with both images and videos

# **Logic Flow:**
# ```
# process() 
#   → Check file type (image or video?)
#   → Route to appropriate handler
#   → _process_image() or _process_video()
#   → Return standardized frames (512×512, RGB)