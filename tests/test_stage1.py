"""
Unit Tests for Stage 1: Input Processing & Depth Estimation
Tests both InputProcessor and DepthEstimator independently
"""


                                      # PyTest will:
                                      #(🧠 Deep learning execution (PyTorch))

# Run this function

# Check if assertion is True

# If True → PASSED

# If False → FAILED

# So tests verify logic correctness.

import pytest
import numpy as np
import cv2
from pathlib import Path
import shutil

from src.stage1_input import InputProcessor, DepthEstimator


class TestInputProcessor:
    """Test suite for InputProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance for tests"""
        return InputProcessor(target_size=(512, 512))
    
    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Create temporary test image"""
        image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
        img_path = tmp_path / "test_room.jpg"
        cv2.imwrite(str(img_path), image)
        return img_path
    
    @pytest.fixture
    def test_video_path(self, tmp_path):
        """Create temporary test video"""
        video_path = tmp_path / "test_video.mp4"
        
        # Create 3-second video at 10 fps = 30 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))
        
        for i in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        return video_path
    
    def test_processor_initialization(self, processor):
        """Test if processor initializes correctly"""
        assert processor.target_size == (512, 512)
        assert processor.cache_dir.exists()
    
    def test_is_image_detection(self, processor):
        """Test image format detection"""
        assert processor._is_image(Path("test.jpg"))
        assert processor._is_image(Path("test.png"))
        assert not processor._is_image(Path("test.mp4"))
    
    def test_is_video_detection(self, processor):
        """Test video format detection"""
        assert processor._is_video(Path("test.mp4"))
        assert processor._is_video(Path("test.avi"))
        assert not processor._is_video(Path("test.jpg"))
    
    def test_process_single_image(self, processor, test_image_path):
        """Test single image processing"""
        frames = processor.process(test_image_path)
        
        assert len(frames) == 1
        assert frames[0].shape == (512, 512, 3)
        assert frames[0].dtype == np.uint8
    
    def test_process_video(self, processor, test_video_path):
        """Test video processing - extract 1 fps"""
        frames = processor.process(test_video_path)
        
        # 3-second video at 1 fps = 3 frames
        assert len(frames) >= 2  # At least 2 frames
        assert frames[0].shape == (512, 512, 3)
        assert all(f.dtype == np.uint8 for f in frames)
    
    def test_process_nonexistent_file(self, processor):
        """Test error handling for missing file"""
        with pytest.raises(FileNotFoundError):
            processor.process("nonexistent_file.jpg")
    
    def test_process_multiple_images(self, processor, tmp_path):
        """Test processing 3-6 images"""
        # Create 4 test images
        image_paths = []
        for i in range(4):
            img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
            img_path = tmp_path / f"room_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            image_paths.append(img_path)
        
        frames = processor.process_multiple_images(image_paths)
        
        assert len(frames) == 4
        assert all(f.shape == (512, 512, 3) for f in frames)
    
    def test_get_info(self, processor):
        """Test configuration retrieval"""
        info = processor.get_info()
        
        assert 'target_size' in info
        assert 'supported_image_formats' in info
        assert info['target_size'] == (512, 512)


class TestDepthEstimator:
    """Test suite for DepthEstimator"""
    
    @pytest.fixture
    def estimator(self):
        """Create estimator instance (downloads model on first run)"""
        return DepthEstimator(model_type="DPT_Hybrid")
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def test_estimator_initialization(self, estimator):
        """Test if estimator initializes correctly"""
        assert estimator.model is not None
        assert estimator.device.type == "cpu"
        assert estimator.cache_dir.exists()
    
    def test_single_depth_estimation(self, estimator, test_image):
        """Test depth map generation for single image"""
        depth_map = estimator.estimate(test_image)
        
        assert depth_map.shape == (512, 512)
        assert depth_map.dtype == np.uint8
        assert depth_map.min() >= 0
        assert depth_map.max() <= 255
    
    def test_batch_depth_estimation(self, estimator):
        """Test batch processing of multiple images"""
        images = [
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        depth_maps = estimator.estimate_batch(images, save_to_cache=False)
        
        assert len(depth_maps) == 3
        assert all(d.shape == (512, 512) for d in depth_maps)
        assert all(d.dtype == np.uint8 for d in depth_maps)
    
    def test_normalize_depth(self, estimator):
        """Test depth normalization"""
        # Create depth array with known range
        raw_depth = np.random.rand(512, 512) * 100
        
        normalized = estimator._normalize_depth(raw_depth)
        
        assert normalized.min() == 0
        assert normalized.max() == 255
        assert normalized.dtype == np.uint8
    
    def test_visualize_depth(self, estimator, test_image):
        """Test depth map visualization"""
        depth_map = estimator.estimate(test_image)
        colorized = estimator.visualize_depth(depth_map)
        
        assert colorized.shape == (512, 512, 3)
        assert colorized.dtype == np.uint8


class TestStage1Integration:
    """Integration tests for complete Stage 1 pipeline"""
    
    @pytest.fixture
    def processor(self):
        return InputProcessor()
    
    @pytest.fixture
    def estimator(self):
        return DepthEstimator(model_type="DPT_Hybrid")
    
    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Create test image file"""
        image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
        img_path = tmp_path / "bedroom.jpg"
        cv2.imwrite(str(img_path), image)
        return img_path
    
    def test_full_pipeline_single_image(self, processor, estimator, test_image_path):
        """Test complete pipeline: Image → Frames → Depth Maps"""
        # Step 1: Process input
        frames = processor.process(test_image_path)
        
        assert len(frames) == 1
        
        # Step 2: Generate depth maps
        depth_maps = estimator.estimate_batch(frames, save_to_cache=False)
        
        assert len(depth_maps) == 1
        assert depth_maps[0].shape == (512, 512)
    
    def test_full_pipeline_video(self, processor, estimator, tmp_path):
        """Test complete pipeline: Video → Frames → Depth Maps"""
        # Create test video
        video_path = tmp_path / "room_walkthrough.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (640, 480))
        
        for i in range(25):  # 5 seconds at 5 fps
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Step 1: Process video
        frames = processor.process(video_path)
        
        assert len(frames) >= 4  # At least 4 frames from 5-second video
        
        # Step 2: Generate depth maps
        depth_maps = estimator.estimate_batch(frames, save_to_cache=False)
        
        assert len(depth_maps) == len(frames)
        assert all(d.shape == (512, 512) for d in depth_maps)
    
    def test_output_format_for_stage2(self, processor, estimator, test_image_path):
        """Test that Stage 1 output is compatible with Stage 2 input"""
        # Process
        frames = processor.process(test_image_path)
        depth_maps = estimator.estimate_batch(frames, save_to_cache=False)
        
        # Verify format for next stage
        for frame, depth in zip(frames, depth_maps):
            # Frame: RGB, uint8, (512, 512, 3)
            assert frame.shape == (512, 512, 3)
            assert frame.dtype == np.uint8
            
            # Depth: Grayscale, uint8, (512, 512)
            assert depth.shape == (512, 512)
            assert depth.dtype == np.uint8


# Cleanup after tests
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_cache():
    """Clean up test cache after all tests"""
    yield
    # Cleanup runs after all tests
    cache_dir = Path("data/cache")
    if cache_dir.exists():
        # Remove test files only, keep structure
        for pattern in ["test_*.jpg", "test_*.png", "frame_*.jpg", "depth_*.png"]:
            for file in cache_dir.rglob(pattern):
                file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])