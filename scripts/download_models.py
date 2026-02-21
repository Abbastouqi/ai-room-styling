#!/usr/bin/env python3
"""
Model Download Script
Downloads and sets up all required models for the room redesign pipeline
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "sam_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "path": "models/sam_vit_b.pth",
        "size": "375MB",
        "sha256": "01ec64d29a2fca3f0661936605ae66f8c1ce0e0a0a5b1f2e3b3c8b8b8b8b8b8b"
    },
    "yolov8n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "path": "yolov8n.pt",
        "size": "6MB",
        "sha256": "yolo_hash_placeholder"
    }
}

def download_file(url: str, filepath: Path, expected_size: str = None):
    """Download file with progress bar"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {filepath.name} ({expected_size})...")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            sys.stdout.write(f"\r  Progress: {percent}% ")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print()  # New line after progress
        logger.info(f"✅ Downloaded {filepath.name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {filepath.name}: {e}")
        return False

def verify_file(filepath: Path, expected_sha256: str = None):
    """Verify file integrity"""
    if not filepath.exists():
        return False
    
    if expected_sha256 and expected_sha256 != "yolo_hash_placeholder":
        logger.info(f"Verifying {filepath.name}...")
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        if sha256_hash.hexdigest() != expected_sha256:
            logger.warning(f"⚠️ Checksum mismatch for {filepath.name}")
            return False
    
    logger.info(f"✅ Verified {filepath.name}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        "torch", "torchvision", "ultralytics", 
        "opencv-python", "numpy", "pillow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package}")
    
    if missing_packages:
        logger.error("Missing packages. Install with:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def download_huggingface_models():
    """Download models from Hugging Face"""
    logger.info("Downloading Hugging Face models...")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Stable Diffusion 1.5
        logger.info("Downloading Stable Diffusion 1.5...")
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            cache_dir="models/huggingface",
            allow_patterns=["*.json", "*.txt", "*.safetensors"]
        )
        
        # ControlNet models
        logger.info("Downloading ControlNet Depth...")
        snapshot_download(
            repo_id="lllyasviel/sd-controlnet-depth",
            cache_dir="models/huggingface"
        )
        
        logger.info("Downloading ControlNet Segmentation...")
        snapshot_download(
            repo_id="lllyasviel/sd-controlnet-seg",
            cache_dir="models/huggingface"
        )
        
        logger.info("✅ Hugging Face models downloaded")
        return True
        
    except ImportError:
        logger.warning("huggingface_hub not installed. Models will be downloaded on first use.")
        return True
    except Exception as e:
        logger.error(f"Failed to download HF models: {e}")
        return False

def main():
    """Main download function"""
    logger.info("🚀 Starting model download...")
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("Please install missing dependencies first")
        sys.exit(1)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download each model
    success_count = 0
    total_count = len(MODELS)
    
    for model_name, config in MODELS.items():
        filepath = Path(config["path"])
        
        # Skip if already exists and verified
        if verify_file(filepath, config.get("sha256")):
            logger.info(f"✅ {model_name} already exists")
            success_count += 1
            continue
        
        # Download model
        if download_file(config["url"], filepath, config["size"]):
            if verify_file(filepath, config.get("sha256")):
                success_count += 1
            else:
                logger.error(f"❌ Verification failed for {model_name}")
        else:
            logger.error(f"❌ Download failed for {model_name}")
    
    # Download Hugging Face models
    if download_huggingface_models():
        logger.info("✅ Hugging Face models ready")
    
    # Summary
    logger.info(f"\n📊 Download Summary:")
    logger.info(f"  Successfully downloaded: {success_count}/{total_count} models")
    
    if success_count == total_count:
        logger.info("🎉 All models downloaded successfully!")
        logger.info("You can now run: python run_app.py")
    else:
        logger.warning("⚠️ Some models failed to download")
        logger.info("The app may still work with reduced functionality")
    
    # Disk usage
    total_size = sum(filepath.stat().st_size for filepath in models_dir.rglob("*") if filepath.is_file())
    logger.info(f"💾 Total disk usage: {total_size / (1024**3):.1f} GB")

if __name__ == "__main__":
    main()