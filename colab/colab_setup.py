#!/usr/bin/env python3
"""
Google Colab Setup Script for AI Room Redesign Studio
Automated setup for running the application in Google Colab
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_colab_environment():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install all required dependencies for Colab"""
    logger.info("📦 Installing dependencies...")
    
    # Core ML libraries with CUDA support
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ], check=True)
    
    # Computer vision and ML libraries
    packages = [
        "diffusers>=0.21.0",
        "transformers>=4.30.0", 
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "timm>=0.9.0",
        "huggingface-hub",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "werkzeug>=2.3.0",
        "pyngrok",  # For web interface tunneling
        "matplotlib",  # For result visualization
        "ipywidgets"  # For interactive widgets
    ]
    
    for package in packages:
        logger.info(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
    
    # Install SAM from GitHub
    logger.info("Installing Segment Anything Model...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/facebookresearch/segment-anything.git"
    ], check=True)
    
    logger.info("✅ All dependencies installed!")

def clone_repository():
    """Clone the AI Room Redesign repository"""
    repo_url = "https://github.com/Abbastouqi/ai-room-styling.git"
    repo_dir = "/content/ai-room-styling"
    
    if os.path.exists(repo_dir):
        logger.info("Repository already exists, pulling latest changes...")
        os.chdir(repo_dir)
        subprocess.run(["git", "pull"], check=True)
    else:
        logger.info("Cloning repository...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    
    os.chdir(repo_dir)
    logger.info(f"✅ Repository ready at {repo_dir}")
    return repo_dir

def download_models():
    """Download required models for the pipeline"""
    logger.info("🔽 Downloading models...")
    
    # Create models directory
    models_dir = Path("/content/ai-room-styling/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download SAM model
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    sam_path = models_dir / "sam_vit_b.pth"
    
    if not sam_path.exists():
        logger.info("Downloading SAM model (375MB)...")
        urllib.request.urlretrieve(sam_url, sam_path)
        logger.info("✅ SAM model downloaded")
    else:
        logger.info("✅ SAM model already exists")
    
    # YOLOv8 will be downloaded automatically by ultralytics
    logger.info("✅ Models ready!")

def setup_colab_interface():
    """Setup Colab-specific interface components"""
    logger.info("🎨 Setting up Colab interface...")
    
    # Create necessary directories
    dirs = [
        "/content/ai-room-styling/data/input",
        "/content/ai-room-styling/data/output", 
        "/content/uploads",
        "/content/results"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Interface setup complete!")

def check_gpu():
    """Check GPU availability and configuration"""
    import torch
    
    logger.info("🔍 Checking GPU configuration...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU detected: {gpu_name}")
        logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        logger.warning("⚠️ No GPU detected!")
        logger.warning("Enable GPU: Runtime → Change runtime type → GPU")
        return False

def create_colab_widgets():
    """Create interactive widgets for Colab"""
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML
        
        # Style selector
        style_selector = widgets.Dropdown(
            options=['modern', 'luxury', 'minimal', 'custom'],
            value='modern',
            description='Style:',
            style={'description_width': 'initial'}
        )
        
        # Custom prompt input
        custom_prompt = widgets.Textarea(
            placeholder='Enter custom prompt for redesign...',
            description='Custom Prompt:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%', height='80px')
        )
        
        # Quality selector
        quality_selector = widgets.IntSlider(
            value=2,
            min=1,
            max=3,
            step=1,
            description='Quality:',
            style={'description_width': 'initial'}
        )
        
        # GPU toggle
        gpu_toggle = widgets.Checkbox(
            value=True,
            description='Use GPU Acceleration',
            style={'description_width': 'initial'}
        )
        
        # Process button
        process_button = widgets.Button(
            description='🎨 Process Images',
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        # Display widgets
        display(HTML("<h3>🎨 AI Room Redesign Configuration</h3>"))
        display(widgets.VBox([
            style_selector,
            custom_prompt,
            quality_selector,
            gpu_toggle,
            process_button
        ]))
        
        return {
            'style': style_selector,
            'custom_prompt': custom_prompt,
            'quality': quality_selector,
            'gpu': gpu_toggle,
            'process_button': process_button
        }
        
    except ImportError:
        logger.warning("ipywidgets not available, using text interface")
        return None

def main():
    """Main setup function for Google Colab"""
    logger.info("🚀 Setting up AI Room Redesign Studio for Google Colab")
    
    # Check if in Colab
    if not check_colab_environment():
        logger.error("❌ This script is designed for Google Colab")
        logger.info("For local installation, use: python run_app.py")
        return False
    
    try:
        # Step 1: Install dependencies
        install_dependencies()
        
        # Step 2: Clone repository
        repo_dir = clone_repository()
        
        # Step 3: Download models
        download_models()
        
        # Step 4: Setup interface
        setup_colab_interface()
        
        # Step 5: Check GPU
        has_gpu = check_gpu()
        
        # Step 6: Create widgets
        widgets = create_colab_widgets()
        
        logger.info("🎉 Setup complete!")
        logger.info("\n📋 Next steps:")
        logger.info("1. Upload your room images/videos")
        logger.info("2. Select your preferred style")
        logger.info("3. Click 'Process Images'")
        logger.info("4. Download your redesigned rooms!")
        
        if not has_gpu:
            logger.warning("\n⚠️ GPU not detected - processing will be slower")
            logger.info("Enable GPU: Runtime → Change runtime type → GPU")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready to redesign rooms with AI!")
    else:
        print("\n❌ Setup failed. Please check the errors above.")