# 🏠 AI Room Redesign Studio - Google Colab Guide

Run the AI Room Redesign Studio directly in Google Colab with free GPU access!

## 🚀 Quick Start Methods

### Method 1: 📓 Jupyter Notebook (Recommended)

**One-Click Setup:**
1. Open the Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abbastouqi/ai-room-styling/blob/main/colab/AI_Room_Redesign_Colab.ipynb)
2. Enable GPU: `Runtime` → `Change runtime type` → `GPU`
3. Run all cells in order
4. Upload your room images/videos
5. Get your redesigned rooms!

### Method 2: 🌐 Web Interface in Colab

```python
# Run this in a Colab cell
!git clone https://github.com/Abbastouqi/ai-room-styling.git
%cd ai-room-styling
!python colab/colab_setup.py

# Start web interface with ngrok
!pip install pyngrok
from pyngrok import ngrok
import threading
import os

# Start backend
def start_backend():
    os.chdir('/content/ai-room-styling/backend')
    os.system('python app.py')

backend_thread = threading.Thread(target=start_backend, daemon=True)
backend_thread.start()

# Create public URLs
backend_url = ngrok.connect(5000)
frontend_url = ngrok.connect(8080)

print(f"🌐 Backend API: {backend_url}")
print(f"🎨 Frontend UI: {frontend_url}")

# Start frontend
os.chdir('/content/ai-room-styling/frontend')
!python -m http.server 8080
```

### Method 3: 📱 Simple Upload Interface

```python
# Simple Colab interface
!git clone https://github.com/Abbastouqi/ai-room-styling.git
%cd ai-room-styling
!python colab/colab_setup.py

# Upload and process
from google.colab import files
import sys
sys.path.append('/content/ai-room-styling')

# Upload files
uploaded = files.upload()

# Process with AI
from src.optimized_pipeline import OptimizedPipeline, OptimizationConfig
import asyncio

config = OptimizationConfig(use_gpu=True, batch_size=4)
pipeline = OptimizedPipeline(config)

# Process uploaded files
for filename in uploaded.keys():
    results = await pipeline.process_batch(
        input_paths=[filename],
        style="modern"  # or "luxury", "minimal", "custom"
    )
    
    # Save and display results
    pipeline.save_results(results, "/content/results")
    print(f"✅ Processed: {filename}")
```

## 🎯 Step-by-Step Instructions

### 1. **Setup Environment**
```python
# Enable GPU (IMPORTANT!)
# Runtime → Change runtime type → Hardware accelerator → GPU

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
```

### 2. **Install Dependencies**
```python
# Clone repository
!git clone https://github.com/Abbastouqi/ai-room-styling.git
%cd ai-room-styling

# Run setup script
!python colab/colab_setup.py
```

### 3. **Upload Your Images/Videos**
```python
from google.colab import files

# Upload files
uploaded = files.upload()

# Or use sample images
!wget https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=512 -O sample_room.jpg
```

### 4. **Process with AI**
```python
import sys
sys.path.append('/content/ai-room-styling')

from src.optimized_pipeline import OptimizedPipeline, OptimizationConfig
import asyncio

# Configure pipeline
config = OptimizationConfig(
    use_gpu=True,           # Use GPU for speed
    batch_size=4,           # Process multiple frames
    use_fp16=True,          # Half precision for speed
    cache_models=True       # Cache models for reuse
)

# Initialize pipeline
pipeline = OptimizedPipeline(config)

# Process files
results = await pipeline.process_batch(
    input_paths=["sample_room.jpg"],
    style="modern"          # Options: modern, luxury, minimal, custom
)

# Save results
pipeline.save_results(results, "/content/results")
```

### 5. **View Results**
```python
import matplotlib.pyplot as plt
from PIL import Image
import os

# Display results
result_files = os.listdir("/content/results")
for file in result_files:
    if file.endswith(('.jpg', '.png')):
        img = Image.open(f"/content/results/{file}")
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Redesigned Room - {file}")
        plt.axis('off')
        plt.show()
```

### 6. **Download Results**
```python
# Download individual files
from google.colab import files
files.download("/content/results/room_redesigned.png")

# Or create zip archive
import zipfile
with zipfile.ZipFile("/content/results.zip", 'w') as zipf:
    for file in os.listdir("/content/results"):
        zipf.write(f"/content/results/{file}", file)

files.download("/content/results.zip")
```

## ⚙️ Configuration Options

### Style Presets
```python
# Available styles
styles = {
    "modern": "Clean lines, minimal furniture, neutral colors",
    "luxury": "Premium materials, elegant design, rich textures", 
    "minimal": "Simple, uncluttered spaces, maximum functionality",
    "custom": "Your own description"
}

# Custom style example
custom_prompt = "Scandinavian living room with light wood furniture, white walls, cozy textiles, and natural lighting"

results = await pipeline.process_batch(
    input_paths=["room.jpg"],
    style="custom",
    custom_prompt=custom_prompt
)
```

### Performance Settings
```python
# Fast processing (lower quality)
config = OptimizationConfig(
    use_gpu=True,
    batch_size=8,
    use_fp16=True,
    diffusion_steps=15      # Fewer steps = faster
)

# High quality (slower)
config = OptimizationConfig(
    use_gpu=True,
    batch_size=2,
    use_fp16=False,
    diffusion_steps=30      # More steps = better quality
)
```

## 📊 Performance Expectations

### Google Colab GPU (T4)
- **Single Image**: 30-60 seconds
- **Video (30 frames)**: 2-5 minutes
- **Memory Usage**: 6-8GB VRAM

### Google Colab CPU (Fallback)
- **Single Image**: 5-10 minutes
- **Video (30 frames)**: 1-3 hours
- **Memory Usage**: 4-6GB RAM

### Colab Pro/Pro+ (A100/V100)
- **Single Image**: 15-30 seconds
- **Video (30 frames)**: 1-2 minutes
- **Memory Usage**: 8-12GB VRAM

## 🔧 Troubleshooting

### Common Issues

**1. GPU Not Available**
```python
# Check GPU status
import torch
print(torch.cuda.is_available())

# If False, enable GPU:
# Runtime → Change runtime type → Hardware accelerator → GPU
```

**2. Out of Memory Error**
```python
# Reduce batch size
config.batch_size = 2

# Enable memory efficient mode
config.memory_efficient = True

# Use CPU for some operations
config.use_gpu = False
```

**3. Model Download Fails**
```python
# Manual model download
import urllib.request
import os

os.makedirs('/content/ai-room-styling/models', exist_ok=True)

# Download SAM model
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
urllib.request.urlretrieve(sam_url, '/content/ai-room-styling/models/sam_vit_b.pth')
```

**4. Slow Processing**
```python
# Check if GPU is being used
import torch
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

# Monitor GPU usage
!nvidia-smi
```

### Performance Tips

1. **Always enable GPU** - 10-50x faster than CPU
2. **Use smaller images** - 512x512 works best
3. **Close other tabs** - Free up GPU memory
4. **Use batch processing** - More efficient for multiple images
5. **Enable FP16** - 2x speedup with minimal quality loss

## 📱 Mobile-Friendly Colab

For mobile users, use the simplified interface:

```python
# Mobile-optimized interface
from IPython.display import HTML, display
import ipywidgets as widgets

# Create simple UI
style_dropdown = widgets.Dropdown(
    options=['modern', 'luxury', 'minimal'],
    description='Style:'
)

upload_button = widgets.FileUpload(
    accept='image/*,video/*',
    multiple=True,
    description='Upload Files'
)

process_button = widgets.Button(
    description='🎨 Redesign Room',
    button_style='success'
)

display(widgets.VBox([
    HTML("<h2>🏠 AI Room Redesign</h2>"),
    style_dropdown,
    upload_button, 
    process_button
]))
```

## 🎉 Examples

### Example 1: Modern Living Room
```python
results = await pipeline.process_batch(
    input_paths=["living_room.jpg"],
    style="modern"
)
# Result: Clean, minimalist design with neutral colors
```

### Example 2: Luxury Bedroom
```python
results = await pipeline.process_batch(
    input_paths=["bedroom.jpg"], 
    style="luxury"
)
# Result: Premium materials, elegant furniture, rich textures
```

### Example 3: Custom Scandinavian Style
```python
results = await pipeline.process_batch(
    input_paths=["room.jpg"],
    style="custom",
    custom_prompt="Scandinavian interior with light wood, white walls, cozy textiles, minimalist furniture, natural lighting"
)
# Result: Nordic-inspired design with specified elements
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Abbastouqi/ai-room-styling/issues)
- **Colab Problems**: Check Runtime → View runtime logs
- **GPU Issues**: Runtime → Factory reset runtime

---

**🎨 Happy room redesigning with AI in Google Colab!**