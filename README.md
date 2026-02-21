# AI Room Redesign Studio

Transform your room images and videos with AI-powered interior design using Stable Diffusion, ControlNet, and advanced computer vision.

## 🚀 Features

- **Fast Processing**: <1 minute for images, <3 minutes for videos (with GPU)
- **Multiple Styles**: Modern, Luxury, Minimal, or Custom prompts
- **Smart Analysis**: Automatic object detection and room understanding
- **High Quality**: Preserves room structure while applying new designs
- **Web Interface**: Beautiful, responsive HTML/CSS frontend
- **Real-time Progress**: Live updates during processing
- **Batch Processing**: Efficient video frame processing

## 🏗️ Architecture

### 4-Stage Pipeline
1. **Input Processing & Depth Estimation** - MiDaS depth maps
2. **Object Detection & Segmentation** - YOLOv8 + SAM
3. **Prompt Generation** - Smart prompt creation
4. **Image Generation** - Stable Diffusion + Dual ControlNet

### Performance Optimizations
- GPU acceleration (CUDA/Apple Silicon)
- Model caching and batch processing
- Half-precision inference (FP16)
- Parallel stage execution
- Memory optimization

## 📋 Requirements

### Minimum (CPU-only)
- Python 3.8+
- 16GB RAM
- 20GB storage
- Processing time: 5-10 minutes per image

### Recommended (GPU)
- NVIDIA RTX 3060+ (8GB VRAM) or Apple M1 Pro+
- 16GB RAM
- 20GB storage (SSD preferred)
- Processing time: 30-60 seconds per image

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd room-redesign-studio
```

2. **Install dependencies**
```bash
pip install -r backend/requirements.txt
```

3. **Download required models**
```bash
python scripts/download_models.py
```

4. **Start the application**
```bash
python run_app.py
```

The app will open automatically at `http://localhost:8080`

## 🎯 Quick Start

1. **Upload** an image or video of your room
2. **Select** a design style (Modern, Luxury, Minimal, or Custom)
3. **Configure** quality vs speed settings
4. **Generate** your redesigned room
5. **Download** the result

## 🔧 Configuration

### Quality Settings
- **Fast**: 15-30 seconds, good quality
- **Balanced**: 30-60 seconds, high quality  
- **High Quality**: 60-120 seconds, best quality

### Advanced Options
- GPU Acceleration (recommended)
- Batch Processing (for videos)
- Custom prompts
- ControlNet weights

## 📊 Performance Benchmarks

| Hardware | Image Time | Video Time (30 frames) |
|----------|------------|------------------------|
| CPU Only | 5-10 min | 2-5 hours |
| RTX 3060 | 45-60s | 2-3 min |
| RTX 4080 | 15-30s | 1-2 min |
| Apple M1 Pro | 60-90s | 3-4 min |

## 🏃‍♂️ API Usage

### Backend API Endpoints

```python
# Upload file
POST /api/upload
Content-Type: multipart/form-data

# Start processing
POST /api/process
{
  "file_id": "uuid",
  "style": "modern",
  "custom_prompt": "optional",
  "options": {
    "quality": 2,
    "gpu_acceleration": true
  }
}

# Check status
GET /api/status/{file_id}

# Download result
GET /api/download/{file_id}/{filename}
```

### Python SDK

```python
from src.optimized_pipeline import OptimizedPipeline, OptimizationConfig

# Configure pipeline
config = OptimizationConfig(
    use_gpu=True,
    batch_size=4,
    use_fp16=True
)

pipeline = OptimizedPipeline(config)

# Process images
results = await pipeline.process_batch(
    input_paths=["room1.jpg", "room2.mp4"],
    style="modern"
)

# Save results
pipeline.save_results(results, "output/")
```

## 🎨 Supported Styles

### Pre-built Styles
- **Modern**: Clean lines, minimal furniture, neutral colors
- **Luxury**: Premium materials, elegant design, rich textures
- **Minimal**: Simple, uncluttered spaces, maximum functionality

### Custom Prompts
Write your own detailed prompts for unique designs:
```
"Scandinavian living room with light wood furniture, 
white walls, cozy textiles, and natural lighting"
```

## 🔍 Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size
config.batch_size = 2
config.memory_efficient = True
```

**Slow Processing (CPU fallback)**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit
# Visit: https://pytorch.org/get-started/locally/
```

**Model Download Failures**
```bash
# Manual download
python scripts/download_models.py --force
```

### Performance Tips

1. **Use GPU**: 10-50x faster than CPU
2. **Enable FP16**: 2x speedup with minimal quality loss
3. **Batch processing**: Process multiple frames together
4. **SSD storage**: Faster I/O for large models
5. **Close other apps**: Free up GPU memory

## 🧪 Development

### Project Structure
```
├── frontend/           # HTML/CSS/JS web interface
├── backend/           # Flask API server
├── src/              # Core pipeline modules
│   ├── stage1_input/     # Input processing & depth
│   ├── stage2_detection/ # Object detection & segmentation  
│   ├── stage3_prompt/    # Prompt generation
│   └── stage4_generation/# Image generation
├── models/           # Downloaded model files
├── data/            # Input/output data
└── scripts/         # Utility scripts
```

### Running Tests
```bash
pytest tests/ -v
```

### Adding New Styles
```python
# In src/stage3_prompt/templates.py
STYLE_TEMPLATES["your_style"] = {
    "prompt": "your style description...",
    "negative_prompt": "things to avoid...",
    "settings": {...}
}
```

## 📈 Monitoring

### Performance Metrics
- Processing time per stage
- GPU utilization and memory usage
- Throughput (images/hour)
- Quality scores

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)

# View detailed pipeline logs
tail -f logs/pipeline.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/ backend/

# Run tests
pytest tests/ --cov=src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stable Diffusion**: Stability AI
- **ControlNet**: Lvmin Zhang
- **YOLOv8**: Ultralytics
- **SAM**: Meta AI
- **MiDaS**: Intel ISL

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**Made with ❤️ for interior design enthusiasts and AI developers**