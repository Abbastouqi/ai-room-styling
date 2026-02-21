#!/usr/bin/env python3
"""
Quick Start Script for Google Colab
One-command setup and launch for AI Room Redesign Studio
"""

def quick_setup():
    """One-click setup for Google Colab"""
    print("🚀 AI Room Redesign Studio - Quick Start")
    print("=" * 50)
    
    # Check if in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("❌ This script is for Google Colab only")
        return
    
    # Step 1: Clone repository
    print("\n📥 Cloning repository...")
    import os
    if not os.path.exists('/content/ai-room-styling'):
        os.system('git clone https://github.com/Abbastouqi/ai-room-styling.git /content/ai-room-styling')
    os.chdir('/content/ai-room-styling')
    print("✅ Repository ready")
    
    # Step 2: Install dependencies
    print("\n📦 Installing dependencies...")
    os.system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
    os.system('pip install diffusers transformers accelerate ultralytics opencv-python pillow numpy flask flask-cors pyngrok')
    os.system('pip install git+https://github.com/facebookresearch/segment-anything.git')
    print("✅ Dependencies installed")
    
    # Step 3: Download models
    print("\n🔽 Downloading models...")
    import urllib.request
    os.makedirs('/content/ai-room-styling/models', exist_ok=True)
    
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    sam_path = "/content/ai-room-styling/models/sam_vit_b.pth"
    
    if not os.path.exists(sam_path):
        urllib.request.urlretrieve(sam_url, sam_path)
    print("✅ Models ready")
    
    # Step 4: Check GPU
    print("\n🔍 Checking GPU...")
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ No GPU - Enable: Runtime → Change runtime type → GPU")
    
    print("\n🎉 Setup complete! Choose your method:")
    print("1. 📱 Simple Interface - Run: simple_interface()")
    print("2. 🌐 Web Interface - Run: web_interface()")
    print("3. 🔧 Advanced Usage - Run: advanced_usage()")

def simple_interface():
    """Simple upload and process interface"""
    print("📱 Simple Interface Mode")
    print("Upload your room images/videos below:")
    
    from google.colab import files
    import sys
    import asyncio
    import matplotlib.pyplot as plt
    from PIL import Image
    
    sys.path.append('/content/ai-room-styling')
    
    # Upload files
    uploaded = files.upload()
    
    if not uploaded:
        print("❌ No files uploaded")
        return
    
    # Setup pipeline
    from src.optimized_pipeline import OptimizedPipeline, OptimizationConfig
    import torch
    
    config = OptimizationConfig(
        use_gpu=torch.cuda.is_available(),
        batch_size=4 if torch.cuda.is_available() else 2,
        use_fp16=torch.cuda.is_available(),
        cache_models=True
    )
    
    pipeline = OptimizedPipeline(config)
    
    # Process files
    print(f"\n🎨 Processing {len(uploaded)} file(s)...")
    print("⏱️ This may take 30-60 seconds with GPU")
    
    import time
    start_time = time.time()
    
    # Save uploaded files and process
    input_files = []
    for filename, content in uploaded.items():
        filepath = f"/content/{filename}"
        with open(filepath, 'wb') as f:
            f.write(content)
        input_files.append(filepath)
    
    # Process with modern style (you can change this)
    async def process_files():
        results = await pipeline.process_batch(
            input_paths=input_files,
            style="modern"  # Change to: luxury, minimal, or custom
        )
        return results
    
    results = asyncio.run(process_files())
    
    # Save results
    pipeline.save_results(results, "/content/results")
    
    processing_time = time.time() - start_time
    print(f"\n✅ Processing complete in {processing_time:.1f} seconds!")
    
    # Display results
    import os
    result_files = [f for f in os.listdir("/content/results") if f.endswith(('.jpg', '.png'))]
    
    for file in result_files:
        print(f"\n🎨 Result: {file}")
        img = Image.open(f"/content/results/{file}")
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Redesigned Room - Modern Style")
        plt.axis('off')
        plt.show()
    
    # Download results
    print("\n💾 Downloading results...")
    for file in result_files:
        files.download(f"/content/results/{file}")
    
    pipeline.cleanup()
    print("🎉 All done!")

def web_interface():
    """Launch web interface with ngrok"""
    print("🌐 Web Interface Mode")
    
    import os
    import threading
    import time
    
    # Install ngrok if needed
    os.system('pip install pyngrok')
    
    from pyngrok import ngrok
    
    # Start backend
    def start_backend():
        os.chdir('/content/ai-room-styling/backend')
        os.system('python app.py')
    
    print("🚀 Starting backend server...")
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait for backend to start
    time.sleep(5)
    
    # Create tunnels
    backend_url = ngrok.connect(5000)
    print(f"🔗 Backend API: {backend_url}")
    
    # Start frontend
    def start_frontend():
        os.chdir('/content/ai-room-styling/frontend')
        os.system('python -m http.server 8080')
    
    print("🎨 Starting frontend server...")
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    frontend_thread.start()
    
    time.sleep(3)
    
    frontend_url = ngrok.connect(8080)
    print(f"🌐 Frontend UI: {frontend_url}")
    
    print("\n🎉 Web interface ready!")
    print("📝 Note: You may need to update the API URL in the frontend")
    print("🔗 Click the Frontend UI link above to access the web interface")
    
    # Keep running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

def advanced_usage():
    """Advanced usage examples"""
    print("🔧 Advanced Usage Mode")
    
    import sys
    sys.path.append('/content/ai-room-styling')
    
    from src.optimized_pipeline import OptimizedPipeline, OptimizationConfig
    import torch
    import asyncio
    
    # Example 1: Process with different styles
    async def style_comparison():
        config = OptimizationConfig(use_gpu=torch.cuda.is_available())
        pipeline = OptimizedPipeline(config)
        
        # Download sample image
        import urllib.request
        sample_url = "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=512&h=512&fit=crop"
        urllib.request.urlretrieve(sample_url, "/content/sample_room.jpg")
        
        styles = ['modern', 'luxury', 'minimal']
        
        for style in styles:
            print(f"\n🎨 Processing with {style} style...")
            
            results = await pipeline.process_batch(
                input_paths=["/content/sample_room.jpg"],
                style=style
            )
            
            pipeline.save_results(results, f"/content/results_{style}")
            print(f"✅ {style.title()} style complete!")
        
        pipeline.cleanup()
        print("\n🎉 Style comparison complete!")
    
    # Example 2: Custom prompt
    async def custom_prompt_example():
        config = OptimizationConfig(use_gpu=torch.cuda.is_available())
        pipeline = OptimizedPipeline(config)
        
        custom_prompt = "Scandinavian living room with light wood furniture, white walls, cozy textiles, plants, natural lighting, minimalist design"
        
        results = await pipeline.process_batch(
            input_paths=["/content/sample_room.jpg"],
            style="custom",
            custom_prompt=custom_prompt
        )
        
        pipeline.save_results(results, "/content/results_custom")
        pipeline.cleanup()
        print("✅ Custom prompt example complete!")
    
    print("Available examples:")
    print("1. Style comparison: await style_comparison()")
    print("2. Custom prompt: await custom_prompt_example()")
    print("\nRun these in separate cells:")
    print("import asyncio")
    print("await style_comparison()")

# Auto-run setup when imported
if __name__ == "__main__":
    quick_setup()
else:
    # When imported in Colab
    try:
        import google.colab
        print("🏠 AI Room Redesign Studio loaded!")
        print("Run: quick_setup() to begin")
    except ImportError:
        pass