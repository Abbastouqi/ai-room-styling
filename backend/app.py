"""
Flask Backend API for Room Redesign
Connects HTML frontend with optimized pipeline
"""

import asyncio
import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import time
from threading import Thread
import json

# Import optimized pipeline
import sys
sys.path.append('..')
from src.optimized_pipeline import OptimizedPipeline, OptimizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}

# Global pipeline instance
pipeline = None
active_jobs = {}  # Track active processing jobs

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_pipeline():
    """Initialize the optimized pipeline"""
    global pipeline
    
    if pipeline is None:
        logger.info("Initializing optimized pipeline...")
        
        config = OptimizationConfig(
            use_gpu=True,  # Will fallback to CPU if GPU unavailable
            batch_size=4,
            use_fp16=True,
            cache_models=True,
            parallel_stages=True,
            max_workers=4,
            memory_efficient=True
        )
        
        pipeline = OptimizedPipeline(config)
        logger.info("✅ Pipeline initialized")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pipeline_ready': pipeline is not None,
        'timestamp': time.time()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        
        # Save file
        file_path = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(file_path)
        
        # Get file info
        file_size = file_path.stat().st_size
        is_video = file_extension in ['mp4', 'avi', 'mov', 'mkv']
        
        # Estimate processing time
        estimated_time = estimate_processing_time(file_size, is_video)
        
        logger.info(f"File uploaded: {filename} -> {unique_filename}")
        
        return jsonify({
            'file_id': file_id,
            'filename': filename,
            'file_size': file_size,
            'is_video': is_video,
            'estimated_time': estimated_time
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

def estimate_processing_time(file_size_bytes, is_video, gpu_enabled=True, quality=2):
    """Estimate processing time based on file characteristics"""
    # Base time in seconds
    base_time = 45 if not is_video else 120
    
    # File size factor (larger files take longer)
    size_mb = file_size_bytes / (1024 * 1024)
    size_factor = min(2.0, 1.0 + (size_mb - 10) / 50)  # Scale with size
    
    # Quality factor
    quality_factors = {1: 0.7, 2: 1.0, 3: 1.5}
    quality_factor = quality_factors.get(quality, 1.0)
    
    # GPU acceleration
    gpu_factor = 0.2 if gpu_enabled else 1.0
    
    # Video processing overhead
    video_factor = 1.5 if is_video else 1.0
    
    estimated_time = base_time * size_factor * quality_factor * gpu_factor * video_factor
    
    return max(10, int(estimated_time))  # Minimum 10 seconds

@app.route('/api/process', methods=['POST'])
def process_file():
    """Start processing a file"""
    try:
        data = request.get_json()
        
        file_id = data.get('file_id')
        style = data.get('style', 'modern')
        custom_prompt = data.get('custom_prompt')
        options = data.get('options', {})
        
        if not file_id:
            return jsonify({'error': 'File ID required'}), 400
        
        # Check if file exists
        file_path = None
        for ext in ALLOWED_EXTENSIONS:
            potential_path = app.config['UPLOAD_FOLDER'] / f"{file_id}.{ext}"
            if potential_path.exists():
                file_path = potential_path
                break
        
        if not file_path:
            return jsonify({'error': 'File not found'}), 404
        
        # Check if already processing
        if file_id in active_jobs:
            return jsonify({'error': 'File already being processed'}), 409
        
        # Initialize pipeline if needed
        initialize_pipeline()
        
        # Create job
        job_id = str(uuid.uuid4())
        job_data = {
            'job_id': job_id,
            'file_id': file_id,
            'file_path': str(file_path),
            'style': style,
            'custom_prompt': custom_prompt,
            'options': options,
            'status': 'queued',
            'progress': 0,
            'stage': 'initializing',
            'start_time': time.time(),
            'estimated_time': estimate_processing_time(
                file_path.stat().st_size,
                file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv'],
                options.get('gpu_acceleration', True),
                options.get('quality', 2)
            )
        }
        
        active_jobs[file_id] = job_data
        
        # Start processing in background thread
        thread = Thread(target=process_file_async, args=(job_data,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started processing job {job_id} for file {file_id}")
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'estimated_time': job_data['estimated_time']
        })
        
    except Exception as e:
        logger.error(f"Process error: {e}")
        return jsonify({'error': str(e)}), 500

def process_file_async(job_data):
    """Process file asynchronously"""
    file_id = job_data['file_id']
    
    try:
        # Update status
        job_data['status'] = 'processing'
        job_data['stage'] = 'stage1'
        job_data['progress'] = 10
        
        # Run the pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            pipeline.process_batch(
                input_paths=[job_data['file_path']],
                style=job_data['style'],
                custom_prompt=job_data['custom_prompt']
            )
        )
        
        # Update progress through stages
        stages = ['stage1', 'stage2', 'stage3', 'stage4']
        for i, stage in enumerate(stages):
            job_data['stage'] = stage
            job_data['progress'] = 25 + (i * 20)
            time.sleep(0.5)  # Small delay to show progress
        
        # Save results
        output_dir = app.config['OUTPUT_FOLDER'] / file_id
        pipeline.save_results(results, output_dir)
        
        # Find output file
        output_files = list(output_dir.glob('*_redesigned.*'))
        if output_files:
            output_file = output_files[0]
            job_data['output_file'] = str(output_file)
            job_data['output_url'] = f"/api/download/{file_id}/{output_file.name}"
        
        # Complete
        job_data['status'] = 'completed'
        job_data['progress'] = 100
        job_data['stage'] = 'completed'
        job_data['end_time'] = time.time()
        job_data['actual_time'] = job_data['end_time'] - job_data['start_time']
        
        logger.info(f"Job {job_data['job_id']} completed in {job_data['actual_time']:.1f}s")
        
    except Exception as e:
        logger.error(f"Processing error for job {job_data['job_id']}: {e}")
        job_data['status'] = 'failed'
        job_data['error'] = str(e)
        job_data['end_time'] = time.time()

@app.route('/api/status/<file_id>', methods=['GET'])
def get_status(file_id):
    """Get processing status"""
    if file_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job_data = active_jobs[file_id]
    
    # Calculate elapsed time
    elapsed_time = time.time() - job_data['start_time']
    
    response = {
        'job_id': job_data['job_id'],
        'status': job_data['status'],
        'progress': job_data['progress'],
        'stage': job_data['stage'],
        'elapsed_time': int(elapsed_time),
        'estimated_time': job_data['estimated_time']
    }
    
    if job_data['status'] == 'completed':
        response['output_url'] = job_data.get('output_url')
        response['actual_time'] = job_data.get('actual_time')
    elif job_data['status'] == 'failed':
        response['error'] = job_data.get('error')
    
    return jsonify(response)

@app.route('/api/cancel/<file_id>', methods=['POST'])
def cancel_processing(file_id):
    """Cancel processing"""
    if file_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job_data = active_jobs[file_id]
    
    if job_data['status'] in ['completed', 'failed']:
        return jsonify({'error': 'Job already finished'}), 400
    
    job_data['status'] = 'cancelled'
    job_data['end_time'] = time.time()
    
    logger.info(f"Job {job_data['job_id']} cancelled")
    
    return jsonify({'status': 'cancelled'})

@app.route('/api/download/<file_id>/<filename>', methods=['GET'])
def download_result(file_id, filename):
    """Download processed result"""
    try:
        file_path = app.config['OUTPUT_FOLDER'] / file_id / filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup/<file_id>', methods=['DELETE'])
def cleanup_files(file_id):
    """Cleanup uploaded and generated files"""
    try:
        # Remove from active jobs
        if file_id in active_jobs:
            del active_jobs[file_id]
        
        # Remove uploaded file
        for ext in ALLOWED_EXTENSIONS:
            upload_path = app.config['UPLOAD_FOLDER'] / f"{file_id}.{ext}"
            if upload_path.exists():
                upload_path.unlink()
        
        # Remove output directory
        output_dir = app.config['OUTPUT_FOLDER'] / file_id
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        
        logger.info(f"Cleaned up files for {file_id}")
        
        return jsonify({'status': 'cleaned'})
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """Get system information"""
    import torch
    
    info = {
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'cpu_count': os.cpu_count(),
        'pipeline_ready': pipeline is not None
    }
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['mps_available'] = True
        info['gpu_name'] = 'Apple Silicon (MPS)'
    
    return jsonify(info)

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 100MB)'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize pipeline on startup
    initialize_pipeline()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )