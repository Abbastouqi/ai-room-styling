"""
Optimized Pipeline for Fast Room Redesign
Performance improvements:
- GPU acceleration
- Batch processing
- Model caching
- Memory optimization
- Parallel processing
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations"""
    use_gpu: bool = True
    batch_size: int = 4
    use_fp16: bool = True  # Half precision for speed
    cache_models: bool = True
    parallel_stages: bool = True
    max_workers: int = 4
    memory_efficient: bool = True

class ModelCache:
    """Singleton model cache to avoid reloading"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str, loader_func):
        """Get cached model or load if not cached"""
        if model_name not in self._models:
            logger.info(f"Loading {model_name} into cache...")
            self._models[model_name] = loader_func()
            logger.info(f"✅ {model_name} cached")
        return self._models[model_name]
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class OptimizedPipeline:
    """
    High-performance room redesign pipeline
    Target: <1 minute for single image, <3 minutes for video
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.device = self._setup_device()
        self.model_cache = ModelCache()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Pre-load models if caching enabled
        if self.config.cache_models:
            self._preload_models()
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device (GPU/CPU)"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ Using GPU: {torch.cuda.get_device_name()}")
            
            # Optimize GPU settings
            torch.backends.cudnn.benchmark = True
            if self.config.use_fp16:
                torch.backends.cudnn.allow_tf32 = True
                
        elif self.config.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon
            logger.info("✅ Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("⚠️ Using CPU (slower)")
            
        return device
    
    def _preload_models(self):
        """Pre-load all models into cache"""
        logger.info("Pre-loading models for faster inference...")
        
        # Load models in parallel
        futures = []
        
        # MiDaS depth estimation
        futures.append(
            self.executor.submit(self._load_depth_model)
        )
        
        # YOLOv8 detection
        futures.append(
            self.executor.submit(self._load_detection_model)
        )
        
        # SAM segmentation
        futures.append(
            self.executor.submit(self._load_segmentation_model)
        )
        
        # Stable Diffusion + ControlNet
        futures.append(
            self.executor.submit(self._load_diffusion_model)
        )
        
        # Wait for all models to load
        for future in futures:
            future.result()
            
        logger.info("✅ All models pre-loaded")
    
    def _load_depth_model(self):
        """Load optimized MiDaS model"""
        def loader():
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            model.to(self.device)
            if self.config.use_fp16 and self.device.type == "cuda":
                model = model.half()
            model.eval()
            
            # Load transforms
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = transforms.small_transform
            
            return {"model": model, "transform": transform}
        
        return self.model_cache.get_model("depth", loader)
    
    def _load_detection_model(self):
        """Load optimized YOLO model"""
        def loader():
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")  # Nano for speed
            
            # Configure for optimal performance
            model.overrides['device'] = self.device
            if self.config.use_fp16 and self.device.type == "cuda":
                model.overrides['half'] = True
                
            return model
        
        return self.model_cache.get_model("detection", loader)
    
    def _load_segmentation_model(self):
        """Load optimized SAM model"""
        def loader():
            try:
                from segment_anything import sam_model_registry, SamPredictor
                
                # Use smallest SAM model for speed
                model_path = "models/sam_vit_b.pth"
                sam = sam_model_registry["vit_b"](checkpoint=model_path)
                sam.to(self.device)
                
                if self.config.use_fp16 and self.device.type == "cuda":
                    sam = sam.half()
                
                predictor = SamPredictor(sam)
                return predictor
                
            except ImportError:
                logger.warning("SAM not available, using fallback segmentation")
                return None
        
        return self.model_cache.get_model("segmentation", loader)
    
    def _load_diffusion_model(self):
        """Load optimized Stable Diffusion + ControlNet"""
        def loader():
            try:
                from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
                import torch
                
                # Load ControlNets
                depth_controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth",
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
                )
                
                seg_controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-seg",
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
                )
                
                # Create pipeline
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    controlnet=[depth_controlnet, seg_controlnet],
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
                    safety_checker=None,  # Disable for speed
                    requires_safety_checker=False
                )
                
                pipe = pipe.to(self.device)
                
                # Optimize pipeline
                if self.config.memory_efficient:
                    pipe.enable_attention_slicing()
                    pipe.enable_model_cpu_offload()
                
                # Enable xformers for speed (if available)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass
                
                return pipe
                
            except ImportError:
                logger.warning("Diffusers not available")
                return None
        
        return self.model_cache.get_model("diffusion", loader)
    
    async def process_batch(self, 
                          input_paths: List[Union[str, Path]], 
                          style: str = "modern",
                          custom_prompt: Optional[str] = None) -> List[Dict]:
        """
        Process multiple images/videos in batch for optimal performance
        
        Args:
            input_paths: List of image/video file paths
            style: Style preset (modern, luxury, minimal)
            custom_prompt: Custom prompt override
            
        Returns:
            List of results with generated images/videos
        """
        start_time = time.time()
        logger.info(f"Starting batch processing of {len(input_paths)} files")
        
        # Stage 1: Input processing and depth estimation (parallel)
        stage1_results = await self._stage1_batch(input_paths)
        
        # Stage 2: Object detection and segmentation (parallel)
        stage2_results = await self._stage2_batch(stage1_results)
        
        # Stage 3: Prompt generation (fast, sequential)
        stage3_results = await self._stage3_batch(stage2_results, style, custom_prompt)
        
        # Stage 4: Image generation (batch)
        final_results = await self._stage4_batch(stage3_results)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Batch processing completed in {total_time:.1f}s")
        
        return final_results
    
    async def _stage1_batch(self, input_paths: List[Union[str, Path]]) -> List[Dict]:
        """Stage 1: Optimized input processing and depth estimation"""
        logger.info("Stage 1: Processing inputs and generating depth maps...")
        start_time = time.time()
        
        # Load models
        depth_models = self._load_depth_model()
        
        # Process files in parallel batches
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(input_paths), batch_size):
            batch_paths = input_paths[i:i + batch_size]
            
            # Process batch in parallel
            batch_futures = []
            for path in batch_paths:
                future = self.executor.submit(self._process_single_input, path, depth_models)
                batch_futures.append(future)
            
            # Collect results
            for future in batch_futures:
                results.append(future.result())
        
        stage_time = time.time() - start_time
        logger.info(f"✅ Stage 1 completed in {stage_time:.1f}s")
        
        return results
    
    def _process_single_input(self, input_path: Union[str, Path], depth_models: Dict) -> Dict:
        """Process single input file optimized"""
        from src.stage1_input.processor import InputProcessor
        from PIL import Image
        import cv2
        
        path = Path(input_path)
        processor = InputProcessor()
        
        # Extract frames (optimized)
        if path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            frames = self._extract_frames_optimized(path)
        else:
            # Single image
            img = cv2.imread(str(path))
            img = cv2.resize(img, (512, 512))
            frames = [img]
        
        # Generate depth maps in batch
        depth_maps = self._generate_depth_batch(frames, depth_models)
        
        return {
            "input_path": str(path),
            "frames": frames,
            "depth_maps": depth_maps,
            "is_video": len(frames) > 1
        }
    
    def _extract_frames_optimized(self, video_path: Path, fps: int = 1) -> List[np.ndarray]:
        """Optimized video frame extraction"""
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Resize immediately to save memory
                frame = cv2.resize(frame, (512, 512))
                frames.append(frame)
                
                # Limit frames for performance (max 30 frames = 30 seconds at 1fps)
                if len(frames) >= 30:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _generate_depth_batch(self, frames: List[np.ndarray], depth_models: Dict) -> List[np.ndarray]:
        """Generate depth maps for batch of frames"""
        model = depth_models["model"]
        transform = depth_models["transform"]
        
        depth_maps = []
        
        # Process frames in batches for GPU efficiency
        batch_size = min(4, len(frames))  # Adjust based on GPU memory
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Prepare batch tensor
            batch_tensors = []
            for frame in batch_frames:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = transform(rgb_frame).unsqueeze(0)
                batch_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            if self.config.use_fp16 and self.device.type == "cuda":
                batch_tensor = batch_tensor.half()
            
            # Inference
            with torch.no_grad():
                batch_depth = model(batch_tensor)
            
            # Convert back to numpy
            for depth in batch_depth:
                depth_np = depth.cpu().numpy().squeeze()
                depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
                depth_np = (depth_np * 255).astype(np.uint8)
                depth_maps.append(depth_np)
        
        return depth_maps
    
    async def _stage2_batch(self, stage1_results: List[Dict]) -> List[Dict]:
        """Stage 2: Optimized object detection and segmentation"""
        logger.info("Stage 2: Detecting objects and generating masks...")
        start_time = time.time()
        
        # Load models
        detection_model = self._load_detection_model()
        segmentation_model = self._load_segmentation_model()
        
        results = []
        
        # Process each input
        for result in stage1_results:
            frames = result["frames"]
            
            # Detect objects in all frames (batch)
            detections = self._detect_objects_batch(frames, detection_model)
            
            # Generate masks (optimized)
            masks = self._generate_masks_batch(frames, detections, segmentation_model)
            
            result.update({
                "detections": detections,
                "masks": masks
            })
            results.append(result)
        
        stage_time = time.time() - start_time
        logger.info(f"✅ Stage 2 completed in {stage_time:.1f}s")
        
        return results
    
    def _detect_objects_batch(self, frames: List[np.ndarray], model) -> List[List[Dict]]:
        """Batch object detection"""
        all_detections = []
        
        # YOLO can process multiple images at once
        results = model(frames, verbose=False)
        
        for result in results:
            frame_detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    if conf > 0.5:  # Confidence threshold
                        frame_detections.append({
                            "bbox": box.tolist(),
                            "confidence": float(conf),
                            "class": int(cls),
                            "class_name": model.names[int(cls)]
                        })
            
            all_detections.append(frame_detections)
        
        return all_detections
    
    def _generate_masks_batch(self, frames: List[np.ndarray], 
                            detections: List[List[Dict]], 
                            segmentation_model) -> List[List[np.ndarray]]:
        """Optimized mask generation"""
        if segmentation_model is None:
            # Fallback: use bounding boxes as masks
            return self._generate_bbox_masks(frames, detections)
        
        all_masks = []
        
        for frame, frame_detections in zip(frames, detections):
            frame_masks = []
            
            if frame_detections:
                # Set image for SAM
                segmentation_model.set_image(frame)
                
                # Generate masks for all objects in one call
                bboxes = [det["bbox"] for det in frame_detections]
                
                if bboxes:
                    # Convert to SAM format
                    input_boxes = torch.tensor(bboxes, device=self.device)
                    
                    # Batch prediction
                    masks, _, _ = segmentation_model.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=input_boxes,
                        multimask_output=False,
                    )
                    
                    # Convert to numpy
                    for mask in masks:
                        mask_np = mask.cpu().numpy().squeeze().astype(np.uint8) * 255
                        frame_masks.append(mask_np)
            
            all_masks.append(frame_masks)
        
        return all_masks
    
    def _generate_bbox_masks(self, frames: List[np.ndarray], 
                           detections: List[List[Dict]]) -> List[List[np.ndarray]]:
        """Fallback: generate masks from bounding boxes"""
        all_masks = []
        
        for frame, frame_detections in zip(frames, detections):
            frame_masks = []
            h, w = frame.shape[:2]
            
            for detection in frame_detections:
                bbox = detection["bbox"]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                x1, y1, x2, y2 = map(int, bbox)
                mask[y1:y2, x1:x2] = 255
                
                frame_masks.append(mask)
            
            all_masks.append(frame_masks)
        
        return all_masks
    
    async def _stage3_batch(self, stage2_results: List[Dict], 
                          style: str, custom_prompt: Optional[str]) -> List[Dict]:
        """Stage 3: Fast prompt generation"""
        logger.info("Stage 3: Generating prompts...")
        start_time = time.time()
        
        from src.stage3_prompt.generator import PromptGenerator
        
        generator = PromptGenerator()
        results = []
        
        for result in stage2_results:
            frames = result["frames"]
            detections = result["detections"]
            depth_maps = result["depth_maps"]
            
            frame_prompts = []
            
            for frame, frame_detections, depth_map in zip(frames, detections, depth_maps):
                # Extract object names
                object_names = [det["class_name"] for det in frame_detections]
                
                # Generate prompt
                if custom_prompt:
                    prompt_result = generator.generate(
                        mode="prompt",
                        detected_objects=object_names,
                        image_array=frame,
                        depth_map=depth_map,
                        custom_prompt=custom_prompt
                    )
                else:
                    prompt_result = generator.generate(
                        mode="generic",
                        detected_objects=object_names,
                        image_array=frame,
                        depth_map=depth_map,
                        style=style
                    )
                
                frame_prompts.append(prompt_result)
            
            result["prompts"] = frame_prompts
            results.append(result)
        
        stage_time = time.time() - start_time
        logger.info(f"✅ Stage 3 completed in {stage_time:.1f}s")
        
        return results
    
    async def _stage4_batch(self, stage3_results: List[Dict]) -> List[Dict]:
        """Stage 4: Optimized image generation"""
        logger.info("Stage 4: Generating redesigned images...")
        start_time = time.time()
        
        # Load diffusion model
        pipe = self._load_diffusion_model()
        
        if pipe is None:
            logger.warning("Diffusion model not available, using placeholders")
            return self._generate_placeholder_results(stage3_results)
        
        results = []
        
        for result in stage3_results:
            frames = result["frames"]
            depth_maps = result["depth_maps"]
            masks = result["masks"]
            prompts = result["prompts"]
            
            generated_frames = []
            
            # Process frames in batches
            batch_size = min(2, len(frames))  # Smaller batch for diffusion
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_depths = depth_maps[i:i + batch_size]
                batch_masks = masks[i:i + batch_size]
                batch_prompts = prompts[i:i + batch_size]
                
                # Generate batch
                batch_generated = self._generate_images_batch(
                    batch_frames, batch_depths, batch_masks, batch_prompts, pipe
                )
                
                generated_frames.extend(batch_generated)
            
            result["generated_frames"] = generated_frames
            results.append(result)
        
        stage_time = time.time() - start_time
        logger.info(f"✅ Stage 4 completed in {stage_time:.1f}s")
        
        return results
    
    def _generate_images_batch(self, frames: List[np.ndarray], 
                             depth_maps: List[np.ndarray],
                             masks: List[List[np.ndarray]], 
                             prompts: List[Dict],
                             pipe) -> List[np.ndarray]:
        """Generate images in batch"""
        from PIL import Image
        
        generated = []
        
        for frame, depth_map, frame_masks, prompt_data in zip(frames, depth_maps, masks, prompts):
            # Prepare control images
            depth_image = Image.fromarray(depth_map).convert("RGB")
            
            # Combine masks into single segmentation map
            seg_map = np.zeros_like(depth_map)
            for i, mask in enumerate(frame_masks):
                seg_map[mask > 0] = (i + 1) * 50  # Different values for different objects
            
            seg_image = Image.fromarray(seg_map).convert("RGB")
            
            # Generate with ControlNet
            prompt = prompt_data.get("prompt", "modern interior design")
            negative_prompt = prompt_data.get("negative_prompt", "blurry, low quality")
            
            # Optimized generation settings
            num_steps = 15 if self.config.use_gpu else 10  # Fewer steps for speed
            
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=[depth_image, seg_image],
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                controlnet_conditioning_scale=[0.8, 0.6],  # depth, segmentation
                height=512,
                width=512,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
            
            # Convert to numpy
            generated_image = np.array(result.images[0])
            generated.append(generated_image)
        
        return generated
    
    def _generate_placeholder_results(self, stage3_results: List[Dict]) -> List[Dict]:
        """Generate placeholder results when diffusion model unavailable"""
        results = []
        
        for result in stage3_results:
            frames = result["frames"]
            
            # Create placeholder generated frames
            generated_frames = []
            for frame in frames:
                # Simple color transformation as placeholder
                placeholder = cv2.applyColorMap(frame, cv2.COLORMAP_VIRIDIS)
                generated_frames.append(placeholder)
            
            result["generated_frames"] = generated_frames
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_dir: Path):
        """Save generated results to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            input_path = Path(result["input_path"])
            is_video = result["is_video"]
            generated_frames = result["generated_frames"]
            
            if is_video:
                # Save as video
                output_path = output_dir / f"{input_path.stem}_redesigned.mp4"
                self._save_video(generated_frames, output_path)
            else:
                # Save as image
                output_path = output_dir / f"{input_path.stem}_redesigned.png"
                cv2.imwrite(str(output_path), generated_frames[0])
            
            logger.info(f"✅ Saved: {output_path}")
    
    def _save_video(self, frames: List[np.ndarray], output_path: Path, fps: int = 1):
        """Save frames as video"""
        import cv2
        
        if not frames:
            return
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        if self.config.cache_models:
            self.model_cache.clear_cache()

# Example usage
async def main():
    """Example usage of optimized pipeline"""
    config = OptimizationConfig(
        use_gpu=True,
        batch_size=4,
        use_fp16=True,
        cache_models=True,
        parallel_stages=True
    )
    
    pipeline = OptimizedPipeline(config)
    
    # Process single image
    input_files = ["data/input/room1.jpg"]
    
    results = await pipeline.process_batch(
        input_paths=input_files,
        style="modern"
    )
    
    # Save results
    pipeline.save_results(results, Path("data/output"))
    
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())