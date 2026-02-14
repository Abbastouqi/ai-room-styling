"""Stage 4 diffusion pipeline (Stable Diffusion + ControlNet, CPU-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .controlnet import ControlNetConfig, controlnet_weights, load_controlnets, validate_controlnet_images


@dataclass
class DiffusionConfig:
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cpu"
    dtype: str = "float32"  # "float32" recommended on CPU
    cache_dir: Optional[str] = None
    steps: int = 30
    cfg_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    enable_attention_slicing: bool = True
    disable_safety_checker: bool = True


def _resolve_torch_dtype(dtype: str):
    try:
        import torch  # local import to avoid hard dependency at import time
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for diffusion pipeline.") from exc

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def build_pipeline(diff_cfg: DiffusionConfig, cn_cfg: ControlNetConfig):
    """Create Stable Diffusion ControlNet pipeline (CPU-only)."""
    try:
        from diffusers import StableDiffusionControlNetPipeline
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("diffusers is required for diffusion pipeline.") from exc

    torch_dtype = _resolve_torch_dtype(diff_cfg.dtype)

    controlnets = load_controlnets(cn_cfg)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        diff_cfg.base_model_id,
        controlnet=controlnets,
        torch_dtype=torch_dtype,
        cache_dir=diff_cfg.cache_dir,
        safety_checker=None if diff_cfg.disable_safety_checker else None,
    )

    if diff_cfg.enable_attention_slicing:
        pipe.enable_attention_slicing()

    pipe = pipe.to(diff_cfg.device)
    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    depth_image,
    seg_image,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    cn_weights: Optional[Sequence[float]] = None,
):
    """Generate a single image using dual ControlNet conditioning."""
    validate_controlnet_images(depth_image, seg_image)

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for diffusion pipeline.") from exc

    if not prompt:
        raise ValueError("prompt is required.")

    steps = steps or 30
    cfg_scale = cfg_scale or 7.5

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg_scale),
        image=[depth_image, seg_image],
        controlnet_conditioning_scale=cn_weights,
        generator=generator,
    ).images

    if not images:
        raise RuntimeError("No images were generated.")

    return images[0]


def generate_with_config(
    diff_cfg: DiffusionConfig,
    cn_cfg: ControlNetConfig,
    prompt: str,
    negative_prompt: Optional[str],
    depth_image,
    seg_image,
):
    """Convenience wrapper: build pipeline and generate one image."""
    pipe = build_pipeline(diff_cfg, cn_cfg)
    return generate_image(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        depth_image=depth_image,
        seg_image=seg_image,
        seed=diff_cfg.seed,
        steps=diff_cfg.steps,
        cfg_scale=diff_cfg.cfg_scale,
        cn_weights=controlnet_weights(cn_cfg),
    )
