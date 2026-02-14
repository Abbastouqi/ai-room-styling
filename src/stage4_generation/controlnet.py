"""Stage 4 ControlNet helpers (CPU-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class ControlNetConfig:
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    depth_model_id: str = "lllyasviel/control_v11f1p_sd15_depth"
    seg_model_id: str = "lllyasviel/control_v11p_sd15_seg"
    device: str = "cpu"
    dtype: str = "float32"  # "float32" recommended for CPU
    cache_dir: Optional[str] = None
    depth_weight: float = 0.8
    seg_weight: float = 0.6


def _resolve_torch_dtype(dtype: str):
    try:
        import torch  # local import to avoid hard dependency at import time
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for ControlNet usage.") from exc

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def load_controlnets(cfg: ControlNetConfig):
    """Load depth and segmentation ControlNet models."""
    try:
        from diffusers import ControlNetModel
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("diffusers is required for ControlNet usage.") from exc

    torch_dtype = _resolve_torch_dtype(cfg.dtype)

    depth = ControlNetModel.from_pretrained(
        cfg.depth_model_id,
        torch_dtype=torch_dtype,
        cache_dir=cfg.cache_dir,
    )
    seg = ControlNetModel.from_pretrained(
        cfg.seg_model_id,
        torch_dtype=torch_dtype,
        cache_dir=cfg.cache_dir,
    )

    return [depth, seg]


def validate_controlnet_images(
    depth_image,
    seg_image,
):
    """Validate ControlNet conditioning images (depth + segmentation)."""
    if depth_image is None or seg_image is None:
        raise ValueError("Both depth_image and seg_image are required.")

    # Avoid importing PIL/NumPy unless needed. Just basic attribute checks.
    for name, img in ("depth_image", depth_image), ("seg_image", seg_image):
        if not hasattr(img, "size") and not hasattr(img, "shape"):
            raise TypeError(f"{name} must be a PIL Image or numpy array.")

    return True


def controlnet_weights(cfg: ControlNetConfig) -> Sequence[float]:
    return [cfg.depth_weight, cfg.seg_weight]
