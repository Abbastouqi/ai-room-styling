"""Stage 4 generation package exports."""

from .controlnet import ControlNetConfig, controlnet_weights, load_controlnets, validate_controlnet_images
from .diffusion import DiffusionConfig, build_pipeline, generate_image, generate_with_config
from .video import images_to_video


def smoke_test(check_deps: bool = False):
    """Basic smoke test for imports; optionally check heavy deps."""
    result = {"status": "ok", "deps": {}}

    if check_deps:
        for name in ("torch", "diffusers", "cv2", "PIL"):
            try:
                __import__(name)
                result["deps"][name] = True
            except Exception:
                result["deps"][name] = False

    return result

__all__ = [
    "ControlNetConfig",
    "DiffusionConfig",
    "controlnet_weights",
    "load_controlnets",
    "validate_controlnet_images",
    "build_pipeline",
    "generate_image",
    "generate_with_config",
    "images_to_video",
    "smoke_test",
]
