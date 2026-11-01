"""
Common image loading and normalization utilities for biomedical visual imaging.
Provides standardized RGB conversion, target resizing, ImageNet normalization constants,
and PIL/tensor transformation helpers.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_normalize_transform() -> T.Normalize:
    """Return PyTorch Normalize transform configured with ImageNet mean and std."""
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load an image from disk, convert to 3-channel RGB, resize to target dimensions,
    convert to float tensor [0.0, 1.0], and optionally apply ImageNet normalization.

    Args:
        image_path: Absolute or relative path to image file.
        target_size: Target (height, width) tuple. Default: (224, 224).
        normalize: Whether to apply ImageNet normalization. Default: True.

    Returns:
        PyTorch Tensor of shape (3, H, W).
    """
    try:
        with Image.open(image_path) as img:
            # Convert single-channel or RGBA images to RGB
            rgb_img = img.convert("RGB")
            
            transform_list = [
                T.Resize(target_size),
                T.ToTensor(),
            ]
            if normalize:
                transform_list.append(get_imagenet_normalize_transform())

            transform = T.Compose(transform_list)
            tensor_img = transform(rgb_img)
            return tensor_img
    except Exception as e:
        raise ValueError(f"Failed to load and preprocess image at {image_path}: {str(e)}")
