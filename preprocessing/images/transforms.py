"""
PyTorch transformation pipelines for medical image datasets.
Enforces strict separation between training data augmentation and validation/testing preprocessing.
Validation and testing pipelines strictly perform non-random resizing, tensor conversion, and normalization.
"""

from typing import Tuple
import torchvision.transforms as T
from .common import IMAGENET_MEAN, IMAGENET_STD


def get_image_transforms(
    target_size: Tuple[int, int] = (224, 224),
    is_training: bool = False,
    augmentation_mode: str = "standard",
) -> T.Compose:
    """
    Construct torchvision transform pipeline for training or evaluation.

    Args:
        target_size: (Height, Width) dimensions for resizing. Default: (224, 224).
        is_training: If True, includes training data augmentation.
                     If False, includes deterministic evaluation preprocessing ONLY.
        augmentation_mode: Augmentation strategy ('standard' or 'mild').

    Returns:
        torchvision.transforms.Compose object.
    """
    if is_training:
        if augmentation_mode == "standard":
            transforms = [
                T.Resize(target_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        elif augmentation_mode == "mild":
            transforms = [
                T.Resize(target_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        else:
            raise ValueError(f"Unknown augmentation mode: {augmentation_mode}")
    else:
        # Strict deterministic validation/testing preprocessing: NO augmentation
        transforms = [
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    return T.Compose(transforms)
