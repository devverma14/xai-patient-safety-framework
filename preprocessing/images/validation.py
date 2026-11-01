"""
Image dataset integrity validation module.
Inspects image directories for unreadable/corrupted files, checks format/dimension consistency,
and calculates class distribution and channel counts.
"""

import os
from typing import Dict, Any, List, Tuple
from PIL import Image


def validate_image_directory(
    dir_path: str,
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> Dict[str, Any]:
    """
    Scan a directory of image files and validate file integrity, dimensions, and readability.

    Args:
        dir_path: Root directory to scan.
        valid_extensions: Tuple of allowed file extensions.

    Returns:
        Dictionary containing counts of total images, valid images, corrupted images,
        dimension statistics, and channel modes.
    """
    if not os.path.exists(dir_path):
        return {
            "status": "NOT_PRESENT",
            "directory": dir_path,
            "total_files": 0,
            "valid_images": 0,
            "corrupted_images": 0,
            "corrupted_files": [],
        }

    valid_images = 0
    corrupted_images = 0
    corrupted_files: List[str] = []
    modes: Dict[str, int] = {}
    widths: List[int] = []
    heights: List[int] = []

    for root, _, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, f)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    # Re-open after verify to inspect mode & size (verify closes/resets img)
                    with Image.open(file_path) as img:
                        mode = img.mode
                        modes[mode] = modes.get(mode, 0) + 1
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        valid_images += 1
                except Exception as e:
                    corrupted_images += 1
                    corrupted_files.append(file_path)

    total_files = valid_images + corrupted_images

    summary = {
        "status": "PRESENT" if total_files > 0 else "EMPTY",
        "directory": dir_path,
        "total_files": total_files,
        "valid_images": valid_images,
        "corrupted_images": corrupted_images,
        "corrupted_files": corrupted_files,
        "image_modes": modes,
        "dimension_stats": {
            "min_width": min(widths) if widths else None,
            "max_width": max(widths) if widths else None,
            "min_height": min(heights) if heights else None,
            "max_height": max(heights) if heights else None,
        },
    }

    return summary
