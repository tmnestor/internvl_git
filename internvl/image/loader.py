"""
Image loading utilities for InternVL Evaluation

This module handles loading and preprocessing images for the InternVL model.
"""

import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from internvl.image.preprocessing import build_transform, dynamic_preprocess
from internvl.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def load_image(
    image_path: str,
    input_size: int = 448,
    max_num: int = 12,
    use_thumbnail: bool = True,
    device_type: str = "auto",
) -> Tuple[torch.Tensor, float, float]:
    """
    Load and process an image for InternVL model inference.

    Args:
        image_path: The file path to the image
        input_size: Size of each image tile
        max_num: Maximum number of tiles
        use_thumbnail: Whether to include thumbnail of whole image
        device_type: Device type ("cpu", "cuda", or "auto" for auto-detection)

    Returns:
        Tuple containing:
        - pixel_values: The processed image tensors batch (moved to appropriate device and dtype)
        - download_time: Time taken to load the image from disk
        - encode_time: Time taken to process and transform the image
    """
    # Verify image file exists - fail explicitly
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Log the resolved path
    logger.info(f"Loading image from path: {path_obj.absolute()}")

    try:
        # Measure time to load the image from disk
        download_start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        download_end_time = time.time()
        download_exe_time = download_end_time - download_start_time
        logger.info(f"Image load time: {download_exe_time:.4f}s")
        logger.info(f"Image dimensions: {image.size}")

        # Measure time to process and transform the image
        encode_start_time = time.time()

        # Create the transformation pipeline
        transform = build_transform(input_size=input_size)

        # Split the image into tiles
        images = dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num
        )

        # Transform all the processed images
        pixel_values = []
        for img in images:
            pixel_values.append(transform(img))

        # Stack the tensors into a batch
        pixel_values = torch.stack(pixel_values)

        # Auto-detect device configuration if needed
        if device_type == "auto":
            if torch.cuda.is_available():
                device_type = "cuda"
            else:
                device_type = "cpu"

        # Move tensors to appropriate device and dtype
        if device_type == "cuda":
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            logger.debug("Image tensors moved to CUDA with bfloat16 precision")
        else:
            pixel_values = pixel_values.to(torch.float32)
            logger.debug("Image tensors kept on CPU with float32 precision")

        encode_end_time = time.time()
        encode_exe_time = encode_end_time - encode_start_time
        logger.debug(f"Image encoding time: {encode_exe_time:.4f}s")

        return pixel_values, download_exe_time, encode_exe_time

    except Exception as e:
        logger.error(f"Error in load_image: {e}")
        # Fail explicitly rather than returning default values
        raise


def get_image_filepaths(
    image_folder: Path, extensions: List[str] = None
) -> List[str]:
    """
    Get a list of image file paths from a folder.

    Args:
        image_folder: Path to the folder containing images
        extensions: List of valid file extensions to include

    Returns:
        List of absolute file paths to images
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]
    if not image_folder.exists():
        logger.error(f"Image folder does not exist: {image_folder}")
        return []

    # Get all files with valid extensions
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(image_folder.glob(f"*{ext}")))

    # Convert to absolute paths and sort
    abs_paths = [str(path.absolute()) for path in image_paths]
    abs_paths.sort()

    logger.info(f"Found {len(abs_paths)} images in {image_folder}")
    return abs_paths
