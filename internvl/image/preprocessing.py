"""
Image preprocessing utilities for InternVL Evaluation

This module handles image preprocessing and transformations for the InternVL model.
"""

from typing import List, Tuple

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from internvl.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    """
    Create a transformation pipeline for preprocessing images for InternVL model.

    Args:
        input_size: The target size (width and height) for the transformed image

    Returns:
        A transformation pipeline that converts images to a format suitable for the model
    """
    # Create the transform pipeline
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """
    Find the closest valid aspect ratio for image tiling.

    Args:
        aspect_ratio: The original image's aspect ratio (width/height)
        target_ratios: List of valid aspect ratios as (width, height) tuples
        width: The original image width
        height: The original image height
        image_size: The target size of each tile

    Returns:
        The best (width, height) ratio for tiling
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """
    Process images with dynamic tiling based on aspect ratio.

    Args:
        image: The input image to process
        min_num: Minimum number of tiles
        max_num: Maximum number of tiles
        image_size: Size of each square tile
        use_thumbnail: Whether to add a thumbnail as an additional tile

    Returns:
        A list of PIL.Image objects representing the image tiles
    """
    # Log beginning of preprocessing
    logger.info(
        f"Starting dynamic preprocessing with parameters: min_num={min_num}, max_num={max_num}, image_size={image_size}"
    )

    # Get and log image dimensions
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    logger.info(
        f"Original image dimensions: {orig_width}x{orig_height}, aspect ratio: {aspect_ratio:.2f}"
    )

    # Calculate all possible tiling patterns that meet the constraints
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the best tiling pattern for this image
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the dimensions for the resized image
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image to fit the target dimensions
    resized_img = image.resize((target_width, target_height))

    # Create individual tiles by cropping the resized image
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # Split the image into tiles
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    # Verify we have the expected number of tiles
    if len(processed_images) != blocks:
        logger.warning(
            f"Tile count mismatch: expected {blocks}, got {len(processed_images)}"
        )
        # Fail explicitly with clear error message
        if len(processed_images) == 0:
            raise ValueError(
                f"Failed to create any image tiles. Expected {blocks} tiles."
            )

    # Add a thumbnail of the entire image if requested
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        logger.info(f"Added thumbnail as tile #{len(processed_images)}")

    logger.info(
        f"Preprocessing complete: created {len(processed_images)} tiles with dimensions {image_size}x{image_size}"
    )
    return processed_images
