"""
Model inference utilities for InternVL Evaluation

This module handles running inference with the InternVL model.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from internvl.image.loader import load_image
from internvl.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def get_raw_prediction(
    image_path: str,
    model: Any,
    tokenizer: Any,
    prompt: str,
    generation_config: Dict[str, Any],
    device: str = "auto",
) -> str:
    """
    Run model inference on a single image and return the raw text output without post-processing.

    Args:
        image_path: Path to the image file (absolute or relative)
        model: The loaded InternVL model
        tokenizer: The loaded tokenizer
        prompt: The prompt to use for inference
        generation_config: Dictionary of generation parameters
        device: The device to run inference on ('cuda' or 'cpu')

    Returns:
        The raw text output from the model
    """
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    logger.info(f"Processing image at path: {image_path}")

    try:
        # Log the image path details for debugging
        image_path_obj = Path(image_path)
        logger.info(
            f"Processing image: {image_path_obj.name} (full path: {image_path_obj.absolute()})"
        )

        # Load and preprocess the image - use environment values if available
        image_size = int(os.environ.get("INTERNVL_IMAGE_SIZE", 448))
        max_tiles = int(os.environ.get("INTERNVL_MAX_TILES", 12))

        logger.info(
            f"Using image_size={image_size}, max_tiles={max_tiles} for preprocessing"
        )

        # Auto-detect device if needed
        actual_device = device
        if device == "auto":
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"

        pixel_values, download_time, encode_time = load_image(
            image_path=image_path,
            input_size=image_size,
            max_num=max_tiles,
            device_type=actual_device,
        )

        # Log detailed timing information
        logger.info(
            f"Image loading time: {download_time:.4f}s, encoding time: {encode_time:.4f}s"
        )

        if len(pixel_values) == 0:
            raise ValueError(f"Failed to load or process image: {image_path}")

        logger.info(f"Image processed successfully: {len(pixel_values)} tiles created")

        # Run inference
        logger.info(f"Running inference on image: {Path(image_path).name}")
        inference_start_time = time.time()

        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
        )

        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        logger.info(f"Inference completed in {inference_time:.2f}s")

        # The raw text output may be first element of tuple or direct response
        raw_output = response[0] if isinstance(response, tuple) else response

        return raw_output

    except Exception as e:
        logger.error(f"Error during inference for {image_path}: {e}")
        # Raise the exception to let the caller handle it
        raise


def run_inference_with_timing(
    model: Any,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt: str,
    generation_config: Dict[str, Any],
) -> Tuple[str, float]:
    """
    Run inference with timing measurement.

    Args:
        model: The model to use for inference
        tokenizer: The tokenizer to use
        pixel_values: Preprocessed image tensor
        prompt: The prompt to use
        generation_config: Generation parameters

    Returns:
        Tuple of (response, inference_time)
    """
    try:
        start_time = time.time()
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
        )
        end_time = time.time()
        inference_time = end_time - start_time

        # Handle different response formats
        if isinstance(response, tuple):
            return response[0], inference_time
        else:
            return response, inference_time

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return f"Error during inference: {str(e)}", 0
