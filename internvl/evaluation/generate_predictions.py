#!/usr/bin/env python3
"""
InternVL Generate Predictions

This script generates predictions on a set of test images for evaluation.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Import from the src directory structure
# This is the correct import path when running as a module
from internvl.config import load_config, setup_argparse
from internvl.extraction.normalization import post_process_prediction
from internvl.image.loader import get_image_filepaths
from internvl.model import load_model_and_tokenizer
from internvl.model.inference import get_raw_prediction
from internvl.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = setup_argparse()
    parser.add_argument(
        "--test-image-dir",
        type=str,
        required=True,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save prediction files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main function for generating predictions.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Load configuration
    config = load_config(args)

    # Get transformers log level from config
    transformers_log_level = config.get("transformers_log_level", "WARNING")

    # Setup logging with appropriate levels
    setup_logging(log_level, transformers_log_level=transformers_log_level)
    logger = get_logger(__name__)

    logger.info(f"Generating predictions for images in: {args.test_image_dir}")

    try:
        # Ensure output directory exists
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get image files
        test_image_dir = Path(args.test_image_dir)
        if not test_image_dir.exists():
            logger.error(f"Test image directory not found: {test_image_dir}")
            return 1

        image_paths = get_image_filepaths(test_image_dir)
        if not image_paths:
            logger.error(f"No images found in {test_image_dir}")
            return 1

        logger.info(f"Found {len(image_paths)} images to process")

        # Get the prompt
        try:
            import yaml

            prompts_path = config.get("prompts_path")
            prompt_name = config.get("prompt_name")

            if prompts_path and Path(prompts_path).exists():
                with Path(prompts_path).open("r") as f:
                    prompts = yaml.safe_load(f)
                prompt = prompts.get(prompt_name, "")
                logger.info(f"Using prompt '{prompt_name}' from {prompts_path}")
            else:
                prompt = "<image>\nExtract information from this receipt and return it in JSON format."
                logger.warning("Prompts file not found, using default prompt")
        except Exception as e:
            logger.error(f"Error loading prompt: {e}")
            prompt = "<image>\nExtract information from this receipt and return it in JSON format."

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_path=config["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Generate predictions
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generation_config = {
            "max_new_tokens": config.get("max_tokens", 1024),
            "do_sample": config.get("do_sample", False),
        }

        # Process each image
        start_time = time.time()
        total_images = len(image_paths)
        success_count = 0
        error_count = 0

        for i, image_path in enumerate(image_paths, 1):
            image_id = Path(image_path).stem
            logger.info(f"Processing [{i}/{total_images}]: {image_id}")

            try:
                # Get raw prediction
                raw_output = get_raw_prediction(
                    image_path=image_path,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    generation_config=generation_config,
                    device=device,
                )

                # Process and normalize
                processed_json = post_process_prediction(raw_output)

                # Save prediction
                output_file = output_dir / f"{image_id}.json"
                with output_file.open("w") as f:
                    json.dump(processed_json, f, indent=2)

                success_count += 1

            except Exception as e:
                logger.error(f"Error processing {image_id}: {e}")
                error_count += 1

                # Save error info
                error_file = output_dir / f"{image_id}_error.json"
                with error_file.open("w") as f:
                    json.dump({"error": str(e)}, f, indent=2)

        # Print statistics
        elapsed_time = time.time() - start_time
        logger.info("Finished generating predictions")
        logger.info(f"Total time: {elapsed_time:.2f}s")
        logger.info(f"Images processed: {total_images}")
        logger.info(f"Successes: {success_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Predictions saved to {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
