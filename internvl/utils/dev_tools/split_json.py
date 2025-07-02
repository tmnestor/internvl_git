#!/usr/bin/env python3
"""
SROIE JSON Splitter

This module replaces the legacy split_json.sh script with a Python equivalent.
It splits a nested SROIE JSON file into individual files matching image filenames.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from internvl.utils.logging import get_logger, setup_logging
from internvl.utils.path import enforce_module_invocation, project_root

# Enforce module invocation pattern
enforce_module_invocation("src.scripts")

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split SROIE JSON into individual files"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/sroie/ground_truth_sroie_v5.json",
        help="Path to the input JSON file (relative to project root)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sroie/ground_truth",
        help="Path to the output directory (relative to project root)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/sroie/images",
        help="Path to the directory containing SROIE images (relative to project root)",
    )
    return parser.parse_args()


def main():
    """Split a nested JSON file into individual files matching image filenames."""
    # Set up logging
    setup_logging()

    # Parse arguments
    args = parse_args()

    # Resolve paths
    input_file = project_root / args.input_file
    output_dir = project_root / args.output_dir
    image_dir = project_root / args.image_dir

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check if input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    # Load the JSON data
    try:
        with input_file.open("r") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {input_file}")
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        return 1

    # Get list of image files
    image_pattern = re.compile(r"sroie_test_\d+\.jpg")
    try:
        image_files = [
            f.name for f in Path(image_dir).iterdir() if image_pattern.match(f.name)
        ]
        image_files.sort()
        logger.info(f"Found {len(image_files)} image files in {image_dir}")
    except Exception as e:
        logger.error(f"Error reading image directory: {e}")
        return 1

    # Process each image file
    success_count = 0
    for image_file in image_files:
        # Extract image name without extension
        image_name = Path(image_file).stem

        # Extract image index (e.g., "sroie_test_033" -> "33")
        match = re.search(r"sroie_test_0*(\d+)", image_name)
        if not match:
            logger.warning(f"Could not extract index from {image_name}")
            continue

        image_index = match.group(1)

        # Get the corresponding JSON data
        if image_index not in data:
            logger.warning(f"No data found for index {image_index}")
            continue

        # Write the JSON data to a file
        output_file = output_dir / f"{image_name}.json"
        try:
            with output_file.open("w") as f:
                json.dump(data[image_index], f, indent=2)
            logger.info(f"Created {output_file}")
            success_count += 1
        except Exception as e:
            logger.error(f"Error writing JSON for {image_name}: {e}")

    logger.info(f"Processed {success_count} out of {len(image_files)} images")
    logger.info(f"All JSON files have been created in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
