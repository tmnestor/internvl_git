#!/usr/bin/env python3
"""
SROIE Evaluation Script

This module replaces the legacy evaluate_sroie.sh script with a Python equivalent.
It runs the complete evaluation pipeline for the SROIE dataset:
1. Generates predictions for all SROIE images
2. Evaluates predictions against ground truth
3. Outputs metrics and visualizations
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

from internvl.utils.logging import get_logger, setup_logging
from internvl.utils.path import enforce_module_invocation

# Enforce module invocation pattern
enforce_module_invocation("internvl.evaluation")

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run SROIE evaluation pipeline")
    parser.add_argument(
        "--prompt-name",
        type=str,
        default=None,
        help="Prompt name to use (if not provided, uses value from .env)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="evaluation_sroie",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Show example predictions in output",
    )
    return parser.parse_args()


def get_env_prompt_name():
    """Get the prompt name from the .env file."""
    try:
        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            return None

        with env_path.open("r") as f:
            for line in f:
                if line.strip().startswith("INTERNVL_PROMPT_NAME="):
                    return line.strip().split("=", 1)[1].strip()
        return None
    except Exception as e:
        logger.error(f"Error reading prompt name from .env: {e}")
        return None


def run_module(module_name, args):
    """Run a Python module using the module invocation pattern."""
    cmd = [sys.executable, "-m", module_name] + args
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {module_name}: {e}")
        return False


def main():
    """Run the SROIE evaluation pipeline."""
    # Set up logging
    setup_logging()

    # Parse arguments
    args = parse_args()

    # Get prompt name (from args or .env)
    prompt_name = args.prompt_name
    if not prompt_name:
        prompt_name = get_env_prompt_name()
        if prompt_name:
            logger.info(f"Using prompt from .env: {prompt_name}")
        else:
            prompt_name = "default_receipt_prompt"
            logger.warning(f"No prompt name provided. Using default: {prompt_name}")

    # Set paths
    project_root = Path.cwd()
    sroie_image_dir = project_root / "data/sroie/images"
    predictions_dir = project_root / "output/predictions_sroie"
    ground_truth_dir = project_root / "data/sroie/ground_truth"

    # Create predictions directory if it doesn't exist
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)

    # Generate timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"{args.output_prefix}_{timestamp}"
    output_path = project_root / "output" / output_prefix

    # Print header and environment info
    logger.info("=" * 60)
    logger.info("SROIE Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Using prompt: {prompt_name}")

    # Step 1: Generate predictions
    logger.info("Step 1: Generating predictions on SROIE images...")

    generate_args = [
        "internvl.evaluation.generate_predictions",
        "--test-image-dir",
        str(sroie_image_dir),
        "--output-dir",
        str(predictions_dir),
    ]

    if prompt_name:
        generate_args.extend(["--prompt-name", prompt_name])

    if not run_module("internvl.evaluation.generate_predictions", generate_args[1:]):
        logger.error("Prediction generation failed!")
        return 1

    # Count prediction files
    prediction_files = list(predictions_dir.glob("*.json"))
    logger.info(f"Generated {len(prediction_files)} prediction files")

    # Step 2: Evaluate predictions
    logger.info("Step 2: Evaluating predictions against ground truth...")

    evaluate_args = [
        "internvl.evaluation.evaluate_extraction",
        "--predictions-dir",
        str(predictions_dir),
        "--ground-truth-dir",
        str(ground_truth_dir),
        "--output-path",
        str(output_path),
    ]

    if args.show_examples:
        evaluate_args.append("--show-examples")

    if not run_module("internvl.evaluation.evaluate_extraction", evaluate_args[1:]):
        logger.error("Evaluation failed!")
        return 1

    # Print success message
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)
    logger.info("Results saved to:")
    logger.info(f"  - CSV: {output_path}.csv")
    logger.info(f"  - JSON: {output_path}.json")
    logger.info(f"  - Visualization: {output_path}.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
