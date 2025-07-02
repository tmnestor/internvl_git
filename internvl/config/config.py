"""
Configuration management for InternVL Evaluation

This module handles loading configuration from environment variables and command-line arguments.
"""

import argparse
import ast
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type

from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Enum representing execution environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    UNKNOWN = "unknown"


def detect_environment() -> str:
    """
    Detect the current execution environment.

    Returns:
        str: The detected environment type
    """
    env_name = os.getenv("INTERNVL_ENV", "").lower()
    if env_name == "production":
        return Environment.PRODUCTION.value
    elif env_name == "staging":
        return Environment.STAGING.value
    elif env_name == "development":
        return Environment.DEVELOPMENT.value
    return Environment.UNKNOWN.value




def ensure_output_directory(output_path: str) -> Path:
    """
    Ensure output directory exists and is writable.

    Args:
        output_path: Path to output directory

    Returns:
        Path: Validated output directory path

    Raises:
        ValueError: If directory cannot be created or is not writable
    """
    output_dir = Path(output_path)
    
    try:
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test if directory is writable
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        
        logger.info(f"Output directory validated: {output_dir}")
        return output_dir
        
    except PermissionError as e:
        raise ValueError(f"Output directory not writable: {output_dir}") from e
    except OSError as e:
        raise ValueError(f"Cannot create output directory: {output_dir}") from e


def get_env(
    key: str, default: Any = None, required: bool = False, var_type: Type = str
) -> Any:
    """
    Enhanced environment variable getter with type conversion and validation.

    Args:
        key: The environment variable name to retrieve
        default: Default value if variable is not found
        required: Whether the variable is required
        var_type: Type to convert the value to
            Supported types: str, int, float, bool, list, tuple, dict

    Returns:
        The environment variable value converted to the specified type,
        or the default value if not found

    Raises:
        ValueError: If the variable is required but not found, or if type conversion fails
    """
    # Add INTERNVL_ prefix if not present
    if not key.startswith("INTERNVL_") and not key == "path":
        key = f"INTERNVL_{key}"

    # Try to get the value
    value = os.getenv(key)

    # Debug output
    logging.debug(f"Looking for env var '{key}', found: {value is not None}")

    # Handle required variables
    if value is None:
        if required:
            # Print environment variables for debugging
            env_keys = [k for k in os.environ.keys() if k.startswith("INTERNVL_")]
            logging.error(f"Required environment variable '{key}' is not set")
            logging.error(f"Available INTERNVL_ variables: {env_keys}")
            raise ValueError(
                f"Required environment variable '{key}' is not set (available: {env_keys})"
            )
        return default

    # Special handling for path variables
    if (
        (key.lower().endswith("path") or key.lower().endswith("_path"))
        and ":" in value
        and Path(value.split(":")[-1]).is_absolute()
    ):
        # If it's a path with colons, and last part is absolute path, take just the last part
        parts = value.split(":")
        original_value = value
        value = parts[-1]  # Take the last part of any PATH-style variable
        logger.warning(
            f"Fixed PATH contamination in '{key}': '{original_value}' -> '{value}'"
        )

    # Handle type conversion
    try:
        if var_type is str:
            return value
        elif var_type is int:
            return int(value)
        elif var_type is float:
            return float(value)
        elif var_type is bool:
            return value.lower() in ("true", "yes", "1", "y")
        elif var_type in (list, tuple, dict):
            # Use ast.literal_eval for safe parsing of Python literals
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, var_type):
                raise ValueError(
                    f"Parsed value {parsed} is not of type {var_type.__name__}"
                )
            return parsed
        else:
            # For custom types, attempt direct type conversion
            return var_type(value)
    except Exception as e:
        logger.error(
            f"Error converting environment variable '{key}' to {var_type.__name__}: {e}"
        )
        if required:
            raise ValueError(
                f"Failed to convert required environment variable '{key}' to {var_type.__name__}"
            ) from e
        return default


def load_config(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables and command-line arguments.

    Args:
        args: Command-line arguments that override environment variables

    Returns:
        Dict containing all configuration settings
    """
    # Load environment variables
    load_dotenv()

    # Base configuration from environment - simple path variables
    config = {
        # Path settings - use environment variables or sensible defaults
        "image_folder_path": get_env("IMAGE_FOLDER_PATH", "./data/images"),
        "output_path": get_env("OUTPUT_PATH", "./output"),
        "input_path": get_env("INPUT_PATH", "./data"),
        "model_path": get_env("MODEL_PATH", "OpenGVLab/InternVL2_5-1B"),
        "data_path": get_env("INPUT_PATH", "./data"),  # Use INPUT_PATH for consistency
        "prompts_path": get_env("PROMPTS_PATH", "./prompts.yaml"),
        # Model settings
        "image_size": get_env("IMAGE_SIZE", 448, var_type=int),
        "max_tiles": get_env("MAX_TILES", 12, var_type=int),
        "max_workers": get_env("MAX_WORKERS", 8, var_type=int),
        "max_tokens": get_env("MAX_TOKENS", 1024, var_type=int),
        "do_sample": get_env("DO_SAMPLE", False, var_type=bool),
        # Prompt settings
        "prompt_name": get_env("PROMPT_NAME", "default_receipt_prompt"),
        # Environment information
        "environment": get_env("ENVIRONMENT", detect_environment()),
        # Logging settings
        "transformers_log_level": get_env("TRANSFORMERS_LOG_LEVEL", "WARNING"),
    }

    # Override with command-line arguments if provided
    if args:
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value

    # Validate and ensure output directory exists
    try:
        validated_output_dir = ensure_output_directory(config["output_path"])
        config["output_path"] = str(validated_output_dir)
    except ValueError as e:
        logger.error(f"Output directory validation failed: {e}")
        logger.warning("Continuing with unvalidated output path")

    return config


def setup_argparse() -> argparse.ArgumentParser:
    """
    Create a base argument parser with common arguments.

    Returns:
        An ArgumentParser with common arguments
    """
    parser = argparse.ArgumentParser(description="InternVL Information Extraction")

    # Common arguments
    parser.add_argument(
        "--image-folder-path", type=str, help="Path to folder containing images"
    )
    parser.add_argument("--output-path", type=str, help="Path to output directory")
    parser.add_argument("--model-path", type=str, help="Path to model or model ID")
    parser.add_argument(
        "--prompt-name", type=str, help="Name of prompt template to use"
    )
    parser.add_argument("--image-size", type=int, help="Size of image for processing")
    parser.add_argument("--max-tiles", type=int, help="Maximum number of image tiles")
    parser.add_argument(
        "--max-workers", type=int, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens for model generation"
    )
    parser.add_argument(
        "--do-sample", action="store_true", help="Enable sampling in text generation"
    )

    return parser
