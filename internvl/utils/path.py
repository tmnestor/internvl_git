"""
Path management utilities for InternVL Evaluation

This module provides centralized path resolution and management supporting both
absolute and relative paths for KFP (Kubeflow Pipelines) compatibility.

It also includes utilities for enforcing module invocation patterns for scripts.
"""

import logging
import os
import sys
from pathlib import Path

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# First make sure we can import from our own project

# Determine project root path - for both direct execution and module invocation
current_dir = Path(__file__).resolve().parent
project_root = (current_dir / "../../..").resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def enforce_module_invocation(expected_package=None):
    """
    Enforce that a script is invoked as a module using the pattern:
    `python -m module1.module2`

    This function should be called at the beginning of any script that needs
    to enforce the module invocation pattern.

    Args:
        expected_package: The expected package name for the script.
                          If None, just checks that __package__ is not None.

    Raises:
        SystemExit: If the script is not invoked as a module.
    """
    import inspect
    import sys

    # Get the caller's frame
    caller_frame = inspect.currentframe().f_back
    caller_module = inspect.getmodule(caller_frame)

    # Check if the script was invoked as a module
    if caller_module.__package__ is None:
        script_name = caller_frame.f_globals.get("__file__", "script.py")
        module_path = (
            expected_package if expected_package else "appropriate.module.path"
        )

        print(
            "Error: This script should be run using the module invocation pattern:",
            file=sys.stderr,
        )
        print(f"  python -m {module_path}", file=sys.stderr)
        print("Instead of:", file=sys.stderr)
        print(f"  python {script_name}", file=sys.stderr)
        sys.exit(1)

    # If an expected package was provided, check it matches
    if expected_package and caller_module.__package__ != expected_package:
        print(
            "Error: This script should be run using the module invocation pattern:",
            file=sys.stderr,
        )
        print(f"  python -m {expected_package}", file=sys.stderr)
        print("Instead of:", file=sys.stderr)
        print(f"  python -m {caller_module.__package__}", file=sys.stderr)
        sys.exit(1)


def resolve_path(env_var, default_relative_path=None):
    """
    Resolve a path from environment variable relative to project root.
    Supports both absolute and relative paths for KFP compatibility.

    Args:
        env_var: The environment variable name to resolve
        default_relative_path: Default relative path if env var not found

    Returns:
        Path object for the resolved path
    """
    # Since we use absolute paths in .env, relative path resolution is minimal
    # Default to current directory for any relative path needs
    proj_root = Path().absolute()

    # Get path from environment variable
    env_key = env_var if env_var.startswith("INTERNVL_") else f"INTERNVL_{env_var}"
    path_value = os.environ.get(env_key, default_relative_path)

    if not path_value:
        return None

    path_obj = Path(path_value)

    # Special case for model paths - always use as-is silently (models are typically external)
    if env_key == "INTERNVL_MODEL_PATH" and path_obj.is_absolute():
        return path_obj

    # If it's already an absolute path, return it as is (backwards compatibility)
    if path_obj.is_absolute():
        logger.warning(
            f"Using absolute path for {env_key}={path_value}. For KFP compatibility, consider using relative paths."
        )
        return path_obj

    # Return resolved absolute path relative to project root
    return proj_root / path_value


# Instead of importing get_env, we'll define a simplified version here
# to avoid circular imports
def get_env_for_path(key, default=None, required=False):
    """
    Simplified environment variable getter for path module.
    Supports both absolute paths and paths relative to project root.
    This avoids circular imports with config.py.
    """
    # Add INTERNVL_ prefix if not present
    if not key.startswith("INTERNVL_"):
        key = f"INTERNVL_{key}"

    # Try to resolve the path from environment
    path = resolve_path(key)

    # If not found in environment, use default
    if path is None:
        if required:
            # Print environment variables for debugging
            env_keys = [k for k in os.environ.keys() if k.startswith("INTERNVL_")]
            raise ValueError(
                f"Required environment variable '{key}' is not set (available: {env_keys})"
            )
        return default

    # Check if it's an output path that might need to be created
    if "OUTPUT" in key and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory for {key}: {path}")

    return str(path)


class PathManager:
    """
    Manages path resolution across different environments.

    This class provides a unified interface for resolving paths for source code,
    data, and model resources in different environments. It uses environment
    detection to determine which base paths to use.

    Attributes:
        environment (str): The detected environment
        base_paths (Dict[str, Path]): Base paths for different resource types
        env_vars (Dict[str, str]): Environment variables loaded from .env file
    """

    def __init__(self):
        """
        Initialize the PathManager with proper environment detection.
        Supports both local development and KFP (Kubeflow Pipelines) environments.
        """
        # Detect environment and set up base paths
        self.environment = get_env_for_path("ENVIRONMENT", "development")
        self.base_paths = {}
        self.env_vars = {}

        # Load environment variables
        self._load_environment_variables()

        # Configure base paths based on detected environment
        self._configure_base_paths()

        logger.info(f"PathManager initialized in {self.environment} environment")
        logger.info(f"Base paths: {self.base_paths}")
        logger.info(f"Project root: {project_root}")

    def _load_environment_variables(self):
        """Load environment variables for path resolution."""
        # Store environment variables in a dictionary for easy access
        for key, value in os.environ.items():
            if key.startswith("INTERNVL_"):
                self.env_vars[key] = value

    def _configure_base_paths(self):
        """
        Configure base paths for different resource types.

        Supports both absolute paths and paths relative to project root
        for KFP (Kubeflow Pipelines) compatibility.
        """
        # Get current module directory as fallback for source path
        module_dir = Path(__file__).parent.parent.parent

        # Configure base paths from environment variables directly (all absolute paths)
        self.base_paths = {
            "source": Path(os.environ.get("INTERNVL_SOURCE_PATH", str(module_dir))),
            "data": Path(os.environ.get("INTERNVL_INPUT_PATH", "./data")),
            "output": Path(os.environ.get("INTERNVL_OUTPUT_PATH", "./output")),
            # Models are accessed directly via INTERNVL_MODEL_PATH
        }

        # Verify all paths are valid
        missing_paths = []
        for path_type, path in self.base_paths.items():
            # Check if path is provided
            if not path:
                missing_paths.append(path_type)
                continue

        # If any critical paths are missing, raise error
        if missing_paths:
            paths_str = ", ".join([f"INTERNVL_{p.upper()}_PATH" for p in missing_paths])
            raise ValueError(
                f"Missing required paths: {paths_str}. These must be set in environment variables."
            )

        # Create output directory if it doesn't exist
        Path(self.base_paths["output"]).mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path_type: str, *parts) -> Path:
        """
        Resolve a path for the specified resource type.

        Args:
            path_type: Type of resource ("source", "data", "output")
            *parts: Additional path components to append

        Returns:
            The resolved absolute path

        Raises:
            ValueError: If the path_type is invalid
        """
        if path_type not in self.base_paths:
            raise ValueError(f"Invalid path type: {path_type}")

        path = self.base_paths[path_type]
        for part in parts:
            path = path / part

        return path

    def get_data_path(self, *parts) -> Path:
        """Get a path in the data directory."""
        return self.resolve_path("data", *parts)

    def get_output_path(self, *parts) -> Path:
        """Get a path in the output directory."""
        return self.resolve_path("output", *parts)

    def get_source_path(self, *parts) -> Path:
        """Get a path in the source directory."""
        return self.resolve_path("source", *parts)

    def get_prompt_path(self) -> Path:
        """Get path to the prompts YAML file."""
        return resolve_path("PROMPTS_PATH", "prompts.yaml")

    def get_synthetic_data_path(self) -> Path:
        """Get path to the synthetic data directory."""
        return self.get_data_path("synthetic")

    def get_synthetic_ground_truth_path(self) -> Path:
        """Get path to the synthetic ground truth directory."""
        return self.get_synthetic_data_path() / "ground_truth"

    def get_synthetic_images_path(self) -> Path:
        """Get path to the synthetic images directory."""
        return self.get_synthetic_data_path() / "images"


# Create singleton instance of PathManager
# We use a try/except to make the imports safer during development/testing
try:
    path_manager = PathManager()
except Exception as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.error(f"Error initializing PathManager: {e}")
    logger.error("Make sure your environment variables are correctly set in .env file")
    logger.error(
        "Ensure all required environment variables are set in .env file"
    )
    path_manager = None
