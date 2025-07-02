"""
Utility modules for InternVL Evaluation
"""

# Import setup_logging and get_logger directly
from internvl.utils.logging import get_logger, setup_logging

# Define what we export first
__all__ = ["setup_logging", "get_logger"]

# Import path module separately to avoid circular imports
try:
    from internvl.utils.path import PathManager, path_manager

    __all__ += ["PathManager", "path_manager"]
except ImportError as e:
    # Log the error but continue
    import logging

    logging.getLogger(__name__).warning(f"Could not import PathManager: {e}")
    # Define dummy values to prevent import errors
    PathManager = None
    path_manager = None
