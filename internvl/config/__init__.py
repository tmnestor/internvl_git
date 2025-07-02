"""
Configuration module for InternVL Evaluation
"""

from .config import (
    Environment,
    detect_environment,
    get_env,
    load_config,
    setup_argparse,
)

__all__ = [
    "Environment",
    "detect_environment",
    "get_env",
    "load_config",
    "setup_argparse",
]
