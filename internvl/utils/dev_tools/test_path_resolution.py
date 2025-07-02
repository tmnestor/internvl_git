#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Path Resolution and Module Invocation

This script tests the path resolution and module invocation pattern.
"""

import os
import sys
from pathlib import Path

# Import our path utilities
from internvl.utils.path import enforce_module_invocation, project_root, resolve_path

# Enforce module invocation pattern
enforce_module_invocation("src.scripts")


def main():
    """
    Main function to test path resolution.
    """
    print("Testing path resolution and module invocation")
    print("=============================================")

    # Project root
    print(f"Project root: {project_root}")

    # Current working directory
    print(f"Current working directory: {Path.cwd()}")

    # Module package
    print(f"Module package: {__package__}")

    # Test resolving paths from environment variables
    print("\nPath resolution from environment variables:")
    paths_to_test = [
        "PROJECT_ROOT",
        "DATA_PATH",
        "OUTPUT_PATH",
        "SOURCE_PATH",
        "PROMPTS_PATH",
        "IMAGE_FOLDER_PATH",
        "MODEL_PATH",
    ]

    for path in paths_to_test:
        resolved = resolve_path(path)
        path_var = f"INTERNVL_{path}"
        env_value = os.environ.get(path_var, "Not set")
        print(f"{path_var}: {env_value} -> Resolved: {resolved}")

    # Test resolving paths with defaults
    print("\nPath resolution with defaults:")
    defaults = [("DEFAULT_TEST", "default/test"), ("NON_EXISTENT", None)]

    for name, default in defaults:
        resolved = resolve_path(name, default)
        print(f"resolve_path('INTERNVL_{name}', '{default}'): {resolved}")

    # Test path existence
    print("\nTesting path existence:")
    existing_paths = ["DATA_PATH", "OUTPUT_PATH", "PROMPTS_PATH"]

    for path in existing_paths:
        resolved = resolve_path(path)
        if resolved:
            exists = resolved.exists()
            print(f"{path}: {resolved} - Exists: {exists}")


if __name__ == "__main__":
    # This will only execute if the script is run directly
    # It will not execute if the script is imported as a module
    if __package__ is None:
        # If run directly, enforce module invocation
        print("Error: This script should be run as a module:")
        print("  python -m src.scripts.test_path_resolution")
        print("Instead of:")
        print("  python src/scripts/test_path_resolution.py")
        sys.exit(1)
    else:
        # If run as a module, execute normally
        main()
