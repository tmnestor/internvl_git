#!/usr/bin/env python3
"""
Test Image Path Resolution (Fail Early)

This script tests the image path resolution with explicit failure.
"""

from pathlib import Path

# Import our path utilities
from internvl.utils.path import project_root


def main():
    """Test image path resolution for different image path formats."""
    print("Testing image path resolution (fail early)")
    print("=========================================")

    # Current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")

    # Project root
    print(f"Project root: {project_root}")

    # Test image path
    test_image = "test_receipt.png"

    # Check if the image exists at path relative to CWD
    path_obj = Path(test_image)
    if path_obj.exists():
        print(f"Image found directly at: {path_obj.absolute()}")
    else:
        print(f"Image NOT found directly at: {path_obj}")

    # Check if the image exists at absolute path from project root
    alt_path = project_root / path_obj
    if alt_path.exists():
        print(f"Image found at project root: {alt_path}")
    else:
        print(f"Image NOT found at project root: {alt_path}")

    # This is the path that should be used at runtime
    print("\nTo run the inference with this image, use the command:")
    if alt_path.exists():
        print(f"python -m src.scripts.internvl_single --image-path {test_image}")
    else:
        print("Image not found in common locations. Please provide a valid path.")


if __name__ == "__main__":
    main()
