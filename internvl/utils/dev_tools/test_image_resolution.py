#!/usr/bin/env python3
"""
Test Image Path Resolution

This script tests the image path resolution strategies in the internvl_single script.
"""

from pathlib import Path

# Import our path utilities
from internvl.utils.path import enforce_module_invocation, project_root

# Enforce module invocation pattern
enforce_module_invocation("src.scripts")


def main():
    """Test image path resolution for different image path formats."""
    print("Testing image path resolution")
    print("============================")

    # Current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")

    # Project root
    print(f"Project root: {project_root}")

    # Test image path
    test_image = "test_receipt.png"

    # Test different resolution strategies
    test_paths = [
        test_image,  # Relative to CWD
        f"./{test_image}",  # Explicit relative to CWD
        f"{project_root}/{test_image}",  # Relative to project root
        f"{cwd}/{test_image}",  # Absolute with CWD
    ]

    for path in test_paths:
        path_obj = Path(path)
        print(f"\nTesting path: {path}")
        print(f"  Is absolute: {path_obj.is_absolute()}")
        print(f"  Exists: {path_obj.exists()}")

        if not path_obj.is_absolute():
            # Try different resolution strategies

            # 1. Directly (might be relative to CWD)
            if path_obj.exists():
                print(f"  Found directly at: {path_obj.absolute()}")

            # 2. Relative to project root
            alt_path = project_root / path_obj
            if alt_path.exists():
                print(f"  Found relative to project root: {alt_path}")

            # 3. Just the filename in CWD
            cwd_path = cwd / path_obj.name
            if cwd_path.exists():
                print(f"  Found in CWD by name: {cwd_path}")

            # If still not found, check all common locations
            print("  Checking common locations:")
            common_locations = [
                cwd,
                project_root,
                project_root / "data",
                project_root / "data/synthetic/images",
                project_root / "data/sroie/images",
            ]

            for loc in common_locations:
                check_path = loc / path_obj.name
                if check_path.exists():
                    print(f"    Found at: {check_path}")


if __name__ == "__main__":
    main()
