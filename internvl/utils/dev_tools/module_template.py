#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Template Script

This file serves as a template for scripts that need to enforce the
module invocation pattern.
"""

import argparse
import sys
from pathlib import Path

# Import path utilities
from internvl.utils.path import resolve_path

# Enforce module invocation pattern (uncomment and set your expected package)
# enforce_module_invocation("src.scripts")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Module template script demonstrating module invocation pattern"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file (absolute or relative to project root)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (absolute or relative to project root)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Enforce correct module invocation
    if __name__ != "__main__" or __package__ is None:
        print("Error: This script should be run as a module:")
        print("  python -m src.scripts.module_template [args]")
        sys.exit(1)

    # Parse arguments
    args = parse_arguments()

    # Resolve paths
    image_path = resolve_path("IMAGE_FOLDER_PATH", "data/images")
    if args.image:
        # Handle both absolute and relative paths
        if Path(args.image).is_absolute():
            image_path = Path(args.image)
        else:
            # If it's a relative path, resolve it relative to project root
            image_path = resolve_path("PROJECT_ROOT", ".") / args.image

    # Output path
    output_path = resolve_path("OUTPUT_PATH", "output")
    if args.output:
        if Path(args.output).is_absolute():
            output_path = Path(args.output)
        else:
            output_path = resolve_path("PROJECT_ROOT", ".") / args.output

    # Print resolved paths
    print(f"Image path: {image_path}")
    print(f"Output path: {output_path}")

    # Actual script logic would go here
    print("Script executed successfully using module invocation pattern")


if __name__ == "__main__":
    # This will only execute if the script is run directly
    # It will not execute if the script is imported as a module
    if __package__ is None:
        # If run directly, enforce module invocation
        print("Error: This script should be run as a module:")
        print("  python -m src.scripts.module_template [args]")
        print("Instead of:")
        print("  python src/scripts/module_template.py [args]")
        sys.exit(1)
    else:
        # If run as a module, execute normally
        main()
