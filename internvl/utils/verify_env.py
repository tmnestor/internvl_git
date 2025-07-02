#!/usr/bin/env python3
"""
Verify that the Python environment has all required packages for InternVL.
Run this after setting up a new environment to confirm it's correctly configured.

Usage: python -m src.internvl.utils.verify_env
"""

import importlib
import sys
from importlib.metadata import version

from internvl.utils.path import enforce_module_invocation

# Enforce module invocation pattern
enforce_module_invocation("src.internvl.utils")

# List of required packages and their minimum versions
REQUIRED_PACKAGES = {
    "torch": "2.0.0",
    "torchvision": "0.15.0",
    "transformers": "4.34.0",
    "einops": "0.6.0",
    "numpy": "1.20.0",
    "pillow": "9.4.0",
    "pandas": "2.0.0",
    "dateparser": "1.1.8",
    "nltk": "3.8.1",
    "scikit-learn": "1.3.0",
    "matplotlib": "3.7.0",
    "opencv-python": "4.7.0",
    "pyyaml": "6.0.0",
    "tqdm": "4.65.0",
    "ipykernel": "6.0.0",
    "ipywidgets": "8.0.0",
    "python-dotenv": "1.0.0",
}

# Optional but recommended packages
OPTIONAL_PACKAGES = {
    "ruff": "0.0.270",
    "pytest": "7.3.1",
    "pytest-cov": "4.1.0",
    "scikit-image": "0.21.0",
}


def check_packages(packages, optional=False):
    """Check if the specified packages are installed and meet version requirements."""
    missing = []
    outdated = []

    for package, min_version in packages.items():
        try:
            importlib.import_module(package)
            try:
                pkg_version = version(package)
                if pkg_version < min_version:
                    outdated.append((package, pkg_version, min_version))
                    status = "⚠️"
                else:
                    status = "✓"
                print(f"{status} {package} (v{pkg_version})")
            except Exception:
                # Can't determine version, just mark as present
                print(f"✓ {package} (version unknown)")
        except ImportError:
            missing.append(package)
            msg = "[OPTIONAL] " if optional else ""
            print(f"✗ {msg}{package}")

    return missing, outdated


def check_gpu_support():
    """Check if GPU support is available."""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "None"
            print(f"\n✓ GPU support available ({device_count} devices)")
            print(f"  Device: {device_name}")
            return True
        else:
            print("\n✗ GPU support not available (CPU only)")
            return False
    except Exception:
        print("\n✗ Could not check GPU support (error importing torch)")
        return False


def main():
    """Main verification function."""
    print("InternVL Environment Verification\n" + "=" * 32)

    print("\nChecking required packages:")
    missing, outdated = check_packages(REQUIRED_PACKAGES)

    print("\nChecking optional packages:")
    opt_missing, opt_outdated = check_packages(OPTIONAL_PACKAGES, optional=True)

    # Check GPU support
    has_gpu = check_gpu_support()

    # Print summary
    print("\nEnvironment Summary:")
    print("-" * 20)

    if not missing and not outdated:
        print("✅ All required packages installed with correct versions!")
    else:
        if missing:
            print(f"❌ Missing required packages: {', '.join(missing)}")
        if outdated:
            print("⚠️ Outdated packages:")
            for pkg, current, required in outdated:
                print(f"  - {pkg}: {current} (required: {required})")

    if opt_missing:
        print(f"ℹ️ Missing optional packages: {', '.join(opt_missing)}")

    if not has_gpu:
        print("ℹ️ Running in CPU-only mode (this is fine if no GPU is required)")

    print("\nEnvironment verification complete!")

    # Return non-zero exit code if there are missing required packages
    return len(missing) > 0


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
