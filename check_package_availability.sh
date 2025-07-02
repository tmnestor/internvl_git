#!/bin/bash

# Script to check package availability in local conda channels
# Usage: ./check_package_availability.sh

source /opt/conda/etc/profile.d/conda.sh

echo "=== CONDA PACKAGE AVAILABILITY CHECK ==="
echo "Checking packages from internvl_env.yml against local channels..."
echo

# Core packages
echo "## Core Python packages:"
conda search python
conda search pip
conda search setuptools
conda search wheel

# echo -e "\n## PyTorch ecosystem:"
# conda search pytorch
# conda search torchvision
# conda search pytorch-cuda
# conda search pytorch-mutex

echo -e "\n## Scientific computing:"
conda search numpy
conda search scipy
conda search pandas
conda search scikit-learn
conda search scikit-image

echo -e "\n## Image processing:"
conda search pillow
conda search opencv-python

# echo -e "\n## ML/AI libraries:"
# conda search transformers
# conda search accelerate
# conda search safetensors
# conda search timm
# conda search einops

echo -e "\n## Jupyter and development:"
conda search ipython
conda search ipykernel
conda search jupyter-client
conda search jupyter-core
conda search ipywidgets

# echo -e "\n## Visualization:"
# conda search matplotlib
# conda search rich

# echo -e "\n## Utilities:"
# conda search pyyaml
# conda search python-dotenv
# conda search requests
# conda search tqdm
# conda search click
# conda search typer

echo -e "\n## Testing and quality:"
conda search pytest
conda search pytest-cov
conda search ruff

echo -e "\n=== PIP PACKAGE AVAILABILITY CHECK ==="
echo "Checking pip packages (requires pip index access):"

echo -e "\n## PyTorch ecosystem:"
pip index versions pytorch 2>/dev/null || echo "pytorch: NOT AVAILABLE"
pip index versions torchvision 2>/dev/null || echo "torchvision: NOT AVAILABLE"
pip index versions pytorch-cuda 2>/dev/null || echo "pytorch-cuda: NOT AVAILABLE"
pip index versions pytorch-mutex 2>/dev/null || echo "pytorch-mutex: NOT AVAILABLE"

echo -e "\n## ML/AI libraries:"
pip index versions transformers 2>/dev/null || echo "transformers: NOT AVAILABLE"
pip index versions accelerate 2>/dev/null || echo "accelerate: NOT AVAILABLE"
pip index versions safetensors 2>/dev/null || echo "safetensors: NOT AVAILABLE"
pip index versions einops 2>/dev/null || echo "einops: NOT AVAILABLE"

pip index versions tokenizers 2>/dev/null || echo "tokenizers: NOT AVAILABLE"
pip index versions pydantic 2>/dev/null || echo "pydantic: NOT AVAILABLE" 
pip index versions pytesseract 2>/dev/null || echo "pytesseract: NOT AVAILABLE"
pip index versions dateparser 2>/dev/null || echo "dateparser: NOT AVAILABLE"

echo -e "\n## Visualization:"
pip index versions matplotlib 2>/dev/null || echo "matplotlib: NOT AVAILABLE"
pip index versions rich 2>/dev/null || echo "rich: NOT AVAILABLE"

echo -e "\n## Utilities:"
pip index versions pyyaml 2>/dev/null || echo "pyyaml: NOT AVAILABLE"
pip index versions python-dotenv 2>/dev/null || echo "python-dotenv: NOT AVAILABLE"
pip index versions requests 2>/dev/null || echo "requests: NOT AVAILABLE"
pip index versions tqdm 2>/dev/null || echo "tqdm: NOT AVAILABLE"
pip index versions click 2>/dev/null || echo "click: NOT AVAILABLE"
pip index versions typer 2>/dev/null || echo "typer: NOT AVAILABLE"

echo -e "\n=== CHANNEL INFORMATION ==="
echo "Available conda channels:"
conda config --show channels

echo -e "\nConfigured pip index:"
pip config list