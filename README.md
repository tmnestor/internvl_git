# InternVL PoC - Document Information Extraction

A Python package for extracting structured information from Australian financial documents using InternVL3 multimodal models with environment-driven configuration and ATO compliance.

## Overview

InternVL PoC processes financial documents for Australian Tax Office (ATO) work expense claims with:

- **Key-Value Extraction**: Production-ready structured data extraction (preferred over JSON)
- **Document Classification**: Automatic type detection (receipts, invoices, bank statements)
- **ATO Compliance**: Built-in Australian tax requirements validation
- **Environment-Driven Configuration**: Cross-platform compatibility via `.env` files
- **Auto Device Detection**: Optimizes for CPU, single GPU, or multi-GPU setups

## Quick Start

### 1. Environment Setup

```bash
# Clone and setup
git clone <your-repo-url>
cd internvl_PoC/internvl_git

# Create conda environment
conda env create -f internvl_env.yml
conda activate internvl_env

# Configure environment
cp .env.example .env  # Edit with your paths
```

### 2. Configuration

Edit your `.env` file with correct paths:

```bash
# Cross-platform base path (change this for your environment)
INTERNVL_BASE_PATH=/home/jovyan/nfs_share/tod
# INTERNVL_BASE_PATH=/Users/tod/Desktop/internvl_PoC

# Model path
INTERNVL_MODEL_PATH=/home/jovyan/nfs_share/models/InternVL3-8B
# INTERNVL_MODEL_PATH=/Users/tod/PretrainedLLM/InternVL3-8B

# Data paths (automatically derived from base path)
INTERNVL_IMAGE_FOLDER_PATH=${INTERNVL_BASE_PATH}/data/examples
INTERNVL_OUTPUT_PATH=${INTERNVL_BASE_PATH}/output

# Processing settings
INTERNVL_PROMPT_NAME=key_value_receipt_prompt
INTERNVL_MAX_WORKERS=6
```

### 3. Basic Usage

```bash
# Verify environment setup
python -m internvl.utils.verify_env

# Process single document (uses environment defaults)
python -m internvl.cli.internvl_single --image-path Costco-petrol.jpg

# Generate predictions for evaluation
python -m internvl.evaluation.generate_predictions

# Complete SROIE evaluation
python -m internvl.evaluation.evaluate_sroie
```

## Key Features

### ✅ Production-Ready
- **Key-Value extraction** (preferred over JSON for reliability)
- **Environment-driven configuration** (no hardcoded paths)
- **Cross-platform support** (Mac M1 local ↔ multi-GPU remote)
- **KFP-ready architecture** (data outside source directory)

### 🎯 Document Processing
- **Automatic classification** with confidence scoring (>0.8 threshold)
- **Specialized extraction** per document type
- **Business document focus** (rejects personal documents correctly)
- **Bank statement processing** with highlight detection

### 🇦🇺 Australian Tax Compliance
- **ATO requirements** validation
- **ABN verification** and GST calculations
- **Work expense categories** classification
- **Compliance scoring** for audit readiness

## Architecture

```
internvl_git/                   # Source code (git repository)
├── internvl/                   # Main package
│   ├── cli/                    # Command-line interfaces
│   ├── classification/         # Document type classification
│   ├── extraction/             # Key-Value extraction (preferred)
│   ├── evaluation/             # SROIE evaluation pipeline
│   └── config/                 # Environment configuration
├── archive/                    # Legacy code (archived)
└── prompts.yaml               # Extraction prompts

data/                          # Data (outside source, KFP-ready)
├── examples/                  # Production test images
├── sroie/                     # SROIE evaluation dataset
└── synthetic/                 # Generated test data

output/                        # Results (outside source, KFP-ready)
```

## Command Examples

### Single Document Processing
```bash
# Environment defaults (no arguments needed)
python -m internvl.cli.internvl_single

# Specific business receipts (these work)
python -m internvl.cli.internvl_single --image-path Costco-petrol.jpg
python -m internvl.cli.internvl_single --image-path Bunnings.png
python -m internvl.cli.internvl_single --image-path Target.png

# Personal documents (correctly rejected)
python -m internvl.cli.internvl_single --image-path driverlicense.jpg
# → "Classification confidence too low: 0.30 (minimum required: 0.8)"
```

### Evaluation Pipeline
```bash
# Use environment defaults (no arguments required)
python -m internvl.evaluation.generate_predictions
python -m internvl.evaluation.evaluate_sroie

# Custom paths
python -m internvl.evaluation.generate_predictions \
  --test-image-dir /path/to/images \
  --output-dir /path/to/output
```

## Environment Variables

All paths configured via `.env` file:

| Variable | Purpose | Example |
|----------|---------|---------|
| `INTERNVL_BASE_PATH` | Root directory | `/home/jovyan/nfs_share/tod` |
| `INTERNVL_MODEL_PATH` | InternVL model location | `/models/InternVL3-8B` |
| `INTERNVL_IMAGE_FOLDER_PATH` | Default image directory | `${BASE_PATH}/data/examples` |
| `INTERNVL_SROIE_DATA_PATH` | SROIE dataset | `${BASE_PATH}/data/sroie` |
| `INTERNVL_OUTPUT_PATH` | Results directory | `${BASE_PATH}/output` |
| `INTERNVL_PROMPT_NAME` | Extraction prompt | `key_value_receipt_prompt` |

## Extraction Methods

### Key-Value Extraction (Recommended)
- ✅ **Production-ready**: Robust parsing, no JSON syntax errors
- ✅ **ATO compliance**: Built-in Australian tax validation
- ✅ **Confidence scoring**: Quality assessment per extraction
- ✅ **Type-specific**: Specialized prompts per document type

```python
from internvl.extraction.key_value_parser import extract_key_value_enhanced

result = extract_key_value_enhanced(model_response)
# Returns structured data with confidence scores and validation
```

### Document Classification
- **Automatic detection**: Receipt, invoice, bank statement, other
- **Confidence thresholding**: Rejects ambiguous documents (< 0.8)
- **Business focus**: Designed for work expense claims
- **Fail-fast approach**: Clear error messages for unsuitable documents

## System Behavior

| Document Type | Confidence | Result |
|---------------|------------|--------|
| **Business receipts** | > 0.8 | ✅ Processed with specialized extraction |
| **Tax invoices** | > 0.8 | ✅ Processed with ABN/GST validation |
| **Bank statements** | > 0.8 | ✅ Processed with highlight detection |
| **Personal documents** | < 0.8 | ❌ Rejected (correct behavior) |

## Package Demo

The `internvl_package_demo.ipynb` notebook demonstrates:
- ✅ Environment-driven configuration
- ✅ Package module utilization
- ✅ Key-Value extraction testing
- ✅ Cross-platform compatibility
- ✅ CLI command examples

## Troubleshooting

### Environment Issues
```bash
# Check package installation
python -m internvl.utils.verify_env

# Test configuration loading
python -c "from internvl.config.config import load_config; print(load_config())"
```

### Path Issues
- Use **absolute paths** in `.env` for production
- Verify `INTERNVL_BASE_PATH` points to correct directory
- Check that data directories exist outside `internvl_git/`

### Model Loading
- Ensure `INTERNVL_MODEL_PATH` is correct
- System auto-detects CPU/GPU configuration
- Multi-GPU: Automatic device mapping
- Single GPU: 8-bit quantization applied

### Document Processing
- **Business documents**: Should have confidence > 0.8
- **Personal documents**: Will be rejected (this is correct)
- **Poor quality images**: May result in low confidence scores

## Advanced Usage

### Custom Evaluation
```bash
# Evaluate your own dataset
python -m internvl.evaluation.generate_predictions \
  --test-image-dir /path/to/your/images \
  --output-dir /path/to/results

python -m internvl.evaluation.evaluate_extraction \
  --predictions-dir /path/to/results \
  --ground-truth-dir /path/to/ground_truth
```

### Development Environment
```bash
# Local development (Mac M1)
uv run python -m internvl.cli.internvl_single --image-path receipt.jpg

# Remote environment (multi-GPU)
python -m internvl.cli.internvl_single --image-path receipt.jpg
```

## Migration from Legacy

If upgrading from older versions:
- ✅ **Use Key-Value extraction** instead of JSON
- ✅ **Configure via .env** instead of hardcoded paths  
- ✅ **Use package modules** instead of custom implementations
- ✅ **Leverage auto-configuration** for device detection

Legacy code has been archived to `archive/` directory for reference.

## License

MIT License - see LICENSE file for details.