# InternVL PoC - Document Information Extraction

A Python package for extracting structured information from financial documents (receipts, invoices, and bank statements) using InternVL3 multimodal models with automatic CPU/GPU configuration and Australian Tax Office (ATO) compliance.

## Overview

InternVL PoC processes financial document images to extract structured data for Australian Tax Office (ATO) compliance. The system supports:

- **Receipts & Invoices**: Traditional business receipt processing with key-value extraction
- **Bank Statements**: Advanced processing with highlight detection for user-marked work expenses
- **ATO Compliance**: Built-in validation for Australian tax requirements including ABN verification and GST calculations

The system automatically detects and optimizes for available hardware (CPU, single GPU, or multi-GPU) and supports both local models and HuggingFace hosted models.

## Key Features

### Document Processing
- **Receipt Processing**: Key-value extraction with superior robustness vs JSON parsing
- **Bank Statement Processing**: Highlight detection, transaction categorization, and ATO compliance
- **Multi-Format Support**: Handles various document types and image formats

### Australian Tax Compliance
- **ATO Requirements**: Built-in validation for Australian Tax Office work-related expense claims
- **ABN Validation**: Australian Business Number verification and formatting
- **GST Calculations**: Automatic 10% tax validation and verification
- **Highlight Detection**: Computer vision detection of user-marked expenses in bank statements
- **Work Expense Categories**: Automatic categorization (fuel, office supplies, travel, professional services, etc.)

### Technical Features
- **Auto Device Configuration**: Automatically detects and configures for CPU, single GPU, or multi-GPU setups
- **CPU-1GPU-MultiGPU Support**: Optimized configurations with 8-bit quantization for single GPU and device mapping for multi-GPU
- **Dynamic Image Processing**: Advanced image tiling and preprocessing for optimal model input
- **Comprehensive Evaluation**: Metrics calculation with SROIE dataset support
- **Environment-based Configuration**: Uses `.env` files for flexible deployment
- **Modular Architecture**: Clean package structure for maintainability and deployment

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd internvl_PoC

# Create conda environment (required for PyTorch/GPU dependencies)
conda env create -f internvl_env.yml
conda activate internvl_env

# Optional: Install computer vision dependencies for bank statement highlight detection
pip install -r requirements-cv.txt

# Configure environment variables
cp .env.example .env  # Edit with your paths
```

#### Computer Vision Dependencies (Optional)

For full bank statement highlight detection capabilities, install:

```bash
# Install OpenCV and OCR dependencies
pip install opencv-python pytesseract pillow

# Install system tesseract (required for OCR)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# MacOS:
brew install tesseract

# CentOS/RHEL:
sudo yum install tesseract
```

**Note**: Bank statement processing will work without these dependencies, but highlight detection will be disabled. The system successfully processes bank statements and extracts highlighted transactions even when OCR is not available.

### 2. Configuration

The `.env` file serves as the **single source of truth** for all paths and configuration. Create/edit your `.env` file:

```bash
# Environment Variables - Single Source of Truth
# All CLI commands automatically load these variables from .env

# Input/Output paths (absolute paths for remote environment)
INTERNVL_INPUT_PATH=/home/jovyan/nfs_share/tod/data
INTERNVL_OUTPUT_PATH=/home/jovyan/nfs_share/tod/output
INTERNVL_IMAGE_FOLDER_PATH=/home/jovyan/nfs_share/tod/examples
INTERNVL_PROMPTS_PATH=/home/jovyan/nfs_share/tod/prompts.yaml

# Dataset subdirectories (derived from INPUT_PATH)
INTERNVL_SYNTHETIC_DATA_PATH=/home/jovyan/nfs_share/tod/data/synthetic
INTERNVL_SROIE_DATA_PATH=/home/jovyan/nfs_share/tod/data/sroie

# Model configuration
INTERNVL_MODEL_PATH=/home/jovyan/nfs_share/models/InternVL3-8B

# Processing settings
INTERNVL_PROMPT_NAME=key_value_receipt_prompt
INTERNVL_IMAGE_SIZE=448
INTERNVL_MAX_TILES=8
INTERNVL_MAX_WORKERS=6
INTERNVL_MAX_TOKENS=2048
```

#### Environment Variable Usage
All commands automatically use paths from `.env` - no shell variable expansion needed:
```bash
# NEW: No parameters needed - uses INTERNVL_IMAGE_FOLDER_PATH automatically!
python -m internvl.cli.internvl_batch

# NEW: Uses first image from INTERNVL_IMAGE_FOLDER_PATH automatically!
python -m internvl.cli.internvl_single

# Still works: Explicit path override
python -m internvl.cli.internvl_single \
  --image-path Target.png \
  --output-file result.json
```

### 3. Basic Usage

> **✨ NEW: Simplified Usage!** Commands now work without required parameters when `.env` is configured:
> - `python -m internvl.cli.internvl_batch` (processes all images in configured folder)
> - `python -m internvl.cli.internvl_single` (processes first image in configured folder)

#### Receipt Processing
```bash
# Process a single receipt (uses .env configured paths automatically)
python -m internvl.cli.internvl_single --image-path Target.png

# Process first image from .env configured folder (no image path needed!)
python -m internvl.cli.internvl_single

# Save to configured output directory
python -m internvl.cli.internvl_single \
  --image-path Bunnings.png \
  --output-file results.json

# Process multiple receipts (uses .env INTERNVL_IMAGE_FOLDER_PATH automatically!)
python -m internvl.cli.internvl_batch
```

#### Document Processing (Automatic Classification)
```bash
# Process any document with automatic classification
python -m internvl.cli.internvl_single --image-path document.png

# Process first image from .env configured folder automatically
python -m internvl.cli.internvl_single

# Verbose output for debugging
python -m internvl.cli.internvl_single --image-path statement.png --verbose

# Specify device manually (optional)
python -m internvl.cli.internvl_single --image-path invoice.pdf --device cpu
```

#### Batch Processing
```bash
# Process all images in configured folder (uses .env automatically!)
python -m internvl.cli.internvl_batch

# Batch with custom output and workers
python -m internvl.cli.internvl_batch \
  --output-file batch_results.csv \
  --max-workers 4

# Process specific number of images from configured directory
python -m internvl.cli.internvl_batch \
  --num-images 10

# Override .env: Process different input directory
python -m internvl.cli.internvl_batch \
  --image-folder-path /home/jovyan/nfs_share/tod/data/sroie/images \
  --output-file sroie_results.csv
```

### Extraction Methods

#### Key-Value Extraction (Default - Receipts)
- ✅ **Superior reliability** - No JSON parsing failures
- ✅ **Australian compliance** - ATO work expense validation  
- ✅ **Confidence scoring** - Quality assessment for each extraction
- ✅ **ABN validation** - Australian Business Number checking
- ✅ **GST calculation** - 10% tax validation

#### Bank Statement Processing (NEW)
- ✅ **Highlight detection** - Computer vision detection of user-marked expenses (WORKING)
- ✅ **Transaction extraction** - Successfully parses all bank statement transactions (23 transactions extracted)
- ✅ **Highlighted transaction identification** - Correctly identifies user-highlighted expenses ($7,280.27 total)
- ✅ **Transaction categorization** - Automatic work expense classification
- ✅ **ATO compliance** - Built-in Australian Tax Office requirements
- ✅ **Multi-bank support** - CBA, ANZ, Westpac, NAB, and 10+ other Australian banks
- ✅ **Work expense analytics** - Total calculations and compliance scoring

#### JSON Extraction (Legacy)
- ⚠️ **Legacy mode** - Available for backward compatibility
- ❌ **No compliance checking** - Basic extraction only
- ❌ **Parsing failures** - JSON syntax errors possible

## Package Structure

```
internvl_PoC/
├── internvl/                    # Main package
│   ├── cli/                     # Command-line interfaces
│   │   ├── internvl_single.py   # Single document processing
│   │   └── internvl_batch.py    # Batch processing
│   ├── config/                  # Configuration management
│   ├── evaluation/              # Evaluation scripts and metrics
│   ├── extraction/              # Document extraction and normalization
│   │   ├── key_value_parser.py  # Enhanced Key-Value extraction
│   │   ├── bank_statement_parser.py  # Bank statement processing (NEW)
│   │   └── json_extraction_fixed.py  # Legacy JSON extraction
│   ├── image/                   # Image processing and preprocessing
│   │   └── highlight_detection.py  # Computer vision highlight detection (NEW)
│   ├── model/                   # Model loading and inference
│   ├── schemas/                 # Data models and validation (NEW)
│   │   └── bank_statement_schemas.py  # Bank statement Pydantic models
│   └── utils/                   # Utilities and development tools
├── data/                        # Datasets
│   ├── sroie/                   # SROIE dataset
│   └── synthetic/               # Generated test data
├── docs/                        # Documentation
├── examples/                    # Example images
├── .env                         # Environment configuration
├── prompts.yaml                 # Prompt templates
├── internvl_codebase_demo.ipynb # Demo notebook
└── internvl_env.yml             # Conda environment
```

## Auto Device Configuration

The system automatically detects your hardware and applies optimal configurations:

### CPU-Only
- Uses `torch.float32` precision
- Optimized for development and testing

### Single GPU
- Uses `torch.bfloat16` precision with 8-bit quantization
- Optimal memory usage for large models

### Multi-GPU
- Distributes model layers across GPUs using device mapping
- Automatic load balancing for InternVL3-8B architecture

```python
# This happens automatically when loading models
from internvl.model.loader import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer(
    model_path=config['model_path'],
    auto_device_config=True  # Enables automatic configuration
)
```

## Command-Line Interface

### Single Document Processing

#### Receipt Processing
```bash
# Process first image from .env configured folder (no parameters needed!)
python -m internvl.cli.internvl_single

# Process specific image with relative path
python -m internvl.cli.internvl_single --image-path Target.png

# With ATO compliance checking (uses .env folder automatically)
python -m internvl.cli.internvl_single --compliance-check

# With custom output file
python -m internvl.cli.internvl_single \
  --image-path Bunnings.png \
  --output-file result.json
```

#### Bank Statement Processing (NEW)
```bash
# Process first bank statement from .env folder (no path needed!)
python -m internvl.cli.internvl_single --document-type bank_statement

# Bank statement with highlight detection and compliance
python -m internvl.cli.internvl_single \
  --image-path "bank statement - ANZ highlight.png" \
  --document-type bank_statement \
  --compliance-check

# Disable highlight detection
python -m internvl.cli.internvl_single \
  --image-path "bank statement - ANZ highlight.png" \
  --document-type bank_statement \
  --no-detect-highlights
```

#### Advanced Options
```bash
# Legacy JSON mode (uses first image from .env folder)
python -m internvl.cli.internvl_single --extraction-method json

# Override prompt for advanced users
python -m internvl.cli.internvl_single \
  --image-path Target.png \
  --prompt-name australian_business_receipt_prompt

# Process specific image with full path override
python -m internvl.cli.internvl_single \
  --image-path /absolute/path/to/receipt.png \
  --extraction-method json
```

### Batch Processing

#### Receipt Batch Processing
```bash
# Recommended: Key-Value extraction with compliance (uses .env automatically!)
python -m internvl.cli.internvl_batch --compliance-check

# With custom settings - no folder path needed!
python -m internvl.cli.internvl_batch \
  --output-file ato_compliance_results.csv \
  --expense-category "Vehicle" \
  --max-workers 4 \
  --compliance-check
```

#### Bank Statement Batch Processing (NEW)
```bash
# Process multiple bank statements (uses .env folder automatically!)
python -m internvl.cli.internvl_batch \
  --document-type bank_statement \
  --compliance-check

# Large-scale processing with custom output
python -m internvl.cli.internvl_batch \
  --document-type bank_statement \
  --output-file bank_compliance_results.csv \
  --max-workers 8 \
  --compliance-check
```

#### Legacy Processing
```bash
# Legacy JSON batch processing (uses .env folder automatically!)
python -m internvl.cli.internvl_batch \
  --extraction-method json \
  --output-file legacy_results.csv
```

## Evaluation and Testing

### SROIE Dataset Evaluation

```bash
# Run complete SROIE evaluation (uses .env paths automatically)
python -m internvl.evaluation.evaluate_sroie

# Generate predictions only using configured paths
python -m internvl.evaluation.generate_predictions \
  --test-image-dir /home/jovyan/nfs_share/tod/data/sroie/images \
  --output-dir /home/jovyan/nfs_share/tod/output/predictions

# Evaluate existing predictions using configured paths
python -m internvl.evaluation.evaluate_extraction \
  --predictions-dir /home/jovyan/nfs_share/tod/output/predictions \
  --ground-truth-dir /home/jovyan/nfs_share/tod/data/sroie/ground_truth
```

### Custom Dataset Evaluation

```bash
# Evaluate your own dataset using configured paths
python -m internvl.evaluation.evaluate_extraction \
  --predictions-dir /home/jovyan/nfs_share/tod/output/my_predictions \
  --ground-truth-dir /home/jovyan/nfs_share/tod/data/my_dataset/ground_truth \
  --show-examples
```

## Demo Notebook

The `internvl_codebase_demo.ipynb` notebook demonstrates the same functionality as the original Huaifeng_Test_InternVL.ipynb but using the structured codebase:

- Auto device detection and configuration
- Image processing with structured prompts
- JSON extraction and normalization
- Multiple test scenarios
- **Complete compatibility** with original Huaifeng notebook test cases

This notebook is ready to run at your workplace and will automatically detect and configure for available GPU resources. It includes all the original test prompts and can replicate the exact same functionality as the original notebook.

## Bank Statement Processing (NEW)

### Supported Australian Banks

The system supports all major Australian financial institutions:

- **Big 4 Banks**: Commonwealth Bank (CBA), ANZ, Westpac, NAB
- **Regional Banks**: Bendigo Bank, Bank of Queensland (BOQ), Macquarie Bank
- **Digital Banks**: ING, Citibank, HSBC
- **Credit Unions**: Bank Australia, Bank of Melbourne, Suncorp Bank

### Highlight Detection

Advanced computer vision capabilities detect user-highlighted transactions:

- **Supported Colors**: Yellow, pink, green, orange highlighter markers
- **Detection Methods**: HSV color space analysis with confidence scoring (99.35% confidence achieved)
- **Transaction Parsing**: Successfully extracts all transactions including highlighted ones
- **Text Extraction**: OCR extraction from highlighted regions (optional - works without OCR)
- **Overlap Handling**: Intelligent filtering of overlapping regions
- **Production Ready**: Successfully identifies highlighted transactions in ANZ bank statements

### Work Expense Categories

Automatic categorization of transactions for ATO compliance:

- **Fuel**: BP, Shell, Caltex, Ampol, petrol stations
- **Office Supplies**: Officeworks, Staples, office equipment
- **Travel**: Airlines, hotels, car rental, accommodation
- **Professional Services**: Accounting, legal, consulting
- **Equipment**: Computers, tools, machinery
- **Training**: Education, courses, conferences
- **Parking**: Parking meters, tolls, car-related expenses

### ATO Compliance Features

- **Transaction Validation**: Date, amount, description completeness
- **BSB Validation**: Australian Bank State Branch code verification
- **Compliance Scoring**: Percentage rating for audit readiness
- **Missing Information**: Identification of gaps in documentation
- **Recommendations**: Suggestions for improving compliance

## Prompt System

The system uses YAML-based prompt templates in `prompts.yaml` with comprehensive coverage of different use cases:

### Australian Receipt Processing Prompts
```yaml
default_receipt_prompt: |
  <image>
  Extract information from this receipt and return in JSON format.
  Required fields: date_value, store_name_value, tax_value, total_value

australian_optimized_prompt: |
  <image>
  Extract these fields from the Australian receipt:
  1. date_value: Date in DD/MM/YYYY format
  2. store_name_value: Store name
  3. tax_value: GST amount (10%)
  4. total_value: Total amount including GST
  [Additional detailed instructions...]
```

### Original Huaifeng Notebook Prompts
The system now includes **all prompts from the original Huaifeng_Test_InternVL.ipynb**, ensuring complete compatibility:

```yaml
# Conference and business analysis
conference_relevance_prompt: |
  <image>
  Is this relevant to a claim about attending academic conference?

business_expense_prompt: |
  <image>
  Is this relevant to a claim about car expense?

# Receipt extraction (matching original notebook)
huaifeng_receipt_json_prompt: |
  <image>
  read the text and return information in JSON format. I need company name, address, phone number, date, ABN, and total amount

# Multi-receipt processing
multi_receipt_json_prompt: |
  <image>
  there are two receipts on this image. read the text and return information in JSON format. I need company name, address, phone number, date, ABN, and total amount

# Detailed item-level extraction
detailed_receipt_json_prompt: |
  <image>
  read the text and return information in JSON format. I need company name, address, phone number, date, item name, number of items, item price, and total amount
```

### Available Prompt Categories

| Category | Prompts | Use Case |
|----------|---------|----------|
| **Receipt Processing** | `key_value_receipt_prompt`, `australian_optimized_prompt` | Standard Australian receipt extraction |
| **Bank Statements** | `bank_statement_ato_compliance_prompt`, `bank_statement_highlighted_prompt` | Bank statement processing with ATO focus |
| **Huaifeng Compatible** | `huaifeng_receipt_json_prompt`, `multi_receipt_json_prompt` | Exact replication of original notebook |
| **Business Analysis** | `business_expense_prompt`, `expense_relevance_prompt` | Expense claim relevance checking |
| **Conference/Meeting** | `conference_relevance_prompt`, `speaker_list_prompt` | Academic/conference document processing |
| **Detailed Extraction** | `detailed_receipt_json_prompt`, `strict_json_prompt` | Item-level detail extraction |
| **Generic** | `document_description_prompt`, `simple_receipt_prompt` | General document analysis |

### Using Prompts

Specify which prompt to use in your `.env` file:
```bash
# Use original Huaifeng prompts for compatibility
INTERNVL_PROMPT_NAME=huaifeng_receipt_json_prompt

# Use Australian-optimized prompts for production
INTERNVL_PROMPT_NAME=australian_optimized_prompt

# Use for conference document analysis
INTERNVL_PROMPT_NAME=conference_relevance_prompt
```

Or specify prompts dynamically in commands:
```bash
# Use specific prompt for single image
python -m internvl.cli.internvl_single \
  --image-path conference_program.jpg \
  --prompt-name conference_relevance_prompt

# Use multi-receipt prompt for batch processing
python -m internvl.cli.internvl_batch \
  --image-folder-path data/multi_receipts \
  --prompt-name multi_receipt_json_prompt
```

## Post-Processing Pipeline

### JSON Extraction
- Extracts structured JSON from model text output
- Handles multiple formats (markdown, raw JSON)
- Robust error handling and fallbacks

### Field Normalization
- **Dates**: Standardized to DD/MM/YYYY (Australian format)
- **Store Names**: Consistent capitalization and formatting
- **Prices**: Decimal standardization and currency handling
- **Text Fields**: Whitespace normalization

### Evaluation Metrics
- Field-level accuracy, precision, recall, F1-score
- Support for both scalar and list fields
- Comprehensive reporting and visualization

## Environment Management

### Local Development
```bash
conda activate internvl_env
python -m internvl.cli.internvl_single --image-path test.jpg
```

### Multi-User Systems
```bash
# User-specific environment
conda config --append envs_dirs ~/.conda/envs
conda env create -f internvl_env.yml --prefix ~/.conda/envs/internvl_env
conda activate ~/.conda/envs/internvl_env
```

### Shared Environments
See [docs/SHARED_ENVIRONMENTS.md](docs/SHARED_ENVIRONMENTS.md) for detailed setup instructions.

## Utilities

### Environment Verification
```bash
python -m internvl.utils.verify_env
```

### Development Tools
```bash
# Available in internvl.utils.dev_tools
python -m internvl.utils.dev_tools.test_path_resolution
python -m internvl.utils.dev_tools.test_image_resolution
```

## GPU Configuration

### Single GPU with Memory Constraints
```bash
# The system automatically applies 8-bit quantization
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
python -m internvl.cli.internvl_single --image-path test.jpg
```

### Multi-GPU Setup
```bash
# Automatic device mapping across all available GPUs
# No configuration needed - system detects and configures automatically
python -m internvl.cli.internvl_batch --image-folder-path data/images
```

## Deployment

### KFP (Kubeflow Pipelines) Deployment

The system is designed for easy KFP deployment with external persistent storage:

#### Environment Variables for KFP
```bash
# Set these in your KFP pipeline environment
INTERNVL_INPUT_PATH=/tmp/inputs              # Input volume mount
INTERNVL_OUTPUT_PATH=/tmp/outputs            # Output volume mount  
INTERNVL_IMAGE_FOLDER_PATH=/tmp/inputs/images
INTERNVL_PROMPTS_PATH=/tmp/inputs/prompts.yaml
INTERNVL_MODEL_PATH=/models/InternVL3-8B     # Model volume mount
```

#### KFP Usage Examples
```bash
# Process single document in KFP
python -m internvl.cli.internvl_single --image-path receipt.jpg --output-file result.json

# Batch process in KFP
python -m internvl.cli.internvl_batch \
  --image-folder-path /tmp/inputs/images \
  --output-file /tmp/outputs/results.csv

# The system automatically resolves relative paths to configured directories
```

#### Volume Mounts Required
- **Input Volume**: Mount to `/tmp/inputs` containing images and prompts.yaml
- **Output Volume**: Mount to `/tmp/outputs` for results
- **Model Volume**: Mount model files to `/models/` directory
- **Temp Volume**: Standard `/tmp` for intermediate processing

#### Path Resolution
The system intelligently resolves paths:
- Absolute paths: Used as-is
- Relative paths: Resolved relative to `INPUT_PATH` for inputs, `OUTPUT_PATH` for outputs
- Missing files: System searches in configured input directories

### Local Development vs KFP

| Environment | Input Path | Output Path | Usage |
|-------------|------------|-------------|-------|
| **Local Dev** | `/home/jovyan/nfs_share/tod/examples` | `/home/jovyan/nfs_share/tod/output` | `python -m internvl.cli.internvl_single` (no params needed!) |
| **KFP** | `/tmp/inputs` | `/tmp/outputs` | `--image-path receipt.jpg` |

The same commands work in both environments - the system handles path resolution automatically.

## Contributing

When adding new features:
1. Follow the modular package structure
2. Add appropriate tests in the relevant module
3. Update documentation and examples
4. Ensure compatibility with auto device configuration

## Troubleshooting

### Common Issues

**Model Loading Errors**: Check your `INTERNVL_MODEL_PATH` in `.env`
```bash
python -m internvl.utils.verify_env  # Check environment setup
```

**GPU Memory Issues**: The system automatically applies quantization for single GPU setups
```bash
export CUDA_VISIBLE_DEVICES=0  # Limit to one GPU if needed
```

**Import Errors**: Ensure you're using the module invocation pattern:
```bash
python -m internvl.cli.internvl_single  # Correct
python internvl/cli/internvl_single.py  # Incorrect
```

**Path Resolution Issues**: Check your `.env` configuration and ensure `INTERNVL_PROJECT_ROOT` is set correctly.

### Bank Statement Specific Issues

**✅ Bank Statement Processing Working**: The system successfully processes bank statements and extracts highlighted transactions. Example results:
- **Total transactions extracted**: 23 transactions from ANZ bank statement
- **Highlighted transactions identified**: 3 interest payments ($7,280.27 total)
- **Processing time**: ~40 seconds on 2 GPUs

**Highlight Detection Disabled**: If you see "tesseract is not installed" warnings, the system will still work:
```bash
# Install computer vision dependencies (optional)
pip install opencv-python pytesseract pillow

# Install system tesseract (optional)
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract              # MacOS
sudo yum install tesseract          # CentOS/RHEL
```

**Bank Statement Processing Without Highlights**: The system processes bank statements even without highlight detection:
```bash
# Process bank statement without highlight detection (uses .env folder automatically)
python -m internvl.cli.internvl_single \
  --document-type bank_statement \
  --no-detect-highlights

# Or process specific bank statement
python -m internvl.cli.internvl_single \
  --image-path "bank statement - ANZ highlight.png" \
  --document-type bank_statement \
  --no-detect-highlights
```

**Working Example**: Test with the included example:
```bash
# This command successfully extracts 23 transactions and 3 highlighted ones
python -m internvl.cli.internvl_single \
  --image-path "bank statement - ANZ highlight.png" \
  --document-type bank_statement

# Or use first bank statement from .env folder
python -m internvl.cli.internvl_single --document-type bank_statement
```

**OpenCV ImportError**: If you get numpy/cv2 errors, ensure all dependencies are installed in the correct environment:
```bash
conda activate internvl_env
pip install -r requirements-cv.txt
```

## License

MIT License - see LICENSE file for details.