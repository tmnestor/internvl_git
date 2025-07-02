# Production CLI Enhancement: Key-Value as Default

## Executive Summary

This document outlines the implementation steps to make the **Enhanced Key-Value Parser** the default extraction method across all CLI tools, with JSON extraction as an optional fallback. Based on successful testing with Target.png (80% ATO compliance) and Bunnings.png (100% ATO compliance), the Key-Value approach has proven superior for Australian business receipt processing.

## Current State vs Target State

### Current State
```bash
# Current behavior - JSON extraction by default
python -m internvl.cli.internvl_single --image-path examples/Target.png
# Uses: JSON prompts + json_extraction_fixed.py

# Key-Value requires manual prompt selection
INTERNVL_PROMPT_NAME=key_value_receipt_prompt python -m internvl.cli.internvl_single --image-path examples/Target.png
```

### Target State  
```bash
# New behavior - Key-Value extraction by default
python -m internvl.cli.internvl_single --image-path examples/Target.png
# Uses: key_value_receipt_prompt + key_value_parser.py

# JSON becomes the optional fallback
python -m internvl.cli.internvl_single --image-path examples/Target.png --extraction-method json
```

## Implementation Plan

### Phase 1: Update CLI Interface (1-2 days)

#### Step 1.1: Modify CLI Argument Structure

**File: `internvl/cli/internvl_single.py`**

```python
# BEFORE: Current argument structure
def main():
    parser = argparse.ArgumentParser(description="InternVL single image extraction")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--prompt-name", default=None)
    parser.add_argument("--verbose", action="store_true")

# AFTER: Enhanced argument structure
def main():
    parser = argparse.ArgumentParser(description="InternVL single image extraction")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--prompt-name", default=None, 
                       help="Override default prompt (advanced users)")
    parser.add_argument("--extraction-method", 
                       choices=["key_value", "json"], 
                       default="key_value",
                       help="Extraction method: key_value (default, robust) or json (legacy)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compliance-check", action="store_true",
                       help="Perform ATO compliance validation (key_value only)")
```

#### Step 1.2: Update Default Configuration Logic

**File: `internvl/cli/internvl_single.py`**

```python
def get_extraction_config(args):
    """Get extraction configuration based on CLI arguments."""
    config = {}
    
    if args.extraction_method == "key_value":
        # NEW DEFAULT: Key-Value extraction
        config.update({
            'prompt_name': args.prompt_name or 'key_value_receipt_prompt',
            'extraction_method': 'key_value',
            'processor_module': 'internvl.extraction.key_value_parser',
            'processor_function': 'extract_work_related_expense',
            'supports_compliance': True
        })
    elif args.extraction_method == "json":
        # LEGACY: JSON extraction
        config.update({
            'prompt_name': args.prompt_name or 'default_receipt_prompt',
            'extraction_method': 'json',
            'processor_module': 'internvl.extraction.json_extraction_fixed',
            'processor_function': 'extract_json_from_text',
            'supports_compliance': False
        })
    
    # Compliance check only available for key_value
    if args.compliance_check and args.extraction_method != "key_value":
        logger.warning("Compliance check only available with --extraction-method key_value")
        config['compliance_check'] = False
    else:
        config['compliance_check'] = args.compliance_check
    
    return config
```

#### Step 1.3: Implement Unified Processing Function

**File: `internvl/cli/internvl_single.py`**

```python
def process_image_unified(image_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Unified image processing supporting both key_value and json methods."""
    
    logger.info(f"Processing {image_path} with {config['extraction_method']} method")
    
    # Load model and generate response
    model, tokenizer = load_model_and_tokenizer()
    
    # Get prompt
    prompt = get_prompt_by_name(config['prompt_name'])
    
    # Generate response
    response = get_raw_prediction(
        image_path=str(image_path),
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        generation_config=get_generation_config(),
        device="auto"
    )
    
    # Process based on extraction method
    if config['extraction_method'] == 'key_value':
        return process_key_value_extraction(response, config)
    elif config['extraction_method'] == 'json':
        return process_json_extraction(response, config)
    else:
        raise ValueError(f"Unknown extraction method: {config['extraction_method']}")

def process_key_value_extraction(response: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process using Enhanced Key-Value Parser."""
    from internvl.extraction.key_value_parser import extract_work_related_expense
    
    # Determine expense category from config or default
    expense_category = config.get('expense_category', 'General')
    
    if config.get('compliance_check', False):
        # Full ATO compliance assessment
        result = extract_work_related_expense(response, expense_category)
        
        # Add CLI-specific formatting
        if result['success']:
            assessment = result['assessment']
            result['cli_summary'] = {
                'extraction_method': 'key_value',
                'ato_compliance': f"{assessment['compliance_score']:.0f}%",
                'ato_ready': assessment['ato_ready'],
                'confidence': result['summary']['extraction_quality']['confidence_score'],
                'quality_grade': result['summary']['validation_status']['quality_grade']
            }
        
        return result
    else:
        # Basic key-value extraction without full compliance
        from internvl.extraction.key_value_parser import extract_key_value_enhanced
        result = extract_key_value_enhanced(response)
        
        # Add CLI-specific formatting
        if result['success']:
            result['cli_summary'] = {
                'extraction_method': 'key_value',
                'confidence': result['summary']['extraction_quality']['confidence_score'],
                'quality_grade': result['summary']['validation_status']['quality_grade'],
                'production_ready': result['summary']['validation_status']['recommended_for_production']
            }
        
        return result

def process_json_extraction(response: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process using legacy JSON extraction."""
    from internvl.extraction.json_extraction_fixed import extract_json_from_text
    
    try:
        json_result = extract_json_from_text(response)
        
        return {
            'success': True,
            'extraction_method': 'json',
            'extracted_data': json_result,
            'cli_summary': {
                'extraction_method': 'json',
                'note': 'Legacy JSON extraction - consider using key_value for better results'
            },
            'raw_response': response
        }
    except Exception as e:
        return {
            'success': False,
            'extraction_method': 'json',
            'error': str(e),
            'cli_summary': {
                'extraction_method': 'json',
                'error': 'JSON extraction failed - try --extraction-method key_value'
            }
        }
```

### Phase 2: Update Batch Processing (1 day)

#### Step 2.1: Enhance Batch CLI

**File: `internvl/cli/internvl_batch.py`**

```python
def main():
    parser = argparse.ArgumentParser(description="InternVL batch image extraction")
    parser.add_argument("--image-folder-path", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default="output/batch_results.csv")
    parser.add_argument("--extraction-method", 
                       choices=["key_value", "json"], 
                       default="key_value",
                       help="Extraction method: key_value (default, robust) or json (legacy)")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--compliance-check", action="store_true",
                       help="Perform ATO compliance validation (key_value only)")
    parser.add_argument("--expense-category", default="General",
                       help="ATO expense category for compliance checking")
    
    args = parser.parse_args()
    
    # Process batch with unified configuration
    config = get_extraction_config(args)
    results = process_image_batch(args.image_folder_path, config)
    
    # Save results with method-specific formatting
    save_batch_results(results, args.output_file, config)
```

#### Step 2.2: Enhanced Batch Results Format

**File: `internvl/cli/internvl_batch.py`**

```python
def save_batch_results(results: List[Dict], output_file: Path, config: Dict[str, Any]):
    """Save batch results with extraction method specific columns."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if config['extraction_method'] == 'key_value':
        # Enhanced columns for key-value results
        columns = [
            'image_name', 'success', 'confidence_score', 'quality_grade',
            'supplier_name', 'supplier_abn', 'invoice_date', 'total_amount', 'gst_amount',
            'items_count', 'extraction_method'
        ]
        
        if config.get('compliance_check'):
            columns.extend(['ato_compliance_score', 'ato_ready', 'compliance_issues'])
            
    elif config['extraction_method'] == 'json':
        # Legacy columns for JSON results
        columns = [
            'image_name', 'success', 'store_name_value', 'date_value', 'total_value', 'tax_value',
            'extraction_method', 'notes'
        ]
    
    # Write CSV with appropriate columns
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            row = format_result_for_csv(result, config)
            writer.writerow(row)
    
    logger.info(f"Batch results saved to: {output_file}")
```

### Phase 3: Update Configuration Defaults (30 minutes)

#### Step 3.1: Update .env.example

**File: `.env.example`**

```bash
# BEFORE
INTERNVL_PROMPT_NAME=default_receipt_prompt

# AFTER  
INTERNVL_PROMPT_NAME=key_value_receipt_prompt
INTERNVL_EXTRACTION_METHOD=key_value
INTERNVL_COMPLIANCE_CHECK=true
```

#### Step 3.2: Update prompts.yaml Priority

**File: `prompts.yaml`**

```yaml
# Move key_value_receipt_prompt to the top as the recommended default
key_value_receipt_prompt: |
  <image>
  Extract information from this Australian receipt and return in KEY-VALUE format.
  # ... (existing prompt content)

# Legacy prompts section
legacy_json_prompts:
  default_receipt_prompt: |
    <image>
    Extract information from this receipt and return in JSON format.
    # ... (existing JSON prompt content)
```

### Phase 4: Enhanced Help and Documentation (1 hour)

#### Step 4.1: Update CLI Help Text

**File: `internvl/cli/internvl_single.py`**

```python
HELP_TEXT = """
InternVL Receipt Extraction - Production Ready

RECOMMENDED USAGE (Key-Value - Default):
  python -m internvl.cli.internvl_single --image-path receipt.png
  
  Key-Value extraction provides:
  ✅ Superior robustness vs JSON parsing
  ✅ Australian business compliance (ATO)
  ✅ Comprehensive validation and confidence scoring
  ✅ Built-in ABN validation and GST calculations

AUSTRALIAN TAX COMPLIANCE:
  python -m internvl.cli.internvl_single --image-path receipt.png --compliance-check
  
  Provides full ATO work-related expense validation including:
  • ABN requirement checking (mandatory >$82.50)
  • Australian date format validation (DD/MM/YYYY)
  • GST calculation verification (10% Australian rate)
  • Work expense category classification

LEGACY MODE (JSON - Optional):
  python -m internvl.cli.internvl_single --image-path receipt.png --extraction-method json
  
  Note: JSON extraction is less reliable and does not support compliance checking.

EXAMPLES:
  # Basic extraction (recommended)
  python -m internvl.cli.internvl_single --image-path examples/Target.png
  
  # With ATO compliance check
  python -m internvl.cli.internvl_single --image-path examples/Bunnings.png --compliance-check
  
  # Legacy JSON mode
  python -m internvl.cli.internvl_single --image-path examples/receipt.png --extraction-method json
"""

def print_help():
    print(HELP_TEXT)
```

#### Step 4.2: Update README.md Quick Start

**File: `README.md`**

```markdown
### 3. Basic Usage

```bash
# Process a single receipt (Key-Value - recommended)
python -m internvl.cli.internvl_single --image-path examples/Target.png

# With Australian Tax Office compliance checking
python -m internvl.cli.internvl_single --image-path examples/Bunnings.png --compliance-check

# Process multiple receipts
python -m internvl.cli.internvl_batch --image-folder-path examples/ --compliance-check

# Legacy JSON mode (less reliable)
python -m internvl.cli.internvl_single --image-path test.png --extraction-method json
```

### Key-Value Extraction (Default)
- ✅ **Superior reliability** - No JSON parsing failures
- ✅ **Australian compliance** - ATO work expense validation  
- ✅ **Confidence scoring** - Quality assessment for each extraction
- ✅ **ABN validation** - Australian Business Number checking
- ✅ **GST calculation** - 10% tax validation

### JSON Extraction (Legacy)
- ⚠️ **Legacy mode** - Available for backward compatibility
- ❌ **No compliance checking** - Basic extraction only
- ❌ **Parsing failures** - JSON syntax errors possible
```

### Phase 5: Testing and Validation (2-3 hours)

#### Step 5.1: Update Existing Tests

**File: `test_cli_integration.py`** (new file)

```python
import subprocess
import tempfile
from pathlib import Path
import json
import pytest

class TestCLIIntegration:
    """Test CLI integration with both extraction methods."""
    
    def test_default_key_value_extraction(self):
        """Test default behavior uses key-value extraction."""
        result = subprocess.run([
            "python", "-m", "internvl.cli.internvl_single",
            "--image-path", "examples/Target.png",
            "--output-file", "/tmp/test_result.json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # Check output contains key-value indicators
        with open("/tmp/test_result.json") as f:
            output = json.load(f)
        
        assert output['cli_summary']['extraction_method'] == 'key_value'
        assert 'confidence' in output['cli_summary']
    
    def test_json_extraction_option(self):
        """Test --extraction-method json works as option."""
        result = subprocess.run([
            "python", "-m", "internvl.cli.internvl_single",
            "--image-path", "examples/Target.png",
            "--extraction-method", "json",
            "--output-file", "/tmp/test_json_result.json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # Check output contains JSON indicators
        with open("/tmp/test_json_result.json") as f:
            output = json.load(f)
        
        assert output['cli_summary']['extraction_method'] == 'json'
    
    def test_compliance_check_feature(self):
        """Test --compliance-check works with key-value."""
        result = subprocess.run([
            "python", "-m", "internvl.cli.internvl_single",
            "--image-path", "examples/Bunnings.png",
            "--compliance-check",
            "--output-file", "/tmp/test_compliance_result.json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # Check compliance results present
        with open("/tmp/test_compliance_result.json") as f:
            output = json.load(f)
        
        assert 'ato_compliance' in output['cli_summary']
        assert 'ato_ready' in output['cli_summary']
    
    def test_compliance_check_requires_key_value(self):
        """Test compliance check fails gracefully with JSON method."""
        result = subprocess.run([
            "python", "-m", "internvl.cli.internvl_single",
            "--image-path", "examples/Target.png",
            "--extraction-method", "json",
            "--compliance-check"
        ], capture_output=True, text=True)
        
        # Should complete but with warning
        assert result.returncode == 0
        assert "Compliance check only available with" in result.stderr
```

#### Step 5.2: Regression Testing

**File: `test_backward_compatibility.py`** (new file)

```python
def test_backward_compatibility():
    """Ensure existing functionality still works."""
    
    # Test that old environment variable still works
    os.environ['INTERNVL_PROMPT_NAME'] = 'default_receipt_prompt'
    result = subprocess.run([
        "python", "-m", "internvl.cli.internvl_single",
        "--image-path", "examples/Target.png",
        "--extraction-method", "json"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    
def test_prompt_override_still_works():
    """Test that --prompt-name override still functions."""
    result = subprocess.run([
        "python", "-m", "internvl.cli.internvl_single",
        "--image-path", "examples/Target.png",
        "--prompt-name", "simple_receipt_prompt"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
```

## Implementation Timeline

| Phase | Duration | Priority |
|-------|----------|----------|
| Phase 1: CLI Interface Updates | 1-2 days | High |
| Phase 2: Batch Processing | 1 day | High |
| Phase 3: Configuration Defaults | 30 minutes | Medium |
| Phase 4: Documentation | 1 hour | Medium |
| Phase 5: Testing | 2-3 hours | High |

**Total Estimated Time: 3-5 days**

## Success Criteria

### ✅ **Primary Success Metrics**
1. **Default behavior changed**: `python -m internvl.cli.internvl_single --image-path receipt.png` uses Key-Value extraction
2. **Backward compatibility**: `--extraction-method json` provides legacy functionality
3. **Enhanced features**: `--compliance-check` provides ATO validation
4. **All tests pass**: No regression in existing functionality

### ✅ **User Experience Improvements**
1. **Better default results**: Higher accuracy with Key-Value extraction
2. **Australian compliance**: Built-in ATO work expense validation
3. **Clear feedback**: Confidence scores and quality grades in output
4. **Migration path**: Easy transition from JSON to Key-Value

### ✅ **Technical Achievements**
1. **Unified architecture**: Single codebase supporting both extraction methods
2. **Robust error handling**: Graceful fallback and clear error messages
3. **Performance maintained**: No significant speed regression
4. **Maintainable code**: Clean separation between extraction methods

## Post-Implementation Benefits

### 🎯 **For End Users**
- **Better results by default** - No configuration needed for superior extraction
- **Australian compliance** - Built-in ATO work expense validation
- **Clear quality indicators** - Confidence scores guide usage decisions
- **Smooth migration** - Existing workflows continue to work

### 🎯 **For Developers**
- **Future-proof architecture** - Key-Value as the primary method
- **Maintained compatibility** - JSON still available when needed
- **Enhanced testing** - Comprehensive test coverage for both methods
- **Clear upgrade path** - Natural progression from JSON to Key-Value

This implementation makes the **Enhanced Key-Value Parser the production default** while maintaining full backward compatibility! 🚀