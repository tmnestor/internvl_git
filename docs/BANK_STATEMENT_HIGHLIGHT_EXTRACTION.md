# Bank Statement Highlight Text Extraction

This document explains how the InternVL PoC system extracts highlighted text from bank statements for Australian work-related expense processing.

## Architecture Overview

The system uses a **two-stage approach**: computer vision for highlight detection + OCR for text extraction, then enhances this with InternVL's multimodal understanding.

## Key Components

### 1. Highlight Detection Module
**File**: `internvl/image/highlight_detection.py`

The `BankStatementHighlightDetector` class implements computer vision techniques to detect highlighted regions:

#### Technical Implementation
- **Color Space Conversion**: Uses HSV color space for better color detection
- **Multi-Color Support**: Detects yellow, pink, green, and orange highlights
- **Morphological Operations**: Applies opening/closing operations to clean up detected regions
- **Contour Detection**: Uses OpenCV contour detection to find highlight boundaries
- **Confidence Scoring**: Calculates confidence based on area and shape regularity
- **Overlap Filtering**: Removes overlapping detections to avoid duplicates

#### Color Detection Parameters
```python
color_ranges = {
    'yellow': {'lower': [15, 50, 50], 'upper': [35, 255, 255]},
    'pink': {'lower': [160, 50, 50], 'upper': [180, 255, 255]},
    'green': {'lower': [35, 50, 50], 'upper': [85, 255, 255]},
    'orange': {'lower': [5, 50, 50], 'upper': [15, 255, 255]}
}
```

### 2. Text Extraction from Highlights

#### OCR Processing
- **Library**: Uses Tesseract OCR via `pytesseract`
- **Region Cropping**: Extracts highlighted regions with padding
- **Optimized Config**: Custom Tesseract configuration for bank statements:
  ```python
  custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$.,/- '
  ```
- **Text Association**: Links extracted text back to specific highlight regions

### 3. Bank Statement Processing Module
**File**: `internvl/extraction/bank_statement_parser.py`

The `BankStatementProcessor` class handles bank statement processing with integrated highlight detection:

- **Australian Bank Recognition**: Pattern matching for major Australian banks (CBA, ANZ, Westpac, NAB, etc.)
- **Work Expense Categorization**: Automatic categorization of transactions into fuel, office supplies, travel, professional services, etc.
- **ATO Compliance Assessment**: Validates extracted data against Australian Tax Office requirements

## Processing Pipeline

The complete bank statement processing pipeline follows these steps:

1. **Image Loading & Validation**: Load bank statement image
2. **Highlight Detection**: 
   - Convert to HSV color space
   - Apply color masks for each highlight color
   - Morphological operations to clean masks
   - Find contours and calculate bounding boxes
   - Filter by area (100-50,000 pixels) and confidence (>0.6)
   - Remove overlapping detections
3. **Text Extraction**: OCR processing of highlighted regions
4. **InternVL Processing**: Send image + context-aware prompt to model
5. **Transaction Parsing**: Parse model response into structured data
6. **Highlight Enhancement**: Match extracted text with detected highlights
7. **ATO Compliance Assessment**: Validate against Australian tax requirements

## Specialized Prompts for Bank Statements

The system uses different prompts based on highlight detection results:

### With Highlights (`bank_statement_highlighted_prompt`)
```
PRIORITY: Focus on extracting highlighted/marked transactions first
HIGHLIGHTED_TRANSACTIONS:
DATE: [DD/MM/YYYY] | DESCRIPTION: [Merchant/Description] | AMOUNT: [Debit amount] | WORK_RELEVANCE: [High/Medium/Low]
```

### Without Highlights (`bank_statement_ato_compliance_prompt`)
```
WORK_RELEVANCE_CRITERIA:
- High: Clear work expenses (fuel, office supplies, professional services)
- Medium: Potentially work-related (meals, equipment, training)
- Low: Possibly work-related (general purchases, subscriptions)
```

## Data Structures

### HighlightRegion Schema
```python
class HighlightRegion(BaseModel):
    x: int                    # X coordinate
    y: int                    # Y coordinate  
    width: int               # Width in pixels
    height: int              # Height in pixels
    color: str               # Detected color
    confidence: float        # Detection confidence (0-1)
    extracted_text: str      # OCR-extracted text
```

### BankTransaction Schema
```python
class BankTransaction(BaseModel):
    transaction_date: str
    description: str
    debit_amount: Optional[str]
    credit_amount: Optional[str] 
    work_related_likelihood: Optional[float]  # AI-assessed relevance
    highlight_detected: bool                  # Was highlighted by user
    highlight_region: Optional[HighlightRegion]
```

## Real-World Example

Looking at the provided ANZ bank statement (`examples/bank statement - ANZ highlight.png`):

- **Yellow highlighted entries**: "INTEREST" transactions on specific dates (29 MAR, 26 APR, 26 MAY, 26 JUL, 26 AUG, 26 SEP)
- **Detection capability**: The system identifies these yellow highlighted regions
- **Text extraction**: OCR extracts "INTEREST" and associated amounts
- **Work relevance**: These are flagged as potentially work-related for tax purposes

## Production Features

- **Multi-platform support**: Handles various bank statement formats
- **Visualization**: Creates annotated images showing detected highlights
- **Confidence scoring**: Provides quality metrics for extraction reliability
- **Error handling**: Graceful fallbacks when highlight detection fails
- **Integration**: Seamlessly integrates with the main InternVL processing pipeline

## Key Implementation Files

- **Core Processing**: `internvl/extraction/bank_statement_parser.py`
- **Highlight Detection**: `internvl/image/highlight_detection.py` 
- **Schema Definitions**: `internvl/schemas/bank_statement_schemas.py`
- **Test Suite**: `test_bank_statement_features.py`
- **Prompts**: Bank-specific prompts in `prompts.yaml`

## Usage

The highlight extraction is automatically integrated into the main processing pipeline. When processing bank statements, the system:

1. Automatically detects if highlights are present
2. Prioritizes highlighted transactions in the extraction
3. Provides metadata about highlight locations and confidence
4. Associates extracted text with specific highlight regions

This implementation provides a comprehensive solution for processing Australian bank statements with highlighted transaction detection, specifically designed for work-related expense claims and ATO compliance.