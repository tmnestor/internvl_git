# Bank Statement Analysis Extension for Work-Related Expense Claims

## Executive Summary

This document outlines the technical approach to extend the InternVL system to process bank statements with highlighted regions for Australian Tax Office (ATO) work-related expense claims. Bank statements represent a critical evidence type for tax deductions, especially when taxpayers highlight specific transactions related to their work expenses.

## Current State Analysis

### Existing Capabilities ✅
- **Key-Value extraction** from standard receipts
- **ATO compliance assessment** for business receipts
- **Multi-format image processing** with InternVL3 vision models
- **Confidence scoring** and quality assessment
- **Australian business validation** (ABN, GST, date formats)

### Critical Gap ❌
- **No bank statement schema** - Current schema focuses on retail receipts
- **No highlight detection** - Cannot identify user-marked regions of interest
- **No transaction parsing** - Cannot extract individual bank transactions
- **No expense correlation** - Cannot link bank transactions to work purposes
- **No financial institution validation** - Missing bank-specific formatting

---

## BANK STATEMENT PROCESSING REQUIREMENTS

### 1. Australian Bank Statement Characteristics

#### Major Australian Banks (Complete List)
```python
AUSTRALIAN_BANKS = {
    # Big 4 Banks
    'cba': {
        'full_name': 'Commonwealth Bank of Australia',
        'bsb_range': ['06', '20', '21', '22'],
        'account_format': 'XXXXXX-XXXXXXXX',
        'logo_identifiers': ['yellow diamond', 'CommBank', 'NetBank'],
        'common_formats': ['netbank_statement', 'commbank_app', 'paper_statement']
    },
    'anz': {
        'full_name': 'Australia and New Zealand Banking Group',
        'bsb_range': ['01', '06', '08'],
        'account_format': 'XXX-XXXXXX',
        'logo_identifiers': ['ANZ', 'blue branding', 'ANZ Internet Banking'],
        'common_formats': ['anz_online', 'mobile_banking', 'paper_statement']
    },
    'westpac': {
        'full_name': 'Westpac Banking Corporation',
        'bsb_range': ['03', '73'],
        'account_format': 'XXX-XXX',
        'logo_identifiers': ['red W logo', 'Westpac', 'Westpac Online'],
        'common_formats': ['westpac_online', 'mobile_app', 'branch_statement']
    },
    'nab': {
        'full_name': 'National Australia Bank',
        'bsb_range': ['08', '09'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['red star', 'NAB', 'NAB Connect'],
        'common_formats': ['nab_connect', 'internet_banking', 'paper_statement']
    },
    
    # Major Regional Banks
    'bendigo': {
        'full_name': 'Bendigo and Adelaide Bank',
        'bsb_range': ['63'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['Bendigo Bank', 'community banking'],
        'common_formats': ['bendigo_online', 'community_branch', 'paper_statement']
    },
    'boq': {
        'full_name': 'Bank of Queensland',
        'bsb_range': ['12', '13'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['BOQ', 'blue orange branding'],
        'common_formats': ['boq_online', 'mobile_banking']
    },
    'macquarie': {
        'full_name': 'Macquarie Bank',
        'bsb_range': ['18'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['Macquarie', 'black logo'],
        'common_formats': ['macquarie_online', 'investment_statement']
    },
    
    # Digital/International Banks
    'ing': {
        'full_name': 'ING',
        'bsb_range': ['92'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['ING', 'orange branding'],
        'common_formats': ['ing_app', 'digital_statement']
    },
    'citibank': {
        'full_name': 'Citibank Australia',
        'bsb_range': ['24'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['citi', 'Citibank'],
        'common_formats': ['citi_online', 'credit_statement']
    },
    'hsbc': {
        'full_name': 'HSBC Bank Australia',
        'bsb_range': ['34'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['HSBC', 'red white logo'],
        'common_formats': ['hsbc_online', 'international_format']
    },
    
    # Credit Unions and Mutuals
    'bank_australia': {
        'full_name': 'Bank Australia',
        'bsb_range': ['31'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['Bank Australia', 'pink logo'],
        'common_formats': ['sustainable_banking', 'online_statement']
    },
    'bank_melbourne': {
        'full_name': 'Bank of Melbourne',
        'bsb_range': ['19'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['Bank of Melbourne', 'purple branding'],
        'common_formats': ['melbourne_online', 'local_banking']
    },
    'suncorp': {
        'full_name': 'Suncorp Bank',
        'bsb_range': ['48'],
        'account_format': 'XXXXXXXX',
        'logo_identifiers': ['Suncorp', 'yellow logo'],
        'common_formats': ['suncorp_app', 'queensland_banking']
    }
}
```

#### Bank Statement Data Structure
```python
# Standard bank transaction fields
BANK_TRANSACTION_FIELDS = {
    'transaction_date': 'DD/MM/YYYY format',
    'description': 'Transaction description/merchant',
    'debit_amount': 'Money withdrawn (negative)',
    'credit_amount': 'Money deposited (positive)',
    'balance': 'Account balance after transaction',
    'reference': 'Transaction reference number',
    'merchant_category': 'Business type (if available)'
}

# Statement metadata fields  
STATEMENT_METADATA_FIELDS = {
    'account_holder': 'Customer name',
    'account_number': 'Bank account number (masked)',
    'bsb': 'Bank State Branch code',
    'statement_period': 'Date range of statement',
    'opening_balance': 'Starting balance',
    'closing_balance': 'Ending balance',
    'bank_name': 'Financial institution'
}
```

### 2. Highlight Detection Requirements

#### Visual Highlight Types
- **Yellow highlighter** - Most common taxpayer marking
- **Pink/green markers** - Alternative highlighting colors
- **Red pen circles** - Manual markup
- **Digital highlights** - PDF annotation tools
- **Arrows and annotations** - Directional indicators

#### Technical Detection Approach
```python
class HighlightDetector:
    """Detect and extract highlighted regions from bank statements."""
    
    def __init__(self):
        self.highlight_colors = {
            'yellow': {'hsv_range': [(15, 50, 50), (35, 255, 255)]},
            'pink': {'hsv_range': [(160, 50, 50), (180, 255, 255)]},
            'green': {'hsv_range': [(35, 50, 50), (85, 255, 255)]},
        }
        self.detection_confidence_threshold = 0.7
    
    def detect_highlights(self, image_path: str) -> List[HighlightRegion]:
        """Detect highlighted regions using computer vision."""
        pass
    
    def extract_highlighted_text(self, image_path: str, regions: List[HighlightRegion]) -> str:
        """Extract text from highlighted areas using OCR."""
        pass
```

---

## TECHNICAL IMPLEMENTATION APPROACH

### Phase 1: Bank Statement Schema Extension

#### Step 1.1: Create Bank Statement Data Models

**File: `internvl/schemas/bank_statement_schemas.py`**

```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import re

class BankTransaction(BaseModel):
    """Individual bank transaction for work expense analysis."""
    
    transaction_date: str = Field(..., description="Transaction date in DD/MM/YYYY format")
    description: str = Field(..., description="Transaction description/merchant name")
    debit_amount: Optional[str] = Field(None, description="Amount debited (withdrawn)")
    credit_amount: Optional[str] = Field(None, description="Amount credited (deposited)")
    balance: Optional[str] = Field(None, description="Account balance after transaction")
    reference: Optional[str] = Field(None, description="Transaction reference/ID")
    merchant_category: Optional[str] = Field(None, description="Business category if identifiable")
    
    # Work expense relevance fields
    work_related_likelihood: Optional[float] = Field(None, description="AI-assessed work relevance (0-1)")
    expense_category: Optional[str] = Field(None, description="ATO expense category")
    highlight_detected: bool = Field(False, description="Was this transaction highlighted by user")
    
    @validator('transaction_date')
    def validate_australian_date(cls, v):
        """Validate Australian DD/MM/YYYY date format."""
        if not re.match(r'^\d{2}/\d{2}/\d{4}$', v):
            raise ValueError('Date must be in DD/MM/YYYY format')
        return v

class BankStatementExtraction(BaseModel):
    """Complete bank statement extraction for ATO compliance."""
    
    # Statement metadata
    bank_name: str = Field(..., description="Name of financial institution")
    account_holder: str = Field(..., description="Account holder name")
    account_number: str = Field(..., description="Masked account number")
    bsb: Optional[str] = Field(None, description="Bank State Branch code")
    statement_period: str = Field(..., description="Statement period (DD/MM/YYYY - DD/MM/YYYY)")
    
    # Financial summary
    opening_balance: str = Field(..., description="Opening balance")
    closing_balance: str = Field(..., description="Closing balance")
    total_debits: str = Field(..., description="Total withdrawals")
    total_credits: str = Field(..., description="Total deposits")
    
    # Transactions
    transactions: List[BankTransaction] = Field(..., description="All transactions in statement")
    highlighted_transactions: List[BankTransaction] = Field(default=[], description="User-highlighted transactions")
    
    # Work expense analysis
    work_expense_summary: Dict[str, Any] = Field(default={}, description="Summary of work-related expenses")
    ato_compliance_assessment: Dict[str, Any] = Field(default={}, description="ATO compliance evaluation")
```

#### Step 1.2: Bank Statement Prompts

**File: `prompts.yaml` - Add bank statement section**

```yaml
# === BANK STATEMENT PROMPTS ===

bank_statement_analysis_prompt: |
  <image>
  Analyze this Australian bank statement and extract transaction information.
  
  BANK STATEMENT FORMAT:
  BANK: [Name of financial institution]
  ACCOUNT_HOLDER: [Customer name]
  ACCOUNT_NUMBER: [Account number - mask middle digits]
  BSB: [Bank State Branch code if visible]
  STATEMENT_PERIOD: [Start date - End date]
  OPENING_BALANCE: [Starting balance]
  CLOSING_BALANCE: [Ending balance]
  
  TRANSACTIONS: [Extract each transaction using format below]
  DATE: [DD/MM/YYYY] | DESCRIPTION: [Transaction description] | DEBIT: [Amount withdrawn] | CREDIT: [Amount deposited] | BALANCE: [Balance after transaction]
  
  HIGHLIGHTED_AREAS: [If any areas are highlighted/marked, extract those transaction details separately]
  
  INSTRUCTIONS:
  - Extract ALL visible transactions in chronological order
  - Pay special attention to highlighted/marked transactions
  - Use Australian date format DD/MM/YYYY
  - Include all transaction descriptions exactly as shown
  - Note any highlighted areas or user markings
  - Identify the bank name from logos/headers
  
  Return in the specified format above.

bank_statement_highlighted_prompt: |
  <image>
  This bank statement contains highlighted transactions that the taxpayer has marked as work-related expenses.
  
  PRIORITY: Focus on extracting highlighted/marked transactions first, then process the full statement.
  
  HIGHLIGHTED_TRANSACTION_FORMAT:
  HIGHLIGHT_DETECTED: [Yes/No]
  HIGHLIGHT_COLOR: [Yellow/Pink/Green/Red/Other]
  HIGHLIGHTED_TRANSACTIONS:
  DATE: [DD/MM/YYYY] | DESCRIPTION: [Merchant/Description] | AMOUNT: [Debit amount] | WORK_RELEVANCE: [High/Medium/Low]
  
  FULL_STATEMENT_ANALYSIS:
  [Use same format as bank_statement_analysis_prompt]
  
  WORK_EXPENSE_ASSESSMENT:
  - Analyze each highlighted transaction for work-related expense potential
  - Identify merchant categories (fuel, office supplies, travel, etc.)
  - Assess ATO deductibility likelihood
  - Note any patterns in highlighted transactions

bank_statement_ato_compliance_prompt: |
  <image>
  Extract bank statement information for Australian Tax Office work-related expense claims.
  
  ATO REQUIREMENTS for bank statement evidence:
  1. Transaction date and description
  2. Amount of expense
  3. Business purpose (if determinable from description)
  4. Account holder name matching taxpayer
  
  EXTRACTION_PRIORITIES:
  1. HIGHLIGHTED TRANSACTIONS (user-marked as work expenses)
  2. Business-relevant merchants (Officeworks, petrol stations, airlines)
  3. Professional services (accounting, legal, consulting)
  4. Travel and transport expenses
  5. Equipment and supply purchases
  
  OUTPUT_FORMAT:
  [Use bank_statement_analysis_prompt format]
  
  COMPLIANCE_ASSESSMENT:
  - Rate each transaction's ATO compliance (0-100%)
  - Identify missing information for full deductibility
  - Suggest additional documentation needed
```

### Phase 2: Highlight Detection Implementation

#### Step 2.1: Computer Vision Highlight Detection

**File: `internvl/image/highlight_detection.py`**

```python
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class HighlightRegion:
    """Detected highlight region with metadata."""
    x: int
    y: int
    width: int
    height: int
    color: str
    confidence: float
    extracted_text: str = ""

class BankStatementHighlightDetector:
    """Detect highlighted regions in bank statement images."""
    
    def __init__(self):
        """Initialize highlight detector with color ranges."""
        self.color_ranges = {
            'yellow': {
                'lower': np.array([15, 50, 50]),
                'upper': np.array([35, 255, 255]),
                'color_name': 'yellow'
            },
            'pink': {
                'lower': np.array([160, 50, 50]),
                'upper': np.array([180, 255, 255]),
                'color_name': 'pink'
            },
            'green': {
                'lower': np.array([35, 50, 50]),
                'upper': np.array([85, 255, 255]),
                'color_name': 'green'
            }
        }
        self.min_highlight_area = 100  # Minimum pixel area for valid highlight
        self.confidence_threshold = 0.6
    
    def detect_highlights(self, image_path: str) -> List[HighlightRegion]:
        """
        Detect highlighted regions in bank statement image.
        
        Args:
            image_path: Path to bank statement image
            
        Returns:
            List of detected highlight regions
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            detected_regions = []
            
            # Detect each highlight color
            for color_name, color_config in self.color_ranges.items():
                regions = self._detect_color_highlights(
                    hsv, image, color_config, color_name
                )
                detected_regions.extend(regions)
            
            # Filter and validate regions
            valid_regions = self._filter_valid_highlights(detected_regions)
            
            logger.info(f"Detected {len(valid_regions)} highlight regions in {image_path}")
            return valid_regions
            
        except Exception as e:
            logger.error(f"Highlight detection failed for {image_path}: {e}")
            return []
    
    def _detect_color_highlights(
        self, 
        hsv: np.ndarray, 
        original: np.ndarray, 
        color_config: Dict[str, Any], 
        color_name: str
    ) -> List[HighlightRegion]:
        """Detect highlights of specific color."""
        
        # Create color mask
        mask = cv2.inRange(hsv, color_config['lower'], color_config['upper'])
        
        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_highlight_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area and shape
                confidence = min(area / 1000.0, 1.0)  # Normalize by expected highlight size
                
                if confidence >= self.confidence_threshold:
                    regions.append(HighlightRegion(
                        x=x, y=y, width=w, height=h,
                        color=color_name,
                        confidence=confidence
                    ))
        
        return regions
    
    def _filter_valid_highlights(self, regions: List[HighlightRegion]) -> List[HighlightRegion]:
        """Filter overlapping and invalid highlight regions."""
        
        if not regions:
            return []
        
        # Sort by confidence
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        # Remove overlapping regions (keep highest confidence)
        filtered = []
        for region in regions:
            is_overlap = False
            for existing in filtered:
                if self._regions_overlap(region, existing):
                    is_overlap = True
                    break
            
            if not is_overlap:
                filtered.append(region)
        
        return filtered
    
    def _regions_overlap(self, r1: HighlightRegion, r2: HighlightRegion, threshold: float = 0.3) -> bool:
        """Check if two regions overlap significantly."""
        
        # Calculate intersection
        x_overlap = max(0, min(r1.x + r1.width, r2.x + r2.width) - max(r1.x, r2.x))
        y_overlap = max(0, min(r1.y + r1.height, r2.y + r2.height) - max(r1.y, r2.y))
        intersection = x_overlap * y_overlap
        
        # Calculate areas
        area1 = r1.width * r1.height
        area2 = r2.width * r2.height
        
        # Check if intersection is significant
        overlap_ratio = intersection / min(area1, area2)
        return overlap_ratio > threshold

    def extract_highlighted_regions_text(
        self, 
        image_path: str, 
        regions: List[HighlightRegion]
    ) -> List[HighlightRegion]:
        """Extract text from highlighted regions using OCR."""
        
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            logger.warning("pytesseract not available for text extraction")
            return regions
        
        try:
            image = Image.open(image_path)
            
            for region in regions:
                # Crop highlighted region
                cropped = image.crop((
                    region.x, 
                    region.y, 
                    region.x + region.width, 
                    region.y + region.height
                ))
                
                # Extract text with OCR
                text = pytesseract.image_to_string(cropped, config='--psm 6')
                region.extracted_text = text.strip()
                
            return regions
            
        except Exception as e:
            logger.error(f"Text extraction from highlights failed: {e}")
            return regions
```

#### Step 2.2: Integration with InternVL Processing

**File: `internvl/extraction/bank_statement_parser.py`**

```python
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from internvl.image.highlight_detection import BankStatementHighlightDetector, HighlightRegion
from internvl.extraction.key_value_parser import KeyValueParser
from internvl.schemas.bank_statement_schemas import BankStatementExtraction, BankTransaction

logger = logging.getLogger(__name__)

class BankStatementProcessor:
    """Process bank statements with highlight detection and ATO compliance."""
    
    def __init__(self):
        """Initialize bank statement processor."""
        self.highlight_detector = BankStatementHighlightDetector()
        self.key_value_parser = KeyValueParser()
        
        # Comprehensive Australian bank patterns
        self.bank_patterns = {
            # Big 4 Banks
            'cba': r'(Commonwealth Bank|CBA|CommBank|NetBank)',
            'anz': r'(ANZ|Australia.*New Zealand|ANZ Internet Banking)',
            'westpac': r'(Westpac|WBC|Westpac Online)',
            'nab': r'(National Australia Bank|NAB|NAB Connect)',
            
            # Major Regional Banks
            'bendigo': r'(Bendigo Bank|Bendigo.*Adelaide|Community Bank)',
            'boq': r'(Bank of Queensland|BOQ)',
            'macquarie': r'(Macquarie Bank|Macquarie)',
            
            # Digital/International Banks
            'ing': r'(ING|ING Direct)',
            'citibank': r'(Citibank|Citi)',
            'hsbc': r'(HSBC|HSBC Bank)',
            
            # Credit Unions and Mutuals
            'bank_australia': r'(Bank Australia)',
            'bank_melbourne': r'(Bank of Melbourne)',
            'suncorp': r'(Suncorp Bank|Suncorp)',
            'rabobank': r'(Rabobank)',
            'bankwest': r'(Bankwest)'
        }
        
        # Work expense keywords
        self.work_expense_keywords = {
            'fuel': ['bp', 'shell', 'caltex', 'mobil', 'petrol', 'fuel'],
            'office_supplies': ['officeworks', 'staples', 'office', 'supplies'],
            'travel': ['qantas', 'jetstar', 'virgin', 'hotel', 'motel', 'taxi', 'uber'],
            'professional': ['accounting', 'legal', 'consulting', 'services'],
            'equipment': ['harvey norman', 'jb hi-fi', 'computer', 'electronics']
        }
    
    def process_bank_statement(
        self, 
        image_path: str, 
        model, 
        tokenizer, 
        use_highlight_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Process bank statement with optional highlight detection.
        
        Args:
            image_path: Path to bank statement image
            model: InternVL model
            tokenizer: Model tokenizer  
            use_highlight_detection: Whether to detect highlights
            
        Returns:
            Processed bank statement data with ATO compliance
        """
        try:
            result = {
                'success': False,
                'bank_statement_data': {},
                'highlighted_transactions': [],
                'ato_compliance': {},
                'processing_metadata': {
                    'highlight_detection_used': use_highlight_detection,
                    'highlights_detected': 0,
                    'total_transactions': 0
                }
            }
            
            # Step 1: Detect highlights if enabled
            highlight_regions = []
            if use_highlight_detection:
                highlight_regions = self.highlight_detector.detect_highlights(image_path)
                highlight_regions = self.highlight_detector.extract_highlighted_regions_text(
                    image_path, highlight_regions
                )
                result['processing_metadata']['highlights_detected'] = len(highlight_regions)
                logger.info(f"Detected {len(highlight_regions)} highlighted regions")
            
            # Step 2: Choose appropriate prompt based on highlights
            if highlight_regions:
                prompt_name = 'bank_statement_highlighted_prompt'
            else:
                prompt_name = 'bank_statement_ato_compliance_prompt'
            
            # Step 3: Process with InternVL
            from internvl.model.inference import get_raw_prediction
            
            # Get prompt (would need to load from prompts.yaml)
            prompt = self._get_bank_statement_prompt(prompt_name, highlight_regions)
            
            # Generate response
            raw_response = get_raw_prediction(
                image_path=image_path,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                generation_config={"max_new_tokens": 2048, "do_sample": False},
                device="auto"
            )
            
            # Step 4: Parse bank statement response
            parsed_data = self._parse_bank_statement_response(raw_response)
            
            # Step 5: Enhance with highlight information
            if highlight_regions:
                parsed_data = self._enhance_with_highlights(parsed_data, highlight_regions)
            
            # Step 6: Assess ATO compliance
            ato_assessment = self._assess_ato_compliance(parsed_data)
            
            result.update({
                'success': True,
                'bank_statement_data': parsed_data,
                'highlighted_transactions': self._extract_highlighted_transactions(parsed_data),
                'ato_compliance': ato_assessment,
                'processing_metadata': {
                    **result['processing_metadata'],
                    'total_transactions': len(parsed_data.get('transactions', [])),
                    'work_related_transactions': len([t for t in parsed_data.get('transactions', []) 
                                                    if t.get('work_related_likelihood', 0) > 0.5])
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Bank statement processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'bank_statement_data': {},
                'highlighted_transactions': [],
                'ato_compliance': {}
            }
    
    def _get_bank_statement_prompt(self, prompt_name: str, highlight_regions: List[HighlightRegion]) -> str:
        """Get appropriate prompt for bank statement processing."""
        
        # Base prompt would be loaded from prompts.yaml
        # For now, return a basic prompt
        if highlight_regions:
            highlight_info = f"\nHighlighted regions detected: {len(highlight_regions)} areas marked by user"
            highlight_details = "\n".join([
                f"- {region.color} highlight at ({region.x}, {region.y}): {region.extracted_text[:50]}..."
                for region in highlight_regions[:3]  # Show first 3
            ])
            highlight_context = f"{highlight_info}\n{highlight_details}"
        else:
            highlight_context = "\nNo highlighted regions detected."
        
        return f"""<image>
Analyze this Australian bank statement for work-related expense transactions.

{highlight_context}

Extract information in this format:
BANK: [Bank name]
ACCOUNT_HOLDER: [Account holder name]
STATEMENT_PERIOD: [Date range]

TRANSACTIONS:
DATE: [DD/MM/YYYY] | DESCRIPTION: [Transaction description] | AMOUNT: [Debit amount] | WORK_RELEVANCE: [High/Medium/Low/None]

Focus on transactions that appear highlighted or marked by the user, as these are likely work-related expenses the taxpayer wants to claim."""

    def _parse_bank_statement_response(self, response: str) -> Dict[str, Any]:
        """Parse InternVL response into structured bank statement data."""
        
        # This would use a specialized parser similar to KeyValueParser
        # For now, return a basic structure
        return {
            'bank_name': 'ANZ',  # Would be extracted from response
            'account_holder': 'Account Holder',
            'transactions': [],
            'statement_period': '01/01/2024 - 31/01/2024'
        }
    
    def _enhance_with_highlights(
        self, 
        parsed_data: Dict[str, Any], 
        highlight_regions: List[HighlightRegion]
    ) -> Dict[str, Any]:
        """Enhance parsed data with highlight detection information."""
        
        # Mark transactions that correspond to highlighted regions
        # This would involve spatial correlation between highlights and extracted transactions
        
        return parsed_data
    
    def _assess_ato_compliance(self, bank_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess ATO compliance for bank statement transactions."""
        
        assessment = {
            'overall_compliance': 0.0,
            'compliant_transactions': 0,
            'total_transactions': len(bank_data.get('transactions', [])),
            'compliance_issues': [],
            'recommendations': []
        }
        
        # Assess each transaction for ATO requirements
        for transaction in bank_data.get('transactions', []):
            # Check for required fields
            has_date = bool(transaction.get('transaction_date'))
            has_description = bool(transaction.get('description'))
            has_amount = bool(transaction.get('debit_amount'))
            
            compliance_score = sum([has_date, has_description, has_amount]) / 3.0
            
            if compliance_score >= 0.8:
                assessment['compliant_transactions'] += 1
        
        if assessment['total_transactions'] > 0:
            assessment['overall_compliance'] = (
                assessment['compliant_transactions'] / assessment['total_transactions'] * 100
            )
        
        return assessment
    
    def _extract_highlighted_transactions(self, bank_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract transactions that were highlighted by the user."""
        
        return [
            t for t in bank_data.get('transactions', [])
            if t.get('highlight_detected', False)
        ]


def extract_bank_statement_with_highlights(
    image_path: str, 
    model, 
    tokenizer, 
    detect_highlights: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for bank statement extraction with highlight detection.
    
    Args:
        image_path: Path to bank statement image
        model: InternVL model
        tokenizer: Model tokenizer
        detect_highlights: Whether to detect highlighted regions
        
    Returns:
        Complete bank statement analysis with ATO compliance
    """
    processor = BankStatementProcessor()
    return processor.process_bank_statement(
        image_path=image_path,
        model=model,
        tokenizer=tokenizer,
        use_highlight_detection=detect_highlights
    )
```

### Phase 3: CLI Integration

#### Step 3.1: Add Bank Statement CLI Option

**File: `internvl/cli/internvl_single.py` - Enhancement**

```python
# Add to existing CLI arguments
@app.command()
def main(
    # ... existing arguments ...
    document_type: str = typer.Option(
        "receipt", "--document-type",
        help="Document type: receipt (default), bank_statement"
    ),
    detect_highlights: bool = typer.Option(
        True, "--detect-highlights",
        help="Detect highlighted regions in bank statements"
    ),
    # ... rest of arguments ...
):
    # ... existing validation ...
    
    # Add document type validation
    if document_type not in ["receipt", "bank_statement"]:
        rich_config.console.print(
            f"{rich_config.fail_style} Invalid document type: {document_type}. Must be 'receipt' or 'bank_statement'"
        )
        raise typer.Exit(1)
    
    # ... existing processing with document type routing ...
```

#### Step 3.2: Update Processing Pipeline

```python
def process_image_unified(
    image_path: Path, 
    model, 
    tokenizer, 
    prompt: str, 
    config: Dict[str, Any],
    generation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Enhanced unified processing supporting bank statements."""
    
    if config.get('document_type') == 'bank_statement':
        # Use bank statement processor
        from internvl.extraction.bank_statement_parser import extract_bank_statement_with_highlights
        
        result = extract_bank_statement_with_highlights(
            image_path=str(image_path),
            model=model,
            tokenizer=tokenizer,
            detect_highlights=config.get('detect_highlights', True)
        )
        
        # Add CLI summary for bank statements
        if result['success']:
            result['cli_summary'] = {
                'document_type': 'bank_statement',
                'highlights_detected': result['processing_metadata']['highlights_detected'],
                'total_transactions': result['processing_metadata']['total_transactions'],
                'ato_compliance': f"{result['ato_compliance']['overall_compliance']:.0f}%",
                'work_related_transactions': result['processing_metadata'].get('work_related_transactions', 0)
            }
        
        return result
    else:
        # Existing receipt processing
        return process_receipt_unified(image_path, model, tokenizer, prompt, config, generation_config)
```

---

## IMPLEMENTATION ROADMAP

### Phase 1: Foundation (1-2 weeks)
- [ ] **Bank statement schema design** - Define data models
- [ ] **Basic prompt engineering** - Create bank statement prompts  
- [ ] **Computer vision setup** - Implement highlight detection
- [ ] **Testing infrastructure** - Unit tests for new components

### Phase 2: Core Processing (2-3 weeks)
- [ ] **Bank statement parser** - Key-value extraction for bank data
- [ ] **Highlight integration** - Connect CV detection with extraction
- [ ] **ATO compliance assessment** - Bank-specific compliance rules
- [ ] **CLI integration** - Add bank statement processing option

### Phase 3: Enhancement (1-2 weeks)
- [ ] **Multi-bank support** - Handle different bank formats
- [ ] **Advanced highlight detection** - Improve accuracy and types
- [ ] **Transaction categorization** - Auto-detect work expense types
- [ ] **Batch processing** - Handle multiple bank statements

### Phase 4: Production (1 week)
- [ ] **Performance optimization** - Speed and accuracy improvements
- [ ] **Error handling** - Robust error recovery
- [ ] **Documentation** - User guides and API docs
- [ ] **Testing** - End-to-end validation with real bank statements

**Total Estimated Time: 5-8 weeks**

---

## SUCCESS METRICS

### Technical Metrics
- **Highlight Detection Accuracy**: >90% for common highlight colors
- **Transaction Extraction Accuracy**: >95% for standard bank formats
- **ATO Compliance Assessment**: >85% accuracy for rule-based checks
- **Processing Speed**: <30 seconds per bank statement page

### Business Metrics  
- **Work Expense Identification**: >80% accuracy for highlighted transactions
- **False Positive Rate**: <10% for work-related classification
- **User Satisfaction**: >90% accuracy in taxpayer feedback
- **ATO Audit Readiness**: 100% compliance with documentation requirements

---

## TECHNICAL CONSIDERATIONS

### Computer Vision Challenges
- **Highlight Color Variation** - Different markers, fading, scanning artifacts
- **Document Quality** - Low resolution, skewed scans, mobile photos
- **Bank Format Diversity** - Different layouts across institutions
- **OCR Accuracy** - Text extraction from highlighted regions

### ATO Compliance Requirements
- **Transaction Completeness** - Date, amount, description minimum
- **Business Purpose** - Inferring work-relatedness from merchant data
- **Documentation Standards** - Meeting audit requirements
- **Privacy Protection** - Masking sensitive account information

### Performance Optimization
- **Parallel Processing** - Highlight detection + text extraction
- **Model Efficiency** - Optimizing InternVL for bank statement layouts
- **Caching Strategy** - Reusing processed regions across similar statements
- **Error Recovery** - Graceful degradation when highlights aren't detected

This extension positions the InternVL system as a comprehensive solution for Australian work-related expense claims, handling both traditional receipts and the critical bank statement evidence type with intelligent highlight detection and ATO compliance assessment.