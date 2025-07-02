# Australian Taxation Office Work-Related Expense Claim Extraction

## Executive Summary

The InternVL system needs to be reoriented from general supermarket receipt processing to **Australian Taxation Office (ATO) compliant work-related expense claim extraction**. This document outlines the critical changes needed to support legitimate business expense claims for Australian tax returns.

## ATO Work-Related Expense Requirements

### Critical ATO Receipt Requirements

According to ATO guidelines, valid work-related expense receipts must contain:

1. **Supplier Details**
   - Business name of the supplier
   - Australian Business Number (ABN) - **MANDATORY for claims >$82.50**
   - Supplier address (for claims >$82.50)

2. **Transaction Details** 
   - Date of purchase (DD/MM/YYYY Australian format)
   - Description of goods/services purchased
   - GST amount (if applicable)
   - Total amount paid

3. **Business Justification Fields**
   - Clear description of business purpose
   - Work-related category classification

### ATO Work Expense Categories

```python
# ATO-compliant work expense categories
ATO_WORK_EXPENSE_CATEGORIES = {
    'vehicle_expenses': {
        'fuel': ['petrol', 'diesel', 'lpg', 'fuel', 'gas'],
        'maintenance': ['service', 'repair', 'oil change', 'tyres', 'battery'],
        'registration': ['registration', 'rego', 'insurance', 'ctp'],
        'parking_tolls': ['parking', 'toll', 'meter', 'garage']
    },
    'travel_expenses': {
        'accommodation': ['hotel', 'motel', 'accommodation', 'lodging'],
        'flights': ['airline', 'flight', 'airfare', 'jetstar', 'qantas', 'virgin'],
        'taxi_uber': ['taxi', 'uber', 'rideshare', 'cab'],
        'public_transport': ['train', 'bus', 'metro', 'opal', 'myki']
    },
    'office_expenses': {
        'stationery': ['pen', 'paper', 'notebook', 'folder', 'stapler'],
        'computer_equipment': ['laptop', 'computer', 'mouse', 'keyboard', 'monitor'],
        'software': ['software', 'subscription', 'license', 'app'],
        'internet_phone': ['internet', 'mobile', 'phone', 'broadband']
    },
    'clothing_equipment': {
        'protective_clothing': ['safety', 'uniform', 'hard hat', 'boots', 'gloves'],
        'tools': ['tool', 'equipment', 'machinery', 'drill', 'hammer']
    },
    'education_training': {
        'courses': ['course', 'training', 'workshop', 'seminar', 'conference'],
        'books_materials': ['book', 'manual', 'textbook', 'materials']
    },
    'meals_entertainment': {
        'client_meals': ['restaurant', 'cafe', 'catering', 'meal'],
        'conference_meals': ['conference', 'seminar', 'workshop']
    }
}
```

---

## ATO-Compliant Extraction Schema

### Enhanced Receipt Schema for ATO Compliance

```python
# internvl/schemas/ato_expense_schemas.py
from pydantic import BaseModel, validator, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import re

class ATOWorkExpenseReceipt(BaseModel):
    """ATO-compliant work-related expense receipt schema."""
    
    # TAXPAYER IDENTIFICATION (NO TAX NUMBERS EVER)
    taxpayer_name: str = Field(..., description="Full name of taxpayer claiming expense")
    taxpayer_address: str = Field(..., description="Residential address of taxpayer")
    
    # MANDATORY ATO FIELDS
    supplier_name: str = Field(..., description="Business name of supplier")
    supplier_abn: Optional[str] = Field(None, description="Australian Business Number (mandatory >$82.50)")
    supplier_address: Optional[str] = Field(None, description="Supplier business address")
    
    transaction_date: str = Field(..., description="Purchase date (DD/MM/YYYY)")
    total_amount: str = Field(..., description="Total amount paid including GST")
    gst_amount: Optional[str] = Field(None, description="GST component if applicable")
    
    # EXPENSE DETAILS
    expense_description: str = Field(..., description="Clear description of goods/services")
    expense_category: str = Field(..., description="ATO work expense category")
    expense_subcategory: Optional[str] = Field(None, description="Specific subcategory")
    
    # BUSINESS JUSTIFICATION
    business_purpose: Optional[str] = Field(None, description="How expense relates to work")
    work_percentage: Optional[int] = Field(100, description="Percentage used for work (0-100)")
    
    # PAYMENT METHOD
    payment_method: Optional[str] = Field(None, description="Cash/Card/EFTPOS/etc")
    
    # DETAILED ITEMS (for itemized receipts)
    line_items: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Individual items if applicable")
    
    @validator('supplier_abn')
    def validate_abn_format(cls, v):
        """Validate Australian Business Number format."""
        if v is None:
            return v
        
        # ABN is 11 digits, often formatted with spaces: "XX XXX XXX XXX"
        abn_digits = re.sub(r'\s+', '', v)
        
        if not re.match(r'^\d{11}$', abn_digits):
            raise ValueError(f"Invalid ABN format: {v}. ABN must be 11 digits.")
        
        # Format with spaces for consistency
        return f"{abn_digits[:2]} {abn_digits[2:5]} {abn_digits[5:8]} {abn_digits[8:11]}"
    
    @validator('transaction_date')
    def validate_australian_date(cls, v):
        """Validate date is in Australian DD/MM/YYYY format."""
        if not v:
            return v
        
        # Try to parse as Australian date format
        try:
            datetime.strptime(v, '%d/%m/%Y')
            return v
        except ValueError:
            # Try other common formats and convert
            formats = ['%d-%m-%Y', '%d.%m.%Y', '%Y-%m-%d', '%m/%d/%Y']
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(v, fmt)
                    return parsed_date.strftime('%d/%m/%Y')
                except ValueError:
                    continue
            
            raise ValueError(f"Invalid date format: {v}. Use DD/MM/YYYY format.")
    
    @validator('total_amount', 'gst_amount')
    def validate_currency_amount(cls, v):
        """Validate currency amounts."""
        if not v:
            return v
        
        # Extract numeric value
        clean_amount = re.sub(r'[^\d.]', '', str(v))
        try:
            amount = float(clean_amount)
            return f"{amount:.2f}"
        except ValueError:
            raise ValueError(f"Invalid currency amount: {v}")
    
    @validator('work_percentage')
    def validate_work_percentage(cls, v):
        """Validate work percentage is between 0-100."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Work percentage must be between 0 and 100")
        return v
    
    @validator('taxpayer_name')
    def validate_taxpayer_name(cls, v):
        """Validate taxpayer name format."""
        if not v or len(v.strip()) < 2:
            raise ValueError("Taxpayer name must be provided")
        
        # Basic name validation - should contain letters
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError("Taxpayer name must contain letters")
        
        return v.strip().title()
    
    @validator('taxpayer_address')
    def validate_taxpayer_address(cls, v):
        """Validate taxpayer address format."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Taxpayer address must be provided and be reasonably complete")
        
        # Check for basic Australian address components
        address_indicators = ['street', 'st', 'road', 'rd', 'avenue', 'ave', 'place', 'pl', 'drive', 'dr', 'nsw', 'vic', 'qld', 'wa', 'sa', 'tas', 'nt', 'act']
        if not any(indicator in v.lower() for indicator in address_indicators):
            raise ValueError("Address should include street and state information")
        
        return v.strip()
    
    @validator('expense_category')
    def validate_ato_category(cls, v):
        """Validate against ATO-approved expense categories."""
        valid_categories = [
            'vehicle_expenses', 'travel_expenses', 'office_expenses',
            'clothing_equipment', 'education_training', 'meals_entertainment',
            'other_work_expenses'
        ]
        
        if v not in valid_categories:
            raise ValueError(f"Invalid expense category: {v}. Must be one of: {valid_categories}")
        
        return v
    
    def is_abn_required(self) -> bool:
        """Check if ABN is required based on amount."""
        try:
            amount = float(self.total_amount)
            return amount > 82.50  # ATO threshold
        except (ValueError, TypeError):
            return True  # Default to requiring ABN if amount unclear
    
    def validate_ato_compliance(self) -> List[str]:
        """Validate full ATO compliance and return any issues."""
        issues = []
        
        # Check ABN requirement
        if self.is_abn_required() and not self.supplier_abn:
            issues.append(f"ABN required for claims over $82.50 (amount: ${self.total_amount})")
        
        # Check mandatory fields
        if not self.supplier_name:
            issues.append("Supplier name is mandatory for ATO claims")
        
        if not self.expense_description:
            issues.append("Expense description is mandatory for ATO claims")
        
        # Check GST validation for amounts > $82.50
        if self.is_abn_required() and self.gst_amount:
            try:
                total = float(self.total_amount)
                gst = float(self.gst_amount)
                expected_gst = total / 11  # GST-inclusive calculation
                
                if abs(gst - expected_gst) > 0.10:  # 10 cent tolerance
                    issues.append(f"GST amount appears incorrect: ${gst:.2f} vs expected ${expected_gst:.2f}")
            except ValueError:
                issues.append("Cannot validate GST calculation - invalid amounts")
        
        return issues
    
    class Config:
        schema_extra = {
            "example": {
                "taxpayer_name": "John Smith",
                "taxpayer_address": "45 Collins Street, Melbourne VIC 3000",
                "supplier_name": "CALTEX AUSTRALIA PETROLEUM",
                "supplier_abn": "11 000 014 675",
                "supplier_address": "123 Pacific Highway, North Sydney NSW 2060",
                "transaction_date": "15/06/2024",
                "total_amount": "85.50",
                "gst_amount": "7.77",
                "expense_description": "Fuel for work vehicle - client visits",
                "expense_category": "vehicle_expenses",
                "expense_subcategory": "fuel",
                "business_purpose": "Travel to client meetings",
                "work_percentage": 100,
                "payment_method": "Credit Card",
                "line_items": [
                    {"item": "Unleaded Petrol", "quantity": "45.2L", "amount": "85.50"}
                ]
            }
        }
```

---

## ATO-Specific Robust Prompts

### Enhanced Prompt for ATO Work Expense Claims

```python
# internvl/prompts/ato_work_expense_prompts.py

ATO_WORK_EXPENSE_EXTRACTION_PROMPT = """
<image>
Extract information from this Australian work-related expense receipt for ATO tax compliance.

Use this EXACT format:

TAXPAYER_NAME: [Full name of person claiming this expense]
TAXPAYER_ADDRESS: [Residential address of taxpayer]
SUPPLIER_NAME: [Business name of the supplier]
SUPPLIER_ABN: [Australian Business Number - format: XX XXX XXX XXX]
SUPPLIER_ADDRESS: [Business address if visible]
DATE: [Purchase date in DD/MM/YYYY format]
TOTAL: [Total amount paid including GST]
GST: [GST amount if shown separately]
DESCRIPTION: [What was purchased - be specific]
CATEGORY: [Work expense category - see below]
PURPOSE: [How this relates to work]
PAYMENT: [Payment method if shown]
ITEMS: [Detailed items if receipt is itemized - separate with |]

WORK EXPENSE CATEGORIES (choose most appropriate):
- vehicle_expenses: Fuel, car maintenance, registration, parking, tolls
- travel_expenses: Accommodation, flights, taxi/Uber, public transport
- office_expenses: Stationery, computer equipment, software, internet/phone
- clothing_equipment: Protective clothing, uniforms, tools, equipment
- education_training: Courses, training, books, professional development
- meals_entertainment: Client meals, conference catering
- other_work_expenses: Other legitimate work-related expenses

ATO COMPLIANCE REQUIREMENTS:
- ABN is MANDATORY for claims over $82.50
- Date must be in Australian DD/MM/YYYY format
- GST should be 10% in Australia (1/11 of GST-inclusive total)
- Description must clearly indicate business purpose
- Supplier name should be the registered business name

AUSTRALIAN BUSINESS CONTEXT:
- Common fuel suppliers: Caltex, Shell, BP, 7-Eleven, United, Ampol
- Airlines: Qantas, Jetstar, Virgin Australia, Tiger
- Car hire: Hertz, Avis, Budget, Redspot
- Accommodation: Hotels, motels, Airbnb
- Office supplies: Officeworks, Staples, Bunnings (tools)
- Technology: JB Hi-Fi, Harvey Norman, Apple Store

Example:
TAXPAYER_NAME: John Smith
TAXPAYER_ADDRESS: 45 Collins Street Melbourne VIC 3000
SUPPLIER_NAME: CALTEX AUSTRALIA PETROLEUM
SUPPLIER_ABN: 11 000 014 675
SUPPLIER_ADDRESS: 123 Pacific Highway North Sydney NSW 2060
DATE: 15/06/2024
TOTAL: 85.50
GST: 7.77
DESCRIPTION: Unleaded petrol for work vehicle
CATEGORY: vehicle_expenses
PURPOSE: Fuel for client visits and work travel
PAYMENT: Credit Card
ITEMS: Unleaded Petrol 45.2L | 85.50

CRITICAL: Extract information exactly as it appears on the receipt. If ABN not visible but amount >$82.50, note "ABN REQUIRED BUT NOT VISIBLE".
"""

ATO_SIMPLE_EXTRACTION_PROMPT = """
<image>
Extract from this Australian work expense receipt:

TAXPAYER_NAME: [Person claiming expense]
TAXPAYER_ADDRESS: [Taxpayer's address]
SUPPLIER_NAME: [Who was paid]
SUPPLIER_ABN: [ABN if visible]
DATE: [When - DD/MM/YYYY]
TOTAL: [Amount paid]
GST: [Tax amount]
DESCRIPTION: [What was bought]
CATEGORY: [Type of work expense]
PURPOSE: [Why work-related]

Focus on ATO tax compliance requirements.
IMPORTANT: NO TAX FILE NUMBERS - only name and address.
"""
```

---

## ATO-Specific Post-Processing

### Enhanced Post-Processor for ATO Compliance

```python
# internvl/extraction/ato_postprocessor.py
from typing import Dict, List, Any, Optional
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ATOCompliancePostProcessor:
    """Post-processor specialized for ATO work expense compliance."""
    
    def __init__(self):
        # Australian fuel/service station chains
        self.fuel_suppliers = {
            'caltex', 'shell', 'bp', '7-eleven', 'seven eleven', 'united', 
            'ampol', 'mobil', 'esso', 'liberty', 'costco fuel'
        }
        
        # Australian business name patterns
        self.business_patterns = {
            'airlines': ['qantas', 'jetstar', 'virgin', 'tiger', 'rex'],
            'car_hire': ['hertz', 'avis', 'budget', 'redspot', 'europcar'],
            'accommodation': ['hotel', 'motel', 'inn', 'resort', 'lodge'],
            'office_retail': ['officeworks', 'staples', 'jb hi-fi', 'harvey norman'],
            'transport': ['uber', 'taxi', 'metro', 'transport', 'parking']
        }
        
        # ATO expense categorization
        self.ato_category_keywords = {
            'vehicle_expenses': [
                'fuel', 'petrol', 'diesel', 'gas', 'service', 'repair', 
                'oil', 'tyre', 'battery', 'registration', 'insurance', 
                'parking', 'toll', 'car wash'
            ],
            'travel_expenses': [
                'hotel', 'motel', 'accommodation', 'flight', 'airline', 
                'taxi', 'uber', 'train', 'bus', 'metro', 'transport'
            ],
            'office_expenses': [
                'stationery', 'paper', 'pen', 'computer', 'laptop', 'software', 
                'internet', 'phone', 'mobile', 'broadband', 'printer'
            ],
            'clothing_equipment': [
                'uniform', 'safety', 'boots', 'helmet', 'tools', 'equipment', 
                'drill', 'hammer', 'protective'
            ],
            'education_training': [
                'course', 'training', 'workshop', 'seminar', 'conference', 
                'book', 'manual', 'certification'
            ],
            'meals_entertainment': [
                'restaurant', 'cafe', 'catering', 'meal', 'lunch', 'dinner'
            ]
        }
    
    def process_ato_extraction(self, raw_extraction: Dict[str, str]) -> Dict[str, Any]:
        """Process extraction results for ATO compliance."""
        
        logger.info("Processing extraction for ATO work expense compliance")
        
        # Stage 1: Basic field extraction and normalization
        processed = {
            'taxpayer_name': self._normalize_taxpayer_name(raw_extraction.get('TAXPAYER_NAME', '')),
            'taxpayer_address': self._normalize_taxpayer_address(raw_extraction.get('TAXPAYER_ADDRESS', '')),
            'supplier_name': self._normalize_supplier_name(raw_extraction.get('SUPPLIER_NAME', '')),
            'supplier_abn': self._normalize_abn(raw_extraction.get('SUPPLIER_ABN', '')),
            'supplier_address': self._normalize_address(raw_extraction.get('SUPPLIER_ADDRESS', '')),
            'transaction_date': self._normalize_australian_date(raw_extraction.get('DATE', '')),
            'total_amount': self._normalize_currency(raw_extraction.get('TOTAL', '')),
            'gst_amount': self._normalize_currency(raw_extraction.get('GST', '')),
            'expense_description': self._normalize_description(raw_extraction.get('DESCRIPTION', '')),
            'business_purpose': raw_extraction.get('PURPOSE', ''),
            'payment_method': self._normalize_payment_method(raw_extraction.get('PAYMENT', ''))
        }
        
        # Stage 2: ATO category classification
        processed['expense_category'] = self._classify_ato_category(
            processed['expense_description'], 
            processed['supplier_name']
        )
        
        # Stage 3: ATO compliance validation
        processed['ato_compliance'] = self._validate_ato_compliance(processed)
        
        # Stage 4: Business context enhancement
        processed['business_context'] = self._enhance_business_context(processed)
        
        logger.info(f"ATO processing complete. Compliance status: {processed['ato_compliance']['compliant']}")
        
        return processed
    
    def _normalize_taxpayer_name(self, name_str: str) -> str:
        """Normalize taxpayer name for ATO compliance."""
        if not name_str:
            return ""
        
        # Clean and format name
        clean_name = name_str.strip()
        
        # Remove any potential sensitive information
        sensitive_patterns = [
            r'\b\d{8,9}\b',  # Potential TFN patterns
            r'\bTFN\b',      # TFN abbreviation
            r'\bTAX\s*FILE\s*NUMBER\b'  # Tax file number text
        ]
        
        for pattern in sensitive_patterns:
            clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
        
        # Format to title case
        clean_name = clean_name.title()
        
        # Basic validation - should look like a name
        if len(clean_name.strip()) < 2:
            logger.warning(f"Taxpayer name appears too short: '{name_str}'")
            return name_str
        
        return clean_name.strip()
    
    def _normalize_taxpayer_address(self, address_str: str) -> str:
        """Normalize taxpayer address for ATO compliance."""
        if not address_str:
            return ""
        
        # Clean address
        clean_address = address_str.strip()
        
        # Remove any potential sensitive information
        sensitive_patterns = [
            r'\b\d{8,9}\b',  # Potential TFN patterns
            r'\bTFN\b',      # TFN abbreviation
        ]
        
        for pattern in sensitive_patterns:
            clean_address = re.sub(pattern, '', clean_address, flags=re.IGNORECASE)
        
        # Standardize Australian state abbreviations
        state_mappings = {
            'NEW SOUTH WALES': 'NSW',
            'VICTORIA': 'VIC', 
            'QUEENSLAND': 'QLD',
            'WESTERN AUSTRALIA': 'WA',
            'SOUTH AUSTRALIA': 'SA',
            'TASMANIA': 'TAS',
            'NORTHERN TERRITORY': 'NT',
            'AUSTRALIAN CAPITAL TERRITORY': 'ACT'
        }
        
        for full_state, abbrev in state_mappings.items():
            clean_address = re.sub(full_state, abbrev, clean_address, flags=re.IGNORECASE)
        
        # Title case for readability
        clean_address = clean_address.title()
        
        return clean_address.strip()
    
    def _normalize_supplier_name(self, supplier_str: str) -> str:
        """Normalize supplier name for ATO compliance."""
        if not supplier_str:
            return ""
        
        # Clean and standardize
        clean_name = supplier_str.strip().upper()
        
        # Remove common suffixes
        suffixes_to_remove = [
            r'\bPTY\s*LTD\b', r'\bLIMITED\b', r'\bLTD\b', 
            r'\bAUSTRALIA\b', r'\bAU\b', r'\bINC\b'
        ]
        
        for suffix in suffixes_to_remove:
            clean_name = re.sub(suffix, '', clean_name).strip()
        
        # Standardize known Australian businesses
        business_mappings = {
            'CALTEX': 'CALTEX AUSTRALIA PETROLEUM',
            'SHELL': 'SHELL AUSTRALIA',
            '7-ELEVEN': '7-ELEVEN STORES',
            'QANTAS': 'QANTAS AIRWAYS',
            'OFFICEWORKS': 'OFFICEWORKS',
            'JB HI-FI': 'JB HI-FI',
            'HARVEY NORMAN': 'HARVEY NORMAN'
        }
        
        for key, value in business_mappings.items():
            if key in clean_name:
                return value
        
        return clean_name
    
    def _normalize_abn(self, abn_str: str) -> str:
        """Normalize ABN to standard format."""
        if not abn_str:
            return ""
        
        # Extract digits only
        digits = re.sub(r'\D', '', abn_str)
        
        if len(digits) != 11:
            logger.warning(f"ABN '{abn_str}' does not contain 11 digits")
            return abn_str  # Return original if invalid
        
        # Format as XX XXX XXX XXX
        return f"{digits[:2]} {digits[2:5]} {digits[5:8]} {digits[8:11]}"
    
    def _normalize_australian_date(self, date_str: str) -> str:
        """Normalize to Australian DD/MM/YYYY format."""
        if not date_str:
            return ""
        
        # Australian date formats
        formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
            '%d/%m/%y', '%d-%m-%y',
            '%Y-%m-%d',  # ISO format
            '%m/%d/%Y'   # US format (less common)
        ]
        
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                return parsed_date.strftime('%d/%m/%Y')
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: '{date_str}'")
        return date_str
    
    def _normalize_currency(self, currency_str: str) -> str:
        """Normalize currency for ATO compliance."""
        if not currency_str:
            return ""
        
        # Extract numeric value
        clean_amount = re.sub(r'[^\d.]', '', currency_str)
        
        try:
            amount = float(clean_amount)
            return f"{amount:.2f}"
        except ValueError:
            logger.warning(f"Could not parse currency: '{currency_str}'")
            return currency_str
    
    def _normalize_description(self, description_str: str) -> str:
        """Normalize expense description for ATO clarity."""
        if not description_str:
            return ""
        
        # Clean and standardize
        clean_desc = description_str.strip().title()
        
        # Expand common abbreviations for clarity
        abbreviations = {
            'Fuel': 'Motor Vehicle Fuel',
            'Petrol': 'Motor Vehicle Petrol', 
            'Diesel': 'Motor Vehicle Diesel',
            'Parking': 'Vehicle Parking',
            'Phone': 'Mobile Phone',
            'Internet': 'Internet Service'
        }
        
        for abbrev, expansion in abbreviations.items():
            if clean_desc.lower().startswith(abbrev.lower()):
                clean_desc = expansion
                break
        
        return clean_desc
    
    def _normalize_payment_method(self, payment_str: str) -> str:
        """Normalize payment method."""
        if not payment_str:
            return ""
        
        payment_mappings = {
            'card': 'Credit Card',
            'credit': 'Credit Card',
            'debit': 'Debit Card',
            'eftpos': 'EFTPOS',
            'cash': 'Cash',
            'cheque': 'Cheque',
            'bank transfer': 'Bank Transfer'
        }
        
        payment_lower = payment_str.lower()
        for key, value in payment_mappings.items():
            if key in payment_lower:
                return value
        
        return payment_str.title()
    
    def _classify_ato_category(self, description: str, supplier: str) -> str:
        """Classify expense into ATO-compliant category."""
        
        # Combine description and supplier for classification
        text_to_analyze = f"{description} {supplier}".lower()
        
        category_scores = {}
        
        # Score each category based on keyword matches
        for category, keywords in self.ato_category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_to_analyze)
            if score > 0:
                category_scores[category] = score
        
        # Return highest scoring category, or 'other_work_expenses' if none match
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'other_work_expenses'
    
    def _validate_ato_compliance(self, processed: Dict[str, str]) -> Dict[str, Any]:
        """Validate ATO compliance requirements."""
        
        compliance_issues = []
        warnings = []
        
        # Check total amount
        try:
            total_amount = float(processed.get('total_amount', '0'))
        except ValueError:
            total_amount = 0
            compliance_issues.append("Invalid total amount format")
        
        # ABN requirement check
        if total_amount > 82.50:
            if not processed.get('supplier_abn'):
                compliance_issues.append(f"ABN required for claims over $82.50 (amount: ${total_amount:.2f})")
            else:
                # Validate ABN format
                abn = processed.get('supplier_abn', '')
                if not re.match(r'^\d{2} \d{3} \d{3} \d{3}$', abn):
                    compliance_issues.append(f"Invalid ABN format: {abn}")
        
        # Mandatory field checks
        mandatory_fields = [
            ('taxpayer_name', 'Taxpayer name'),
            ('taxpayer_address', 'Taxpayer address'),
            ('supplier_name', 'Supplier name'),
            ('transaction_date', 'Transaction date'),
            ('total_amount', 'Total amount'),
            ('expense_description', 'Expense description')
        ]
        
        for field, field_name in mandatory_fields:
            if not processed.get(field):
                compliance_issues.append(f"{field_name} is mandatory for ATO claims")
        
        # GST validation
        if processed.get('gst_amount') and processed.get('total_amount'):
            try:
                gst = float(processed['gst_amount'])
                total = float(processed['total_amount'])
                expected_gst = total / 11  # GST-inclusive calculation
                
                if abs(gst - expected_gst) > 0.50:  # 50 cent tolerance
                    warnings.append(f"GST amount may be incorrect: ${gst:.2f} vs expected ${expected_gst:.2f}")
            except ValueError:
                warnings.append("Could not validate GST calculation")
        
        # Date validation
        if processed.get('transaction_date'):
            try:
                claim_date = datetime.strptime(processed['transaction_date'], '%d/%m/%Y')
                current_date = datetime.now()
                
                # Check if date is in the future
                if claim_date > current_date:
                    compliance_issues.append("Transaction date cannot be in the future")
                
                # Check if date is more than 5 years old (ATO record keeping requirement)
                years_old = (current_date - claim_date).days / 365.25
                if years_old > 5:
                    warnings.append(f"Receipt is {years_old:.1f} years old - check ATO record keeping requirements")
                    
            except ValueError:
                compliance_issues.append("Invalid date format for ATO compliance")
        
        return {
            'compliant': len(compliance_issues) == 0,
            'issues': compliance_issues,
            'warnings': warnings,
            'requires_abn': total_amount > 82.50,
            'total_amount': total_amount
        }
    
    def _enhance_business_context(self, processed: Dict[str, str]) -> Dict[str, Any]:
        """Add business context for ATO purposes."""
        
        supplier = processed.get('supplier_name', '').lower()
        description = processed.get('expense_description', '').lower()
        category = processed.get('expense_category', '')
        
        context = {
            'business_type': 'unknown',
            'likely_deductible': True,
            'common_work_expense': False,
            'requires_evidence': False
        }
        
        # Identify business type
        for business_type, businesses in self.business_patterns.items():
            if any(business in supplier for business in businesses):
                context['business_type'] = business_type
                break
        
        # Check if it's a common work expense
        common_expenses = [
            'fuel', 'parking', 'toll', 'stationery', 'computer', 'phone', 
            'internet', 'training', 'conference', 'tools'
        ]
        
        context['common_work_expense'] = any(expense in description for expense in common_expenses)
        
        # Flag expenses that require additional evidence
        evidence_required_categories = ['meals_entertainment', 'travel_expenses']
        if category in evidence_required_categories:
            context['requires_evidence'] = True
        
        # Special handling for vehicle expenses
        if category == 'vehicle_expenses':
            context['vehicle_expense_method'] = self._determine_vehicle_method(description)
        
        return context
    
    def _determine_vehicle_method(self, description: str) -> str:
        """Determine if vehicle expense should use logbook or cents per km method."""
        
        # If fuel/ongoing costs, likely logbook method
        ongoing_costs = ['fuel', 'petrol', 'diesel', 'service', 'repair', 'oil', 'registration', 'insurance']
        if any(cost in description.lower() for cost in ongoing_costs):
            return 'logbook_method_recommended'
        
        # If one-off travel, could be cents per km
        travel_costs = ['parking', 'toll']
        if any(cost in description.lower() for cost in travel_costs):
            return 'cents_per_km_or_logbook'
        
        return 'method_depends_on_usage'
```

---

## ATO-Specific CLI Tools

### Enhanced CLI for ATO Work Expense Processing

```python
# internvl/cli/ato_work_expense_extractor.py
import logging
from pathlib import Path
from typing import Dict, Any
import json

from internvl.extraction.ato_postprocessor import ATOCompliancePostProcessor
from internvl.schemas.ato_expense_schemas import ATOWorkExpenseReceipt
from internvl.prompts.ato_work_expense_prompts import ATO_WORK_EXPENSE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

class ATOWorkExpenseExtractor:
    """Extractor specialized for ATO work-related expense claims."""
    
    def __init__(self, model, parser):
        self.model = model
        self.parser = parser
        self.ato_processor = ATOCompliancePostProcessor()
    
    def extract_work_expense(self, image_path: Path) -> Dict[str, Any]:
        """Extract work expense information for ATO compliance."""
        
        logger.info(f"Extracting ATO work expense from: {image_path}")
        
        # Stage 1: Primary extraction
        response = self.model.generate(str(image_path), ATO_WORK_EXPENSE_EXTRACTION_PROMPT)
        kv_result = self.parser.parse_key_value_response(response)
        
        # Stage 2: ATO-specific post-processing
        ato_processed = self.ato_processor.process_ato_extraction(kv_result.extracted_fields)
        
        # Stage 3: Schema validation
        try:
            validated_receipt = ATOWorkExpenseReceipt(**ato_processed)
            ato_compliance_issues = validated_receipt.validate_ato_compliance()
            
            ato_processed['schema_validation'] = {
                'valid': len(ato_compliance_issues) == 0,
                'issues': ato_compliance_issues
            }
            
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
            ato_processed['schema_validation'] = {
                'valid': False,
                'issues': [str(e)]
            }
        
        # Stage 4: Generate ATO compliance report
        ato_processed['ato_compliance_report'] = self._generate_compliance_report(ato_processed)
        
        return ato_processed
    
    def _generate_compliance_report(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive ATO compliance report."""
        
        compliance = processed.get('ato_compliance', {})
        schema_validation = processed.get('schema_validation', {})
        
        report = {
            'compliant': compliance.get('compliant', False) and schema_validation.get('valid', False),
            'ready_for_ato_claim': False,
            'required_actions': [],
            'recommendations': []
        }
        
        # Collect all issues
        all_issues = compliance.get('issues', []) + schema_validation.get('issues', [])
        all_warnings = compliance.get('warnings', [])
        
        # Determine if ready for ATO claim
        critical_issues = [issue for issue in all_issues if 'mandatory' in issue.lower() or 'required' in issue.lower()]
        report['ready_for_ato_claim'] = len(critical_issues) == 0
        
        # Generate required actions
        if all_issues:
            report['required_actions'] = [
                f"Fix: {issue}" for issue in all_issues
            ]
        
        # Generate recommendations
        if all_warnings:
            report['recommendations'].extend([
                f"Review: {warning}" for warning in all_warnings
            ])
        
        # Add category-specific recommendations
        category = processed.get('expense_category')
        if category == 'vehicle_expenses':
            report['recommendations'].append("Consider whether logbook method or cents per km is more beneficial")
        elif category == 'meals_entertainment':
            report['recommendations'].append("Ensure you have evidence of business purpose for meal expenses")
        elif category == 'travel_expenses':
            report['recommendations'].append("Keep evidence of business travel purpose and itinerary")
        
        return report

def main():
    """Main function for ATO work expense extraction."""
    import argparse
    from internvl.model.loader import load_model
    from internvl.extraction.key_value_parser import KeyValueParser
    from internvl.config import get_config
    
    parser = argparse.ArgumentParser(description="ATO Work Expense Receipt Extractor")
    parser.add_argument("--image-path", type=Path, required=True, help="Path to work expense receipt")
    parser.add_argument("--output-path", type=Path, help="Output path for ATO extraction results")
    parser.add_argument("--compliance-check", action="store_true", help="Perform detailed ATO compliance check")
    
    args = parser.parse_args()
    
    if not args.image_path.exists():
        logger.error(f"Receipt image not found: {args.image_path}")
        return 1
    
    # Setup extraction pipeline
    config = get_config()
    model = load_model(config.model_path)
    kv_parser = KeyValueParser()
    
    extractor = ATOWorkExpenseExtractor(model, kv_parser)
    
    # Extract work expense
    result = extractor.extract_work_expense(args.image_path)
    
    # Save results
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ATO extraction results saved to: {args.output_path}")
    
    # Display compliance summary
    compliance_report = result.get('ato_compliance_report', {})
    
    print(f"\n=== ATO WORK EXPENSE EXTRACTION ===")
    print(f"Receipt: {args.image_path.name}")
    print(f"Taxpayer: {result.get('taxpayer_name', 'N/A')}")
    print(f"Address: {result.get('taxpayer_address', 'N/A')}")
    print(f"Supplier: {result.get('supplier_name', 'N/A')}")
    print(f"Date: {result.get('transaction_date', 'N/A')}")
    print(f"Amount: ${result.get('total_amount', 'N/A')}")
    print(f"Category: {result.get('expense_category', 'N/A')}")
    print(f"Description: {result.get('expense_description', 'N/A')}")
    
    print(f"\n=== ATO COMPLIANCE STATUS ===")
    print(f"Compliant: {'✅ YES' if compliance_report.get('compliant') else '❌ NO'}")
    print(f"Ready for Claim: {'✅ YES' if compliance_report.get('ready_for_ato_claim') else '❌ NO'}")
    
    if compliance_report.get('required_actions'):
        print(f"\n⚠️  REQUIRED ACTIONS:")
        for action in compliance_report['required_actions']:
            print(f"  • {action}")
    
    if compliance_report.get('recommendations'):
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in compliance_report['recommendations']:
            print(f"  • {rec}")
    
    # Return appropriate exit code
    return 0 if compliance_report.get('ready_for_ato_claim') else 1

if __name__ == "__main__":
    exit(main())
```

---

## PRIVACY AND SECURITY POLICY

### 🔒 **CRITICAL: NO TAX FILE NUMBERS (TFN) POLICY**

**ABSOLUTE PROHIBITION**: This system **NEVER** captures, stores, or processes Australian Tax File Numbers (TFN) or any tax identification numbers.

**Privacy Protection Measures:**
- ✅ **Taxpayer name and address** - Required for ATO compliance
- ❌ **Tax File Numbers (TFN)** - NEVER captured or stored
- ❌ **Medicare numbers** - Not relevant for expense claims
- ❌ **Social security numbers** - Not used in Australia
- ✅ **Business ABN only** - Required for supplier identification

**Validation Safeguards:**
- Automatic detection and removal of potential TFN patterns
- Schema validation prevents TFN field creation
- Post-processing filters remove sensitive number patterns
- Logging warnings for any suspicious number patterns

**ATO Compliance Note**: The ATO requires taxpayer identification (name/address) for expense claims but **explicitly prohibits** requiring TFN disclosure for receipt processing.

---

## KEY CHANGES FOR ATO FOCUS

### 🎯 **Critical Shifts from Supermarket to ATO Focus**

1. **Taxpayer Identification**: Name and address (NO tax numbers)
2. **ABN Requirement**: Mandatory for claims >$82.50 (ATO threshold)
3. **Business Purpose**: Every expense must have clear work-related justification  
4. **ATO Categories**: Proper classification into tax-deductible categories
5. **Date Format**: Australian DD/MM/YYYY format compliance
6. **GST Validation**: 10% Australian GST calculation verification
7. **Privacy Protection**: Automatic removal of sensitive number patterns
8. **Record Keeping**: 5-year retention requirement awareness

### 🏢 **Business Types Supported**

- **Vehicle Expenses**: Fuel, maintenance, parking, tolls
- **Travel Expenses**: Flights, accommodation, transport
- **Office Expenses**: Equipment, software, stationery  
- **Education**: Training, courses, professional development
- **Tools/Equipment**: Work-related tools and protective clothing
- **Meals**: Client entertainment, conference meals

### 📊 **ATO Compliance Features**

- **Automatic ABN validation** and formatting
- **Expense categorization** into ATO-approved categories  
- **Business purpose** extraction and validation
- **GST calculation** verification
- **Compliance reporting** with required actions
- **Record keeping** recommendations

This transformation makes the system **Australian Taxation Office compliant** and suitable for legitimate work-related expense claims! 🇦🇺📋
