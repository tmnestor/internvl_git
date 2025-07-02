# InternVL Robust Information Extraction: Implementation Guide

## Executive Summary

While the evaluation system improvements focus on robust assessment of extraction quality, this document outlines complementary improvements to make the information extraction pipeline itself more robust and reliable. These enhancements will reduce extraction failures, improve consistency, and provide better handling of real-world receipt variations in Australian business contexts.

## Current Extraction Pipeline Assessment

### Strengths
- ✅ Key-value prompting eliminates JSON parsing issues
- ✅ Modular architecture with clear separation
- ✅ Australian-specific prompt optimization
- ✅ Support for multiple prompt templates

### Critical Gaps
- ❌ Single-shot extraction without fallback strategies
- ❌ No post-processing normalization for format variations
- ❌ Limited handling of date/currency format diversity
- ❌ No consistency validation between extracted fields
- ❌ No confidence scoring for extraction quality
- ❌ Brittle handling of Australian business naming variations

---

## ROBUST EXTRACTION IMPROVEMENTS

### 1. Enhanced Key-Value Prompt Engineering

**Current State:** Basic key-value format prompts

**Enhancement:** Multi-format tolerant prompting with explicit robustness instructions

#### Implementation Plan

**Robust Australian Receipt Prompt:**
```python
# internvl/prompts/robust_extraction_prompts.py

ROBUST_AUSTRALIAN_RECEIPT_PROMPT = """
<image>
Extract information from this Australian receipt using this EXACT format:

DATE: [Find purchase date - accept ANY format: DD/MM/YYYY, DD-MM-YYYY, "15th June 2024", etc.]
STORE: [Store name - WOOLWORTHS, Coles, ALDI, IGA, Bunnings, etc.]
TAX: [GST amount - look for "GST", "TAX", or calculate 10% of subtotal]
TOTAL: [Final total including GST - accept $XX.XX, XX.XX AUD, "XX dollars", etc.]
PRODUCTS: [ALL items - separate with | - "Milk 2L | Bread | Eggs"]
QUANTITIES: [Quantities for each item - separate with | - "1 | 2 | 1"]
PRICES: [Price for each item - separate with | - "4.50 | 8.00 | 7.60"]

EXTRACTION RULES:
- DATE: Accept any date format, prefer DD/MM/YYYY for Australia
- STORE: Extract main store name (WOOLWORTHS not "Woolworths Metro Store 123")
- TAX: GST is 10% in Australia, look for GST/TAX lines
- TOTAL: Final amount paid, include currency symbol if visible
- PRODUCTS: Use Title Case ("Milk 2L" not "MILK 2L")
- QUANTITIES: Include units if shown ("2kg", "1L", or just "2")
- PRICES: Individual item prices, match currency format from receipt

ROBUSTNESS INSTRUCTIONS:
- If date has multiple formats, choose the clearest one
- If store name varies (Woolworths vs WOOLWORTHS), use capitals
- If GST not labeled, calculate 1/11 of total for GST-inclusive amounts
- If product names unclear, extract what's readable
- Ensure PRODUCTS, QUANTITIES, PRICES have same number of items
- If currency symbols missing, add $ for Australian context

AUSTRALIAN BUSINESS CONTEXT:
- Major retailers: WOOLWORTHS, COLES, ALDI, IGA, BUNNINGS, KMART, TARGET
- GST (Goods and Services Tax) is always 10% in Australia
- Date format preference: DD/MM/YYYY (day first)
- Currency: Australian Dollars (AUD) with $ symbol
- Common product units: kg, g, L, mL, pack, each

Example:
DATE: 16/03/2023
STORE: WOOLWORTHS
TAX: 3.82
TOTAL: $42.08
PRODUCTS: Milk Full Cream 2L | Bread Multigrain | Free Range Eggs 12pk
QUANTITIES: 1 | 2 | 1
PRICES: 4.50 | 8.00 | 7.60
"""

FALLBACK_SIMPLE_PROMPT = """
<image>
Extract from this Australian receipt:

DATE: [purchase date]
STORE: [store name]
TAX: [GST/tax amount]
TOTAL: [total amount]
PRODUCTS: [item1 | item2 | item3]
QUANTITIES: [qty1 | qty2 | qty3]
PRICES: [price1 | price2 | price3]

Use | to separate multiple items in lists.
Return empty if field not found.
"""

MINIMAL_EXTRACTION_PROMPT = """
<image>
Find these from the receipt:
DATE: 
STORE: 
TOTAL: 
PRODUCTS: 
"""
```

**Prompt Selection Strategy:**
```python
class PromptSelector:
    """Select appropriate prompt based on image quality and complexity."""
    
    def __init__(self):
        self.prompts = {
            'robust': ROBUST_AUSTRALIAN_RECEIPT_PROMPT,
            'simple': FALLBACK_SIMPLE_PROMPT,
            'minimal': MINIMAL_EXTRACTION_PROMPT
        }
    
    def select_prompt(self, image_analysis: Dict[str, Any]) -> str:
        """Select prompt based on image characteristics."""
        
        # Analyze image quality factors
        text_density = image_analysis.get('text_density', 'medium')
        image_quality = image_analysis.get('quality', 'good')
        receipt_complexity = image_analysis.get('complexity', 'standard')
        
        if image_quality == 'excellent' and receipt_complexity == 'standard':
            return self.prompts['robust']
        elif image_quality in ['good', 'fair']:
            return self.prompts['simple']
        else:
            return self.prompts['minimal']
```

---

### 2. Comprehensive Post-Processing Pipeline

**Current State:** Basic key-value parsing without normalization

**Enhancement:** Multi-stage post-processing with format normalization and consistency validation

#### Implementation Plan

**Robust Post-Processor:**
```python
# internvl/extraction/robust_postprocessor.py
from typing import Dict, List, Any, Optional
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RobustPostProcessor:
    """Post-process extraction results for maximum robustness and consistency."""
    
    def __init__(self):
        self.australian_stores = {
            'woolworths', 'coles', 'aldi', 'iga', 'bunnings', 'kmart', 
            'target', 'big w', 'jb hi-fi', 'harvey norman', 'officeworks',
            'costco', 'ikea', 'spotlight', 'super cheap auto', 'supercheap',
            'repco', 'autobarn', 'bcf', 'rebel', 'amart', 'fantastic furniture'
        }
        
        self.store_variations = {
            # Woolworths variations
            'woolworths metro': 'WOOLWORTHS',
            'woolworths supermarket': 'WOOLWORTHS',
            'woolworths petrol': 'WOOLWORTHS',
            'woolies': 'WOOLWORTHS',
            
            # Coles variations
            'coles express': 'COLES',
            'coles supermarket': 'COLES',
            'coles local': 'COLES',
            
            # Other common variations
            'bunnings warehouse': 'BUNNINGS',
            'iga supermarket': 'IGA',
            'target australia': 'TARGET',
            'kmart australia': 'KMART',
            'jb hifi': 'JB HI-FI',
            'harvey norman': 'HARVEY NORMAN',
            'super cheap auto': 'SUPERCHEAP AUTO',
            'supercheap auto': 'SUPERCHEAP AUTO',
        }
        
    def robust_postprocess(self, raw_extraction: Dict[str, str]) -> Dict[str, Any]:
        """Apply comprehensive post-processing to extraction results."""
        
        logger.info("Starting robust post-processing of extraction results")
        
        # Stage 1: Basic field normalization
        processed = {
            'date_value': self._normalize_date(raw_extraction.get('DATE', '')),
            'store_name_value': self._normalize_store_name(raw_extraction.get('STORE', '')),
            'tax_value': self._normalize_currency(raw_extraction.get('TAX', '')),
            'total_value': self._normalize_currency(raw_extraction.get('TOTAL', '')),
            'prod_item_value': self._normalize_product_list(raw_extraction.get('PRODUCTS', '')),
            'prod_quantity_value': self._normalize_quantity_list(raw_extraction.get('QUANTITIES', '')),
            'prod_price_value': self._normalize_price_list(raw_extraction.get('PRICES', ''))
        }
        
        # Stage 2: Consistency validation and correction
        processed = self._apply_consistency_corrections(processed)
        
        # Stage 3: Australian business rule validation
        processed = self._apply_australian_business_rules(processed)
        
        # Stage 4: Quality scoring
        processed['extraction_quality'] = self._calculate_extraction_quality(processed)
        
        logger.info(f"Post-processing complete. Quality score: {processed['extraction_quality']['overall_score']:.2f}")
        
        return processed
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to Australian DD/MM/YYYY format with maximum robustness."""
        if not date_str or date_str.strip() == '':
            return ""
        
        clean_str = date_str.strip()
        
        # Australian date formats (DD/MM/YYYY is standard)
        formats = [
            # Primary Australian formats
            '%d/%m/%Y',    # 15/06/2024
            '%d-%m-%Y',    # 15-06-2024
            '%d.%m.%Y',    # 15.06.2024
            '%d %m %Y',    # 15 06 2024
            
            # Short year formats
            '%d/%m/%y',    # 15/06/24
            '%d-%m-%y',    # 15-06-24
            
            # Month name formats
            '%d %B %Y',    # 15 June 2024
            '%d %b %Y',    # 15 Jun 2024
            '%B %d, %Y',   # June 15, 2024
            '%b %d, %Y',   # Jun 15, 2024
            
            # ISO format
            '%Y-%m-%d',    # 2024-06-15
            
            # US format (less common in Australia but possible)
            '%m/%d/%Y',    # 06/15/2024
        ]
        
        # Try direct parsing first
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(clean_str.upper(), fmt)
                return parsed_date.strftime('%d/%m/%Y')
            except ValueError:
                continue
        
        # Try regex extraction for complex cases
        date_patterns = [
            # DD/MM/YYYY or similar with separators
            (r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})', 
             lambda m: f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/{m.group(3)}"),
            
            # DD/MM/YY
            (r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2})', 
             lambda m: f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}/20{m.group(3)}"),
            
            # Handle ordinal dates like "15th June 2024"
            (r'(\d{1,2})(?:st|nd|rd|th)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})', 
             lambda m: self._parse_ordinal_date(m.group())),
        ]
        
        for pattern, formatter in date_patterns:
            match = re.search(pattern, clean_str.upper())
            if match:
                try:
                    result = formatter(match)
                    # Validate the result
                    datetime.strptime(result, '%d/%m/%Y')
                    return result
                except (ValueError, AttributeError):
                    continue
        
        logger.warning(f"Could not parse date: '{date_str}', returning original")
        return date_str
    
    def _parse_ordinal_date(self, date_str: str) -> str:
        """Parse ordinal dates like '15th June 2024'."""
        # Remove ordinal suffixes
        clean_date = re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', date_str, flags=re.IGNORECASE)
        
        # Try to parse the cleaned date
        formats = ['%d %B %Y', '%d %b %Y']
        for fmt in formats:
            try:
                parsed = datetime.strptime(clean_date, fmt)
                return parsed.strftime('%d/%m/%Y')
            except ValueError:
                continue
        
        return date_str
    
    def _normalize_store_name(self, store_str: str) -> str:
        """Normalize store name for Australian retail consistency."""
        if not store_str or store_str.strip() == '':
            return ""
        
        # Clean and normalize
        clean_store = store_str.strip().upper()
        
        # Remove common suffixes and prefixes
        cleanup_patterns = [
            r'\b(PTY\s*LTD|LIMITED|LTD|AUSTRALIA|AU)\b',
            r'\b(STORE\s*\d+|SHOP\s*\d+|\d+\s*STORE|\d+\s*SHOP)\b',
            r'\b(SUPERMARKET|METRO|EXPRESS|WAREHOUSE|OUTLET)\b$'
        ]
        
        for pattern in cleanup_patterns:
            clean_store = re.sub(pattern, '', clean_store).strip()
        
        # Check exact mappings first
        clean_lower = clean_store.lower()
        if clean_lower in self.store_variations:
            return self.store_variations[clean_lower]
        
        # Check for partial matches with known Australian stores
        for store in self.australian_stores:
            store_upper = store.upper()
            if store_upper in clean_store or clean_store in store_upper:
                return store_upper
        
        # Handle special cases
        if 'WOOLWORTH' in clean_store or 'WOOLIES' in clean_store:
            return 'WOOLWORTHS'
        elif 'COLE' in clean_store and len(clean_store) <= 10:
            return 'COLES'
        elif 'BUNNING' in clean_store:
            return 'BUNNINGS'
        
        return clean_store
    
    def _normalize_currency(self, currency_str: str) -> str:
        """Normalize currency to Australian standard format with maximum robustness."""
        if not currency_str or currency_str.strip() == '':
            return ""
        
        # Handle special cases
        if currency_str.strip().lower() in ['free', 'no charge', 'complimentary']:
            return "0.00"
        
        # Remove common Australian currency indicators
        clean_text = currency_str.upper().strip()
        currency_patterns = ['AUD', '(AUD)', '$', 'DOLLARS', 'DOLLAR', 'CENTS', 'CENT']
        
        for pattern in currency_patterns:
            clean_text = clean_text.replace(pattern, ' ')
        
        # Handle negative amounts (refunds, discounts)
        is_negative = (
            '-' in clean_text or 
            'REFUND' in currency_str.upper() or 
            'DISCOUNT' in currency_str.upper() or
            'CREDIT' in currency_str.upper()
        )
        
        # Extract numeric value with comma handling
        # Support formats: 12.34, 1,234.56, 1234, 12.3, .50
        amount_patterns = [
            r'(\d{1,3}(?:,\d{3})*\.\d{2})',  # 1,234.56
            r'(\d{1,3}(?:,\d{3})*\.\d{1})',  # 1,234.5
            r'(\d{1,3}(?:,\d{3})*)',         # 1,234
            r'(\d+\.\d{2})',                 # 123.45
            r'(\d+\.\d{1})',                 # 123.4
            r'(\.\d{2})',                    # .50
            r'(\d+)',                        # 123
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, clean_text.replace(' ', ''))
            if match:
                try:
                    amount_str = match.group(1)
                    if amount_str.startswith('.'):
                        amount = float('0' + amount_str)
                    else:
                        amount = float(amount_str.replace(',', ''))
                    
                    final_amount = -amount if is_negative else amount
                    return f"{final_amount:.2f}"
                except ValueError:
                    continue
        
        logger.warning(f"Could not parse currency: '{currency_str}', returning original")
        return currency_str
    
    def _normalize_product_list(self, products_str: str) -> List[str]:
        """Normalize product list to consistent Title Case format."""
        if not products_str or products_str.strip() == '':
            return []
        
        # Split by pipe separator
        items = [item.strip() for item in products_str.split('|')]
        normalized_items = []
        
        for item in items:
            if item:
                # Convert to proper Title Case for products
                normalized = self._to_title_case_product(item)
                normalized_items.append(normalized)
        
        return normalized_items
    
    def _to_title_case_product(self, product: str) -> str:
        """Convert product name to proper Australian retail Title Case."""
        if not product:
            return ""
        
        # Handle special cases for Australian products
        special_cases = {
            # Units
            'ML': 'mL', 'KG': 'kg', 'GM': 'g', 'LTR': 'L', 'LITRE': 'L',
            'GRAM': 'g', 'GRAMS': 'g', 'KILOGRAM': 'kg', 'KILOGRAMS': 'kg',
            
            # Product descriptors
            'FREE RANGE': 'Free Range', 'ORGANIC': 'Organic',
            'LOW FAT': 'Low Fat', 'FULL CREAM': 'Full Cream',
            'EXTRA VIRGIN': 'Extra Virgin', 'OLIVE OIL': 'Olive Oil',
            
            # Pack sizes
            'PACK': 'Pack', 'PK': 'pk', 'EACH': 'each',
            
            # Common Australian brands/products
            'SANITARIUM': 'Sanitarium', 'ARNOTT': 'Arnott',
            'CADBURY': 'Cadbury', 'NESTLE': 'Nestle',
        }
        
        # Start with title case
        title_case = product.title()
        
        # Apply special cases
        for old, new in special_cases.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(old) + r'\b'
            title_case = re.sub(pattern, new, title_case, flags=re.IGNORECASE)
        
        # Handle numeric patterns (e.g., "2L", "500Ml")
        title_case = re.sub(r'(\d+)\s*([a-zA-Z]+)', r'\1\2', title_case)
        
        return title_case
    
    def _normalize_quantity_list(self, quantities_str: str) -> List[str]:
        """Normalize quantity list with unit standardization."""
        if not quantities_str or quantities_str.strip() == '':
            return []
        
        quantities = [qty.strip() for qty in quantities_str.split('|')]
        normalized_qtys = []
        
        for qty in quantities:
            if qty:
                # Normalize units and format
                normalized_qty = self._normalize_quantity_format(qty)
                normalized_qtys.append(normalized_qty)
        
        return normalized_qtys
    
    def _normalize_quantity_format(self, qty_str: str) -> str:
        """Normalize individual quantity format."""
        if not qty_str:
            return "1"  # Default quantity
        
        # Clean the quantity string
        clean_qty = qty_str.strip().lower()
        
        # Handle special cases
        if clean_qty in ['each', 'ea', 'piece', 'pc']:
            return "1"
        
        # Normalize units using regex
        unit_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(kg|kilogram)', r'\1kg'),
            (r'(\d+(?:\.\d+)?)\s*(g|gram)', r'\1g'),
            (r'(\d+(?:\.\d+)?)\s*(l|litre|ltr)', r'\1L'),
            (r'(\d+(?:\.\d+)?)\s*(ml|millilitre)', r'\1mL'),
            (r'(\d+(?:\.\d+)?)\s*(pack|pk)', r'\1'),
        ]
        
        normalized = clean_qty
        for pattern, replacement in unit_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Extract just the numeric part if no units
        if re.match(r'^\d+(?:\.\d+)?$', normalized):
            return normalized
        
        # If we have units, keep them
        unit_match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]*)', normalized)
        if unit_match:
            return unit_match.group(1) + unit_match.group(2)
        
        return qty_str  # Return original if can't normalize
    
    def _normalize_price_list(self, prices_str: str) -> List[str]:
        """Normalize price list to consistent Australian currency format."""
        if not prices_str or prices_str.strip() == '':
            return []
        
        prices = [price.strip() for price in prices_str.split('|')]
        normalized_prices = []
        
        for price in prices:
            if price:
                normalized_price = self._normalize_currency(price)
                normalized_prices.append(normalized_price)
        
        return normalized_prices
    
    def _apply_consistency_corrections(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """Apply consistency checks and auto-corrections."""
        
        # Ensure list lengths match
        products = processed.get('prod_item_value', [])
        quantities = processed.get('prod_quantity_value', [])
        prices = processed.get('prod_price_value', [])
        
        max_len = max(len(products), len(quantities), len(prices), 1)
        
        # If we have products but missing quantities/prices, fill with defaults
        if len(products) > 0:
            while len(quantities) < len(products):
                quantities.append("1")  # Default quantity
            while len(prices) < len(products):
                prices.append("0.00")  # Default price
        
        # If we have more quantities/prices than products, trim
        if len(products) > 0:
            quantities = quantities[:len(products)]
            prices = prices[:len(products)]
        
        # Update the processed data
        processed['prod_quantity_value'] = quantities
        processed['prod_price_value'] = prices
        
        return processed
    
    def _apply_australian_business_rules(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Australian business rule validations and corrections."""
        
        # GST validation and correction
        total_str = processed.get('total_value', '')
        tax_str = processed.get('tax_value', '')
        
        if total_str and not tax_str:
            try:
                total_amount = float(total_str)
                if total_amount > 0:
                    # Calculate GST from total (GST-inclusive calculation)
                    estimated_gst = total_amount / 11  # 1/11 for GST-inclusive
                    processed['tax_value'] = f"{estimated_gst:.2f}"
                    processed['gst_calculated'] = True
                    logger.info(f"Calculated GST: ${estimated_gst:.2f} from total: ${total_amount:.2f}")
            except ValueError:
                logger.warning(f"Could not calculate GST from total: '{total_str}'")
        
        # Validate GST calculation if both total and tax are present
        if total_str and tax_str:
            try:
                total_amount = float(total_str)
                tax_amount = float(tax_str)
                
                # Check if GST is reasonable (should be ~9.09% of total for GST-inclusive)
                expected_gst_rate = tax_amount / total_amount if total_amount > 0 else 0
                
                if 0.08 <= expected_gst_rate <= 0.12:  # Allow some tolerance
                    processed['gst_validation'] = 'valid'
                elif expected_gst_rate > 0.15:
                    processed['gst_validation'] = 'too_high'
                elif expected_gst_rate < 0.05 and expected_gst_rate > 0:
                    processed['gst_validation'] = 'too_low'
                else:
                    processed['gst_validation'] = 'invalid'
                    
            except ValueError:
                processed['gst_validation'] = 'unparseable'
        
        return processed
    
    def _calculate_extraction_quality(self, processed: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive quality scores for the extraction."""
        
        scores = {
            'field_completeness': 0.0,
            'data_consistency': 0.0,
            'format_compliance': 0.0,
            'business_rule_compliance': 0.0,
            'overall_score': 0.0
        }
        
        # Field completeness (40% of total score)
        required_fields = ['date_value', 'store_name_value', 'total_value']
        optional_fields = ['tax_value', 'prod_item_value', 'prod_quantity_value', 'prod_price_value']
        
        required_complete = sum(1 for field in required_fields if processed.get(field))
        optional_complete = sum(1 for field in optional_fields if processed.get(field))
        
        scores['field_completeness'] = (
            (required_complete / len(required_fields)) * 0.7 +
            (optional_complete / len(optional_fields)) * 0.3
        )
        
        # Data consistency (25% of total score)
        consistency_score = 1.0
        
        products = processed.get('prod_item_value', [])
        quantities = processed.get('prod_quantity_value', [])
        prices = processed.get('prod_price_value', [])
        
        if products:
            if len(products) != len(quantities) or len(products) != len(prices):
                consistency_score -= 0.3
        
        scores['data_consistency'] = max(0.0, consistency_score)
        
        # Format compliance (20% of total score)
        format_score = 0.0
        
        # Check date format
        date_val = processed.get('date_value', '')
        if re.match(r'\d{2}/\d{2}/\d{4}', date_val):
            format_score += 0.3
        
        # Check currency format
        total_val = processed.get('total_value', '')
        if re.match(r'\d+\.\d{2}', total_val):
            format_score += 0.4
        
        # Check store name format
        store_val = processed.get('store_name_value', '')
        if store_val and store_val.isupper():
            format_score += 0.3
        
        scores['format_compliance'] = format_score
        
        # Business rule compliance (15% of total score)
        business_score = 0.0
        
        gst_validation = processed.get('gst_validation', 'unknown')
        if gst_validation == 'valid':
            business_score += 1.0
        elif gst_validation in ['too_high', 'too_low']:
            business_score += 0.5
        
        scores['business_rule_compliance'] = business_score
        
        # Calculate overall score
        scores['overall_score'] = (
            scores['field_completeness'] * 0.4 +
            scores['data_consistency'] * 0.25 +
            scores['format_compliance'] * 0.2 +
            scores['business_rule_compliance'] * 0.15
        )
        
        return scores
```

---

### 3. Multi-Stage Extraction Pipeline with Fallback Strategies

**Current State:** Single-shot extraction without error recovery

**Enhancement:** Progressive fallback extraction with quality validation

#### Implementation Plan

**Robust Extraction Pipeline:**
```python
# internvl/extraction/robust_extraction_pipeline.py
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RobustExtractionPipeline:
    """Multi-stage robust extraction pipeline with fallback strategies."""
    
    def __init__(self, model, parser, postprocessor, prompt_selector):
        self.model = model
        self.parser = parser
        self.postprocessor = postprocessor
        self.prompt_selector = prompt_selector
        
        # Quality thresholds for fallback decisions
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'acceptable': 0.4,
            'poor': 0.2
        }
        
    def extract_with_progressive_fallback(self, image_path: Path) -> Dict[str, Any]:
        """Extract with progressive fallback strategies."""
        
        logger.info(f"Starting robust extraction for: {image_path}")
        
        extraction_attempts = []
        
        # Stage 1: Primary extraction with robust prompt
        primary_result = self._attempt_extraction(
            image_path, 'robust', "Primary extraction with robust prompt"
        )
        extraction_attempts.append(primary_result)
        
        if self._is_extraction_acceptable(primary_result):
            logger.info("Primary extraction successful")
            return self._finalize_result(primary_result, extraction_attempts)
        
        # Stage 2: Fallback with simpler prompt
        logger.info("Primary extraction insufficient, trying fallback")
        fallback_result = self._attempt_extraction(
            image_path, 'simple', "Fallback extraction with simplified prompt"
        )
        extraction_attempts.append(fallback_result)
        
        if self._is_extraction_acceptable(fallback_result):
            logger.info("Fallback extraction successful")
            return self._finalize_result(fallback_result, extraction_attempts)
        
        # Stage 3: Minimal extraction for basic fields
        logger.info("Fallback insufficient, trying minimal extraction")
        minimal_result = self._attempt_extraction(
            image_path, 'minimal', "Minimal extraction for essential fields"
        )
        extraction_attempts.append(minimal_result)
        
        # Stage 4: Hybrid approach - combine best results
        logger.info("Creating hybrid result from all attempts")
        hybrid_result = self._create_hybrid_result(extraction_attempts)
        
        return self._finalize_result(hybrid_result, extraction_attempts)
    
    def _attempt_extraction(self, image_path: Path, prompt_type: str, description: str) -> Dict[str, Any]:
        """Attempt extraction with specified prompt type."""
        
        try:
            logger.debug(f"Attempting: {description}")
            
            # Get appropriate prompt
            prompt = self.prompt_selector.get_prompt(prompt_type)
            
            # Generate response
            response = self.model.generate(str(image_path), prompt)
            
            # Parse key-value response
            kv_result = self.parser.parse_key_value_response(response)
            
            # Post-process for robustness
            processed_result = self.postprocessor.robust_postprocess(kv_result.extracted_fields)
            
            # Add extraction metadata
            processed_result['extraction_metadata'] = {
                'prompt_type': prompt_type,
                'description': description,
                'parsing_errors': len(kv_result.validation_errors),
                'confidence_score': kv_result.confidence_score,
                'raw_response': response[:500],  # Truncated for logging
                'extraction_successful': True
            }
            
            logger.debug(f"Extraction attempt completed. Quality: {processed_result['extraction_quality']['overall_score']:.2f}")
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Extraction attempt failed: {e}")
            return {
                'extraction_metadata': {
                    'prompt_type': prompt_type,
                    'description': description,
                    'extraction_successful': False,
                    'error': str(e)
                },
                'extraction_quality': {'overall_score': 0.0}
            }
    
    def _is_extraction_acceptable(self, result: Dict[str, Any]) -> bool:
        """Check if extraction result meets acceptable quality threshold."""
        
        if not result or 'extraction_quality' not in result:
            return False
        
        overall_score = result['extraction_quality'].get('overall_score', 0.0)
        return overall_score >= self.quality_thresholds['acceptable']
    
    def _create_hybrid_result(self, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create hybrid result by combining best aspects of all attempts."""
        
        # Initialize hybrid result
        hybrid = {
            'date_value': '',
            'store_name_value': '',
            'tax_value': '',
            'total_value': '',
            'prod_item_value': [],
            'prod_quantity_value': [],
            'prod_price_value': []
        }
        
        # For each field, choose the best value from all attempts
        fields_to_evaluate = [
            'date_value', 'store_name_value', 'tax_value', 'total_value'
        ]
        
        for field in fields_to_evaluate:
            best_value = self._choose_best_field_value(field, attempts)
            if best_value:
                hybrid[field] = best_value
        
        # For list fields, choose the most complete set
        list_fields = ['prod_item_value', 'prod_quantity_value', 'prod_price_value']
        best_lists = self._choose_best_list_values(list_fields, attempts)
        hybrid.update(best_lists)
        
        # Reprocess the hybrid result
        hybrid_processed = self.postprocessor.robust_postprocess({
            'DATE': hybrid['date_value'],
            'STORE': hybrid['store_name_value'],
            'TAX': hybrid['tax_value'],
            'TOTAL': hybrid['total_value'],
            'PRODUCTS': ' | '.join(hybrid['prod_item_value']),
            'QUANTITIES': ' | '.join(hybrid['prod_quantity_value']),
            'PRICES': ' | '.join(hybrid['prod_price_value'])
        })
        
        hybrid_processed['extraction_metadata'] = {
            'prompt_type': 'hybrid',
            'description': 'Hybrid result from multiple extraction attempts',
            'extraction_successful': True,
            'source_attempts': len(attempts)
        }
        
        return hybrid_processed
    
    def _choose_best_field_value(self, field: str, attempts: List[Dict[str, Any]]) -> str:
        """Choose the best value for a specific field from all attempts."""
        
        candidates = []
        
        for attempt in attempts:
            value = attempt.get(field, '')
            if value and value.strip():
                quality_score = attempt.get('extraction_quality', {}).get('overall_score', 0.0)
                candidates.append((value, quality_score))
        
        if not candidates:
            return ''
        
        # Sort by quality score and return the best
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _choose_best_list_values(self, list_fields: List[str], attempts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Choose the best list values from all attempts."""
        
        best_lists = {field: [] for field in list_fields}
        best_score = 0.0
        
        for attempt in attempts:
            # Check if this attempt has complete list data
            products = attempt.get('prod_item_value', [])
            quantities = attempt.get('prod_quantity_value', [])
            prices = attempt.get('prod_price_value', [])
            
            if products and len(products) == len(quantities) == len(prices):
                quality_score = attempt.get('extraction_quality', {}).get('overall_score', 0.0)
                
                # Also consider list completeness
                list_completeness = len(products) / 10.0  # Normalize assuming max 10 items
                combined_score = quality_score + min(list_completeness, 0.2)  # Add up to 0.2 for completeness
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_lists['prod_item_value'] = products
                    best_lists['prod_quantity_value'] = quantities
                    best_lists['prod_price_value'] = prices
        
        return best_lists
    
    def _finalize_result(self, result: Dict[str, Any], all_attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize the extraction result with comprehensive metadata."""
        
        result['extraction_summary'] = {
            'total_attempts': len(all_attempts),
            'successful_attempts': sum(1 for attempt in all_attempts 
                                     if attempt.get('extraction_metadata', {}).get('extraction_successful', False)),
            'final_quality_score': result.get('extraction_quality', {}).get('overall_score', 0.0),
            'extraction_strategy': result.get('extraction_metadata', {}).get('prompt_type', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add confidence indicators
        quality_score = result['extraction_summary']['final_quality_score']
        
        if quality_score >= self.quality_thresholds['excellent']:
            result['confidence_level'] = 'excellent'
        elif quality_score >= self.quality_thresholds['good']:
            result['confidence_level'] = 'good'
        elif quality_score >= self.quality_thresholds['acceptable']:
            result['confidence_level'] = 'acceptable'
        else:
            result['confidence_level'] = 'poor'
        
        logger.info(f"Extraction finalized. Confidence: {result['confidence_level']}, Score: {quality_score:.2f}")
        
        return result
```

---

### 4. Integration with Existing Infrastructure

**Current State:** Basic CLI tools with single extraction method

**Enhancement:** Robust extraction integration with existing tools

#### Implementation Plan

**Enhanced CLI Integration:**
```python
# internvl/cli/robust_internvl_single.py
import logging
from pathlib import Path
from typing import Dict, Any

from internvl.config import get_config
from internvl.model.loader import load_model
from internvl.extraction.key_value_parser import KeyValueParser
from internvl.extraction.robust_postprocessor import RobustPostProcessor
from internvl.extraction.robust_extraction_pipeline import RobustExtractionPipeline
from internvl.prompts.robust_extraction_prompts import PromptSelector
from internvl.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def setup_robust_pipeline() -> RobustExtractionPipeline:
    """Setup the robust extraction pipeline with all components."""
    
    config = get_config()
    
    # Load model
    model = load_model(config.model_path)
    
    # Initialize components
    parser = KeyValueParser()
    postprocessor = RobustPostProcessor()
    prompt_selector = PromptSelector()
    
    # Create pipeline
    pipeline = RobustExtractionPipeline(
        model=model,
        parser=parser,
        postprocessor=postprocessor,
        prompt_selector=prompt_selector
    )
    
    return pipeline

def process_single_image_robust(image_path: Path, output_path: Path = None) -> Dict[str, Any]:
    """Process single image with robust extraction pipeline."""
    
    logger.info(f"Processing image with robust pipeline: {image_path}")
    
    # Setup pipeline
    pipeline = setup_robust_pipeline()
    
    # Extract with fallback strategies
    result = pipeline.extract_with_progressive_fallback(image_path)
    
    # Save result if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
    
    # Log summary
    confidence = result.get('confidence_level', 'unknown')
    quality_score = result.get('extraction_summary', {}).get('final_quality_score', 0.0)
    
    logger.info(f"Extraction complete. Confidence: {confidence}, Quality: {quality_score:.2f}")
    
    return result

def main():
    """Main function for robust single image processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust InternVL single image extraction")
    parser.add_argument("--image-path", type=Path, required=True, help="Path to receipt image")
    parser.add_argument("--output-path", type=Path, help="Output path for extraction results")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    setup_logging(level=args.log_level)
    
    if not args.image_path.exists():
        logger.error(f"Image file not found: {args.image_path}")
        return 1
    
    try:
        result = process_single_image_robust(args.image_path, args.output_path)
        
        print(f"\n=== EXTRACTION RESULTS ===")
        print(f"Confidence Level: {result.get('confidence_level', 'unknown')}")
        print(f"Quality Score: {result.get('extraction_summary', {}).get('final_quality_score', 0.0):.2f}")
        print(f"Store: {result.get('store_name_value', 'N/A')}")
        print(f"Date: {result.get('date_value', 'N/A')}")
        print(f"Total: ${result.get('total_value', 'N/A')}")
        print(f"Products: {len(result.get('prod_item_value', []))} items")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
```

**Enhanced Batch Processing:**
```python
# internvl/cli/robust_internvl_batch.py
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from internvl.cli.robust_internvl_single import setup_robust_pipeline

logger = logging.getLogger(__name__)

class RobustBatchProcessor:
    """Robust batch processing with parallel execution and error recovery."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.pipeline = None
        
    def process_image_batch(self, image_paths: List[Path], output_dir: Path) -> Dict[str, Any]:
        """Process batch of images with robust extraction."""
        
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track results
        results = {
            'total_images': len(image_paths),
            'successful': 0,
            'failed': 0,
            'excellent_quality': 0,
            'good_quality': 0,
            'acceptable_quality': 0,
            'poor_quality': 0,
            'processing_time': 0.0,
            'failed_images': [],
            'quality_distribution': {}
        }
        
        start_time = time.time()
        
        # Process images in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self._process_single_image_wrapper, image_path, output_dir): image_path
                for image_path in image_paths
            }
            
            # Collect results
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        results['successful'] += 1
                        
                        # Track quality distribution
                        confidence = result['confidence_level']
                        if confidence == 'excellent':
                            results['excellent_quality'] += 1
                        elif confidence == 'good':
                            results['good_quality'] += 1
                        elif confidence == 'acceptable':
                            results['acceptable_quality'] += 1
                        else:
                            results['poor_quality'] += 1
                    else:
                        results['failed'] += 1
                        results['failed_images'].append({
                            'image': str(image_path),
                            'error': result.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    results['failed'] += 1
                    results['failed_images'].append({
                        'image': str(image_path),
                        'error': str(e)
                    })
        
        results['processing_time'] = time.time() - start_time
        
        # Calculate success rate
        results['success_rate'] = results['successful'] / results['total_images'] if results['total_images'] > 0 else 0.0
        
        # Save batch summary
        summary_path = output_dir / 'batch_processing_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing complete. Success rate: {results['success_rate']:.1%}")
        
        return results
    
    def _process_single_image_wrapper(self, image_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Wrapper for single image processing (for multiprocessing)."""
        
        try:
            # Setup pipeline if not already done
            if self.pipeline is None:
                self.pipeline = setup_robust_pipeline()
            
            # Process image
            result = self.pipeline.extract_with_progressive_fallback(image_path)
            
            # Save individual result
            output_file = output_dir / f"{image_path.stem}_extraction.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'confidence_level': result.get('confidence_level', 'unknown'),
                'quality_score': result.get('extraction_summary', {}).get('final_quality_score', 0.0),
                'output_file': str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Processing failed for {image_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Main function for robust batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust InternVL batch image extraction")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing receipt images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for extraction results")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--file-pattern", default="*.png", help="File pattern for images (e.g., '*.jpg')")
    
    args = parser.parse_args()
    
    if not args.image_dir.exists():
        logger.error(f"Image directory not found: {args.image_dir}")
        return 1
    
    # Find all image files
    image_paths = list(args.image_dir.glob(args.file_pattern))
    if not image_paths:
        logger.error(f"No images found matching pattern: {args.file_pattern}")
        return 1
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process batch
    processor = RobustBatchProcessor(max_workers=args.max_workers)
    results = processor.process_image_batch(image_paths, args.output_dir)
    
    # Print summary
    print(f"\n=== BATCH PROCESSING SUMMARY ===")
    print(f"Total Images: {results['total_images']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Processing Time: {results['processing_time']:.1f} seconds")
    print(f"\nQuality Distribution:")
    print(f"  Excellent: {results['excellent_quality']}")
    print(f"  Good: {results['good_quality']}")
    print(f"  Acceptable: {results['acceptable_quality']}")
    print(f"  Poor: {results['poor_quality']}")
    
    if results['failed_images']:
        print(f"\nFailed Images:")
        for failed in results['failed_images'][:5]:  # Show first 5
            print(f"  {failed['image']}: {failed['error']}")
        if len(results['failed_images']) > 5:
            print(f"  ... and {len(results['failed_images']) - 5} more")
    
    return 0 if results['success_rate'] > 0.8 else 1

if __name__ == "__main__":
    exit(main())
```

---

## IMPLEMENTATION ROADMAP

### Phase 1: Foundation (2-3 weeks)
1. **Enhanced Prompt Engineering**
   - Implement robust Australian receipt prompts
   - Create fallback prompt hierarchy
   - Add prompt selection logic

2. **Post-Processing Pipeline**
   - Build comprehensive normalization functions
   - Implement Australian business rules
   - Add quality scoring framework

### Phase 2: Robust Pipeline (2-3 weeks)
3. **Multi-Stage Extraction**
   - Implement progressive fallback strategies
   - Add hybrid result combination logic
   - Create quality validation frameworks

4. **Integration & Testing**
   - Update CLI tools for robust extraction
   - Add batch processing capabilities
   - Implement comprehensive error handling

### Phase 3: Optimization (1-2 weeks)
5. **Performance Enhancement**
   - Add parallel processing capabilities
   - Optimize prompt selection algorithms
   - Implement caching strategies

6. **Monitoring & Metrics**
   - Add extraction quality monitoring
   - Implement performance metrics
   - Create automated reporting

## SUCCESS METRICS

- **Extraction Success Rate:** >95% (up from current ~85%)
- **Quality Consistency:** >90% of extractions rated "good" or better
- **Australian Format Compliance:** >98% for dates, currency, store names
- **Processing Speed:** <2x slower than current (acceptable for robustness gains)
- **Error Recovery:** >80% of failed primary extractions recovered via fallback

## KEY BENEFITS

### 🎯 **Robustness Improvements**
1. **Multi-Format Tolerance**: Handles various date, currency, and store name formats
2. **Progressive Fallback**: Multiple extraction strategies prevent total failures
3. **Australian Optimization**: Specialized for Australian business contexts
4. **Quality Assurance**: Comprehensive validation and scoring

### 🚀 **Operational Benefits**
1. **Reduced Manual Review**: Higher quality extractions need less human verification
2. **Better Error Recovery**: Fallback strategies recover from initial failures
3. **Consistent Output**: Standardized formatting regardless of input variations
4. **Confidence Indicators**: Quality scores help prioritize manual review

### 📊 **Business Impact**
1. **Higher Accuracy**: More reliable receipt processing for business applications
2. **Reduced Costs**: Less manual correction and re-processing needed
3. **Scalability**: Robust pipeline handles diverse receipt formats automatically
4. **Compliance**: Australian business rule validation ensures regulatory compliance

## CONCLUSION

These robust extraction improvements directly address the input side of the pipeline, complementing the robust evaluation metrics already planned. Together, they create a comprehensive system that both extracts information reliably and evaluates results accurately.

The progressive fallback approach ensures that even when primary extraction methods struggle, the system can still deliver usable results. The Australian-specific optimizations make it particularly well-suited for local business requirements, while the comprehensive post-processing ensures consistent, high-quality outputs regardless of input variations.

**Recommendation**: Implement these robust extraction improvements in parallel with the evaluation enhancements, as they work synergistically to create a production-ready receipt processing system.