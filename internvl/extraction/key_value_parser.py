"""
Enhanced Key-Value Parser for InternVL Evaluation

This module provides a comprehensive key-value parser with robust validation,
confidence scoring, and error handling optimized for Australian receipt processing.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from internvl.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KeyValueExtractionResult:
    """Result of key-value extraction with comprehensive validation."""
    raw_text: str
    extracted_fields: Dict[str, str]
    parsed_lists: Dict[str, List[str]]
    validation_errors: List[str]
    confidence_score: float
    field_completeness: Dict[str, bool]


class KeyValueParser:
    """Robust parser for key-value receipt format with Australian-specific validation."""
    
    def __init__(self):
        """Initialize parser with field patterns and validation rules."""
        self.field_patterns = {
            'DATE': r'DATE:\s*(.+)',
            'STORE': r'STORE:\s*(.+)',
            'ABN': r'ABN:\s*(.+)',
            'PAYER': r'PAYER:\s*(.+)',
            'TAX': r'TAX:\s*(.+)', 
            'TOTAL': r'TOTAL:\s*(.+)',
            'PRODUCTS': r'PRODUCTS:\s*(.+)',
            'QUANTITIES': r'QUANTITIES:\s*(.+)',
            'PRICES': r'PRICES:\s*(.+)'
        }
        
        # Required fields for valid extraction
        self.required_fields = ['DATE', 'STORE', 'TAX', 'TOTAL']
        
        # List fields that should contain pipe-separated values
        self.list_fields = ['PRODUCTS', 'QUANTITIES', 'PRICES']
        
    def parse_key_value_response(self, response_text: str) -> KeyValueExtractionResult:
        """
        Parse key-value response with comprehensive validation.
        
        Args:
            response_text: Raw text response from the model
            
        Returns:
            KeyValueExtractionResult with parsed data and validation
        """
        logger.debug(f"Parsing key-value response: {len(response_text)} characters")
        
        extracted_fields = {}
        validation_errors = []
        field_completeness = {}
        
        # Clean the input text
        cleaned_text = self._clean_response_text(response_text)
        
        # Extract each field using regex patterns
        for field_name, pattern in self.field_patterns.items():
            match = re.search(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            if match:
                field_value = match.group(1).strip()
                extracted_fields[field_name] = field_value
                field_completeness[field_name] = len(field_value) > 0
                logger.debug(f"Extracted {field_name}: '{field_value}'")
            else:
                extracted_fields[field_name] = ""
                field_completeness[field_name] = False
                if field_name in self.required_fields:
                    validation_errors.append(f"Missing required field: {field_name}")
                    logger.warning(f"Missing required field: {field_name}")
        
        # Parse pipe-separated lists
        parsed_lists = self._parse_pipe_separated_lists(extracted_fields)
        
        # Validate list consistency
        list_validation_errors = self._validate_list_consistency(parsed_lists)
        validation_errors.extend(list_validation_errors)
        
        # Validate field formats
        format_validation_errors = self._validate_field_formats(extracted_fields)
        validation_errors.extend(format_validation_errors)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            parsed_lists, validation_errors, field_completeness
        )
        
        result = KeyValueExtractionResult(
            raw_text=response_text,
            extracted_fields=extracted_fields,
            parsed_lists=parsed_lists,
            validation_errors=validation_errors,
            confidence_score=confidence_score,
            field_completeness=field_completeness
        )
        
        logger.info(f"Key-value parsing completed. Confidence: {confidence_score:.2f}, "
                   f"Errors: {len(validation_errors)}")
        
        return result
    
    def _clean_response_text(self, text: str) -> str:
        """Clean response text for better parsing."""
        # Remove extra whitespace and normalize line endings
        cleaned = text.strip().replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove common artifacts from model responses
        artifacts_to_remove = [
            r'```.*?```',  # Remove code blocks
            r'Here is the.*?format:',  # Remove explanatory text
            r'The extracted.*?is:',  # Remove explanatory text
        ]
        
        for artifact_pattern in artifacts_to_remove:
            cleaned = re.sub(artifact_pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        return cleaned.strip()
    
    def _parse_pipe_separated_lists(self, fields: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Parse pipe-separated values into lists with enhanced cleaning.
        
        Args:
            fields: Dictionary of extracted field values
            
        Returns:
            Dictionary with parsed lists for PRODUCTS, QUANTITIES, PRICES
        """
        parsed_lists = {}
        
        for field in self.list_fields:
            field_value = fields.get(field, "")
            if field_value:
                # Split by pipe and clean each item
                items = [self._clean_list_item(item) for item in field_value.split('|')]
                # Remove empty items
                items = [item for item in items if item]
                parsed_lists[field] = items
                logger.debug(f"Parsed {field}: {len(items)} items")
            else:
                parsed_lists[field] = []
        
        return parsed_lists
    
    def _clean_list_item(self, item: str) -> str:
        """Clean individual list items."""
        # Strip whitespace
        cleaned = item.strip()
        
        # Remove common artifacts
        cleaned = re.sub(r'^["\']|["\']$', '', cleaned)  # Remove quotes
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        
        return cleaned
    
    def _validate_list_consistency(self, parsed_lists: Dict[str, List[str]]) -> List[str]:
        """
        Validate that product lists have consistent lengths.
        
        Args:
            parsed_lists: Dictionary with parsed lists
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        products = parsed_lists.get('PRODUCTS', [])
        quantities = parsed_lists.get('QUANTITIES', [])
        prices = parsed_lists.get('PRICES', [])
        
        # If no products, that's okay for some receipts
        if len(products) == 0:
            if len(quantities) > 0 or len(prices) > 0:
                errors.append("No products but quantities/prices found")
            return errors
        
        # Check length consistency
        if len(products) != len(quantities):
            errors.append(
                f"Product count mismatch: {len(products)} products, "
                f"{len(quantities)} quantities"
            )
        
        if len(products) != len(prices):
            errors.append(
                f"Product count mismatch: {len(products)} products, "
                f"{len(prices)} prices"
            )
        
        # Validate individual items
        for i, qty in enumerate(quantities):
            if not self._is_valid_quantity(qty):
                errors.append(f"Invalid quantity format at position {i+1}: '{qty}'")
        
        for i, price in enumerate(prices):
            if not self._is_valid_price(price):
                errors.append(f"Invalid price format at position {i+1}: '{price}'")
        
        return errors
    
    def _validate_field_formats(self, fields: Dict[str, str]) -> List[str]:
        """
        Validate field formats for Australian standards.
        
        Args:
            fields: Dictionary of extracted field values
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate date format (Australian DD/MM/YYYY)
        date_value = fields.get('DATE', '')
        if date_value and not self._is_valid_australian_date(date_value):
            errors.append(f"Date format may be incorrect for Australian standard: '{date_value}'")
        
        # Validate store name (should be in uppercase for receipts)
        store_value = fields.get('STORE', '')
        if store_value and not any(c.isupper() for c in store_value):
            errors.append(f"Store name should typically be in uppercase: '{store_value}'")
        
        # Validate currency amounts
        for field_name in ['TAX', 'TOTAL']:
            amount = fields.get(field_name, '')
            if amount and not self._is_valid_currency_amount(amount):
                errors.append(f"Invalid currency format for {field_name}: '{amount}'")
        
        # Validate ABN format (Australian Business Number)
        abn_value = fields.get('ABN', '')
        if abn_value and not self._is_valid_abn(abn_value):
            errors.append(f"Invalid ABN format: '{abn_value}' (should be XX XXX XXX XXX)")
        
        return errors
    
    def _is_valid_quantity(self, qty_str: str) -> bool:
        """
        Check if quantity string is valid.
        
        Args:
            qty_str: Quantity string to validate
            
        Returns:
            True if valid quantity format
        """
        if not qty_str:
            return False
        
        try:
            # Allow integers, floats, and units like "2.5kg", "1L", "32.230L"
            # Common Australian quantity patterns
            patterns = [
                r'^\d+$',  # Simple integer: "1", "2"
                r'^\d+\.\d+$',  # Simple decimal: "2.5", "32.230"
                r'^\d+(\.\d+)?\s*[a-zA-Z]+$',  # With units: "2.5kg", "1L", "32.230L"
            ]
            
            for pattern in patterns:
                if re.match(pattern, qty_str.strip()):
                    return True
            
            return False
        except Exception:
            return False
    
    def _is_valid_price(self, price_str: str) -> bool:
        """
        Check if price string is valid Australian currency.
        
        Args:
            price_str: Price string to validate
            
        Returns:
            True if valid price format
        """
        if not price_str:
            return False
        
        try:
            # Remove common currency symbols and validate numeric
            clean_price = re.sub(r'[$AUD\s]', '', price_str.strip())
            
            # Should be a valid decimal number
            if re.match(r'^\d+(\.\d{1,2})?$', clean_price):
                amount = float(clean_price)
                # Reasonable price range for individual items (0.01 to 10000 AUD)
                return 0.01 <= amount <= 10000.0
            
            return False
        except Exception:
            return False
    
    def _is_valid_australian_date(self, date_str: str) -> bool:
        """
        Check if date string matches Australian format patterns.
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if likely Australian date format
        """
        if not date_str:
            return False
        
        # Common Australian date patterns
        australian_patterns = [
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # DD/MM/YYYY or D/M/YYYY
            r'^\d{1,2}-\d{1,2}-\d{4}$',  # DD-MM-YYYY
            r'^\d{1,2}\.\d{1,2}\.\d{4}$',  # DD.MM.YYYY
            r'^\d{1,2}\s+\d{1,2}\s+\d{4}$',  # DD MM YYYY
        ]
        
        for pattern in australian_patterns:
            if re.match(pattern, date_str.strip()):
                return True
        
        return False
    
    def _is_valid_currency_amount(self, amount_str: str) -> bool:
        """
        Check if currency amount is valid for Australian context.
        
        Args:
            amount_str: Currency amount string to validate
            
        Returns:
            True if valid currency amount
        """
        if not amount_str:
            return False
        
        try:
            # Remove currency symbols and clean
            clean_amount = re.sub(r'[$AUD\s,]', '', amount_str.strip())
            
            # Should be a valid decimal
            if re.match(r'^\d+(\.\d{1,2})?$', clean_amount):
                amount = float(clean_amount)
                # Reasonable amount range (0.01 to 100000 AUD)
                return 0.01 <= amount <= 100000.0
            
            return False
        except Exception:
            return False
    
    def _is_valid_abn(self, abn_str: str) -> bool:
        """
        Check if ABN string matches Australian Business Number format.
        
        Args:
            abn_str: ABN string to validate
            
        Returns:
            True if valid ABN format (11 digits total or XX XXX XXX XXX format)
        """
        if not abn_str:
            return False
        
        # Clean ABN string - remove all non-digits
        clean_abn = re.sub(r'[^\d]', '', abn_str.strip())
        
        # ABN must be exactly 11 digits
        if len(clean_abn) != 11:
            return False
        
        # Check for valid format patterns
        abn_patterns = [
            r'^\d{2}\s\d{3}\s\d{3}\s\d{3}$',  # XX XXX XXX XXX (standard format)
            r'^\d{11}$',                       # XXXXXXXXXXX (no spaces)
            r'^\d{2}\s\d{9}$',                # XX XXXXXXXXX (partial spacing)
            r'^\d{2}-\d{3}-\d{3}-\d{3}$',     # XX-XXX-XXX-XXX (with dashes)
        ]
        
        # Check if it matches any of the standard patterns
        for pattern in abn_patterns:
            if re.match(pattern, abn_str.strip()):
                return True
        
        # If it doesn't match standard patterns but has exactly 11 digits, accept it
        # This covers any other formatting variations
        return len(clean_abn) == 11
    
    def _calculate_confidence_score(self, 
                                  lists: Dict[str, List[str]], 
                                  errors: List[str],
                                  completeness: Dict[str, bool]) -> float:
        """
        Calculate confidence score for extraction quality.
        
        Args:
            lists: Parsed lists
            errors: Validation errors
            completeness: Field completeness status
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base score from field completeness
        total_fields = len(self.field_patterns)
        completed_fields = sum(completeness.values())
        field_score = completed_fields / total_fields
        
        # Penalty for validation errors
        error_penalty = min(len(errors) * 0.1, 0.5)  # Cap at 50% penalty
        
        # Bonus for list consistency
        consistency_bonus = 0.0
        products = lists.get('PRODUCTS', [])
        quantities = lists.get('QUANTITIES', [])
        prices = lists.get('PRICES', [])
        
        if products and quantities and prices:
            if len(products) == len(quantities) == len(prices):
                consistency_bonus = 0.2
        
        # Bonus for required fields
        required_bonus = 0.0
        required_complete = sum(1 for field in self.required_fields 
                              if completeness.get(field, False))
        if required_complete == len(self.required_fields):
            required_bonus = 0.1
        
        # Calculate final confidence
        confidence = max(0.0, min(1.0, 
                                field_score - error_penalty + consistency_bonus + required_bonus))
        
        logger.debug(f"Confidence calculation: field_score={field_score:.2f}, "
                    f"error_penalty={error_penalty:.2f}, consistency_bonus={consistency_bonus:.2f}, "
                    f"required_bonus={required_bonus:.2f}, final={confidence:.2f}")
        
        return confidence
    
    def convert_to_expense_claim_format(self, result: KeyValueExtractionResult) -> Dict[str, Any]:
        """
        Convert key-value result to Australian Tax Expense Claim format.
        
        Args:
            result: KeyValueExtractionResult to convert
            
        Returns:
            Dictionary in Australian expense claim format
        """
        expense_data = {
            # Core expense claim fields
            "invoice_date": result.extracted_fields.get('DATE', ''),
            "supplier_name": result.extracted_fields.get('STORE', ''),
            "supplier_abn": result.extracted_fields.get('ABN', ''),
            "payer_name": result.extracted_fields.get('PAYER', ''),
            "gst_amount": result.extracted_fields.get('TAX', ''),
            "total_amount": result.extracted_fields.get('TOTAL', ''),
            "items": result.parsed_lists.get('PRODUCTS', []),
            "quantities": result.parsed_lists.get('QUANTITIES', []),
            "item_prices": result.parsed_lists.get('PRICES', [])
        }
        
        logger.debug("Converted to Australian Tax Expense Claim format")
        return expense_data
    
    def assess_work_related_expense(self, result: KeyValueExtractionResult, expense_category: str = "General") -> Dict[str, Any]:
        """
        Assess extracted receipt data for Australian work-related expense compliance.
        
        Args:
            result: KeyValueExtractionResult from parsing
            expense_category: Category of expense (e.g., "Office Supplies", "Tools & Equipment")
            
        Returns:
            Dictionary with ATO compliance assessment
        """
        expense_data = self.convert_to_expense_claim_format(result)
        
        # Required fields for ATO compliance
        required_fields = ['supplier_name', 'supplier_abn', 'invoice_date', 'gst_amount', 'total_amount']
        
        # Assess field validity
        field_assessment = {}
        for field in required_fields:
            value = expense_data.get(field, '')
            if field == 'supplier_abn' and value:
                field_assessment[field] = {
                    'present': bool(value),
                    'valid': self._is_valid_abn(value),
                    'value': value
                }
            elif field == 'invoice_date' and value:
                field_assessment[field] = {
                    'present': bool(value),
                    'valid': self._is_valid_australian_date(value),
                    'value': value
                }
            elif field in ['gst_amount', 'total_amount'] and value:
                field_assessment[field] = {
                    'present': bool(value),
                    'valid': self._is_valid_currency_amount(value),
                    'value': value
                }
            else:
                field_assessment[field] = {
                    'present': bool(value),
                    'valid': bool(value),
                    'value': value
                }
        
        # Calculate compliance score
        valid_fields = sum(1 for field_info in field_assessment.values() 
                          if field_info['present'] and field_info['valid'])
        compliance_score = (valid_fields / len(required_fields)) * 100
        
        # Determine ATO readiness
        ato_ready = compliance_score >= 80 and all(
            field_assessment[field]['present'] and field_assessment[field]['valid']
            for field in ['supplier_name', 'supplier_abn', 'invoice_date', 'total_amount']
        )
        
        assessment = {
            'expense_category': expense_category,
            'compliance_score': compliance_score,
            'ato_ready': ato_ready,
            'field_assessment': field_assessment,
            'expense_data': expense_data,
            'validation_summary': {
                'total_fields': len(required_fields),
                'valid_fields': valid_fields,
                'missing_fields': [field for field, info in field_assessment.items() 
                                 if not info['present']],
                'invalid_fields': [field for field, info in field_assessment.items() 
                                 if info['present'] and not info['valid']]
            }
        }
        
        logger.info(f"Work-related expense assessment: {compliance_score:.0f}% compliance, "
                   f"ATO ready: {ato_ready}")
        
        return assessment
    
    def get_extraction_summary(self, result: KeyValueExtractionResult) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of extraction results.
        
        Args:
            result: KeyValueExtractionResult to summarize
            
        Returns:
            Dictionary with extraction summary
        """
        products = result.parsed_lists.get('PRODUCTS', [])
        quantities = result.parsed_lists.get('QUANTITIES', [])
        prices = result.parsed_lists.get('PRICES', [])
        
        summary = {
            'extraction_quality': {
                'confidence_score': result.confidence_score,
                'validation_errors_count': len(result.validation_errors),
                'fields_extracted': sum(result.field_completeness.values()),
                'total_fields': len(result.field_completeness),
                'completeness_percentage': sum(result.field_completeness.values()) / len(result.field_completeness) * 100
            },
            'content_summary': {
                'date': result.extracted_fields.get('DATE', 'Not extracted'),
                'store': result.extracted_fields.get('STORE', 'Not extracted'),
                'tax_amount': result.extracted_fields.get('TAX', 'Not extracted'),
                'total_amount': result.extracted_fields.get('TOTAL', 'Not extracted'),
                'product_count': len(products),
                'has_consistent_lists': len(products) == len(quantities) == len(prices) if products else True
            },
            'validation_status': {
                'is_valid_extraction': len(result.validation_errors) == 0,
                'errors': result.validation_errors,
                'quality_grade': self._get_quality_grade(result.confidence_score),
                'recommended_for_production': result.confidence_score >= 0.7 and len(result.validation_errors) <= 2
            }
        }
        
        return summary
    
    def _get_quality_grade(self, confidence: float) -> str:
        """Get quality grade based on confidence score."""
        if confidence >= 0.9:
            return "Excellent"
        elif confidence >= 0.7:
            return "Good"
        elif confidence >= 0.5:
            return "Fair"
        else:
            return "Poor"


def extract_key_value_enhanced(response: str) -> Dict[str, Any]:
    """
    Main function for enhanced key-value extraction from model response.
    
    This is the recommended function to use for production extraction.
    
    Args:
        response: Raw text response from the model
        
    Returns:
        Dictionary with comprehensive extraction results
    """
    parser = KeyValueParser()
    
    try:
        # Parse the response
        extraction_result = parser.parse_key_value_response(response)
        
        # Convert to Australian expense claim format
        expense_data = parser.convert_to_expense_claim_format(extraction_result)
        
        # Generate summary
        summary = parser.get_extraction_summary(extraction_result)
        
        return {
            'success': True,
            'expense_claim_format': expense_data,
            'extraction_result': extraction_result,
            'summary': summary,
            'parser_version': 'enhanced_v1.1_australian_focus'
        }
        
    except Exception as e:
        logger.error(f"Enhanced key-value extraction failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'expense_claim_format': {},
            'extraction_result': None,
            'summary': None,
            'parser_version': 'enhanced_v1.1_australian_focus'
        }


def extract_work_related_expense(response: str, expense_category: str = "General") -> Dict[str, Any]:
    """
    Extract and assess work-related expense information from model response.
    
    Specifically designed for Australian Tax Office work-related expense claims.
    
    Args:
        response: Raw text response from the model
        expense_category: Category of expense (e.g., "Office Supplies", "Tools & Equipment")
        
    Returns:
        Dictionary with expense data and ATO compliance assessment
    """
    parser = KeyValueParser()
    
    try:
        # Parse the response
        extraction_result = parser.parse_key_value_response(response)
        
        # Assess for work-related expense compliance
        assessment = parser.assess_work_related_expense(extraction_result, expense_category)
        
        # Generate summary
        summary = parser.get_extraction_summary(extraction_result)
        
        return {
            'success': True,
            'assessment': assessment,
            'extraction_result': extraction_result,
            'summary': summary,
            'parser_version': 'enhanced_v1.1_work_expense_focus'
        }
        
    except Exception as e:
        logger.error(f"Work-related expense extraction failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'assessment': {},
            'extraction_result': None,
            'summary': None,
            'parser_version': 'enhanced_v1.1_work_expense_focus'
        }