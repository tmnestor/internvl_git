"""
JSON extraction utilities for InternVL Evaluation

This module provides functions for extracting JSON from model output text.
"""

import json
import re
from typing import Any, Dict

from internvl.utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from model output text with robust error handling.

    Args:
        text: Raw model output text

    Returns:
        Parsed JSON object
    """
    # Default structure for fallback
    default_json = {
        "date_value": "",
        "store_name_value": "",
        "tax_value": "",
        "total_value": "",
        "prod_item_value": [],
        "prod_quantity_value": [],
        "prod_price_value": [],
    }

    if not text:
        return default_json

    try:
        # First try to find JSON enclosed in triple backticks
        json_pattern = r"```(?:json)?(.*?)```"
        markdown_matches = re.findall(json_pattern, text, re.DOTALL)

        if markdown_matches:
            # Try each potential JSON block (prioritize longer ones)
            markdown_matches.sort(key=len, reverse=True)
            for potential_json in markdown_matches:
                original_json = potential_json.strip()
                logger.info(f"Processing JSON block of length {len(original_json)}")
                logger.debug(f"Original JSON:\n{repr(original_json)}")
                
                cleaned_json = _clean_and_fix_json(original_json)
                logger.debug(f"Cleaned JSON:\n{repr(cleaned_json)}")
                
                try:
                    result = json.loads(cleaned_json)
                    logger.info("Successfully parsed JSON after cleaning")
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed after cleaning: {e}")
                    logger.warning(f"Failed at character position {e.pos if hasattr(e, 'pos') else 'unknown'}")
                    continue

        # If markdown format failed, try general JSON pattern
        json_pattern = r"({[\s\S]*?})"
        matches = re.findall(json_pattern, text)

        for potential_json in matches:
            cleaned_json = _clean_and_fix_json(potential_json)
            try:
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        logger.error(f"Error extracting JSON from text: {e}")

    return default_json


def _clean_and_fix_json(json_text: str) -> str:
    """
    Clean and fix common JSON formatting issues from model output.
    
    Args:
        json_text: Raw JSON text that may have formatting issues
        
    Returns:
        Cleaned JSON text
    """
    # Remove leading/trailing whitespace
    cleaned = json_text.strip()
    
    # CRITICAL DEBUG: Identify problematic characters causing "Invalid control character" errors
    problematic_chars = []
    for i, char in enumerate(cleaned):
        char_code = ord(char)
        # Identify characters that could cause JSON parsing issues
        if char_code < 32 and char not in '\t\n\r':
            problematic_chars.append((i, char, char_code, repr(char)))
    
    if problematic_chars:
        logger.warning(f"Found {len(problematic_chars)} problematic control characters:")
        for pos, _char, code, repr_char in problematic_chars[:5]:  # Log first 5
            logger.warning(f"  Position {pos}: char={repr_char}, code={code}")
    
    # AGGRESSIVE: Direct reconstruction for these specific malformed patterns
    # The patterns are so consistent that we can target them directly
    if '",\n  ",' in cleaned or '"\n  "' in cleaned or '",\n             ' in cleaned:
        logger.info("Detected specific malformed patterns, using direct reconstruction")
        return _reconstruct_json(cleaned)
    
    # ULTRA-AGGRESSIVE: Remove ALL control characters and problematic bytes
    # This is more aggressive than before - remove anything suspicious
    cleaned = _ultra_sanitize_json(cleaned)
    
    # Fix the most common patterns first
    
    # 1. Fix orphaned comma lines: ",\n
    cleaned = re.sub(r'\n\s*",\s*\n', ',\n', cleaned)
    
    # 2. Fix unclosed quotes followed by newlines and closing quotes
    # Pattern: "value\n  "\n becomes "value",\n
    cleaned = re.sub(r':\s*"([^"]*?)\s*\n\s*"\s*\n', r': "\1",\n', cleaned)
    
    # 3. Fix missing commas between properly quoted fields
    cleaned = re.sub(r'"\s*\n\s*"([^"]+)":', r'",\n  "\1":', cleaned)
    
    # 4. Fix missing commas between numeric values and next key
    cleaned = re.sub(r'(\d+(?:\.\d+)?)\s*\n\s*"([^"]+)":', r'\1,\n  "\2":', cleaned)
    
    # 5. Fix multi-line addresses that span multiple lines
    # Pattern: "address": "Line1\n             Line2\n             Line3",
    def fix_multiline_address(match):
        key = match.group(1)
        first_line = match.group(2)
        rest = match.group(3)
        # Clean up the rest and escape newlines
        lines = [line.strip() for line in rest.split('\n') if line.strip()]
        if lines:
            full_address = first_line + '\\n' + '\\n'.join(lines)
        else:
            full_address = first_line
        return f'"{key}": "{full_address}"'
    
    cleaned = re.sub(r'"(address)":\s*"([^"]*?)"\s*\n\s*([^,}]+?)(?=\s*[,}])', 
                    fix_multiline_address, cleaned, flags=re.DOTALL)
    
    # 6. Fix unescaped newlines in any quoted strings
    def escape_newlines_in_quotes(match):
        content = match.group(1)
        # Replace literal newlines with escaped ones
        content = content.replace('\n', '\\n').replace('\r', '\\r')
        return f'"{content}"'
    
    # Apply to any quoted strings that contain literal newlines
    cleaned = re.sub(r'"([^"]*\n[^"]*)"', escape_newlines_in_quotes, cleaned)
    
    # 7. Remove trailing commas before closing braces
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # 8. Final validation with detailed error reporting
    try:
        json.loads(cleaned)
        logger.info("JSON validation passed after cleaning")
        return cleaned
    except json.JSONDecodeError as e:
        logger.error(f"JSON still invalid after cleaning: {e}")
        logger.error(f"Error at position {getattr(e, 'pos', 'unknown')}")
        if hasattr(e, 'pos') and e.pos < len(cleaned):
            start = max(0, e.pos - 20)
            end = min(len(cleaned), e.pos + 20)
            context = cleaned[start:end]
            logger.error(f"Context around error: {repr(context)}")
        
        # Last resort: full reconstruction
        logger.info("Standard cleaning failed, falling back to reconstruction")
        return _reconstruct_json(json_text)  # Use original text for reconstruction


def _ultra_sanitize_json(text: str) -> str:
    """
    Ultra-aggressive JSON sanitization to remove problematic characters.
    
    This function removes or replaces characters that commonly cause
    "Invalid control character" errors in JSON parsing.
    """
    # Step 1: Remove or replace control characters more aggressively
    sanitized = []
    for char in text:
        char_code = ord(char)
        
        # Allow only safe ASCII characters and essential whitespace
        if char_code == 9:  # Tab -> space
            sanitized.append(' ')
        elif char_code == 10:  # Newline -> keep
            sanitized.append(char)
        elif char_code == 13:  # Carriage return -> ignore
            continue
        elif char_code < 32:  # Other control characters -> space
            sanitized.append(' ')
        elif char_code == 127:  # DEL character -> space
            sanitized.append(' ')
        elif char_code > 127:  # Non-ASCII -> try to preserve if printable
            # For now, replace with space to be safe
            sanitized.append(' ')
        else:
            sanitized.append(char)
    
    cleaned = ''.join(sanitized)
    
    # Step 2: Clean up excessive whitespace created by replacements
    # Replace multiple spaces with single space
    cleaned = re.sub(r' +', ' ', cleaned)
    
    # Step 3: Fix specific problematic sequences that can arise
    # Remove orphaned quote characters that can cause issues
    cleaned = re.sub(r'\n\s*"\s*\n', '\n', cleaned)
    
    # Step 4: Ensure proper JSON structure after sanitization
    # Fix broken quoted strings
    cleaned = re.sub(r'"([^"]*?)\s+"\s*:', r'"\1":', cleaned)
    
    return cleaned


def _is_valid_json_structure(text: str) -> bool:
    """Check if text has basic JSON structure."""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _reconstruct_json(malformed_json: str) -> str:
    """
    Last resort: reconstruct JSON from severely malformed output.
    This handles cases where the model completely botches the format.
    """
    logger.info("Attempting JSON reconstruction as last resort")
    
    # Initialize with expected fields for receipt extraction
    expected_fields = {
        "company_name": "",
        "address": "",
        "phone_number": "",
        "date": "",
        "ABN": "",
        "total_amount": ""
    }
    
    # More aggressive pattern matching for key-value extraction
    # Look for various patterns that might represent key-value pairs
    patterns = [
        r'"([^"]+)"\s*:\s*"([^"]*)"',  # "key": "value" (standard)
        r'"([^"]+)"\s*:\s*([^,\n}]+)',  # "key": value (unquoted value)
        r'"([^"]+)"\s*:\s*"([^"]*?)\s*\n[^"]*?"',  # "key": "value with newlines"
        r'([a-zA-Z_]+)\s*:\s*"([^"]*)"',  # key: "value" (unquoted key)
        r'([a-zA-Z_]+)\s*:\s*([^,\n}]+)',  # key: value (both unquoted)
        # Specific patterns for this model's output
        r'"(company_name|address|phone_number|date|ABN|total_amount)"\s*:\s*"([^"]*?)(?:\s*\n[^"]*?)*"',
        r'"(company_name|address|phone_number|date|ABN|total_amount)"\s*:\s*([^,\n}]+)',
    ]
    
    extracted_pairs = {}
    
    # First, try to extract using the exact malformed pattern this model produces
    # Pattern: "key": "start_of_value\n  "\n becomes "key": "start_of_value"
    malformed_pattern = r'"([^"]+)"\s*:\s*"([^"]*?)(?:\s*\n\s*"|\s*\n\s*$)'
    malformed_matches = re.findall(malformed_pattern, malformed_json)
    
    for key, value in malformed_matches:
        key = key.strip().lower().replace(' ', '_')
        value = value.strip()
        if key in expected_fields and value:
            extracted_pairs[key] = value
            logger.debug(f"Extracted from malformed pattern: {key} = {value}")
    
    # If we got some good matches, use them; otherwise fall back to general patterns
    if len(extracted_pairs) >= 3:  # If we got at least 3 fields, consider it successful
        logger.info(f"Successfully extracted {len(extracted_pairs)} fields using malformed pattern matching")
    else:
        # Fall back to general pattern matching
        for pattern in patterns:
            matches = re.findall(pattern, malformed_json)
            for key, value in matches:
                # Clean the key
                key = key.strip().lower().replace(' ', '_')
                
                # Map common variations to standard field names
                key_mappings = {
                    'company': 'company_name',
                    'store': 'company_name',
                    'business': 'company_name',
                    'phone': 'phone_number',
                    'tel': 'phone_number',
                    'telephone': 'phone_number',
                    'total': 'total_amount',
                    'amount': 'total_amount',
                    'price': 'total_amount',
                }
                
                # Apply key mapping
                for variant, standard in key_mappings.items():
                    if variant in key:
                        key = standard
                        break
                
                # Clean the value
                value = value.strip()
                
                # Remove trailing quotes, newlines, and extra whitespace
                value = re.sub(r'["\']\s*$', '', value)
                value = re.sub(r'\s*\n.*$', '', value, flags=re.DOTALL)
                value = value.strip()
                
                # Only keep non-empty values and valid field names
                if value and key in expected_fields:
                    extracted_pairs[key] = value
    
    # Build the reconstructed JSON
    reconstructed = "{\n"
    pairs_added = 0
    
    for field in expected_fields:
        if field in extracted_pairs:
            value = extracted_pairs[field]
            
            # Format the value appropriately
            if value.replace('.', '').replace('$', '').replace(',', '').isdigit():
                # Numeric value - remove quotes and currency symbols
                value = value.replace('$', '').replace(',', '')
                if '.' in value:
                    value = str(float(value))
                else:
                    value = str(int(value))
            else:
                # String value - ensure it's quoted and escape newlines
                value = value.replace('\n', '\\n').replace('\r', '\\r')
                if not value.startswith('"'):
                    value = f'"{value}"'
            
            comma = "," if pairs_added < len(extracted_pairs) - 1 else ""
            reconstructed += f'  "{field}": {value}{comma}\n'
            pairs_added += 1
    
    reconstructed += "}"
    
    logger.debug(f"Reconstructed JSON with {pairs_added} fields:\n{reconstructed}")
    return reconstructed


def extract_json_from_response(
    output: str, extraction_pattern: str = r'{\s*"(date|store|tax|total|prod).*?}'
) -> Dict:
    """
    Extract and parse JSON output from the model's text response.

    Args:
        output: The raw text output from the model
        extraction_pattern: Regex pattern to find the JSON object

    Returns:
        The parsed JSON object containing the extracted fields
    """
    # Default parsed JSON to return in case of failure
    parsed_json = {
        "date_value": "",
        "store_name_value": "",
        "tax_value": "",
        "total_value": "",
        "prod_item_value": [],
        "prod_quantity_value": [],
        "prod_price_value": [],
    }

    try:
        # Search for JSON pattern in the response
        json_match = re.search(extraction_pattern, output, re.DOTALL)
        if json_match:
            # Extract and parse the JSON
            extracted_json = json_match.group(0)
            parsed_json = json.loads(extracted_json)
    except Exception as e:
        logger.error(f"LLM did not return a valid JSON format: {e}")

    # Always return parsed_json, whether default or extracted
    return parsed_json
