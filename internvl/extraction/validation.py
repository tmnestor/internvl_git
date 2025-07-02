"""
Post-processing validation and error correction for JSON extraction.

This module provides JSON schema validation and automatic error correction
to improve extraction reliability.
"""

import re
from typing import Any, Dict, List, Tuple

from internvl.utils import get_logger

logger = get_logger(__name__)


def validate_and_fix_json(extracted_json: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate extracted JSON and apply automatic fixes.
    
    Args:
        extracted_json: The extracted JSON dictionary
        
    Returns:
        Tuple of (fixed_json, list_of_fixes_applied)
    """
    fixes_applied = []
    fixed_json = extracted_json.copy()
    
    # Define expected schema
    expected_schema = {
        "date_value": str,
        "store_name_value": str, 
        "tax_value": str,
        "total_value": str,
        "prod_item_value": list,
        "prod_quantity_value": list,
        "prod_price_value": list
    }
    
    # Fix 1: Ensure all required fields exist
    for field, field_type in expected_schema.items():
        if field not in fixed_json:
            if field_type is str:
                fixed_json[field] = ""
            elif field_type is list:
                fixed_json[field] = []
            fixes_applied.append(f"Added missing field: {field}")
    
    # Fix 2: Ensure correct data types
    for field, expected_type in expected_schema.items():
        if field in fixed_json:
            current_value = fixed_json[field]
            if not isinstance(current_value, expected_type):
                if expected_type is str:
                    fixed_json[field] = str(current_value)
                    fixes_applied.append(f"Converted {field} to string")
                elif expected_type is list and not isinstance(current_value, list):
                    fixed_json[field] = [str(current_value)] if current_value else []
                    fixes_applied.append(f"Converted {field} to list")
    
    # Fix 3: Clean string values
    string_fields = ["date_value", "store_name_value", "tax_value", "total_value"]
    for field in string_fields:
        if field in fixed_json and isinstance(fixed_json[field], str):
            original = fixed_json[field]
            # Remove control characters and extra whitespace
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', original)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned != original:
                fixed_json[field] = cleaned
                fixes_applied.append(f"Cleaned {field} string")
    
    # Fix 4: Validate and fix date format
    if "date_value" in fixed_json:
        date_value = fixed_json["date_value"]
        if date_value:
            fixed_date = fix_date_format(date_value)
            if fixed_date != date_value:
                fixed_json["date_value"] = fixed_date
                fixes_applied.append(f"Fixed date format: {date_value} -> {fixed_date}")
    
    # Fix 5: Ensure list field consistency
    list_fields = ["prod_item_value", "prod_quantity_value", "prod_price_value"]
    list_lengths = []
    
    for field in list_fields:
        if field in fixed_json and isinstance(fixed_json[field], list):
            list_lengths.append(len(fixed_json[field]))
    
    if list_lengths and len(set(list_lengths)) > 1:
        # Lists have different lengths - fix by truncating to shortest
        min_length = min(list_lengths)
        for field in list_fields:
            if field in fixed_json and isinstance(fixed_json[field], list):
                if len(fixed_json[field]) > min_length:
                    fixed_json[field] = fixed_json[field][:min_length]
                    fixes_applied.append(f"Truncated {field} to length {min_length}")
    
    # Fix 6: Clean list items
    for field in list_fields:
        if field in fixed_json and isinstance(fixed_json[field], list):
            cleaned_list = []
            for item in fixed_json[field]:
                if isinstance(item, str):
                    # Clean item string
                    cleaned_item = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', str(item))
                    cleaned_item = re.sub(r'\s+', ' ', cleaned_item).strip()
                    cleaned_list.append(cleaned_item)
                else:
                    cleaned_list.append(str(item))
            
            if cleaned_list != fixed_json[field]:
                fixed_json[field] = cleaned_list
                fixes_applied.append(f"Cleaned {field} list items")
    
    # Fix 7: Validate numeric values
    numeric_fields = ["tax_value", "total_value"]
    for field in numeric_fields:
        if field in fixed_json and fixed_json[field]:
            original = fixed_json[field]
            # Extract numeric value using regex
            numeric_match = re.search(r'[\d,]+\.?\d*', str(original))
            if numeric_match:
                numeric_str = numeric_match.group().replace(',', '')
                try:
                    # Validate it's a valid number
                    float(numeric_str)
                    if numeric_str != original:
                        fixed_json[field] = numeric_str
                        fixes_applied.append(f"Cleaned numeric {field}: {original} -> {numeric_str}")
                except ValueError:
                    pass
    
    # Fix 8: Validate GST calculation if both tax and total are present
    if (fixed_json.get("tax_value") and fixed_json.get("total_value") and
        is_numeric(fixed_json["tax_value"]) and is_numeric(fixed_json["total_value"])):
        
        tax_val = float(fixed_json["tax_value"])
        total_val = float(fixed_json["total_value"])
        
        # Check if GST is approximately 10% (1/11 of total)
        expected_gst = total_val / 11
        if abs(tax_val - expected_gst) / expected_gst > 0.1:  # More than 10% off
            # Try to fix if total seems reasonable
            if total_val > 0:
                calculated_gst = round(total_val / 11, 2)
                logger.warning(f"GST validation failed: {tax_val} vs expected {calculated_gst}")
                # Don't auto-fix GST as it might be correct in edge cases
    
    return fixed_json, fixes_applied


def fix_date_format(date_str: str) -> str:
    """
    Fix common date format issues.
    
    Args:
        date_str: Original date string
        
    Returns:
        Cleaned date string in DD/MM/YYYY format
    """
    if not date_str:
        return ""
    
    # Remove extra whitespace and control characters
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', date_str).strip()
    
    # Try to match various date patterns
    patterns = [
        r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})',  # DD/MM/YYYY or DD-MM-YYYY
        r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2})',  # DD/MM/YY or DD-MM-YY
        r'(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            day, month, year = match.groups()
            
            # Handle 2-digit years
            if len(year) == 2:
                year_int = int(year)
                if year_int > 50:  # Assume 1950-1999
                    year = f"19{year}"
                else:  # Assume 2000-2049
                    year = f"20{year}"
            
            # Handle YYYY/MM/DD format
            if len(match.group(1)) == 4:  # First group is year
                year, month, day = match.groups()
            
            # Ensure 2-digit day and month
            day = day.zfill(2)
            month = month.zfill(2)
            
            # Basic validation
            try:
                day_int = int(day)
                month_int = int(month)
                year_int = int(year)
                
                if 1 <= day_int <= 31 and 1 <= month_int <= 12 and 1900 <= year_int <= 2100:
                    return f"{day}/{month}/{year}"
            except ValueError:
                continue
    
    # If no pattern matched, return original
    return cleaned


def is_numeric(value: str) -> bool:
    """
    Check if a string represents a valid numeric value.
    
    Args:
        value: String to check
        
    Returns:
        True if numeric, False otherwise
    """
    if not value:
        return False
    
    # Remove common formatting
    cleaned = str(value).replace(',', '').replace('$', '').strip()
    
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def validate_json_schema(json_obj: Dict[str, Any]) -> List[str]:
    """
    Validate JSON against expected schema and return list of errors.
    
    Args:
        json_obj: JSON object to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Required fields
    required_fields = {
        "date_value": str,
        "store_name_value": str,
        "tax_value": str,
        "total_value": str,
        "prod_item_value": list,
        "prod_quantity_value": list,
        "prod_price_value": list
    }
    
    # Check required fields
    for field, expected_type in required_fields.items():
        if field not in json_obj:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(json_obj[field], expected_type):
            errors.append(f"Field {field} should be {expected_type.__name__}, got {type(json_obj[field]).__name__}")
    
    # Check list field consistency
    list_fields = ["prod_item_value", "prod_quantity_value", "prod_price_value"]
    lengths = []
    
    for field in list_fields:
        if field in json_obj and isinstance(json_obj[field], list):
            lengths.append(len(json_obj[field]))
    
    if lengths and len(set(lengths)) > 1:
        errors.append(f"Product list fields have inconsistent lengths: {dict(zip(list_fields, lengths, strict=False))}")
    
    return errors
