"""
Field normalization utilities for InternVL Evaluation

This module provides functions for normalizing field values extracted from model output.
"""

import re
from typing import Any, Dict

# Note: Legacy JSON extraction removed - use Key-Value extraction instead
from internvl.utils import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Confidence scoring imports temporarily removed for emergency restore


def normalize_date(date_str: str) -> str:
    """
    Normalize dates to a standard format.

    Args:
        date_str: Input date string

    Returns:
        Normalized date string in DD/MM/YYYY format or original if parsing fails
    """
    try:
        if not date_str:
            return ""
        
        # Remove any time component
        if " " in date_str:
            date_str = date_str.split(" ")[0]
        
        # Try to match DD/MM/YYYY pattern
        date_pattern = r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})'
        match = re.match(date_pattern, date_str.strip())
        
        if match:
            day, month, year = match.groups()
            # Ensure 2-digit day and month
            day = day.zfill(2)
            month = month.zfill(2)
            return f"{day}/{month}/{year}"
        
        return date_str
    except Exception as e:
        logger.error(f"Error normalizing date '{date_str}': {e}")
        return date_str


def normalize_store_name(name_str: str) -> str:
    """
    Normalize store names to uppercase.

    Args:
        name_str: Input store name

    Returns:
        Normalized store name string
    """
    if not name_str:
        return ""
    # Convert to uppercase and remove leading/trailing spaces
    return name_str.upper().strip()


def normalize_number(value_str: str) -> str:
    """
    Normalize numeric values by removing currency symbols and formatting.

    Args:
        value_str: Input numeric string

    Returns:
        Normalized number as string with two decimal places
    """
    try:
        # Handle empty or None values
        if not value_str:
            return ""

        # Extract digits and decimal point
        matches = re.search(r"([\d,.]+)", str(value_str))
        if not matches:
            return value_str

        number_str = matches.group(1)

        # Handle different number formats
        number_str = number_str.replace(",", ".")

        # Make sure we only have one decimal point
        parts = number_str.split(".")
        if len(parts) > 2:
            # Keep only the first decimal point
            number_str = parts[0] + "." + "".join(parts[1:]).replace(".", "")

        # Convert to float and format to 2 decimal places
        try:
            number = float(number_str)
            return f"{number:.2f}"
        except ValueError:
            return value_str
    except Exception as e:
        logger.error(f"Error normalizing number '{value_str}': {e}")
        return value_str


def post_process_prediction(raw_text: str) -> Dict[str, Any]:
    """
    Process raw model output to extract and normalize structured data.
    
    DEPRECATED: This function is maintained for compatibility with legacy evaluation scripts.
    For new development, use extract_key_value_enhanced from key_value_parser instead.

    Args:
        raw_text: Raw text output from the model

    Returns:
        Normalized structured data object
    """
    import json
    
    # Simple JSON extraction for legacy compatibility
    try:
        # Try to find JSON in the text
        lines = raw_text.strip().split('\n')
        json_text = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                json_text = line
                break
        
        if json_text:
            data = json.loads(json_text)
        else:
            # Fallback: return empty structure
            data = {}
    except (json.JSONDecodeError, Exception):
        logger.warning(f"Could not parse JSON from: {raw_text[:100]}...")
        return {"error": "Could not extract valid JSON", "raw_text": raw_text[:200]}

    # Normalize fields if they exist
    if "date_value" in data:
        data["date_value"] = normalize_date(data["date_value"])

    if "store_name_value" in data:
        data["store_name_value"] = normalize_store_name(data["store_name_value"])

    if "tax_value" in data:
        data["tax_value"] = normalize_number(data["tax_value"])

    if "total_value" in data:
        data["total_value"] = normalize_number(data["total_value"])

    return data


# All confidence scoring functions temporarily removed for emergency restore
