"""
Extraction module for InternVL Evaluation

This module handles Key-Value extraction (preferred) and data normalization.
Key-Value extraction is the recommended approach for production use.
"""

# Key-Value extraction (PREFERRED for production)
from .key_value_parser import extract_key_value_enhanced

# Bank statement parsing
from .bank_statement_parser import extract_bank_statement_with_highlights

# Data normalization and validation
from .normalization import (
    normalize_date,
    normalize_number,
    normalize_store_name,
    post_process_prediction,
)
from .validation import validate_extraction_result

__all__ = [
    # Key-Value extraction (PREFERRED)
    "extract_key_value_enhanced",
    
    # Bank statement parsing
    "extract_bank_statement_with_highlights",
    
    # Data normalization
    "normalize_date",
    "normalize_store_name", 
    "normalize_number",
    "post_process_prediction",
    
    # Validation
    "validate_extraction_result",
]
