"""
Confidence scoring for JSON extraction predictions.

This module provides confidence scoring and selective reprocessing
to improve extraction accuracy.
"""

import re
from typing import Any, Dict, Tuple

from internvl.utils import get_logger

logger = get_logger(__name__)


def calculate_confidence_score(prediction: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate confidence score for a prediction.
    
    Args:
        prediction: The extracted JSON prediction
        
    Returns:
        Tuple of (overall_confidence, component_scores)
    """
    if not prediction:
        return 0.0, {}
    
    component_scores = {}
    
    # 1. Field Completeness Score (40% weight)
    completeness_score = _calculate_completeness_score(prediction)
    component_scores['completeness'] = completeness_score
    
    # 2. GST Validation Score (30% weight)
    gst_score = _calculate_gst_validation_score(prediction)
    component_scores['gst_validation'] = gst_score
    
    # 3. List Consistency Score (20% weight)
    consistency_score = _calculate_list_consistency_score(prediction)
    component_scores['list_consistency'] = consistency_score
    
    # 4. Format Validation Score (10% weight)
    format_score = _calculate_format_validation_score(prediction)
    component_scores['format_validation'] = format_score
    
    # Calculate weighted overall score
    overall_confidence = (
        completeness_score * 0.4 +
        gst_score * 0.3 +
        consistency_score * 0.2 +
        format_score * 0.1
    )
    
    component_scores['overall'] = overall_confidence
    
    logger.debug(f"Confidence scores: {component_scores}")
    
    return overall_confidence, component_scores


def _calculate_completeness_score(prediction: Dict[str, Any]) -> float:
    """Calculate score based on field completeness."""
    required_fields = [
        "date_value", "store_name_value", "tax_value", "total_value",
        "prod_item_value", "prod_quantity_value", "prod_price_value"
    ]
    
    filled_fields = 0
    for field in required_fields:
        if field in prediction:
            value = prediction[field]
            if isinstance(value, str) and value.strip():
                filled_fields += 1
            elif isinstance(value, list) and value:
                filled_fields += 1
    
    return filled_fields / len(required_fields)


def _calculate_gst_validation_score(prediction: Dict[str, Any]) -> float:
    """Calculate score based on GST validation (Australian 10% tax)."""
    tax_value = prediction.get("tax_value", "")
    total_value = prediction.get("total_value", "")
    
    if not tax_value or not total_value:
        return 0.5  # Neutral score if missing values
    
    try:
        # Extract numeric values
        tax_num = _extract_numeric_value(tax_value)
        total_num = _extract_numeric_value(total_value)
        
        if tax_num <= 0 or total_num <= 0:
            return 0.3
        
        # Calculate expected GST (1/11 of total in Australia)
        expected_gst = total_num / 11
        
        # Calculate percentage difference
        if expected_gst > 0:
            diff_percentage = abs(tax_num - expected_gst) / expected_gst
            
            # Score based on accuracy
            if diff_percentage <= 0.05:  # Within 5%
                return 1.0
            elif diff_percentage <= 0.10:  # Within 10%
                return 0.8
            elif diff_percentage <= 0.20:  # Within 20%
                return 0.6
            elif diff_percentage <= 0.50:  # Within 50%
                return 0.4
            else:
                return 0.2
        
        return 0.5
        
    except (ValueError, ZeroDivisionError):
        return 0.3


def _calculate_list_consistency_score(prediction: Dict[str, Any]) -> float:
    """Calculate score based on list field consistency."""
    list_fields = ["prod_item_value", "prod_quantity_value", "prod_price_value"]
    
    lengths = []
    for field in list_fields:
        if field in prediction and isinstance(prediction[field], list):
            lengths.append(len(prediction[field]))
        else:
            lengths.append(0)
    
    # Check if all lists have the same length
    if len(set(lengths)) == 1:
        # All same length - check if meaningful
        if lengths[0] == 0:
            return 0.2  # All empty
        elif lengths[0] >= 1:
            return 1.0  # Good consistency
    
    # Different lengths - penalize based on variance
    if lengths:
        max_len = max(lengths)
        min_len = min(lengths)
        if max_len > 0:
            consistency_ratio = min_len / max_len
            return max(0.1, consistency_ratio)
    
    return 0.1


def _calculate_format_validation_score(prediction: Dict[str, Any]) -> float:
    """Calculate score based on format validation."""
    format_score = 0.0
    checks = 0
    
    # Date format validation (DD/MM/YYYY)
    date_value = prediction.get("date_value", "")
    if date_value:
        checks += 1
        if _is_valid_date_format(date_value):
            format_score += 1.0
    
    # Numeric values validation
    numeric_fields = ["tax_value", "total_value"]
    for field in numeric_fields:
        value = prediction.get(field, "")
        if value:
            checks += 1
            if _is_numeric_value(value):
                format_score += 1.0
    
    # Store name validation (should be non-empty string)
    store_name = prediction.get("store_name_value", "")
    if store_name:
        checks += 1
        if isinstance(store_name, str) and len(store_name.strip()) > 0:
            format_score += 1.0
    
    return format_score / checks if checks > 0 else 0.5


def _extract_numeric_value(value: str) -> float:
    """Extract numeric value from string."""
    if not value:
        return 0.0
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[$,]', '', str(value))
    
    # Extract first number found
    match = re.search(r'\d+\.?\d*', cleaned)
    if match:
        return float(match.group())
    
    return 0.0


def _is_numeric_value(value: str) -> bool:
    """Check if value represents a valid numeric amount."""
    try:
        num = _extract_numeric_value(value)
        return num > 0
    except (ValueError, TypeError):
        return False


def _is_valid_date_format(date_str: str) -> bool:
    """Check if date string matches DD/MM/YYYY format."""
    if not date_str:
        return False
    
    # Check for DD/MM/YYYY pattern
    pattern = r'^\d{1,2}/\d{1,2}/\d{4}$'
    if re.match(pattern, date_str):
        # Basic validation of ranges
        parts = date_str.split('/')
        try:
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            return (1 <= day <= 31 and 
                    1 <= month <= 12 and 
                    1900 <= year <= 2100)
        except (ValueError, IndexError):
            return False
    
    return False


def should_reprocess_prediction(confidence: float, threshold: float) -> bool:
    """
    Determine if a prediction should be reprocessed based on confidence.
    
    Args:
        confidence: Confidence score (0.0-1.0)
        threshold: Minimum acceptable confidence threshold
        
    Returns:
        True if prediction should be reprocessed
    """
    return confidence < threshold


def get_confidence_summary(confidence: float, components: Dict[str, float]) -> str:
    """
    Generate a human-readable confidence summary.
    
    Args:
        confidence: Overall confidence score
        components: Component scores dictionary
        
    Returns:
        Summary string
    """
    level = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.6 else "LOW"
    
    issues = []
    if components.get('completeness', 0) < 0.7:
        issues.append("missing fields")
    if components.get('gst_validation', 0) < 0.7:
        issues.append("GST validation")
    if components.get('list_consistency', 0) < 0.7:
        issues.append("list consistency")
    if components.get('format_validation', 0) < 0.7:
        issues.append("format validation")
    
    summary = f"Confidence: {confidence:.2f} ({level})"
    if issues:
        summary += f" - Issues: {', '.join(issues)}"
    
    return summary