"""
Evaluation metrics for InternVL Evaluation

This module provides metrics for evaluating extraction results.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dateparser
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from internvl.evaluation.schema_converter import ensure_sroie_schema
from internvl.utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


def clean_numeric(value: str) -> str:
    """
    Extract and format numerical values from a string.

    Args:
        value: Input string containing numbers

    Returns:
        Formatted string with extracted numbers
    """
    numbers = re.findall(r"\d*\.?\d+", value)
    formatted_numbers = [f"{float(num):.2f}" for num in numbers]
    formatted_str = " ".join(formatted_numbers)
    return formatted_str


def normalize_evaluation_date(date_str: str) -> str:
    """
    Normalize a date string to a standardized format for evaluation.

    Args:
        date_str: The date string to normalize

    Returns:
        Normalized date string or original if parsing fails
    """
    try:
        # Clean incorrect double colons (if any)
        date_str = date_str.replace(": :", ":")

        # Attempt to parse the date
        parsed_date = dateparser.parse(date_str)  # Auto-detect language and format

        if parsed_date:
            # Use ISO format for consistent comparison
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S")  # Standardized format
        else:
            raise ValueError(f"Could not parse date: {date_str}")

    except Exception as e:
        logger.error(f"Error parsing the date field: {date_str} | {e}")
        return date_str  # Return original string if parsing fails


def normalize_store_name(name_str: str) -> str:
    """
    Normalize a store name for consistent comparison.

    Args:
        name_str: The store name to normalize

    Returns:
        Normalized store name (uppercase with no extra spaces)
    """
    if not name_str:
        return ""

    # Convert to uppercase and strip spaces
    return name_str.upper().strip()


def normalize_number(number_str: str) -> str:
    """
    Normalize a numeric string (used for tax, total, etc).

    Args:
        number_str: The numeric string to normalize

    Returns:
        Normalized number string (as a formatted float with 2 decimal places)
    """
    if not number_str:
        return ""

    # Remove non-numeric characters except decimal point
    clean_str = re.sub(r"[^\d.]", "", str(number_str))
    try:
        # Convert to float and format with 2 decimal places
        number_float = float(clean_str)
        return f"{number_float:.2f}"
    except (ValueError, TypeError):
        return number_str.strip()


def normalize_list_field(items: List[str], field_type: str = "items") -> List[str]:
    """
    Normalize a list field (products, quantities, prices).

    Args:
        items: The list of items to normalize
        field_type: The type of field ('items', 'quantities', 'prices')

    Returns:
        Normalized list of items
    """
    if not items:
        return []

    normalized = []

    for item in items:
        if field_type == "items":
            # For product items, lowercase and strip spaces
            normalized.append(str(item).lower().strip())
        elif field_type == "quantities":
            # For quantities, extract numeric values
            clean_str = re.sub(r"[^\d.]", "", str(item))
            try:
                qty_float = float(clean_str)
                normalized.append(f"{qty_float}")
            except (ValueError, TypeError):
                normalized.append(str(item).strip())
        elif field_type == "prices":
            # For prices, format as currency with 2 decimal places
            clean_str = re.sub(r"[^\d.]", "", str(item))
            try:
                price_float = float(clean_str)
                normalized.append(f"{price_float:.2f}")
            except (ValueError, TypeError):
                normalized.append(str(item).strip())
        else:
            # Default normalization
            normalized.append(str(item).strip())

    return normalized


def calculate_metrics(actual: str, predicted: str, image_id: str) -> Dict[str, float]:
    """
    Calculate evaluation metrics for a single instance.

    Args:
        actual: The ground truth string
        predicted: The predicted string from the model
        image_id: The ID of the image being evaluated

    Returns:
        Dictionary of metrics including Precision, Recall, F1-score, and BLEU score
    """
    logger.debug(f"Image {image_id} | Actual: {actual} | Predicted: {predicted}")

    # Convert strings to character sets for evaluation
    actual_chars = Counter(actual)
    predicted_chars = Counter(predicted)

    # Calculate Precision, Recall, and F1-score
    common = (
        actual_chars & predicted_chars
    )  # Intersection (common characters with counts)
    tp = sum(common.values())  # True Positives (correctly predicted characters)
    fp = (
        sum(predicted_chars.values()) - tp
    )  # False Positives (extra predicted characters)
    fn = sum(actual_chars.values()) - tp  # False Negatives (missed actual characters)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Calculate BLEU score with smoothing
    smoothing = SmoothingFunction().method1  # Apply smoothing method
    bleu = sentence_bleu(
        [list(actual)],
        list(predicted),
        weights=(1, 0, 0, 0),  # Use only unigram matching to avoid warnings
        smoothing_function=smoothing,
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "BLEU Score": bleu,
    }


def average_metrics(
    predicts: List[str], actuals: List[str], image_ids: List[str]
) -> Dict[str, float]:
    """
    Calculate average metrics across multiple predictions.

    Args:
        predicts: List of predicted values as strings
        actuals: List of actual (ground truth) values as strings
        image_ids: List of image IDs for logging

    Returns:
        Dictionary of average metrics
    """
    total_metrics = {"Precision": 0, "Recall": 0, "F1-score": 0, "BLEU Score": 0}
    n = len(predicts)

    for pred, act, img_id in zip(predicts, actuals, image_ids, strict=False):
        metrics = calculate_metrics(act, pred, img_id)
        for key in total_metrics:
            total_metrics[key] += metrics[key]

    avg_metrics = {key: total / n for key, total in total_metrics.items() if n > 0}
    return avg_metrics


def process_results_for_eval(
    results: List[Dict], ground_truth: Dict, fields: List[str] = None
) -> Tuple[Dict, Dict, List[str]]:
    """
    Process results for evaluation by organizing predictions and ground truth data.

    Args:
        results: List of dictionaries containing inference results
        ground_truth: Dictionary containing ground truth data
        fields: List of fields to process

    Returns:
        Tuple containing organized predictions, actuals, and image IDs
    """
    try:
        predictions = {}
        actuals = {}
        image_ids = []

        # Use default fields if none provided
        evaluation_fields = (
            fields
            if fields is not None
            else [
                "date_value",
                "store_name_value",
                "tax_value",
                "total_value",
                "prod_item_value",
                "prod_quantity_value",
                "prod_price_value",
            ]
        )

        # Print stats about what we're trying to match
        result_ids = [r["image_id"] for r in results]
        gt_ids = list(ground_truth.keys())

        logger.info(
            f"Matching {len(result_ids)} prediction files with {len(gt_ids)} ground truth files"
        )
        logger.info(f"Prediction IDs (first 5): {result_ids[:5]}")
        logger.info(f"Ground truth IDs (first 5): {gt_ids[:5]}")

        # Count how many prediction files match ground truth
        matched_count = sum(1 for r_id in result_ids if r_id in gt_ids)
        logger.info(
            f"Found {matched_count} matching files out of {len(result_ids)} predictions"
        )

        # Look for SROIE vs synthetic mismatches
        if "sample_receipt" in ":".join(result_ids) and "sroie_test" in ":".join(
            gt_ids
        ):
            logger.error(
                "MISMATCHED DATASETS: Predictions appear to be for synthetic receipts but ground truth is for SROIE"
            )
        elif "sroie_test" in ":".join(result_ids) and "sample_receipt" in ":".join(
            gt_ids
        ):
            logger.error(
                "MISMATCHED DATASETS: Predictions appear to be for SROIE but ground truth is for synthetic receipts"
            )

        for field in evaluation_fields:
            f_predictions = []
            f_actuals = []

            for result in results:
                image_id = str(result["image_id"])

                if image_id in ground_truth.keys():
                    image_ids.append(image_id)

                    # Get prediction and ground truth values
                    pred_value = result.get("extracted_info", {}).get(field, "")
                    gt_value = ground_truth[image_id].get(field, "")

                    # Debug: log the values being compared (for first few only)
                    if len(f_predictions) < 3:  # Only log first 3 for brevity
                        logger.debug(f"Comparing {field} for {image_id}:")
                        logger.debug(f"  Prediction: {pred_value}")
                        logger.debug(f"  Ground Truth: {gt_value}")

                    # Append values
                    f_predictions.append(pred_value)
                    f_actuals.append(gt_value)

            predictions[field] = f_predictions
            actuals[field] = f_actuals

        # Final stats
        logger.info(f"Collected {len(image_ids)} matching image IDs for evaluation")

        return predictions, actuals, image_ids

    except Exception as e:
        logger.error(f"Exception in function process_results_for_eval(): {e}")
        return None, None, None  # Return None values in case of exception


def get_average_metrics_per_field(
    predictions: Dict[str, List],
    actuals: Dict[str, List],
    image_ids: List[str],
    fields: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate average raw metrics for each field.

    Args:
        predictions: Dictionary of prediction values by field
        actuals: Dictionary of ground truth values by field
        image_ids: List of image IDs used in evaluation
        fields: List of fields to evaluate

    Returns:
        Dictionary of accuracy metrics by field
    """
    accuracy_per_field = {}

    # Use default fields if none provided
    evaluation_fields = (
        fields
        if fields is not None
        else [
            "date_value",
            "store_name_value",
            "tax_value",
            "total_value",
            "prod_item_value",
            "prod_quantity_value",
            "prod_price_value",
        ]
    )

    for field in evaluation_fields:
        # Convert all values to strings, join list items with spaces
        predicted_output = [
            " ".join(str(x) for x in item) if isinstance(item, list) else str(item)
            for item in predictions[field]
        ]
        ground_truth = [
            " ".join(str(x) for x in item) if isinstance(item, list) else str(item)
            for item in actuals[field]
        ]

        # Calculate average metrics for this field
        avg_results = average_metrics(predicted_output, ground_truth, image_ids)
        accuracy_per_field[field] = avg_results

    return accuracy_per_field


def get_average_metrics_per_parsed_fields(
    predictions: Dict[str, List],
    actuals: Dict[str, List],
    image_ids: List[str],
    fields: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate average metrics after normalizing field values.

    Args:
        predictions: Dictionary of prediction values by field
        actuals: Dictionary of ground truth values by field
        image_ids: List of image IDs used in evaluation
        fields: List of fields to evaluate

    Returns:
        Dictionary of accuracy metrics by field after normalization
    """
    accuracy_per_field = {}

    # Use default fields if none provided
    evaluation_fields = (
        fields
        if fields is not None
        else [
            "date_value",
            "store_name_value",
            "tax_value",
            "total_value",
            "prod_item_value",
            "prod_quantity_value",
            "prod_price_value",
        ]
    )

    for field in evaluation_fields:
        if field == "date_value":
            # Normalize dates for proper comparison
            predicted_output = [
                " ".join(normalize_evaluation_date(str(x)) for x in item)
                if isinstance(item, list)
                else normalize_evaluation_date(str(item))
                for item in predictions[field]
            ]
            ground_truth = [
                " ".join(normalize_evaluation_date(str(x)) for x in item)
                if isinstance(item, list)
                else normalize_evaluation_date(str(item))
                for item in actuals[field]
            ]
        elif field in ["prod_price_value", "tax_value", "total_value"]:
            # Clean numeric values for proper comparison
            predicted_output = [
                " ".join(clean_numeric(str(x)) for x in item)
                if isinstance(item, list)
                else clean_numeric(str(item))
                for item in predictions[field]
            ]
            ground_truth = [
                " ".join(clean_numeric(str(x)) for x in item)
                if isinstance(item, list)
                else clean_numeric(str(item))
                for item in actuals[field]
            ]
        else:
            # Standard string formatting for other fields
            predicted_output = [
                " ".join(str(x) for x in item) if isinstance(item, list) else str(item)
                for item in predictions[field]
            ]
            ground_truth = [
                " ".join(str(x) for x in item) if isinstance(item, list) else str(item)
                for item in actuals[field]
            ]

        # Calculate average metrics with normalized values
        avg_results = average_metrics(predicted_output, ground_truth, image_ids)
        accuracy_per_field[field] = avg_results

    return accuracy_per_field


def normalize_field_values(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize all field values in a dictionary for better comparison.

    Args:
        data_dict: Dictionary containing field values to normalize

    Returns:
        Dictionary with normalized field values
    """
    if not data_dict:
        return {}

    normalized = data_dict.copy()

    # Apply field-specific normalization
    if "date_value" in normalized:
        normalized["date_value"] = normalize_evaluation_date(
            str(normalized["date_value"])
        )

    if "store_name_value" in normalized:
        normalized["store_name_value"] = normalize_store_name(
            str(normalized["store_name_value"])
        )

    if "tax_value" in normalized:
        normalized["tax_value"] = normalize_number(str(normalized["tax_value"]))

    if "total_value" in normalized:
        normalized["total_value"] = normalize_number(str(normalized["total_value"]))

    # Handle list fields
    if "prod_item_value" in normalized and isinstance(
        normalized["prod_item_value"], list
    ):
        normalized["prod_item_value"] = normalize_list_field(
            normalized["prod_item_value"], "items"
        )

    if "prod_quantity_value" in normalized and isinstance(
        normalized["prod_quantity_value"], list
    ):
        normalized["prod_quantity_value"] = normalize_list_field(
            normalized["prod_quantity_value"], "quantities"
        )

    if "prod_price_value" in normalized and isinstance(
        normalized["prod_price_value"], list
    ):
        normalized["prod_price_value"] = normalize_list_field(
            normalized["prod_price_value"], "prices"
        )

    return normalized


def calculate_field_metrics(
    predictions_dir: Path,
    ground_truth_dir: Path,
    fields: List[str] = None,
    normalize: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Calculate metrics by comparing predictions with ground truth.

    Args:
        predictions_dir: Directory containing prediction files
        ground_truth_dir: Directory containing ground truth files
        fields: Fields to evaluate
        normalize: Whether to normalize fields before comparison

    Returns:
        Tuple of (overall_metrics, field_metrics)
    """
    # Use default fields if none provided
    evaluation_fields = (
        fields
        if fields is not None
        else [
            "date_value",
            "store_name_value",
            "tax_value",
            "total_value",
            "prod_item_value",
            "prod_quantity_value",
            "prod_price_value",
        ]
    )

    # Load ground truth data
    ground_truth_dict = {}
    for gt_file in ground_truth_dir.glob("*.json"):
        try:
            with gt_file.open("r", encoding="utf-8") as f:
                gt_data = json.load(f)

            # Standardize ground truth data
            image_id = gt_file.stem

            # Debug log: Check ground truth format
            logger.debug(
                f"Ground truth file {gt_file.name} raw keys: {list(gt_data.keys())}"
            )

            # Check if the ground truth already uses the expected field names
            if all(
                field in gt_data
                for field in [
                    "date_value",
                    "store_name_value",
                    "tax_value",
                    "total_value",
                ]
            ):
                logger.debug(
                    f"Ground truth file {gt_file.name} already uses expected field names"
                )
                standardized_gt = gt_data  # Use as-is
            else:
                # Standardize from conventional field names
                logger.debug(
                    f"Ground truth file {gt_file.name} needs field name standardization"
                )
                standardized_gt = {
                    "date_value": gt_data.get("date", ""),
                    "store_name_value": gt_data.get("store_name", ""),
                    "tax_value": gt_data.get("tax", ""),
                    "total_value": gt_data.get("total", ""),
                    "prod_item_value": gt_data.get("items", []),
                    "prod_quantity_value": gt_data.get("quantities", []),
                    "prod_price_value": gt_data.get("prices", []),
                }

            # Apply normalization if enabled
            if normalize:
                standardized_gt = normalize_field_values(standardized_gt)

            ground_truth_dict[image_id] = standardized_gt
            logger.debug(
                f"Added {image_id} to ground truth dict with keys: {list(standardized_gt.keys())}"
            )
        except Exception as e:
            logger.error(f"Error loading ground truth file {gt_file}: {e}")

    # Load predictions
    results = []
    for pred_file in predictions_dir.glob("*.json"):
        try:
            with pred_file.open("r", encoding="utf-8") as f:
                prediction = json.load(f)

            # Convert prediction to SROIE schema if needed
            prediction = ensure_sroie_schema(prediction)

            # Format results for evaluation
            image_id = pred_file.stem

            # Debug log for prediction file
            logger.debug(
                f"Loading prediction file: {pred_file.name}, image_id: {image_id}"
            )
            logger.debug(f"Prediction keys: {list(prediction.keys())}")

            # Check if the corresponding ground truth exists
            if image_id in ground_truth_dict:
                logger.debug(f"Found matching ground truth for {image_id}")
            else:
                logger.warning(
                    f"No matching ground truth found for prediction: {image_id}"
                )
                # Find partial matches (debug only)
                partial_matches = [
                    gt_id
                    for gt_id in ground_truth_dict.keys()
                    if gt_id in image_id or image_id in gt_id
                ]
                if partial_matches:
                    logger.debug(f"Possible matches for {image_id}: {partial_matches}")

            # Apply normalization if enabled
            if normalize:
                prediction = normalize_field_values(prediction)

            results.append({"image_id": image_id, "extracted_info": prediction})
        except Exception as e:
            logger.error(f"Error loading prediction file {pred_file}: {e}")

    # Process results for evaluation
    predictions, actuals, image_ids = process_results_for_eval(
        results, ground_truth_dict, fields=evaluation_fields
    )

    # Calculate metrics (with or without additional field-specific normalization)
    if normalize:
        # We've already normalized the field values before processing
        field_metrics = get_average_metrics_per_field(
            predictions, actuals, image_ids, fields=evaluation_fields
        )
    else:
        # Standard evaluation without normalization
        field_metrics = get_average_metrics_per_field(
            predictions, actuals, image_ids, fields=evaluation_fields
        )

    # Calculate overall metrics (average across fields)
    overall = {}
    for metric in ["Precision", "Recall", "F1-score", "BLEU Score"]:
        values = [metrics[metric] for metrics in field_metrics.values()]
        overall[metric] = sum(values) / len(values) if values else 0

    # Add GST calculation validation
    if (
        len(ground_truth_dict) > 0
        and "tax_value" in evaluation_fields
        and "total_value" in evaluation_fields
    ):
        tax_validation_metrics = validate_gst_calculation(results, ground_truth_dict)
        if tax_validation_metrics:
            field_metrics["gst_calculation"] = tax_validation_metrics
            # Update overall metrics to include GST validation
            updated_values = {}
            for metric in ["Precision", "Recall", "F1-score", "BLEU Score"]:
                all_values = [metrics[metric] for metrics in field_metrics.values()]
                updated_values[metric] = (
                    sum(all_values) / len(all_values) if all_values else 0
                )
            overall = updated_values

    return overall, field_metrics


def validate_gst_calculation(
    results: List[Dict], ground_truth: Dict
) -> Dict[str, float]:
    """
    Validate GST (tax) calculation accuracy.

    In Australia, GST is 10% of the pre-tax amount, which equals 1/11 of the total.

    Args:
        results: List of prediction results
        ground_truth: Dictionary of ground truth values

    Returns:
        Dictionary of metrics for GST calculation accuracy
    """
    correct_count = 0
    total_count = 0

    for result in results:
        image_id = result["image_id"]
        if image_id not in ground_truth:
            continue

        prediction = result["extracted_info"]
        gt = ground_truth[image_id]

        # Get tax and total values
        try:
            pred_tax = float(
                re.sub(r"[^\d.]", "", str(prediction.get("tax_value", "")))
            )
            float(
                re.sub(r"[^\d.]", "", str(prediction.get("total_value", "")))
            )
            float(re.sub(r"[^\d.]", "", str(gt.get("tax_value", ""))))
            gt_total = float(re.sub(r"[^\d.]", "", str(gt.get("total_value", ""))))

            # Calculate expected GST (1/11 of total)
            expected_gst = round(gt_total / 11, 2)

            # Check if predicted tax matches the expected GST calculation
            if abs(pred_tax - expected_gst) < 0.02:  # Allow small rounding difference
                correct_count += 1

            total_count += 1
        except (ValueError, TypeError):
            # Skip if values can't be converted to float
            continue

    # Calculate metrics
    if total_count > 0:
        precision = correct_count / total_count
        recall = precision  # Same value for this calculation
        f1_score = precision  # Also same since precision=recall

        return {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score,
            "BLEU Score": precision,  # Use same value for BLEU for consistency
        }

    return None
