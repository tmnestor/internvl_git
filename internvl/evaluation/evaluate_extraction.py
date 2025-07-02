#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_extraction.py

Script to evaluate extraction results against ground truth.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Import from the src directory structure
# This is the correct import path when running as a module
from internvl.config.config import load_config
from internvl.evaluation.metrics import calculate_field_metrics
from internvl.evaluation.schema_converter import detect_schema_type, ensure_sroie_schema
from internvl.utils.logging import get_logger, setup_logging
from internvl.utils.path import PathManager

# Initialize logger
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Australian receipt extraction results against ground truth. Defaults to synthetic Australian data."
    )

    parser.add_argument(
        "--predictions-dir", help="Directory containing model predictions (default: output/predictions_synthetic)", type=str
    )
    parser.add_argument(
        "--ground-truth-dir", help="Directory containing ground truth files (default: data/synthetic/ground_truth)", type=str
    )
    parser.add_argument(
        "--output-path", help="Path to save evaluation results (default: output/evaluation_results)", type=str
    )
    parser.add_argument(
        "--fields",
        help="Comma-separated list of fields to evaluate",
        type=str,
        default="date_value,store_name_value,tax_value,total_value,prod_item_value,prod_quantity_value,prod_price_value",
    )
    parser.add_argument(
        "--normalize", help="Normalize fields before comparison", action="store_true"
    )
    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")
    parser.add_argument(
        "--save-prompt-template",
        help="Save the optimized Australian receipt prompt template",
        action="store_true",
    )
    parser.add_argument(
        "--show-examples",
        help="Show examples of field comparisons for diagnosis",
        action="store_true",
    )

    return parser.parse_args()


def run_evaluation(
    predictions_dir: Path,
    ground_truth_dir: Path,
    output_path: Path = None,
    fields: List[str] = None,
    normalize: bool = True,
    show_examples: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation comparing predictions against ground truth.

    Args:
        predictions_dir: Directory containing prediction files
        ground_truth_dir: Directory containing ground truth files
        output_path: Path to save evaluation results (optional)
        fields: List of fields to evaluate
        normalize: Whether to normalize fields before comparison

    Returns:
        Dictionary containing evaluation results
    """
    logger.info("Starting evaluation...")
    logger.info(f"Predictions directory: {predictions_dir}")
    logger.info(f"Ground truth directory: {ground_truth_dir}")

    # Ensure directories exist
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

    # Count files in directories
    pred_files = list(predictions_dir.glob("*.json"))
    gt_files = list(ground_truth_dir.glob("*.json"))
    logger.info(
        f"Found {len(pred_files)} prediction files and {len(gt_files)} ground truth files"
    )

    # Print out both prediction and ground truth files for comparison
    logger.info("\n" + "=" * 80)
    logger.info("COMPARING PREDICTIONS AND GROUND TRUTH FILES:")
    logger.info("=" * 80)

    # Get a sample of files to compare (up to 5)
    sample_pred_files = pred_files[:5] if len(pred_files) >= 5 else pred_files

    for pred_file in sample_pred_files:
        image_id = pred_file.stem
        gt_file = ground_truth_dir / f"{image_id}.json"

        logger.info(f"\nFile: {image_id}")
        logger.info("-" * 40)

        # Load prediction file
        if pred_file.exists():
            try:
                with pred_file.open("r", encoding="utf-8") as f:
                    prediction = json.load(f)
                
                # Convert prediction to SROIE schema if needed
                original_schema = detect_schema_type(prediction)
                prediction = ensure_sroie_schema(prediction)
                if original_schema != "sroie":
                    logger.info(f"Converted prediction from {original_schema} to SROIE schema")
                
                logger.info(f"PREDICTION ({pred_file}):")
                logger.info(json.dumps(prediction, indent=2))
            except Exception as e:
                logger.error(f"Error loading prediction file {pred_file}: {e}")
                logger.info("PREDICTION: Error loading file")
        else:
            logger.info("PREDICTION: File not found")

        # Load ground truth file
        if gt_file.exists():
            try:
                with gt_file.open("r", encoding="utf-8") as f:
                    ground_truth = json.load(f)
                logger.info(f"GROUND TRUTH ({gt_file}):")
                logger.info(json.dumps(ground_truth, indent=2))
            except Exception as e:
                logger.error(f"Error loading ground truth file {gt_file}: {e}")
                logger.info("GROUND TRUTH: Error loading file")
        else:
            logger.info(f"GROUND TRUTH: File not found for {image_id}")

    # For example display, we'll need to load some raw data
    examples = {}
    if show_examples:
        # Load a sample prediction and ground truth for comparison
        try:
            # Get first few files as examples
            sample_pred_files = list(predictions_dir.glob("*.json"))[:3]
            for pred_file in sample_pred_files:
                image_id = pred_file.stem
                gt_file = ground_truth_dir / f"{image_id}.json"

                if pred_file.exists() and gt_file.exists():
                    with pred_file.open("r", encoding="utf-8") as f:
                        prediction = json.load(f)
                    # Convert prediction to SROIE schema if needed
                    prediction = ensure_sroie_schema(prediction)
                    
                    with gt_file.open("r", encoding="utf-8") as f:
                        ground_truth = json.load(f)

                    # Store raw examples for display
                    examples[image_id] = {
                        "prediction": prediction,
                        "ground_truth": ground_truth,  # Use the raw ground truth directly
                    }
        except Exception as e:
            logger.warning(f"Could not load examples for comparison: {e}")

    # Calculate metrics
    overall_metrics, field_metrics = calculate_field_metrics(
        predictions_dir, ground_truth_dir, fields=fields if fields else None, normalize=normalize
    )

    # Create DataFrame for better display
    metrics_df = pd.DataFrame(
        {
            field: {metric: value for metric, value in metrics.items()}
            for field, metrics in field_metrics.items()
        }
    ).T

    # Calculate summary statistics by field type
    scalar_fields = ["date_value", "store_name_value", "tax_value", "total_value"]
    list_fields = ["prod_item_value", "prod_quantity_value", "prod_price_value"]

    # Filter fields that exist in our results
    scalar_fields = [f for f in scalar_fields if f in metrics_df.index]
    list_fields = [f for f in list_fields if f in metrics_df.index]

    # Calculate averages by field type if fields exist
    scalar_metrics = {}
    list_metrics = {}

    if scalar_fields:
        scalar_df = metrics_df.loc[scalar_fields]
        for metric in scalar_df.columns:
            scalar_metrics[metric] = scalar_df[metric].mean()

    if list_fields:
        list_df = metrics_df.loc[list_fields]
        for metric in list_df.columns:
            list_metrics[metric] = list_df[metric].mean()

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("OVERALL EVALUATION METRICS:")
    for metric, value in overall_metrics.items():
        logger.info(f"- {metric}: {value:.4f} ({value:.2%})")

    if scalar_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("SCALAR FIELD METRICS (date, store, tax, total):")
        for metric, value in scalar_metrics.items():
            logger.info(f"- {metric}: {value:.4f} ({value:.2%})")

    if list_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("LIST FIELD METRICS (products, quantities, prices):")
        for metric, value in list_metrics.items():
            logger.info(f"- {metric}: {value:.4f} ({value:.2%})")

    logger.info("\n" + "=" * 60)
    logger.info("FIELD-LEVEL METRICS:")
    for field, metrics in field_metrics.items():
        logger.info(f"\nField: {field}")
        for metric, value in metrics.items():
            logger.info(f"- {metric}: {value:.4f} ({value:.2%})")

    # Add GST calculation validation note if present
    if "gst_calculation" in field_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("GST CALCULATION VALIDATION:")
        logger.info(
            "- In Australia, GST is 10% of the pre-tax amount (1/11 of the total)"
        )
        for metric, value in field_metrics["gst_calculation"].items():
            logger.info(f"- {metric}: {value:.4f} ({value:.2%})")

    # Save results if output path provided
    if output_path:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metrics as CSV
        metrics_df.to_csv(output_path.with_suffix(".csv"))
        logger.info(f"Saved metrics to {output_path.with_suffix('.csv')}")

        # Save detailed JSON results
        results = {
            "overall_metrics": overall_metrics,
            "field_metrics": field_metrics,
            "scalar_metrics": scalar_metrics,
            "list_metrics": list_metrics,
        }
        with output_path.with_suffix(".json").open("w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved detailed results to {output_path.with_suffix('.json')}")

        # Create a visualization if matplotlib is available
        try:
            import matplotlib.pyplot as plt

            # Create F1-score bar chart
            plt.figure(figsize=(12, 6))
            fields = metrics_df.index
            f1_scores = metrics_df["F1-score"]

            plt.bar(fields, f1_scores, color="skyblue")
            plt.xlabel("Fields")
            plt.ylabel("F1-score")
            plt.title("F1-scores by Field")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # Add value labels
            for i, v in enumerate(f1_scores):
                plt.text(i, v + 0.01, f"{v:.2%}", ha="center")

            # Save the figure
            plt.savefig(output_path.with_suffix(".png"))
            logger.info(f"Saved visualization to {output_path.with_suffix('.png')}")
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")

    # Display examples if requested
    if show_examples and examples:
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLES OF FIELD COMPARISONS:")

        for image_id, data in examples.items():
            logger.info(f"\nExamples from image: {image_id}")

            # Compare each field
            default_fields = [
                "date_value", "store_name_value", "tax_value", "total_value",
                "prod_item_value", "prod_quantity_value", "prod_price_value"
            ]
            # Convert pandas Index to list if needed
            if hasattr(fields, 'tolist'):
                fields_list = fields.tolist()
            else:
                fields_list = fields if isinstance(fields, list) else []
            fields_to_use = fields_list if fields_list and len(fields_list) > 0 else default_fields
            for field in fields_to_use:
                pred_val = data["prediction"].get(field, "")
                gt_val = data["ground_truth"].get(field, "")

                # Special handling for lists to make output more readable
                if isinstance(pred_val, list) and isinstance(gt_val, list):
                    logger.info(f"\nField: {field}")
                    logger.info(f"  Ground Truth ({len(gt_val)} items): {gt_val}")
                    logger.info(f"  Prediction ({len(pred_val)} items): {pred_val}")
                else:
                    logger.info(f"\nField: {field}")
                    logger.info(f"  Ground Truth: '{gt_val}'")
                    logger.info(f"  Prediction:   '{pred_val}'")

                # Show field-specific issues
                if field == "date_value":
                    logger.info(
                        "  Note: Check date format - Australian format is DD/MM/YYYY"
                    )
                elif field == "tax_value":
                    # Calculate expected GST
                    try:
                        total = float(
                            re.sub(
                                r"[^\d.]",
                                "",
                                str(data["ground_truth"].get("total_value", "")),
                            )
                        )
                        expected_gst = round(
                            total / 11, 2
                        )  # GST is 1/11 of total in Australia
                        logger.info(
                            f"  Expected GST (1/11 of total): ${expected_gst:.2f}"
                        )
                    except (ValueError, TypeError):
                        pass

    return {
        "overall_metrics": overall_metrics,
        "field_metrics": field_metrics,
        "metrics_df": metrics_df,
        "scalar_metrics": scalar_metrics,
        "list_metrics": list_metrics,
        "examples": examples,
    }


def get_australian_prompt_example() -> str:
    """
    Return an example prompt optimized for Australian receipts.
    """
    return """<image>
Extract these seven fields from the provided Australian receipt image:
1. date_value
2. store_name_value
3. tax_value (GST amount)
4. total_value
5. prod_item_value
6. prod_quantity_value
7. prod_price_value

Return the results in JSON format. An example JSON format is:

JSON Output:
{
"date_value": "16/3/2023",
"store_name_value": "WOOLWORTHS METRO",
"tax_value": "3.82",
"total_value": "42.08",
"prod_item_value": [
"MILK 2L",
"BREAD MULTIGRAIN",
"EGGS FREE RANGE 12PK"
],
"prod_quantity_value": [
"1",
"2",
"1"
],
"prod_price_value": [
"4.50",
"8.00",
"7.60"
]
}

Important Notes for Australian Receipts:
- For "tax_value" extract the GST (Goods and Services Tax) amount, typically 10% of the pre-tax total
- GST is often shown as a separate line item on Australian receipts
- Dates in Australia typically use the DD/MM/YYYY format
- Prices should include the '$' symbol and two decimal places
- Ensure product items, quantities, and prices are aligned in the same order

Only return the values for the seven keys specified. Do not return any additional key-value pairs.
Do not output any additional information, notes, reasoning, or explanations. Output only the valid JSON.
"""


def main():
    """Main function to run evaluation."""
    # Parse arguments
    args = parse_args()

    # Setup logger with appropriate verbosity
    log_level = "DEBUG" if args.verbose else "INFO"

    # Load configuration
    config = load_config(args)

    # Get transformers log level from config
    transformers_log_level = config.get("transformers_log_level", "WARNING")

    # Setup logging with appropriate levels
    setup_logging(level=log_level, transformers_log_level=transformers_log_level)

    # Initialize path manager
    path_manager = PathManager()

    # Resolve paths
    predictions_dir = (
        Path(args.predictions_dir)
        if args.predictions_dir
        else path_manager.get_output_path("predictions_synthetic")
    )
    ground_truth_dir = (
        Path(args.ground_truth_dir)
        if args.ground_truth_dir
        else path_manager.get_synthetic_ground_truth_path()
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else path_manager.get_output_path("evaluation_results")
    )

    # Parse fields
    fields = args.fields.split(",") if args.fields else []

    # Run evaluation
    try:
        results = run_evaluation(
            predictions_dir=predictions_dir,
            ground_truth_dir=ground_truth_dir,
            output_path=output_path,
            fields=fields,
            normalize=args.normalize,
            show_examples=args.show_examples,
        )

        # Analyze results and provide suggestions
        f1_score = results["overall_metrics"]["F1-score"]
        if f1_score < 0.9:
            print("\nSuggestions to improve extraction accuracy:")
            print("1. Update your model prompt to be specific for Australian receipts:")
            print("   - Specify GST (Goods and Services Tax) instead of generic tax")
            print("   - Specify DD/MM/YYYY date format for Australian receipts")
            print("   - Ask for complete product names with quantities")
            print("\nRecommended prompt template:")
            print("-" * 80)
            print(get_australian_prompt_example())
            print("-" * 80)

            # Save prompt template if requested
            if args.save_prompt_template:
                prompt_file = (
                    path_manager.get_source_path() / "australian_receipt_prompt.txt"
                )
                with prompt_file.open("w") as f:
                    f.write(get_australian_prompt_example())
                print(f"\nAustralian receipt prompt template saved to: {prompt_file}")
            else:
                prompt_file = (
                    path_manager.get_source_path() / "australian_receipt_prompt.txt"
                )
                print(
                    f"\nTip: Run with --save-prompt-template to save this prompt template to {prompt_file}"
                )

        # Print summary for quick reference
        print("\nEvaluation Summary:")
        print(f"Overall F1-score: {results['overall_metrics']['F1-score']:.2%}")

        # Print field type metrics if available
        if "scalar_metrics" in results and "list_metrics" in results:
            print(
                f"Scalar fields F1-score (date, store, tax, total): {results['scalar_metrics'].get('F1-score', 0):.2%}"
            )
            print(
                f"List fields F1-score (products, quantities, prices): {results['list_metrics'].get('F1-score', 0):.2%}"
            )

        print(f"\nDetailed results saved to: {output_path.with_suffix('.csv')}")
        if output_path.with_suffix(".png").exists():
            print(f"Visualization saved to: {output_path.with_suffix('.png')}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
