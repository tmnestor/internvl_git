#!/usr/bin/env python3
"""
InternVL Batch Image Information Extraction

This script processes multiple images in parallel with InternVL and extracts structured information.
"""

import concurrent.futures as cf
import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
import yaml
from rich.console import Console
from rich.progress import Progress

# Import from the internvl package
from internvl.config import load_config
from internvl.extraction.normalization import post_process_prediction
from internvl.image.loader import get_image_filepaths
from internvl.model import load_model_and_tokenizer
from internvl.model.inference import get_raw_prediction
from internvl.utils.logging import get_logger, setup_logging


@dataclass
class RichConfig:
    """Configuration for rich console output."""

    console: Console = Console()
    success_style: str = "[bold green]✅[/bold green]"
    fail_style: str = "[bold red]❌[/bold red]"
    warning_style: str = "[bold yellow]⚠[/bold yellow]"
    info_style: str = "[bold blue]ℹ[/bold blue]"


rich_config = RichConfig()
app = typer.Typer(
    help="Process multiple images in parallel with InternVL for information extraction."
)


def get_extraction_config_batch(args, _config: Dict[str, Any]) -> Dict[str, Any]:
    """Get extraction configuration based on CLI arguments for batch processing."""
    extraction_config = {}
    
    if args.extraction_method == "key_value":
        # NEW DEFAULT: Key-Value extraction
        extraction_config.update({
            'prompt_name': 'key_value_receipt_prompt',
            'extraction_method': 'key_value',
            'processor_module': 'internvl.extraction.key_value_parser',
            'processor_function': 'extract_work_related_expense',
            'supports_compliance': True,
            'expense_category': args.expense_category
        })
    elif args.extraction_method == "json":
        # LEGACY: JSON extraction
        extraction_config.update({
            'prompt_name': 'default_receipt_prompt',
            'extraction_method': 'json',
            'processor_module': 'internvl.extraction.json_extraction_fixed',
            'processor_function': 'extract_json_from_text',
            'supports_compliance': False
        })
    
    # Compliance check only available for key_value
    extraction_config['compliance_check'] = args.compliance_check and args.extraction_method == "key_value"
    
    return extraction_config


def get_prompt_from_config_batch(extraction_config: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Get prompt from configuration for batch processing."""
    try:
        prompts_path = config.get("prompts_path")
        prompt_name = extraction_config.get("prompt_name")

        if prompts_path and Path(prompts_path).exists():
            with Path(prompts_path).open("r") as f:
                prompts = yaml.safe_load(f)
            prompt = prompts.get(prompt_name, "")
            if prompt:
                return prompt
        
        # Fallback prompts based on extraction method
        if extraction_config['extraction_method'] == 'key_value':
            return """<image>
Extract information from this Australian receipt using this EXACT format:

DATE: [purchase date in DD/MM/YYYY format]
STORE: [store/supplier name]
ABN: [Australian Business Number if visible]
TAX: [GST/tax amount]
TOTAL: [total amount including GST]
PRODUCTS: [item1 | item2 | item3]
QUANTITIES: [qty1 | qty2 | qty3]
PRICES: [price1 | price2 | price3]

Use | to separate multiple items in lists.
Return empty if field not found."""
        else:
            return "<image>\nExtract information from this receipt and return it in JSON format."
            
    except Exception:
        return "<image>\nExtract information from this receipt and return it in JSON format."


def process_key_value_extraction_batch(response: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process using Enhanced Key-Value Parser for batch processing."""
    try:
        from internvl.extraction.key_value_parser import (
            extract_key_value_enhanced,
            extract_work_related_expense,
        )
        
        if config.get('compliance_check', False):
            # Full ATO compliance assessment
            result = extract_work_related_expense(response, config.get('expense_category', 'General'))
        else:
            # Basic key-value extraction
            result = extract_key_value_enhanced(response)
        
        # Convert KeyValueExtractionResult to JSON-serializable format
        if result['success'] and result.get('extraction_result'):
            extraction_result = result['extraction_result']
            
            # Convert to serializable format for batch processing
            serializable_result = {
                'success': result['success'],
                'summary': result['summary'],
                'extracted_data': {
                    'supplier_name': extraction_result.extracted_fields.get('STORE', ''),
                    'supplier_abn': extraction_result.extracted_fields.get('ABN', ''),
                    'invoice_date': extraction_result.extracted_fields.get('DATE', ''),
                    'total_amount': extraction_result.extracted_fields.get('TOTAL', ''),
                    'gst_amount': extraction_result.extracted_fields.get('TAX', ''),
                    'payer_name': extraction_result.extracted_fields.get('PAYER', ''),
                    'items': extraction_result.parsed_lists.get('PRODUCTS', []),
                    'quantities': extraction_result.parsed_lists.get('QUANTITIES', []),
                    'item_prices': extraction_result.parsed_lists.get('PRICES', [])
                },
                'extraction_metadata': {
                    'confidence_score': extraction_result.confidence_score,
                    'validation_errors': extraction_result.validation_errors,
                    'field_completeness': extraction_result.field_completeness,
                    'raw_text': extraction_result.raw_text
                }
            }
            
            # Add assessment if compliance check was performed
            if config.get('compliance_check', False) and 'assessment' in result:
                serializable_result['assessment'] = result['assessment']
            
            return serializable_result
        else:
            # Handle error case
            return {
                'success': result['success'],
                'error': result.get('error', 'Unknown error'),
                'extraction_method': 'key_value'
            }
        
    except Exception as e:
        return {
            'success': False,
            'extraction_method': 'key_value',
            'error': str(e)
        }


def save_batch_results(results: List[Dict], output_file: Path, config: Dict[str, Any]):
    """Save batch results with extraction method specific columns."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if config['extraction_method'] == 'key_value':
        # Enhanced columns for key-value results
        columns = [
            'image_name', 'success', 'confidence_score', 'quality_grade',
            'supplier_name', 'supplier_abn', 'invoice_date', 'total_amount', 'gst_amount',
            'items_count', 'extraction_method'
        ]
        
        if config.get('compliance_check'):
            columns.extend(['ato_compliance_score', 'ato_ready', 'compliance_issues'])
            
    elif config['extraction_method'] == 'json':
        # Legacy columns for JSON results
        columns = [
            'image_name', 'success', 'store_name_value', 'date_value', 'total_value', 'tax_value',
            'extraction_method', 'notes'
        ]
    
    # Write CSV with appropriate columns
    with output_file.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            row = format_result_for_csv(result, config)
            writer.writerow(row)


def format_result_for_csv(result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    """Format a single result for CSV output based on extraction method."""
    
    base_row = {
        'image_name': result.get('image_name', ''),
        'success': result.get('success', False),
        'extraction_method': config['extraction_method']
    }
    
    if not result.get('success', False):
        # Handle errors
        base_row.update({col: '' for col in ['confidence_score', 'quality_grade', 'supplier_name'] 
                        if col not in base_row})
        return base_row
    
    if config['extraction_method'] == 'key_value':
        # Key-value specific formatting
        extracted = result.get('extracted_data', {})
        summary = result.get('summary', {})
        
        row = {
            **base_row,
            'confidence_score': summary.get('extraction_quality', {}).get('confidence_score', ''),
            'quality_grade': summary.get('validation_status', {}).get('quality_grade', ''),
            'supplier_name': extracted.get('supplier_name', ''),
            'supplier_abn': extracted.get('supplier_abn', ''),
            'invoice_date': extracted.get('invoice_date', ''),
            'total_amount': extracted.get('total_amount', ''),
            'gst_amount': extracted.get('gst_amount', ''),
            'items_count': len(extracted.get('items', []) if extracted.get('items') else [])
        }
        
        if config.get('compliance_check') and 'assessment' in result:
            assessment = result['assessment']
            row.update({
                'ato_compliance_score': f"{assessment.get('compliance_score', 0):.0f}%",
                'ato_ready': assessment.get('ato_ready', False),
                'compliance_issues': '; '.join(assessment.get('issues', []))
            })
        
        return row
        
    elif config['extraction_method'] == 'json':
        # Legacy JSON formatting
        extracted = result.get('extracted_data', {})
        
        return {
            **base_row,
            'store_name_value': extracted.get('store_name_value', ''),
            'date_value': extracted.get('date_value', ''),
            'total_value': extracted.get('total_value', ''),
            'tax_value': extracted.get('tax_value', ''),
            'notes': 'Legacy JSON extraction'
        }
    
    return base_row


def process_image(
    image_path: str,
    model,
    tokenizer,
    prompt: str,
    generation_config: Dict[str, Any],
    extraction_config: Dict[str, Any],
    device: str = "auto",
) -> Dict[str, Any]:
    """Process a single image and return extracted information."""
    start_time = time.time()
    image_id = Path(image_path).stem

    try:
        # Get raw prediction
        raw_output = get_raw_prediction(
            image_path=image_path,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=generation_config,
            device=device,
        )

        # Process based on extraction method
        if extraction_config['extraction_method'] == 'key_value':
            processed_result = process_key_value_extraction_batch(raw_output, extraction_config)
        else:
            # Legacy JSON processing
            processed_json = post_process_prediction(raw_output)
            processed_result = {
                'success': True,
                'extraction_method': 'json',
                'extracted_data': processed_json
            }

        # Add metadata
        result = {
            "image_id": image_id,
            "image_name": Path(image_path).name,
            "processing_time": time.time() - start_time,
            **processed_result
        }

        return result

    except Exception as e:
        return {
            "image_id": image_id,
            "error": str(e),
            "processing_time": time.time() - start_time,
        }


def process_images_in_batch(
    image_paths: List[str],
    model,
    tokenizer,
    prompt: str,
    generation_config: Dict[str, Any],
    extraction_config: Dict[str, Any],
    device: str = "auto",
    max_workers: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Process multiple images in parallel."""
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_image,
                image_path=image_path,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                generation_config=generation_config,
                extraction_config=extraction_config,
                device=device,
            ): image_path
            for image_path in image_paths
        }

        # Process results as they complete with progress bar
        with Progress() as progress:
            task = progress.add_task("Processing images...", total=len(image_paths))

            for future in cf.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)

                    # Log progress
                    img_path = futures[future]
                    rich_config.console.print(
                        f"{rich_config.success_style} Processed: {Path(img_path).name}"
                    )
                    progress.advance(task)
                except Exception as e:
                    rich_config.console.print(
                        f"{rich_config.fail_style} Error processing image: {e}"
                    )
                    progress.advance(task)

    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time
    processing_times = [r.get("processing_time", 0) for r in results]
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

    stats = {
        "total_time": total_time,
        "avg_processing_time": avg_time,
        "num_images": len(results),
        "num_errors": sum(1 for r in results if "error" in r),
    }

    return results, stats


@app.command()
def main(
    image_folder_path: Optional[str] = typer.Option(
        None,
        "--image-folder-path",
        "-i",
        help="Path to the folder containing images to process (default: from .env INTERNVL_IMAGE_FOLDER_PATH)",
    ),
    output_file: Optional[str] = typer.Option(
        "batch_results.csv", "--output-file", "-o", help="Path to the output CSV file (relative to configured output directory)"
    ),
    document_type: str = typer.Option(
        "receipt", "--document-type",
        help="Document type: receipt (default), bank_statement"
    ),
    extraction_method: str = typer.Option(
        "key_value", "--extraction-method", 
        help="Extraction method: key_value (default, robust) or json (legacy)"
    ),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", help="Number of parallel workers (default: from config)"
    ),
    compliance_check: bool = typer.Option(
        False, "--compliance-check", 
        help="Perform ATO compliance validation (key_value only)"
    ),
    detect_highlights: bool = typer.Option(
        True, "--detect-highlights",
        help="Detect highlighted regions in bank statements"
    ),
    expense_category: str = typer.Option(
        "General", "--expense-category",
        help="ATO expense category for compliance checking"
    ),
    num_images: Optional[int] = typer.Option(
        None, "--num-images", "-n", help="Number of images to process (default: all)"
    ),
    save_individual: bool = typer.Option(
        False,
        "--save-individual",
        "-s",
        help="Save individual JSON files for each image",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """
    Process multiple images in parallel with InternVL and extract structured information.
    """
    # Validate document type
    if document_type not in ["receipt", "bank_statement"]:
        rich_config.console.print(
            f"{rich_config.fail_style} Invalid document type: {document_type}. Must be 'receipt' or 'bank_statement'"
        )
        raise typer.Exit(1)
    
    # Validate extraction method
    if extraction_method not in ["key_value", "json"]:
        rich_config.console.print(
            f"{rich_config.fail_style} Invalid extraction method: {extraction_method}. Must be 'key_value' or 'json'"
        )
        raise typer.Exit(1)
    
    # Compliance check validation
    if compliance_check and extraction_method != "key_value" and document_type != "bank_statement":
        rich_config.console.print(
            f"{rich_config.warning_style} Compliance check only available with --extraction-method key_value or --document-type bank_statement"
        )
        compliance_check = False
        
    # Highlight detection validation
    if detect_highlights and document_type != "bank_statement":
        rich_config.console.print(
            f"{rich_config.warning_style} Highlight detection only available for bank statements"
        )
        detect_highlights = False
    
    rich_config.console.print(
        f"{rich_config.info_style} Starting batch image processing with {extraction_method} extraction..."
    )
    rich_config.console.print(f"Image folder: {image_folder_path}")
    rich_config.console.print(f"Output file: {output_file}")
    rich_config.console.print(f"Extraction method: {extraction_method}")
    rich_config.console.print(f"Compliance check: {compliance_check}")
    rich_config.console.print(f"Save individual: {save_individual}")
    rich_config.console.print(f"Verbose: {verbose}")

    try:
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO

        # Create mock args for config loading (temporary compatibility)
        class MockArgs:
            def __init__(self):
                self.image_folder_path = image_folder_path
                self.num_images = num_images
                self.output_file = output_file
                self.save_individual = save_individual
                self.verbose = verbose
                self.extraction_method = extraction_method
                self.compliance_check = compliance_check
                self.expense_category = expense_category
                self.max_workers = max_workers

        # Load config first to get defaults
        config = load_config()
        
        # Use default from config if image_folder_path not provided
        if image_folder_path is None:
            image_folder_path = config.get('image_folder_path')
            if image_folder_path is None:
                rich_config.console.print(
                    f"{rich_config.fail_style} No image folder path provided and INTERNVL_IMAGE_FOLDER_PATH not set in .env"
                )
                raise typer.Exit(1)
            rich_config.console.print(
                f"{rich_config.info_style} Using image folder from .env: {image_folder_path}"
            )

        args = MockArgs()
        config = load_config(args)
        
        # Get extraction configuration
        extraction_config = get_extraction_config_batch(args, config)
        transformers_log_level = config.get("transformers_log_level", "ERROR")

        setup_logging(log_level, transformers_log_level=transformers_log_level)
        get_logger(__name__)

        # Get image folder path
        image_folder_obj = Path(image_folder_path)
        if not image_folder_obj.exists():
            rich_config.console.print(
                f"{rich_config.fail_style} Image folder not found: {image_folder_path}"
            )
            raise typer.Exit(1)

        # Get image paths
        image_paths = get_image_filepaths(image_folder_obj)
        if not image_paths:
            rich_config.console.print(
                f"{rich_config.fail_style} No images found in {image_folder_path}"
            )
            raise typer.Exit(1)

        # Limit number of images if specified
        if num_images is not None and num_images > 0:
            image_paths = image_paths[:num_images]

        # Determine worker count
        worker_count = max_workers or config.get('max_workers', 6)
        
        rich_config.console.print(
            f"{rich_config.info_style} Processing {len(image_paths)} images with {worker_count} workers"
        )

        # Get the prompt from extraction config
        prompt = get_prompt_from_config_batch(extraction_config, config)
        rich_config.console.print(
            f"{rich_config.info_style} Using prompt '{extraction_config['prompt_name']}'"
        )

        # Load model and tokenizer with auto-configuration
        rich_config.console.print(
            f"{rich_config.info_style} Loading model with auto-configuration..."
        )
        model, tokenizer = load_model_and_tokenizer(
            model_path=config["model_path"], auto_device_config=True
        )
        rich_config.console.print(
            f"{rich_config.success_style} Model loaded successfully!"
        )

        # Process images with auto device detection
        generation_config = {
            "max_new_tokens": config.get("max_tokens", 1024),
            "do_sample": config.get("do_sample", False),
        }

        results, stats = process_images_in_batch(
            image_paths=image_paths,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=generation_config,
            extraction_config=extraction_config,
            device="auto",
            max_workers=worker_count,
        )

        # Save results with method-specific formatting
        if output_file:
            # Handle relative paths by resolving against configured output directory
            output_path = Path(output_file)
            if not output_path.is_absolute():
                configured_output_dir = Path(config.get("output_path", "."))
                output_path = configured_output_dir / output_path
            
            save_batch_results(results, output_path, extraction_config)
            rich_config.console.print(
                f"{rich_config.success_style} Results saved to {output_path}"
            )

        # Save individual JSON files if requested
        if save_individual:
            # Use configured output directory for individual files
            configured_output_dir = Path(config.get("output_path", "."))
            output_dir = configured_output_dir / "predictions"
            output_dir.mkdir(parents=True, exist_ok=True)

            for result in results:
                if result.get('success', False):
                    output_json_file = output_dir / f"{result['image_id']}.json"
                    with output_json_file.open("w") as f:
                        # Save the full result for individual files
                        json.dump(result, f, indent=2)

            rich_config.console.print(
                f"{rich_config.success_style} Individual JSON files saved to {output_dir}"
            )

        # Print statistics
        rich_config.console.print("\n[bold]Processing Statistics:[/bold]")
        rich_config.console.print(f"Total time: {stats['total_time']:.2f}s")
        rich_config.console.print(
            f"Average time per image: {stats['avg_processing_time']:.2f}s"
        )
        rich_config.console.print(f"Images processed: {stats['num_images']}")
        rich_config.console.print(f"Errors: {stats['num_errors']}")

        rich_config.console.print(
            f"{rich_config.success_style} Batch processing completed successfully!"
        )

    except Exception as e:
        rich_config.console.print(f"{rich_config.fail_style} Error: {e}")
        if verbose:
            import traceback

            rich_config.console.print(traceback.format_exc())
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
