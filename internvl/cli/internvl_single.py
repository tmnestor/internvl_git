#!/usr/bin/env python3
"""
InternVL Single Document Processing with Automatic Classification.

This script processes a single document image with automatic document type
classification and specialized KEY-VALUE extraction for Australian work-related
expense claims with ATO compliance.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console

# Import classification system
from internvl.classification import (
    ClassificationFailedException,
    DocumentType,
    WorkExpenseDocumentClassifier,
)

# Import from the internvl package
from internvl.config import load_config
from internvl.model import load_model_and_tokenizer
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
    help="Process a single document with automatic classification and specialized KEY-VALUE extraction."
)

# Help text for the new clean CLI
HELP_TEXT = """
InternVL Document Classification & Extraction - Clean Architecture

AUTOMATIC PROCESSING (NEW DEFAULT):
  python -m internvl.cli.internvl_single --image-path document.png
  
  Features:
  ✅ Automatic document type classification
  ✅ Specialized KEY-VALUE extraction per document type
  ✅ ATO compliance assessment built-in
  ✅ Clean single-path processing - no complex options
  ✅ Fail-fast with clear error messages

SUPPORTED DOCUMENT TYPES:
  • Business receipts (Woolworths, Coles, Target, etc.)
  • Tax invoices (GST invoices with ABN)
  • Bank statements (with highlight detection)
  • Fuel receipts (BP, Shell, Caltex, etc.)
  • Meal receipts (restaurants, cafes)
  • Accommodation receipts (hotels, Airbnb)
  • Travel documents (flights, trains, buses)
  • Parking/toll receipts
  • Equipment/supplies receipts
  • Professional services invoices

EXAMPLES:
  # Basic usage - automatic classification and extraction
  python -m internvl.cli.internvl_single --image-path receipt.png
  
  # Save results to file
  python -m internvl.cli.internvl_single --image-path invoice.pdf --output-file result.json
  
  # Verbose processing for debugging
  python -m internvl.cli.internvl_single --image-path statement.png --verbose

ARCHITECTURE PRINCIPLES:
  • No manual document type selection - classification is automatic
  • No extraction method options - KEY-VALUE format only for reliability
  • No fallback mechanisms - fail fast with clear error messages
  • Single processing path per document type for predictable results
"""


def process_single_document(
    image_path: Path,
    model,
    tokenizer,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Clean document processing with mandatory classification.
    
    This function implements the clean architecture from the implementation plan:
    1. Mandatory classification (no manual override)
    2. Single processor per document type
    3. Fail fast on classification uncertainty
    4. Type-specific ATO compliance assessment
    """
    
    try:
        # Step 1: MANDATORY Classification (no manual override)
        rich_config.console.print(f"{rich_config.info_style} Classifying document type...")
        
        classifier = WorkExpenseDocumentClassifier()
        try:
            classification = classifier.classify_document(
                image_path=str(image_path),
                model=model,
                tokenizer=tokenizer
            )
        except ClassificationFailedException as e:
            # FAIL FAST - no fallbacks
            error_msg = f"Document classification failed: {e}"
            rich_config.console.print(f"{rich_config.fail_style} {error_msg}")
            return {
                "success": False,
                "error": "Classification failed",
                "details": str(e),
                "confidence": getattr(e, 'confidence', 0.0),
                "evidence": getattr(e, 'evidence', {})
            }
        
        rich_config.console.print(
            f"{rich_config.success_style} Document classified as: {classification.document_type.value} "
            f"(confidence: {classification.confidence:.2f})"
        )
        rich_config.console.print(f"Reasoning: {classification.classification_reasoning}")
        
        # Step 2: Route to SINGLE processor per type
        processor_result = process_classified_document(
            image_path=image_path,
            classification=classification,
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        # Step 3: Type-specific ATO compliance (built into processors)
        
        # Add classification metadata to result
        processor_result['classification'] = {
            'document_type': classification.document_type.value,
            'confidence': classification.confidence,
            'confidence_level': classification.confidence_level,
            'reasoning': classification.classification_reasoning,
            'processing_prompt': classification.processing_prompt,
            'is_definitive': classification.is_definitive
        }
        
        return processor_result
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Document processing failed: {e}")
        
        rich_config.console.print(f"{rich_config.fail_style} Processing error: {e}")
        return {
            "success": False,
            "error": "Processing failed",
            "details": str(e)
        }


def process_classified_document(
    image_path: Path,
    classification,
    model,
    tokenizer,
    config: Dict[str, Any]  # noqa: ARG001
) -> Dict[str, Any]:
    """Process document using the appropriate processor for its classified type."""
    
    document_type = classification.document_type
    prompt_name = classification.processing_prompt
    
    rich_config.console.print(
        f"{rich_config.info_style} Processing {document_type.value} with specialized extraction..."
    )
    
    # Route to appropriate processor based on document type
    if document_type == DocumentType.BANK_STATEMENT:
        return process_bank_statement(image_path, model, tokenizer, prompt_name)
    else:
        return process_key_value_document(image_path, model, tokenizer, prompt_name, document_type, config)


def process_bank_statement(
    image_path: Path,
    model,
    tokenizer,
    prompt_name: str
) -> Dict[str, Any]:
    """Process bank statement with highlight detection and ATO compliance."""
    
    try:
        from internvl.extraction.bank_statement_parser import (
            extract_bank_statement_with_highlights,
        )
        
        result = extract_bank_statement_with_highlights(
            image_path=str(image_path),
            model=model,
            tokenizer=tokenizer,
            detect_highlights=True,  # Always enabled for bank statements
            prompt_name=prompt_name
        )
        
        # Add processing summary
        if result['success']:
            result['processing_summary'] = {
                'processor_type': 'bank_statement',
                'highlights_detected': result['processing_metadata']['highlights_detected'],
                'total_transactions': result['processing_metadata']['total_transactions'],
                'ato_compliance_score': f"{result['ato_compliance']['overall_compliance']:.0f}%",
                'work_related_transactions': result['processing_metadata'].get('work_related_transactions', 0),
                'highlighted_work_transactions': result['processing_metadata'].get('highlighted_work_transactions', 0),
                'extraction_format': 'KEY-VALUE'
            }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Bank statement processing failed: {e}",
            "processor_type": "bank_statement"
        }


def process_key_value_document(
    image_path: Path,
    model,
    tokenizer,
    prompt_name: str,
    document_type: DocumentType,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process document using specialized KEY-VALUE extraction."""
    
    try:
        # Get the prompt content from config
        prompt = get_document_prompt(prompt_name, config)
        
        # Generate response using InternVL
        from internvl.model.inference import get_raw_prediction
        
        rich_config.console.print(f"{rich_config.info_style} Running specialized {document_type.value} extraction...")
        
        response = get_raw_prediction(
            image_path=str(image_path),
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config={
                "max_new_tokens": 1024,
                "do_sample": False,
                "temperature": 0.1  # Low temperature for consistent extraction
            },
            device="auto"
        )
        
        # Parse KEY-VALUE response
        parsed_result = parse_key_value_response(response, document_type)
        
        # Assess ATO compliance for this document type
        compliance_assessment = assess_document_compliance(parsed_result, document_type)
        
        return {
            "success": True,
            "document_type": document_type.value,
            "extraction_method": "KEY-VALUE",
            "extracted_data": parsed_result,
            "ato_compliance": compliance_assessment,
            "processing_summary": {
                "processor_type": "specialized_key_value",
                "prompt_used": prompt_name,
                "extraction_format": "KEY-VALUE",
                "ato_compliance_score": f"{compliance_assessment.get('overall_score', 0):.0f}%",
                "fields_extracted": len([k for k, v in parsed_result.items() if v]),
                "compliance_ready": compliance_assessment.get('ato_ready', False)
            },
            "raw_response": response[:500] + "..." if len(response) > 500 else response
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"KEY-VALUE extraction failed: {e}",
            "document_type": document_type.value,
            "processor_type": "specialized_key_value"
        }


def get_document_prompt(prompt_name: str, config: Dict[str, Any]) -> str:
    """Get prompt content from configured prompts file or use fallback."""
    
    try:
        import yaml
        
        # Try to load from configured prompts path
        prompts_path = Path(config.get("prompts_path", "./prompts.yaml"))
        
        # If relative path and doesn't exist, try in input/data directory
        if not prompts_path.is_absolute() and not prompts_path.exists():
            data_path = Path(config.get("input_path", "./data"))
            alt_prompts_path = data_path / prompts_path.name
            if alt_prompts_path.exists():
                prompts_path = alt_prompts_path
        
        if prompts_path.exists():
            with prompts_path.open("r") as f:
                prompts = yaml.safe_load(f)
            
            prompt = prompts.get(prompt_name)
            if prompt:
                return prompt
        
        # Fallback to basic KEY-VALUE prompt
        return """<image>
Extract information from this Australian document in KEY-VALUE format.

Use this format:
FIELD_NAME: [value]

Extract all visible information relevant to business expense claims."""
        
    except Exception as e:
        rich_config.console.print(f"{rich_config.warning_style} Error loading prompt: {e}")
        return """<image>
Extract information from this document in KEY-VALUE format."""


def parse_key_value_response(response: str, document_type: DocumentType) -> Dict[str, Any]:  # noqa: ARG001
    """Parse KEY-VALUE response into structured data."""
    
    parsed_data = {}
    
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line and not line.startswith('#'):
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            
            if value and value not in ['', 'N/A', 'Not visible', 'None']:
                parsed_data[key] = value
    
    return parsed_data


def assess_document_compliance(data: Dict[str, Any], document_type: DocumentType) -> Dict[str, Any]:
    """Assess ATO compliance for the extracted data."""
    
    from internvl.classification import get_ato_requirements
    
    requirements = get_ato_requirements(document_type)
    mandatory_fields = requirements.get('mandatory_fields', [])
    
    compliance_score = 0
    total_checks = len(mandatory_fields)
    missing_fields = []
    
    # Check mandatory fields with flexible matching
    for field in mandatory_fields:
        field_variations = [
            field.upper(),
            field.upper().replace('_', ' '),
            field.upper().replace('_', ''),
            field.replace('_', ' ').upper(),
            # Common field mappings
            'SUPPLIER' if field == 'supplier_name' else '',
            'SUPPLIER_ABN' if field == 'supplier_abn' else '',
            'GST' if field == 'gst_amount' else '',
            'TOTAL' if field == 'total_amount' else '',
            'DATE' if field == 'date' else ''
        ]
        
        found = any(
            variation and variation in data.keys() 
            for variation in field_variations
        )
        
        if found:
            compliance_score += 1
        else:
            missing_fields.append(field)
    
    overall_score = (compliance_score / total_checks * 100) if total_checks > 0 else 0
    
    return {
        'overall_score': overall_score,
        'ato_ready': overall_score >= 80,
        'mandatory_fields_found': compliance_score,
        'mandatory_fields_total': total_checks,
        'missing_fields': missing_fields,
        'document_type': document_type.value,
        'gst_applicable': requirements.get('gst_applicable', True),
        'max_claim_without_receipt': requirements.get('max_claim_without_receipt', 300.0)
    }


@app.command()
def main(
    image_path: Optional[str] = typer.Option(
        None, "--image-path", "-i", 
        help="Path to the document image file to process (if not provided, will look for images in INTERNVL_IMAGE_FOLDER_PATH)"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output-file", "-o", 
        help="Path to the output JSON file (default: stdout)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", 
        help="Enable verbose output for debugging"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", 
        help="Device to use: auto (default), cpu, cuda"
    ),
) -> None:
    """
    Process a single document with automatic classification and specialized extraction.
    
    This tool automatically:
    1. Classifies the document type (receipts, invoices, bank statements, etc.)
    2. Applies specialized KEY-VALUE extraction for that document type
    3. Assesses ATO compliance for Australian work expense claims
    
    No manual document type selection needed - classification is automatic and definitive.
    """
    
    try:
        # Configure logging with transformers suppression
        log_level = "DEBUG" if verbose else "INFO"
        
        # Load config to get transformers log level setting
        initial_config = load_config()
        transformers_log_level = initial_config.get("transformers_log_level", "ERROR")
        
        setup_logging(log_level, transformers_log_level=transformers_log_level)
        logger = get_logger(__name__)
        
        # Load config first to get defaults if needed
        if image_path is None:
            config = load_config()
            image_folder_path = config.get('image_folder_path')
            if image_folder_path:
                # Look for the first image in the configured folder
                from internvl.image.loader import get_image_filepaths
                image_paths = get_image_filepaths(Path(image_folder_path))
                if image_paths:
                    image_path = str(image_paths[0])
                    rich_config.console.print(
                        f"{rich_config.info_style} No image path provided, using first image from .env folder: {image_path}"
                    )
                else:
                    rich_config.console.print(
                        f"{rich_config.fail_style} No image path provided and no images found in INTERNVL_IMAGE_FOLDER_PATH: {image_folder_path}"
                    )
                    raise typer.Exit(1)
            else:
                rich_config.console.print(
                    f"{rich_config.fail_style} No image path provided and INTERNVL_IMAGE_FOLDER_PATH not set in .env"
                )
                raise typer.Exit(1)
        
        # Load configuration with resolved image path
        args_namespace = type('Args', (), {
            'image_path': image_path,
            'output_file': output_file,
            'verbose': verbose,
            'device': device
        })()
        
        config = load_config(args_namespace)
        
        # Resolve and validate image path using configured paths
        image_path_obj = Path(image_path)
        
        if not image_path_obj.is_absolute() and not image_path_obj.exists():
            # Try relative to configured input path
            input_path = Path(config.get("input_path", "./data"))
            alt_path = input_path / image_path_obj
            
            if alt_path.exists():
                image_path = str(alt_path)
                logger.info(f"Resolved relative path to input directory: {image_path}")
            else:
                # Try relative to image folder path
                image_folder = Path(config.get("image_folder_path", "./data/images"))
                alt_path = image_folder / image_path_obj
                
                if alt_path.exists():
                    image_path = str(alt_path)
                    logger.info(f"Resolved relative path to image folder: {image_path}")
        
        if not Path(image_path).exists():
            rich_config.console.print(
                f"{rich_config.fail_style} Image file not found: {image_path}"
            )
            raise typer.Exit(1)
        
        rich_config.console.print(
            f"{rich_config.info_style} Processing document: {image_path}"
        )
        
        # Load model with auto-configuration
        rich_config.console.print(
            f"{rich_config.info_style} Loading InternVL model with auto-configuration..."
        )
        
        model_kwargs = {
            "model_path": config["model_path"],
            "auto_device_config": True
        }
        if device and device != "auto":
            model_kwargs["device"] = device
            
        model, tokenizer = load_model_and_tokenizer(**model_kwargs)
        rich_config.console.print(
            f"{rich_config.success_style} Model loaded successfully!"
        )
        
        # Process document with clean pipeline
        result = process_single_document(
            image_path=Path(image_path),
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        # Output results
        if output_file:
            # Handle relative paths by resolving against configured output directory
            output_path = Path(output_file)
            if not output_path.is_absolute():
                configured_output_dir = Path(config.get("output_path", "."))
                output_path = configured_output_dir / output_path
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with output_path.open("w") as f:
                json.dump(result, f, indent=2)
            rich_config.console.print(
                f"{rich_config.success_style} Result saved to {output_path}"
            )
        else:
            # Print result to stdout
            rich_config.console.print("\n[bold]Result:[/bold]")
            rich_config.console.print(json.dumps(result, indent=2))
            
            # Show processing summary if available
            if 'processing_summary' in result:
                rich_config.console.print("\n[bold]Processing Summary:[/bold]")
                for key, value in result['processing_summary'].items():
                    rich_config.console.print(f"  {key}: {value}")
        
        # Final status
        if result.get('success', False):
            rich_config.console.print(
                f"{rich_config.success_style} Document processing completed successfully!"
            )
        else:
            rich_config.console.print(
                f"{rich_config.fail_style} Document processing failed: {result.get('error', 'Unknown error')}"
            )
            raise typer.Exit(1)
        
    except Exception as e:
        rich_config.console.print(f"{rich_config.fail_style} Error: {e}")
        if verbose:
            import traceback
            rich_config.console.print(traceback.format_exc())
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()