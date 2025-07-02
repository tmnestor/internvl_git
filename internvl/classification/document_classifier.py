"""
Document Classification for Australian Work Expense Documents.

This module provides document classification capabilities for the InternVL PoC,
automatically categorizing documents for optimal extraction prompt selection
and ATO compliance assessment.
"""

import logging
import re
from pathlib import Path
from typing import Dict

import yaml

from .confidence_scorer import (
    ClassificationFailedException,
    ClassificationResult,
    ConfidenceLevel,
    ConfidenceScorer,
    create_classification_result,
)
from .document_types import (
    DocumentType,
    DocumentTypeMetadata,
    get_key_value_prompt_name,
)

logger = logging.getLogger(__name__)


class WorkExpenseDocumentClassifier:
    """
    Classifies Australian work-related expense documents for optimal processing.
    
    This classifier uses InternVL to analyze document images and determine
    the most appropriate document type for specialized extraction processing.
    """
    
    def __init__(self):
        """Initialize the document classifier."""
        self.confidence_scorer = ConfidenceScorer()
        self.classification_prompt = self._load_classification_prompt()
        
    def classify_document(
        self, 
        image_path: str, 
        model, 
        tokenizer
    ) -> ClassificationResult:
        """
        Classify document - MUST return a definitive classification or raise error.
        
        Args:
            image_path: Path to document image
            model: InternVL model instance
            tokenizer: Model tokenizer
            
        Returns:
            ClassificationResult with definitive classification
            
        Raises:
            ClassificationFailedException: If classification is uncertain or fails
        """
        
        try:
            logger.info(f"Classifying document: {image_path}")
            
            # Step 1: Get classification response from model
            classification_response = self._get_classification_response(
                image_path, model, tokenizer
            )
            
            logger.debug(f"Classification response: {classification_response[:200]}...")
            
            # Step 2: Parse classification response
            parsed_result = self._parse_classification_response(classification_response)
            
            # Step 3: Validate confidence and create result
            if parsed_result['confidence'] >= ConfidenceLevel.HIGH:
                # High confidence - proceed with classification
                result = create_classification_result(
                    document_type=parsed_result['document_type'],
                    confidence=parsed_result['confidence'],
                    reasoning=parsed_result['reasoning'],
                    prompt_name=get_key_value_prompt_name(parsed_result['document_type']),
                    secondary_type=parsed_result.get('secondary_type'),
                    evidence=parsed_result.get('evidence', {})
                )
                
                logger.info(f"Document classified as {result.document_type.value} "
                           f"with {result.confidence:.2f} confidence")
                return result
                
            else:
                # Low confidence - fail fast
                error_msg = (
                    f"Classification confidence too low: {parsed_result['confidence']:.2f} "
                    f"(minimum required: {ConfidenceLevel.HIGH}). "
                    f"Primary type: {parsed_result['document_type'].value}, "
                    f"Reasoning: {parsed_result['reasoning']}"
                )
                
                logger.error(error_msg)
                raise ClassificationFailedException(
                    message=error_msg,
                    confidence=parsed_result['confidence'],
                    evidence=parsed_result.get('evidence', {})
                )
                
        except ClassificationFailedException:
            # Re-raise classification failures
            raise
            
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            raise ClassificationFailedException(
                message=f"Classification error: {str(e)}",
                confidence=0.0,
                evidence={'error': str(e)}
            ) from e
    
    def _get_classification_response(self, image_path: str, model, tokenizer) -> str:
        """Get classification response from InternVL model."""
        
        try:
            from internvl.model.inference import get_raw_prediction
            
            response = get_raw_prediction(
                image_path=image_path,
                model=model,
                tokenizer=tokenizer,
                prompt=self.classification_prompt,
                generation_config={
                    "max_new_tokens": 512,  # Classification shouldn't need many tokens
                    "do_sample": False,
                    "temperature": 0.1      # Low temperature for consistent classification
                },
                device="auto"
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to get classification response: {e}")
            raise ClassificationFailedException(f"Model inference failed: {e}") from e
    
    def _parse_classification_response(self, response: str) -> Dict:
        """Parse classification response into structured result."""
        
        try:
            result = {
                'document_type': DocumentType.OTHER,
                'confidence': 0.0,
                'reasoning': '',
                'secondary_type': None,
                'evidence': {}
            }
            
            # Parse structured response
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('DOCUMENT_TYPE:'):
                    type_str = line.replace('DOCUMENT_TYPE:', '').strip()
                    result['document_type'] = self._parse_document_type(type_str)
                    
                elif line.startswith('CONFIDENCE:'):
                    confidence_str = line.replace('CONFIDENCE:', '').strip()
                    result['confidence'] = self._parse_confidence(confidence_str)
                    
                elif line.startswith('REASONING:'):
                    result['reasoning'] = line.replace('REASONING:', '').strip()
                    
                elif line.startswith('SECONDARY_TYPE:'):
                    secondary_str = line.replace('SECONDARY_TYPE:', '').strip()
                    if secondary_str and secondary_str.lower() != 'none':
                        result['secondary_type'] = self._parse_document_type(secondary_str)
            
            # Score confidence using evidence from response
            confidence_score, evidence = self.confidence_scorer.score_classification(
                response_text=response,
                candidate_type=result['document_type'],
                secondary_candidates=[result['secondary_type']] if result['secondary_type'] else None
            )
            
            # Use the higher of parsed confidence or scored confidence
            result['confidence'] = max(result['confidence'], confidence_score)
            result['evidence'] = evidence
            
            # Validate we have minimum required information
            if not result['reasoning']:
                result['reasoning'] = f"Classified as {result['document_type'].value} based on document analysis"
            
            logger.debug(f"Parsed classification: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse classification response: {e}")
            raise ClassificationFailedException(f"Response parsing failed: {e}") from e
    
    def _parse_document_type(self, type_str: str) -> DocumentType:
        """Parse document type from string."""
        
        type_str = type_str.lower().strip()
        
        # Direct enum mapping
        for doc_type in DocumentType:
            if doc_type.value == type_str:
                return doc_type
        
        # Fuzzy matching for common variations
        type_mappings = {
            'receipt': DocumentType.BUSINESS_RECEIPT,
            'business receipt': DocumentType.BUSINESS_RECEIPT,
            'invoice': DocumentType.TAX_INVOICE,
            'tax invoice': DocumentType.TAX_INVOICE,
            'statement': DocumentType.BANK_STATEMENT,
            'bank statement': DocumentType.BANK_STATEMENT,
            'fuel': DocumentType.FUEL_RECEIPT,
            'petrol': DocumentType.FUEL_RECEIPT,
            'meal': DocumentType.MEAL_RECEIPT,
            'restaurant': DocumentType.MEAL_RECEIPT,
            'hotel': DocumentType.ACCOMMODATION,
            'accommodation': DocumentType.ACCOMMODATION,
            'travel': DocumentType.TRAVEL_DOCUMENT,
            'parking': DocumentType.PARKING_TOLL,
            'equipment': DocumentType.EQUIPMENT_SUPPLIES,
            'professional': DocumentType.PROFESSIONAL_SERVICES
        }
        
        for key, doc_type in type_mappings.items():
            if key in type_str:
                return doc_type
        
        logger.warning(f"Unknown document type: {type_str}, defaulting to OTHER")
        return DocumentType.OTHER
    
    def _parse_confidence(self, confidence_str: str) -> float:
        """Parse confidence from string."""
        
        confidence_str = confidence_str.lower().strip()
        
        # Direct confidence mappings
        confidence_mappings = {
            'high': 0.9,
            'medium': 0.6,
            'low': 0.3,
            'none': 0.0
        }
        
        if confidence_str in confidence_mappings:
            return confidence_mappings[confidence_str]
        
        # Try to parse as percentage
        percentage_match = re.search(r'(\d+)%', confidence_str)
        if percentage_match:
            return float(percentage_match.group(1)) / 100.0
        
        # Try to parse as decimal
        decimal_match = re.search(r'(\d*\.?\d+)', confidence_str)
        if decimal_match:
            value = float(decimal_match.group(1))
            # If value > 1, assume it's a percentage
            if value > 1:
                value = value / 100.0
            return min(max(value, 0.0), 1.0)
        
        logger.warning(f"Could not parse confidence: {confidence_str}")
        return 0.0
    
    def _load_classification_prompt(self) -> str:
        """Load classification prompt from prompts.yaml or use fallback."""
        
        try:
            # Try to load from prompts.yaml
            prompts_path = Path("prompts.yaml")
            if not prompts_path.exists():
                prompts_path = Path.cwd() / "prompts.yaml"
            
            if prompts_path.exists():
                with prompts_path.open("r") as f:
                    prompts = yaml.safe_load(f)
                
                # Look for classification prompt
                prompt = prompts.get("document_classification_prompt")
                if prompt:
                    logger.info("Loaded document classification prompt from prompts.yaml")
                    return prompt
            
            logger.warning("Classification prompt not found in prompts.yaml, using fallback")
            
        except Exception as e:
            logger.error(f"Error loading classification prompt: {e}")
        
        # Fallback classification prompt
        return self._get_fallback_classification_prompt()
    
    def _get_fallback_classification_prompt(self) -> str:
        """Get fallback classification prompt if YAML loading fails."""
        
        # Generate document type descriptions
        type_descriptions = []
        for i, (doc_type, description) in enumerate(DocumentTypeMetadata.DESCRIPTIONS.items(), 1):
            type_descriptions.append(f"{i}. {doc_type.value} - {description}")
        
        type_list = "\n".join(type_descriptions)
        
        return f"""<image>
Analyze this Australian work-related expense document and classify its type.

DOCUMENT_TYPES:
{type_list}

CLASSIFICATION_CRITERIA:
- Look for business names, logos, and document layout
- Identify specific industry indicators (fuel company logos, hotel chains, etc.)
- Check for formal invoice elements (ABN, tax invoice headers)
- Consider document structure and typical content

RESPONSE_FORMAT:
DOCUMENT_TYPE: [type from list above]
CONFIDENCE: [High/Medium/Low]
REASONING: [Brief explanation of classification decision]
SECONDARY_TYPE: [Alternative type if confidence is not High]

Focus on Australian businesses and document formats. Be definitive in your classification."""


def classify_document_type(
    image_path: str, 
    model, 
    tokenizer
) -> ClassificationResult:
    """
    Convenience function for document classification.
    
    Args:
        image_path: Path to document image
        model: InternVL model instance
        tokenizer: Model tokenizer
        
    Returns:
        ClassificationResult with definitive classification
        
    Raises:
        ClassificationFailedException: If classification fails or is uncertain
    """
    classifier = WorkExpenseDocumentClassifier()
    return classifier.classify_document(image_path, model, tokenizer)