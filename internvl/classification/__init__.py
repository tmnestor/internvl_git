"""
Document Classification Module for Australian Work Expense Documents.

This module provides automatic document classification capabilities for
the InternVL PoC, enabling specialized processing based on document type
with ATO compliance and KEY-VALUE extraction.
"""

from .confidence_scorer import (
    ClassificationFailedException,
    ClassificationResult,
    ConfidenceLevel,
    ConfidenceScorer,
    create_classification_result,
)
from .document_classifier import WorkExpenseDocumentClassifier, classify_document_type
from .document_types import (
    DocumentType,
    DocumentTypeMetadata,
    get_ato_requirements,
    get_classification_keywords,
    get_document_type_by_name,
    get_key_value_prompt_name,
    get_max_claim_without_receipt,
    is_gst_applicable,
)

__all__ = [
    # Core classification
    'WorkExpenseDocumentClassifier',
    'classify_document_type',
    
    # Document types
    'DocumentType',
    'DocumentTypeMetadata',
    'get_ato_requirements',
    'get_classification_keywords', 
    'get_document_type_by_name',
    'get_key_value_prompt_name',
    'get_max_claim_without_receipt',
    'is_gst_applicable',
    
    # Confidence and results
    'ClassificationFailedException',
    'ClassificationResult',
    'ConfidenceLevel',
    'ConfidenceScorer',
    'create_classification_result'
]