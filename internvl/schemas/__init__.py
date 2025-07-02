"""
Schema definitions for InternVL data models.

This module provides Pydantic models for structured data validation
across different document types processed by InternVL.
"""

from .bank_statement_schemas import (
    AustralianBankInfo,
    BankStatementExtraction,
    BankTransaction,
    HighlightRegion,
)

__all__ = [
    'BankTransaction',
    'BankStatementExtraction', 
    'HighlightRegion',
    'AustralianBankInfo'
]
