"""
Bank Statement Schema Definitions for Australian ATO Compliance.

This module provides dataclass models for validating and structuring
bank statement data extracted from images, with specific support for
Australian Tax Office work-related expense claim requirements.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class HighlightRegion(BaseModel):
    """Detected highlight region with metadata."""
    
    x: int = Field(..., description="X coordinate of highlight region")
    y: int = Field(..., description="Y coordinate of highlight region") 
    width: int = Field(..., description="Width of highlight region in pixels")
    height: int = Field(..., description="Height of highlight region in pixels")
    color: str = Field(..., description="Detected highlight color (yellow/pink/green/red)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
    extracted_text: str = Field(default="", description="OCR-extracted text from highlighted region")
    
    def area(self) -> int:
        """Calculate area of highlight region."""
        return self.width * self.height
    
    def center(self) -> tuple[int, int]:
        """Get center coordinates of highlight region."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def overlaps_with(self, other: 'HighlightRegion', threshold: float = 0.3) -> bool:
        """Check if this region overlaps significantly with another."""
        # Calculate intersection
        x_overlap = max(0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x))
        y_overlap = max(0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y))
        intersection = x_overlap * y_overlap
        
        # Calculate areas
        area1 = self.area()
        area2 = other.area()
        
        # Check if intersection is significant
        overlap_ratio = intersection / min(area1, area2) if min(area1, area2) > 0 else 0
        return overlap_ratio > threshold


class AustralianBankInfo(BaseModel):
    """Australian banking institution information."""
    
    bank_code: str = Field(..., description="Bank identifier code (cba/anz/westpac/nab/etc)")
    full_name: str = Field(..., description="Full legal name of bank")
    bsb_ranges: List[str] = Field(..., description="BSB number ranges for this bank") 
    account_format: str = Field(..., description="Expected account number format")
    logo_identifiers: List[str] = Field(..., description="Visual identifiers in statements")


class BankTransaction(BaseModel):
    """Individual bank transaction for work expense analysis."""
    
    transaction_date: str = Field(..., description="Transaction date in DD/MM/YYYY format")
    description: str = Field(..., description="Transaction description/merchant name")
    debit_amount: Optional[str] = Field(None, description="Amount debited (withdrawn)")
    credit_amount: Optional[str] = Field(None, description="Amount credited (deposited)")
    balance: Optional[str] = Field(None, description="Account balance after transaction")
    reference: Optional[str] = Field(None, description="Transaction reference/ID")
    merchant_category: Optional[str] = Field(None, description="Business category if identifiable")
    
    # Work expense relevance fields
    work_related_likelihood: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="AI-assessed work relevance (0-1)"
    )
    expense_category: Optional[str] = Field(None, description="ATO expense category")
    highlight_detected: bool = Field(False, description="Was this transaction highlighted by user")
    highlight_region: Optional[HighlightRegion] = Field(None, description="Associated highlight region")
    
    @validator('transaction_date')
    def validate_australian_date(cls, v):
        """Validate Australian DD/MM/YYYY date format."""
        if not v:
            return v
        if not re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', v):
            raise ValueError('Date must be in DD/MM/YYYY format')
        
        # Additional validation: ensure date is parseable
        try:
            day, month, year = v.split('/')
            datetime(int(year), int(month), int(day))
        except ValueError as e:
            raise ValueError(f'Invalid date: {v}') from e
        
        return v
    
    @validator('debit_amount', 'credit_amount', 'balance')
    def validate_monetary_amounts(cls, v):
        """Validate monetary amounts are properly formatted."""
        if not v:
            return v
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[^\d.,\-]', '', str(v).strip())
        
        # Validate format (allow negative for debits)
        if not re.match(r'^-?\d+(\.\d{2})?$', cleaned):
            raise ValueError(f'Invalid monetary amount format: {v}')
        
        return cleaned


class BankStatementExtraction(BaseModel):
    """Complete bank statement extraction for ATO compliance."""
    
    # Statement metadata
    bank_name: str = Field(..., description="Name of financial institution")
    account_holder: str = Field(..., description="Account holder name")
    account_number: str = Field(..., description="Masked account number")
    bsb: Optional[str] = Field(None, description="Bank State Branch code")
    statement_period: str = Field(..., description="Statement period (DD/MM/YYYY - DD/MM/YYYY)")
    
    # Financial summary
    opening_balance: str = Field(..., description="Opening balance")
    closing_balance: str = Field(..., description="Closing balance")
    total_debits: Optional[str] = Field(None, description="Total withdrawals")
    total_credits: Optional[str] = Field(None, description="Total deposits")
    
    # Transactions
    transactions: List[BankTransaction] = Field(..., description="All transactions in statement")
    highlighted_transactions: List[BankTransaction] = Field(
        default=[], description="User-highlighted transactions"
    )
    
    # Work expense analysis
    work_expense_summary: Dict[str, Any] = Field(
        default={}, description="Summary of work-related expenses"
    )
    ato_compliance_assessment: Dict[str, Any] = Field(
        default={}, description="ATO compliance evaluation"
    )
    
    # Processing metadata
    highlight_detection_enabled: bool = Field(
        default=False, description="Whether highlight detection was used"
    )
    highlights_detected: int = Field(default=0, description="Number of highlights found")
    processing_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall processing confidence"
    )
    
    @validator('statement_period')
    def validate_statement_period(cls, v):
        """Validate statement period format."""
        if not v:
            return v
        
        # Expected format: "DD/MM/YYYY - DD/MM/YYYY"
        if not re.match(r'^\d{1,2}/\d{1,2}/\d{4}\s*-\s*\d{1,2}/\d{1,2}/\d{4}$', v):
            raise ValueError('Statement period must be in format "DD/MM/YYYY - DD/MM/YYYY"')
        
        return v
    
    @validator('bsb')
    def validate_bsb(cls, v):
        """Validate Australian BSB format."""
        if not v:
            return v
        
        # BSB format: XXX-XXX or XXXXXX
        cleaned = re.sub(r'[^\d]', '', v)
        if len(cleaned) != 6:
            raise ValueError(f'BSB must be 6 digits: {v}')
        
        return v
    
    @validator('account_number')
    def validate_account_number(cls, v):
        """Validate account number (may be masked)."""
        if not v:
            raise ValueError('Account number is required')
        
        # Allow masked formats like "XXXX-XXXX-1234" or partial masking
        if not re.match(r'^[X\d\-\s]+$', v):
            raise ValueError(f'Invalid account number format: {v}')
        
        return v
    
    def get_work_related_transactions(self, min_likelihood: float = 0.5) -> List[BankTransaction]:
        """Get transactions likely to be work-related based on likelihood score."""
        return [
            t for t in self.transactions 
            if t.work_related_likelihood and t.work_related_likelihood >= min_likelihood
        ]
    
    def get_ato_compliance_score(self) -> float:
        """Calculate overall ATO compliance score for the statement."""
        if not self.ato_compliance_assessment:
            return 0.0
        
        return self.ato_compliance_assessment.get('overall_compliance', 0.0)
    
    def get_total_work_expenses(self) -> float:
        """Calculate total amount of work-related expenses (debits only)."""
        total = 0.0
        for transaction in self.get_work_related_transactions():
            if transaction.debit_amount:
                try:
                    amount = float(transaction.debit_amount.replace(',', ''))
                    if amount > 0:  # Debits are positive numbers but represent money out
                        total += amount
                except (ValueError, AttributeError):
                    continue
        return total
    
    def get_highlighted_expense_total(self) -> float:
        """Calculate total amount from user-highlighted transactions."""
        total = 0.0
        for transaction in self.highlighted_transactions:
            if transaction.debit_amount:
                try:
                    amount = float(transaction.debit_amount.replace(',', ''))
                    if amount > 0:
                        total += amount
                except (ValueError, AttributeError):
                    continue
        return total


# Australian Bank Registry - Major financial institutions
AUSTRALIAN_BANKS = {
    'cba': AustralianBankInfo(
        bank_code='cba',
        full_name='Commonwealth Bank of Australia',
        bsb_ranges=['06', '20', '21', '22'],
        account_format='XXXXXX-XXXXXXXX',
        logo_identifiers=['yellow diamond', 'CommBank', 'NetBank', 'Commonwealth Bank']
    ),
    'anz': AustralianBankInfo(
        bank_code='anz', 
        full_name='Australia and New Zealand Banking Group',
        bsb_ranges=['01', '06', '08'],
        account_format='XXX-XXXXXX',
        logo_identifiers=['ANZ', 'blue branding', 'ANZ Internet Banking']
    ),
    'westpac': AustralianBankInfo(
        bank_code='westpac',
        full_name='Westpac Banking Corporation', 
        bsb_ranges=['03', '73'],
        account_format='XXX-XXX',
        logo_identifiers=['red W logo', 'Westpac', 'Westpac Online']
    ),
    'nab': AustralianBankInfo(
        bank_code='nab',
        full_name='National Australia Bank',
        bsb_ranges=['08', '09'],
        account_format='XXXXXXXX',
        logo_identifiers=['red star', 'NAB', 'NAB Connect']
    ),
    'bendigo': AustralianBankInfo(
        bank_code='bendigo',
        full_name='Bendigo and Adelaide Bank',
        bsb_ranges=['63'],
        account_format='XXXXXXXX', 
        logo_identifiers=['Bendigo Bank', 'community banking']
    ),
    'ing': AustralianBankInfo(
        bank_code='ing',
        full_name='ING',
        bsb_ranges=['92'],
        account_format='XXXXXXXX',
        logo_identifiers=['ING', 'orange branding']
    )
}


def identify_bank_from_text(text: str) -> Optional[AustralianBankInfo]:
    """
    Identify Australian bank from statement text.
    
    Args:
        text: Raw text from bank statement
        
    Returns:
        Identified bank information or None if not found
    """
    text_upper = text.upper()
    
    for _bank_code, bank_info in AUSTRALIAN_BANKS.items():
        for identifier in bank_info.logo_identifiers:
            if identifier.upper() in text_upper:
                return bank_info
    
    return None


def validate_australian_bsb(bsb: str) -> bool:
    """
    Validate Australian BSB against known bank ranges.
    
    Args:
        bsb: BSB number to validate
        
    Returns:
        True if BSB is valid for a known Australian bank
    """
    if not bsb:
        return False
        
    # Clean BSB (remove hyphens, spaces)
    cleaned_bsb = re.sub(r'[^\d]', '', bsb)
    
    if len(cleaned_bsb) != 6:
        return False
    
    # Check against known bank BSB ranges
    bsb_prefix = cleaned_bsb[:2]
    
    for bank_info in AUSTRALIAN_BANKS.values():
        if bsb_prefix in bank_info.bsb_ranges:
            return True
            
    return False