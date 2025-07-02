"""
Document Type Definitions for Australian Work Expense Classification.

This module defines the document types supported by the InternVL PoC
for Australian Tax Office (ATO) compliance and work-related expense claims.
"""

from enum import Enum
from typing import Dict, List


class DocumentType(Enum):
    """Australian work expense document types with ATO compliance focus."""
    
    BUSINESS_RECEIPT = "business_receipt"
    TAX_INVOICE = "tax_invoice" 
    BANK_STATEMENT = "bank_statement"
    FUEL_RECEIPT = "fuel_receipt"
    MEAL_RECEIPT = "meal_receipt"
    ACCOMMODATION = "accommodation"
    TRAVEL_DOCUMENT = "travel_document"
    PARKING_TOLL = "parking_toll"
    EQUIPMENT_SUPPLIES = "equipment_supplies"
    PROFESSIONAL_SERVICES = "professional_services"
    OTHER = "other"


class DocumentTypeMetadata:
    """Metadata and configuration for each document type."""
    
    # Document type descriptions for classification
    DESCRIPTIONS = {
        DocumentType.BUSINESS_RECEIPT: "General retail receipt from businesses like Woolworths, Coles, Target, etc.",
        DocumentType.TAX_INVOICE: "Formal GST tax invoice with ABN, showing business-to-business transactions",
        DocumentType.BANK_STATEMENT: "Bank account statement showing transaction history",
        DocumentType.FUEL_RECEIPT: "Petrol/diesel station receipt from BP, Shell, Caltex, Ampol, etc.",
        DocumentType.MEAL_RECEIPT: "Restaurant, cafe, or catering receipt for business meals",
        DocumentType.ACCOMMODATION: "Hotel, motel, or Airbnb receipt for business travel",
        DocumentType.TRAVEL_DOCUMENT: "Flight, train, bus ticket or travel booking confirmation",
        DocumentType.PARKING_TOLL: "Parking meter, garage, or toll road receipt",
        DocumentType.EQUIPMENT_SUPPLIES: "Office supplies, tools, or equipment purchase receipt",
        DocumentType.PROFESSIONAL_SERVICES: "Legal, accounting, consulting, or professional service invoice",
        DocumentType.OTHER: "Other work-related document not fitting above categories"
    }
    
    # Prompt names for each document type (KEY-VALUE format only)
    KEY_VALUE_PROMPTS = {
        DocumentType.BUSINESS_RECEIPT: "business_receipt_extraction_prompt",
        DocumentType.TAX_INVOICE: "tax_invoice_extraction_prompt",
        DocumentType.BANK_STATEMENT: "bank_statement_ato_compliance_prompt",
        DocumentType.FUEL_RECEIPT: "fuel_receipt_extraction_prompt",
        DocumentType.MEAL_RECEIPT: "meal_receipt_extraction_prompt",
        DocumentType.ACCOMMODATION: "accommodation_receipt_extraction_prompt",
        DocumentType.TRAVEL_DOCUMENT: "travel_document_extraction_prompt",
        DocumentType.PARKING_TOLL: "parking_toll_extraction_prompt",
        DocumentType.EQUIPMENT_SUPPLIES: "equipment_supplies_extraction_prompt",
        DocumentType.PROFESSIONAL_SERVICES: "professional_services_extraction_prompt",
        DocumentType.OTHER: "other_document_extraction_prompt"
    }
    
    # Industry keywords for classification
    CLASSIFICATION_KEYWORDS = {
        DocumentType.BUSINESS_RECEIPT: [
            "woolworths", "coles", "aldi", "iga", "target", "kmart", "bunnings",
            "harvey norman", "jb hi-fi", "officeworks", "big w", "david jones",
            "myer", "retail", "supermarket", "department store"
        ],
        DocumentType.TAX_INVOICE: [
            "tax invoice", "gst invoice", "abn", "tax invoice number", 
            "invoice number", "professional services", "consulting",
            "business services", "contractor"
        ],
        DocumentType.BANK_STATEMENT: [
            "commonwealth bank", "anz", "westpac", "nab", "bendigo bank",
            "account statement", "transaction history", "bank statement",
            "bsb", "account number"
        ],
        DocumentType.FUEL_RECEIPT: [
            "bp", "shell", "caltex", "ampol", "mobil", "petrol", "diesel",
            "fuel", "servo", "united petroleum", "7-eleven fuel", "costco fuel",
            "liberty fuel", "unleaded", "premium"
        ],
        DocumentType.MEAL_RECEIPT: [
            "restaurant", "cafe", "catering", "lunch", "dinner", "meal",
            "food", "beverage", "mcdonald's", "kfc", "subway", "pizza",
            "coffee", "dining"
        ],
        DocumentType.ACCOMMODATION: [
            "hotel", "motel", "accommodation", "booking", "airbnb",
            "lodging", "stay", "room", "suite", "resort", "inn",
            "crown", "hilton", "marriott"
        ],
        DocumentType.TRAVEL_DOCUMENT: [
            "qantas", "jetstar", "virgin", "tiger", "flight", "airline",
            "boarding pass", "ticket", "travel", "transport", "train",
            "bus", "taxi", "uber", "ola"
        ],
        DocumentType.PARKING_TOLL: [
            "parking", "car park", "wilson parking", "secure parking",
            "meter", "toll", "citylink", "eastlink", "m7", "westconnex"
        ],
        DocumentType.EQUIPMENT_SUPPLIES: [
            "computer", "laptop", "software", "electronics", "equipment",
            "machinery", "tools", "supplies", "stationery", "printer",
            "phone", "tablet"
        ],
        DocumentType.PROFESSIONAL_SERVICES: [
            "accounting", "legal", "consulting", "lawyer", "accountant",
            "solicitor", "barrister", "consultant", "advisory", "audit",
            "tax agent", "bookkeeper"
        ],
        DocumentType.OTHER: []  # Fallback category
    }
    
    # ATO compliance requirements by document type
    ATO_REQUIREMENTS = {
        DocumentType.BUSINESS_RECEIPT: {
            "mandatory_fields": ["supplier_name", "date", "total_amount"],
            "required_for_claim_over_82_50": ["gst_amount", "supplier_abn"],
            "max_claim_without_receipt": 300.0,
            "gst_applicable": True
        },
        DocumentType.TAX_INVOICE: {
            "mandatory_fields": ["supplier_name", "supplier_abn", "date", "gst_amount", "total_amount"],
            "required_for_claim_over_82_50": ["document_type_contains_tax_invoice"],
            "max_claim_without_receipt": 0.0,  # Tax invoices always required
            "gst_applicable": True
        },
        DocumentType.BANK_STATEMENT: {
            "mandatory_fields": ["transaction_date", "description", "amount"],
            "required_for_claim_over_82_50": ["merchant_details"],
            "max_claim_without_receipt": 82.50,  # Need receipts for amounts over this
            "gst_applicable": False  # Bank statements don't show GST breakdown
        },
        DocumentType.FUEL_RECEIPT: {
            "mandatory_fields": ["station_name", "date", "fuel_type", "total_amount"],
            "required_for_claim_over_82_50": ["litres", "price_per_litre", "gst_amount"],
            "max_claim_without_receipt": 300.0,
            "gst_applicable": True
        },
        DocumentType.MEAL_RECEIPT: {
            "mandatory_fields": ["restaurant_name", "date", "total_amount"],
            "required_for_claim_over_82_50": ["gst_amount", "business_purpose"],
            "max_claim_without_receipt": 300.0,
            "gst_applicable": True
        },
        DocumentType.ACCOMMODATION: {
            "mandatory_fields": ["hotel_name", "date", "total_amount"],
            "required_for_claim_over_82_50": ["gst_amount", "business_purpose"],
            "max_claim_without_receipt": 300.0,
            "gst_applicable": True
        },
        DocumentType.TRAVEL_DOCUMENT: {
            "mandatory_fields": ["carrier_name", "date", "destination", "total_amount"],
            "required_for_claim_over_82_50": ["gst_amount", "business_purpose"],
            "max_claim_without_receipt": 300.0,
            "gst_applicable": True
        },
        DocumentType.PARKING_TOLL: {
            "mandatory_fields": ["operator_name", "date", "total_amount"],
            "required_for_claim_over_82_50": ["location", "business_purpose"],
            "max_claim_without_receipt": 300.0,
            "gst_applicable": True
        },
        DocumentType.EQUIPMENT_SUPPLIES: {
            "mandatory_fields": ["supplier_name", "date", "item_description", "total_amount"],
            "required_for_claim_over_82_50": ["gst_amount", "supplier_abn"],
            "max_claim_without_receipt": 300.0,
            "gst_applicable": True
        },
        DocumentType.PROFESSIONAL_SERVICES: {
            "mandatory_fields": ["service_provider", "supplier_abn", "date", "service_description", "total_amount"],
            "required_for_claim_over_82_50": ["gst_amount", "invoice_number"],
            "max_claim_without_receipt": 0.0,  # Professional services always need documentation
            "gst_applicable": True
        },
        DocumentType.OTHER: {
            "mandatory_fields": ["supplier_name", "date", "total_amount"],
            "required_for_claim_over_82_50": ["description", "business_purpose"],
            "max_claim_without_receipt": 300.0,
            "gst_applicable": True
        }
    }


def get_document_type_by_name(type_name: str) -> DocumentType:
    """Get DocumentType enum by string name."""
    try:
        return DocumentType(type_name.lower())
    except ValueError:
        return DocumentType.OTHER


def get_classification_keywords(document_type: DocumentType) -> List[str]:
    """Get classification keywords for a document type."""
    return DocumentTypeMetadata.CLASSIFICATION_KEYWORDS.get(document_type, [])


def get_key_value_prompt_name(document_type: DocumentType) -> str:
    """Get the KEY-VALUE prompt name for a document type."""
    return DocumentTypeMetadata.KEY_VALUE_PROMPTS.get(document_type, "other_document_extraction_prompt")


def get_ato_requirements(document_type: DocumentType) -> Dict:
    """Get ATO compliance requirements for a document type."""
    return DocumentTypeMetadata.ATO_REQUIREMENTS.get(document_type, {})


def is_gst_applicable(document_type: DocumentType) -> bool:
    """Check if GST is applicable for this document type."""
    requirements = get_ato_requirements(document_type)
    return requirements.get("gst_applicable", True)


def get_max_claim_without_receipt(document_type: DocumentType) -> float:
    """Get maximum claimable amount without receipt for document type."""
    requirements = get_ato_requirements(document_type)
    return requirements.get("max_claim_without_receipt", 300.0)