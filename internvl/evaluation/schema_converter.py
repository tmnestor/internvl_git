"""
Schema Converter for SROIE Evaluation

Converts between different JSON schemas used in predictions and ground truth.
"""

from typing import Any, Dict


def convert_huaifeng_to_sroie_schema(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert from Huaifeng schema to SROIE schema.
    
    Huaifeng schema:
    - company_name, address, phone_number, date, ABN, total_amount
    
    SROIE schema:
    - date_value, store_name_value, tax_value, total_value, 
      prod_item_value, prod_quantity_value, prod_price_value
    """
    converted = {
        "date_value": "",
        "store_name_value": "",
        "tax_value": "",
        "total_value": "",
        "prod_item_value": [],
        "prod_quantity_value": [],
        "prod_price_value": [],
    }
    
    # Map fields
    field_mapping = {
        "date": "date_value",
        "company_name": "store_name_value",
        "total_amount": "total_value",
        # Note: Huaifeng schema doesn't have product details or tax
    }
    
    for huaifeng_field, sroie_field in field_mapping.items():
        if huaifeng_field in prediction:
            value = prediction[huaifeng_field]
            if isinstance(value, (int, float)):
                value = str(value)
            converted[sroie_field] = value
    
    return converted


def convert_sroie_to_huaifeng_schema(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert from SROIE schema to Huaifeng schema.
    """
    converted = {
        "company_name": "",
        "address": "",
        "phone_number": "",
        "date": "",
        "ABN": "",
        "total_amount": ""
    }
    
    # Map fields
    field_mapping = {
        "date_value": "date",
        "store_name_value": "company_name", 
        "total_value": "total_amount",
    }
    
    for sroie_field, huaifeng_field in field_mapping.items():
        if sroie_field in prediction:
            value = prediction[sroie_field]
            if isinstance(value, list) and len(value) > 0:
                value = value[0]  # Take first value if list
            converted[huaifeng_field] = str(value)
    
    return converted


def detect_schema_type(data: Dict[str, Any]) -> str:
    """
    Detect which schema type the data uses.
    
    Returns:
        'huaifeng' or 'sroie'
    """
    huaifeng_fields = {"company_name", "address", "phone_number", "date", "total_amount"}
    sroie_fields = {"date_value", "store_name_value", "tax_value", "total_value"}
    
    data_fields = set(data.keys())
    
    huaifeng_overlap = len(data_fields.intersection(huaifeng_fields))
    sroie_overlap = len(data_fields.intersection(sroie_fields))
    
    return "huaifeng" if huaifeng_overlap > sroie_overlap else "sroie"


def ensure_sroie_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure data is in SROIE schema format, converting if necessary.
    """
    schema_type = detect_schema_type(data)
    
    if schema_type == "huaifeng":
        return convert_huaifeng_to_sroie_schema(data)
    else:
        return data


def ensure_huaifeng_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure data is in Huaifeng schema format, converting if necessary.
    """
    schema_type = detect_schema_type(data)
    
    if schema_type == "sroie":
        return convert_sroie_to_huaifeng_schema(data)
    else:
        return data