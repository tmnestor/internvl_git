# Work Related Expense Document Classification Implementation Plan

## Executive Summary

This document outlines the implementation of a **Work Related Expense Document Class Categorizer** to enhance the InternVL PoC pipeline for Australian Tax Office (ATO) compliance. The system will automatically classify documents and route them to specialized extraction prompts, significantly improving accuracy and compliance.

## Current State Analysis

### Existing Pipeline
```
Input Document → Generic Prompt → Extraction → Output
```

### Issues with Current Approach
- Single prompt tries to handle all document types
- **JSON prompts are unreliable** - parsing failures and malformed output
- Lower accuracy due to generic nature
- No document-specific ATO compliance rules
- Manual document type specification required
- Suboptimal extraction for specialized documents (petrol receipts, tax invoices, etc.)

## Proposed Architecture

### New Pipeline Flow
```
Input Document → Document Classifier → Key-Value Extractor → ATO Compliance Validator → Output
       ↓                 ↓                      ↓                        ↓
   Image File    Classification Result    KEY-VALUE Extraction     Compliance Score
                  (Definitive Type)      (Reliable Parsing)       + Recommendations
```

### Document Classification Hierarchy

#### Primary Categories
1. **Business Receipts** - General retail purchases
2. **Tax Invoices** - GST invoices with ABN requirements
3. **Bank Statements** - Account statements (already implemented)
4. **Fuel Receipts** - Petrol/diesel purchases
5. **Meal Receipts** - Restaurant/cafe/catering
6. **Accommodation** - Hotels/motels/Airbnb
7. **Travel Documents** - Flights/trains/buses/taxis
8. **Parking/Tolls** - Parking meters/garages/toll roads
9. **Equipment/Supplies** - Office supplies/tools/equipment
10. **Professional Services** - Legal/accounting/consulting
11. **Other** - Fallback category

#### Document Classification Logic
- **Classification Required**: Every document MUST be classified - no fallbacks
- **Single Path Processing**: Each document type follows ONE clear processing path
- **KEY-VALUE ONLY**: All prompts use KEY-VALUE format - NO JSON prompts
- **No Generic Prompts**: Remove all generic/fallback prompts - use specialized KEY-VALUE prompts only
- **Failed Classification**: Explicit failure with clear error message - no silent fallbacks

#### Key-Value Extraction Benefits
- **Reliable Parsing**: No JSON syntax errors or malformed output
- **Consistent Format**: Every document type uses same KEY-VALUE structure
- **Error Resilience**: Partial extraction still provides usable data
- **Human Readable**: Easy to debug and validate extraction results
- **Production Proven**: Existing bank statement processing uses KEY-VALUE successfully

## Technical Implementation

### 1. Document Classification Module

#### File Structure
```
internvl/
├── classification/
│   ├── __init__.py
│   ├── document_classifier.py       # Main classification logic
│   ├── document_types.py           # Document type definitions
│   ├── confidence_scorer.py        # Confidence calculation
│   └── classification_prompts.py   # Classification prompt templates
```

#### Core Components

##### DocumentClassifier Class
```python
class WorkExpenseDocumentClassifier:
    """
    Classifies Australian work-related expense documents for optimal processing
    """
    
    def __init__(self):
        self.document_types = DocumentTypes()
        self.confidence_scorer = ConfidenceScorer()
        
    def classify_document(self, image_path: str, model, tokenizer) -> ClassificationResult:
        """
        Classify document - MUST return a definitive classification or raise error
        
        Returns:
            ClassificationResult(
                document_type: DocumentType,  # Enum - no strings
                classification_reasoning: str,
                processing_prompt: str  # Deterministic prompt selection
            )
            
        Raises:
            ClassificationFailedException: If classification is uncertain
        """
```

##### DocumentTypes Enum
```python
class DocumentType(Enum):
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
```

### 2. Classification Prompt System

#### Classification Prompt Design
```yaml
document_classification_prompt: |
  <image>
  Analyze this Australian work-related expense document and classify its type.
  
  DOCUMENT_TYPES:
  1. business_receipt - General retail receipt (Woolworths, Coles, Target, etc.)
  2. tax_invoice - GST tax invoice with ABN (formal business invoice)
  3. bank_statement - Bank account statement
  4. fuel_receipt - Petrol/diesel station receipt (BP, Shell, Caltex, etc.)
  5. meal_receipt - Restaurant/cafe/catering receipt
  6. accommodation - Hotel/motel/Airbnb receipt
  7. travel_document - Flight/train/bus ticket or travel booking
  8. parking_toll - Parking meter/garage or toll road receipt
  9. equipment_supplies - Office supplies/tools/equipment receipt
  10. professional_services - Legal/accounting/consulting invoice
  11. other - Any other work-related document
  
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
  
  Focus on Australian businesses and document formats.
```

### 3. Specialized Extraction Prompts

#### Business Receipt Key-Value Prompt
```yaml
business_receipt_extraction_prompt: |
  <image>
  Extract information from this Australian business receipt in KEY-VALUE format.
  
  CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.
  
  STORE: [Business name in CAPITALS]
  ABN: [Australian Business Number if visible - XX XXX XXX XXX format]
  DATE: [DD/MM/YYYY format]
  GST: [GST amount - 10% component]
  TOTAL: [Total amount including GST]
  ITEMS: [Product1 | Product2 | Product3]
  QUANTITIES: [Qty1 | Qty2 | Qty3]  
  PRICES: [Price1 | Price2 | Price3]
  
  EXAMPLE OUTPUT:
  STORE: WOOLWORTHS SUPERMARKETS
  ABN: 88 000 014 675
  DATE: 15/06/2024
  GST: 4.25
  TOTAL: 46.75
  ITEMS: Bread White | Milk 2L | Eggs Free Range 12pk
  QUANTITIES: 1 | 1 | 1
  PRICES: 3.50 | 5.20 | 8.95
  
  ATO_COMPLIANCE_REQUIREMENTS:
  - Business name and date are mandatory
  - GST component required for claims over $82.50
  - ABN validates legitimate business expense
  - Use pipe (|) separator for multiple items
```

#### Fuel Receipt Key-Value Prompt
```yaml
fuel_receipt_extraction_prompt: |
  <image>
  Extract information from this Australian fuel receipt for work vehicle expense claims.
  
  CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.
  
  STATION: [Fuel station name - BP, Shell, Caltex, Ampol, etc.]
  STATION_ADDRESS: [Station location if visible]
  DATE: [DD/MM/YYYY]
  TIME: [HH:MM if visible]
  FUEL_TYPE: [Unleaded, Premium, Diesel, etc.]
  LITRES: [Fuel quantity in litres]
  PRICE_PER_LITRE: [Rate per litre - cents format]
  TOTAL_FUEL_COST: [Total fuel amount before other items]
  GST: [GST component]
  TOTAL: [Total amount including GST]
  PUMP_NUMBER: [Pump number if visible]
  VEHICLE_KM: [Odometer reading if visible]
  
  EXAMPLE OUTPUT:
  STATION: BP AUSTRALIA
  STATION_ADDRESS: 123 Main Street, Melbourne VIC
  DATE: 15/06/2024
  TIME: 14:35
  FUEL_TYPE: Unleaded 91
  LITRES: 45.20
  PRICE_PER_LITRE: 189.9
  TOTAL_FUEL_COST: 85.85
  GST: 7.81
  TOTAL: 85.85
  PUMP_NUMBER: 3
  VEHICLE_KM: 45230
  
  ATO_FUEL_REQUIREMENTS:
  - Date, station name, and total amount are mandatory
  - Litres and rate per litre support logbook method claims
  - GST breakdown essential for business vehicle deductions
  - Vehicle odometer helps verify business vs personal use
```

#### Tax Invoice Key-Value Prompt
```yaml
tax_invoice_extraction_prompt: |
  <image>
  Extract information from this Australian GST tax invoice for business expense claims.
  
  CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.
  
  DOCUMENT_TYPE: [Must contain "TAX INVOICE" or "INVOICE"]
  SUPPLIER: [Business/company name]
  SUPPLIER_ABN: [Supplier's ABN - XX XXX XXX XXX format]
  SUPPLIER_ADDRESS: [Supplier's business address]
  CUSTOMER: [Customer/client name]
  CUSTOMER_ABN: [Customer's ABN if visible]
  INVOICE_NUMBER: [Invoice reference number]
  DATE: [Invoice date DD/MM/YYYY]
  DUE_DATE: [Payment due date if specified]
  DESCRIPTION: [Services/goods description]
  SUBTOTAL: [Amount before GST]
  GST: [GST amount - must be specified separately]
  TOTAL: [Total amount including GST]
  
  EXAMPLE OUTPUT:
  DOCUMENT_TYPE: TAX INVOICE
  SUPPLIER: ACME CONSULTING PTY LTD
  SUPPLIER_ABN: 12 345 678 901
  SUPPLIER_ADDRESS: 456 Business Street, Sydney NSW 2000
  CUSTOMER: CLIENT COMPANY PTY LTD
  INVOICE_NUMBER: INV-2024-0156
  DATE: 15/06/2024
  DUE_DATE: 15/07/2024
  DESCRIPTION: Professional consulting services
  SUBTOTAL: 500.00
  GST: 50.00
  TOTAL: 550.00
  
  TAX_INVOICE_REQUIREMENTS:
  - Must contain "TAX INVOICE" text on document
  - Supplier ABN mandatory for invoices over $82.50
  - GST amount must be specified separately from subtotal
  - Essential for business expense claims and BAS reporting
```

### 4. Integration with Existing CLI

#### Clean CLI Flow (No Backward Compatibility)
```python
# internvl/cli/internvl_single.py - COMPLETELY REWRITTEN

def process_single_document(args):
    """Clean document processing with mandatory classification"""
    
    # Step 1: Load model
    model, tokenizer = load_model_and_tokenizer(...)
    
    # Step 2: MANDATORY Classification (no manual override)
    classifier = WorkExpenseDocumentClassifier()
    try:
        classification = classifier.classify_document(
            image_path=args.image_path,
            model=model,
            tokenizer=tokenizer
        )
    except ClassificationFailedException as e:
        # FAIL FAST - no fallbacks
        logger.error(f"Classification failed: {e}")
        return {"error": "Document classification failed", "details": str(e)}
    
    # Step 3: Route to SINGLE processor per type
    processor = get_processor_for_type(classification.document_type)
    result = processor.process(
        image_path=args.image_path,
        prompt=classification.processing_prompt,
        model=model,
        tokenizer=tokenizer
    )
    
    # Step 4: Type-specific ATO compliance
    compliance = assess_document_compliance(result, classification.document_type)
    
    return {
        "document_type": classification.document_type.value,
        "classification_reasoning": classification.classification_reasoning,
        "extraction": result,
        "ato_compliance": compliance
    }

# REMOVE ALL LEGACY CODE - no generic prompts, no fallbacks, no manual type override
```

#### Simplified CLI Arguments (Remove Complex Options)
```python
# REMOVE these confusing options:
# --document-type (manual override removed)
# --auto-classify (classification is mandatory)
# --extraction-method (only specialized prompts)
# --confidence-threshold (classification must be definitive)

# KEEP only essential options:
parser.add_argument("--image-path", required=True, help="Path to document image")
parser.add_argument("--output-file", help="Output file path (default: stdout)")
parser.add_argument("--verbose", action="store_true", help="Verbose logging")

# Classification is ALWAYS automatic, extraction method is ALWAYS document-specific
```

### 5. ATO Compliance Framework

#### Document-Specific Compliance Rules
```python
class ATOComplianceValidator:
    """Validates documents against ATO requirements by type"""
    
    def __init__(self):
        self.compliance_rules = {
            DocumentType.BUSINESS_RECEIPT: BusinessReceiptRules(),
            DocumentType.TAX_INVOICE: TaxInvoiceRules(),
            DocumentType.FUEL_RECEIPT: FuelReceiptRules(),
            # ... other types
        }
    
    def validate_document(self, extraction_result: dict, document_type: DocumentType) -> ComplianceResult:
        """Validate extracted data against ATO requirements"""
        
        rules = self.compliance_rules[document_type]
        return rules.validate(extraction_result)
```

#### Compliance Rule Examples
```python
class TaxInvoiceRules(ComplianceRuleBase):
    """ATO requirements for tax invoices"""
    
    def validate(self, data: dict) -> ComplianceResult:
        issues = []
        score = 100
        
        # Tax invoice must contain "TAX INVOICE" text
        if not data.get('document_type') or 'tax invoice' not in data['document_type'].lower():
            issues.append("Document must contain 'TAX INVOICE' text")
            score -= 20
            
        # ABN required for invoices >$82.50
        total = float(data.get('total', 0))
        if total > 82.50 and not data.get('supplier_abn'):
            issues.append("Supplier ABN required for tax invoices over $82.50")
            score -= 25
            
        # GST must be specified separately
        if not data.get('gst'):
            issues.append("GST amount must be specified separately on tax invoices")
            score -= 15
            
        return ComplianceResult(
            score=max(0, score),
            issues=issues,
            recommendations=self._generate_recommendations(issues)
        )
```

### 6. Performance Optimization

#### Classification Caching
```python
class ClassificationCache:
    """Cache classification results to avoid re-processing"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.max_size = cache_size
        
    def get_classification(self, image_hash: str) -> Optional[ClassificationResult]:
        """Retrieve cached classification if available"""
        return self.cache.get(image_hash)
        
    def store_classification(self, image_hash: str, result: ClassificationResult):
        """Store classification result in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[image_hash] = result
```

#### Batch Processing Optimization
```python
def process_batch_with_classification(image_paths: List[str]) -> List[dict]:
    """Optimized batch processing with document classification"""
    
    # Step 1: Batch classify all documents
    classifications = batch_classify_documents(image_paths)
    
    # Step 2: Group by document type for efficient processing
    typed_groups = group_by_document_type(classifications)
    
    # Step 3: Process each group with optimized settings
    results = []
    for doc_type, group_items in typed_groups.items():
        group_results = process_document_group(group_items, doc_type)
        results.extend(group_results)
    
    return results
```

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- [ ] Create document classification module structure
- [ ] Implement basic DocumentClassifier class
- [ ] Design and test classification prompts
- [ ] Create document type definitions and enums

### Phase 2: Specialized Prompts (Week 2-3)
- [ ] Develop specialized extraction prompts for each document type
- [ ] Test prompts with real document examples
- [ ] Refine prompts based on accuracy testing
- [ ] Create prompt validation and quality checks

### Phase 3: CLI Integration (Week 3-4)
- [ ] Modify internvl_single.py to support classification
- [ ] Update internvl_batch.py for batch classification
- [ ] Add new CLI arguments and options
- [ ] Implement document routing logic

### Phase 4: ATO Compliance Framework (Week 4-5)
- [ ] Design compliance validation system
- [ ] Implement document-specific compliance rules
- [ ] Create compliance scoring and reporting
- [ ] Test compliance validation accuracy

### Phase 5: Optimization & Testing (Week 5-6)
- [ ] Implement classification caching
- [ ] Optimize batch processing performance
- [ ] Comprehensive testing with real documents
- [ ] Performance benchmarking and tuning

### Phase 6: Documentation & Deployment (Week 6)
- [ ] Update README.md with new features
- [ ] Create usage examples and tutorials
- [ ] Update prompts.yaml with all new prompts
- [ ] Prepare deployment documentation

## Testing Strategy

### 1. Classification Accuracy Testing
- **Test Dataset**: 100+ real Australian work expense documents
- **Document Types**: Balanced representation of all supported types
- **Metrics**: Classification accuracy, confidence calibration, confusion matrix
- **Target**: >95% accuracy for primary document types

### 2. Extraction Quality Testing
- **Key-Value vs JSON Testing**: Compare KEY-VALUE prompts vs JSON prompts
- **Metrics**: Parsing success rate, field-level accuracy, extraction completeness, ATO compliance scores
- **Target**: 95%+ parsing success rate with KEY-VALUE (vs <80% with JSON)
- **Target**: 20%+ improvement in extraction accuracy for specialized KEY-VALUE prompts

### 3. Performance Testing
- **Load Testing**: Batch processing with 1000+ documents
- **Latency Testing**: Single document processing time
- **Memory Testing**: Memory usage with classification caching
- **Target**: <2s additional overhead per document for classification

### 4. ATO Compliance Testing
- **Real-world Scenarios**: Test with actual tax office requirements
- **Edge Cases**: Partially compliant documents, missing fields
- **Validation**: Cross-check with ATO guidelines and accounting professionals
- **Target**: 100% accuracy in compliance rule application

## Risk Mitigation

### Technical Risks
1. **Classification Errors**: FAIL FAST with clear error messages - no silent failures
2. **Performance Degradation**: Use caching and batch optimization
3. **Prompt Complexity**: Design specialized prompts that work consistently
4. **Model Compatibility**: Test prompts thoroughly with InternVL - one working configuration

### Business Risks
1. **ATO Compliance**: Regular review of ATO guidelines and rule updates
2. **Document Variability**: Continuous testing with new document formats  
3. **Clear User Experience**: Simple, predictable behavior - document goes in, structured data comes out

### Operational Risks
1. **Clear Architecture**: Single decision path per document type - no complex deployment
2. **Maintenance Burden**: Automated testing and monitoring
3. **Documentation**: Comprehensive documentation for each document type processor

## Success Metrics

### Primary KPIs
- **Classification Accuracy**: >95% for common document types
- **KEY-VALUE Parsing Success**: 95%+ vs <80% for JSON prompts
- **Extraction Improvement**: 20%+ accuracy gain vs. generic JSON prompts
- **ATO Compliance Score**: 90%+ documents meet compliance requirements
- **Processing Performance**: <10% overhead for classification

### Secondary KPIs
- **User Satisfaction**: Feedback on document type detection accuracy
- **Error Reduction**: 50% reduction in manual document type corrections
- **Coverage**: Support for 95% of common Australian work expense documents
- **Maintainability**: <4 hours average time to add new document type support

## Future Enhancements

### Advanced Classification
- **Multi-page Documents**: Handle complex invoices and statements
- **Hybrid Documents**: Receipts that combine multiple expense types
- **Digital Documents**: PDF and electronic receipt processing
- **OCR Enhancement**: Pre-classification text extraction for better accuracy

### AI/ML Improvements
- **Custom Classification Models**: Train document-specific classifiers
- **Confidence Learning**: Adaptive confidence threshold adjustment
- **Few-shot Learning**: Quick adaptation to new document types
- **Active Learning**: User feedback integration for classification improvement

### Integration Enhancements
- **Accounting Software**: Direct integration with Xero, MYOB, QuickBooks
- **Mobile App**: Camera-based document capture and classification
- **API Gateway**: RESTful API for third-party integrations
- **Real-time Processing**: Streaming document processing capabilities

This implementation plan provides a comprehensive roadmap for implementing a robust Work Related Expense Document Classification system that will significantly enhance the InternVL PoC's accuracy and ATO compliance capabilities.