# Robust Output Formats for LLM Receipt Extraction

## Problem Statement

JSON malformation has been a persistent challenge throughout the development of this receipt extraction system. The InternVL model frequently generates syntactically invalid JSON with:

- Missing closing quotes (`"12.82,` instead of `"12.82",`)
- Incomplete structures (truncated arrays)
- Control characters in output
- Unbalanced brackets and braces

While we've achieved 90.91% F1-score with aggressive JSON reconstruction, these malformation issues represent a fundamental weakness in using JSON as the primary output format for LLM-generated structured data.

## Alternative Structured Formats Analysis

### 1. YAML Format

**Advantages:**
- ✅ More forgiving syntax - No strict quote requirements
- ✅ Human readable - Cleaner for LLMs to generate  
- ✅ Flexible indentation - Less brittle than JSON brackets
- ✅ Built-in lists - Natural array syntax without brackets

**Disadvantages:**
- ❌ Indentation sensitive - Spacing errors can break parsing
- ❌ Multiple valid syntaxes - Creates ambiguity for LLMs
- ❌ Slower parsing - More complex than JSON
- ❌ Less universal - JSON has broader tooling support

**Example YAML Output:**
```yaml
date_value: 05/05/2025
store_name_value: WOOLWORTHS
tax_value: 12.82
total_value: 140.98
prod_item_value:
  - Milk 2L
  - Chicken Breast
  - Rice 1kg
prod_quantity_value: [1, 2, 1]
prod_price_value: [4.50, 8.00, 7.60]
```

### 2. Key-Value Pairs Format (Recommended)

**Advantages:**
- ✅ **Extremely robust** - Nearly impossible to malform
- ✅ **Easy to parse** - Simple regex patterns
- ✅ **LLM friendly** - Natural format for language models
- ✅ **Fault tolerant** - Partial failures don't break everything
- ✅ **Human readable** - Easy to debug and validate
- ✅ **Flexible delimiters** - Multiple separator options

**Example Key-Value Output:**
```
DATE: 05/05/2025
STORE: WOOLWORTHS  
TAX: 12.82
TOTAL: 140.98
PRODUCTS: Milk 2L | Chicken Breast | Rice 1kg
QUANTITIES: 1 | 2 | 1
PRICES: 4.50 | 8.00 | 7.60
```

**Parsing Strategy:**
```python
def parse_kv_format(text: str) -> Dict[str, Any]:
    data = {}
    for line in text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Handle arrays with pipe delimiter
            if '|' in value:
                data[key] = [item.strip() for item in value.split('|')]
            else:
                data[key] = value
    return data
```

### 3. Structured Text with Delimiters

**Example:**
```
=== RECEIPT INFO ===
Date: 05/05/2025
Store: WOOLWORTHS
Tax: 12.82
Total: 140.98

=== PRODUCTS ===
1. Milk 2L (qty: 1, price: 4.50)
2. Chicken Breast (qty: 2, price: 8.00)
3. Rice 1kg (qty: 1, price: 7.60)
```

**Advantages:**
- ✅ Very robust and readable
- ✅ Self-documenting structure
- ✅ Easy to validate sections

**Disadvantages:**
- ❌ More complex parsing logic
- ❌ Verbose output format

### 4. CSV-Style Format

**Example:**
```
FIELD,VALUE
date_value,05/05/2025
store_name_value,WOOLWORTHS
tax_value,12.82
total_value,140.98

PRODUCTS
Milk 2L,1,4.50
Chicken Breast,2,8.00
Rice 1kg,1,7.60
```

**Advantages:**
- ✅ Standard format with robust parsers
- ✅ Handles arrays naturally

**Disadvantages:**
- ❌ Less human readable
- ❌ Requires specific structure knowledge

## Implementation Recommendation

### Primary Strategy: Key-Value Format with JSON Fallback

1. **Update Prompts**
   - Modify `australian_optimized_prompt` to request key-value format
   - Keep JSON format as secondary option
   - Provide clear examples and format specifications

2. **Add KV Parser**
   - Implement robust key-value parsing alongside existing JSON extraction
   - Use multiple delimiter strategies (`:`, `|`, `,`)
   - Handle both single values and arrays gracefully

3. **Hybrid Extraction Pipeline**
   ```python
   def extract_structured_data(text: str) -> Dict[str, Any]:
       # Try key-value format first
       kv_result = try_parse_kv_format(text)
       if is_valid_extraction(kv_result):
           return kv_result
       
       # Fallback to enhanced JSON reconstruction
       json_result = extract_json_from_text(text)
       return json_result
   ```

4. **Maintain Backward Compatibility**
   - Keep existing JSON reconstruction for legacy support
   - Monitor performance metrics across both formats
   - Gradual migration strategy

### Expected Performance Improvement

- **Current**: 90.91% F1-score with aggressive JSON reconstruction
- **Expected**: 95%+ F1-score by eliminating JSON syntax errors
- **Robustness**: Dramatically reduced malformation issues
- **Maintainability**: Simpler parsing logic and error handling

### Implementation Priority

1. **Phase 1**: Implement key-value parser as alternative extraction method
2. **Phase 2**: Create dual-format prompts and test performance comparison
3. **Phase 3**: Optimize format based on real-world performance data
4. **Phase 4**: Consider migrating to key-value as primary format

## Prompt Template Example

```yaml
key_value_receipt_prompt: |
  <image>
  Extract information from this Australian receipt and return in KEY-VALUE format.
  
  Use this exact format:
  DATE: [purchase date in DD/MM/YYYY format]
  STORE: [store name in capitals]
  TAX: [GST amount]
  TOTAL: [total amount including GST]
  PRODUCTS: [item1 | item2 | item3]
  QUANTITIES: [qty1 | qty2 | qty3]
  PRICES: [price1 | price2 | price3]
  
  Example:
  DATE: 16/03/2023
  STORE: WOOLWORTHS
  TAX: 3.82
  TOTAL: 42.08
  PRODUCTS: MILK 2L | BREAD MULTIGRAIN | EGGS FREE RANGE 12PK
  QUANTITIES: 1 | 2 | 1
  PRICES: 4.50 | 8.00 | 7.60
  
  Return ONLY the key-value pairs above. No explanations.
```

## Conclusion

While JSON remains the standard for structured data, its strict syntax requirements make it fragile for LLM-generated output. Key-value formats offer superior robustness and could represent the next evolution of this extraction system, potentially pushing performance above 95% F1-score while dramatically reducing parsing complexity and error handling overhead.

The current 90.91% F1-score with JSON reconstruction demonstrates the system's maturity, but implementing alternative formats could be the key to achieving near-perfect extraction reliability.