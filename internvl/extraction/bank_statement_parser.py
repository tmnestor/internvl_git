"""
Bank Statement Processing with ATO Compliance Assessment.

This module provides comprehensive bank statement processing capabilities
with highlight detection, transaction extraction, and Australian Tax Office
compliance assessment for work-related expense claims.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from internvl.image.highlight_detection import (
    BankStatementHighlightDetector,
    HighlightRegion,
    detect_bank_statement_highlights,
)
from internvl.schemas.bank_statement_schemas import (
    validate_australian_bsb,
)

logger = logging.getLogger(__name__)


class BankStatementProcessor:
    """Process bank statements with highlight detection and ATO compliance."""
    
    def __init__(self):
        """Initialize bank statement processor."""
        self.highlight_detector = BankStatementHighlightDetector()
        
        # Australian bank patterns for identification
        self.bank_patterns = {
            'cba': r'(Commonwealth Bank|CBA|CommBank|NetBank)',
            'anz': r'(ANZ|Australia.*New Zealand|ANZ Internet Banking)',
            'westpac': r'(Westpac|WBC|Westpac Online)',
            'nab': r'(National Australia Bank|NAB|NAB Connect)',
            'bendigo': r'(Bendigo Bank|Bendigo.*Adelaide|Community Bank)',
            'boq': r'(Bank of Queensland|BOQ)',
            'macquarie': r'(Macquarie Bank|Macquarie)',
            'ing': r'(ING|ING Direct)',
            'citibank': r'(Citibank|Citi)',
            'hsbc': r'(HSBC|HSBC Bank)',
            'bank_australia': r'(Bank Australia)',
            'suncorp': r'(Suncorp Bank|Suncorp)',
        }
        
        # Work expense keywords for transaction categorization
        self.work_expense_keywords = {
            'fuel': [
                'bp', 'shell', 'caltex', 'ampol', 'mobil', 'petrol', 'fuel', 'servo',
                'united petroleum', '7-eleven fuel', 'costco fuel', 'liberty fuel'
            ],
            'office_supplies': [
                'officeworks', 'staples', 'office', 'supplies', 'stationery',
                'bunnings warehouse', 'harvey norman', 'jb hi-fi'
            ],
            'travel': [
                'qantas', 'jetstar', 'virgin', 'tiger', 'hotel', 'motel', 'accommodation',
                'taxi', 'uber', 'ola', 'didi', 'car rental', 'hertz', 'avis', 'budget'
            ],
            'professional_services': [
                'accounting', 'legal', 'consulting', 'services', 'accountant',
                'lawyer', 'solicitor', 'barrister', 'consultant'
            ],
            'equipment': [
                'computer', 'laptop', 'software', 'electronics', 'phone',
                'equipment', 'machinery', 'tools'
            ],
            'training': [
                'training', 'course', 'seminar', 'workshop', 'conference',
                'education', 'university', 'tafe', 'certification'
            ],
            'parking': [
                'parking', 'car park', 'wilson parking', 'secure parking',
                'meter', 'toll', 'citylink', 'eastlink'
            ]
        }
        
        # ATO thresholds and requirements
        self.ato_requirements = {
            'min_receipt_amount': 82.50,  # ATO requirement for receipts above this amount
            'gst_rate': 0.10,  # Australian GST rate
            'max_claim_without_receipt': 300.00,  # Maximum claim without receipt
            'required_fields': ['transaction_date', 'description', 'debit_amount'],
            'preferred_fields': ['merchant_category', 'reference']
        }
    
    def process_bank_statement(
        self, 
        image_path: str, 
        model, 
        tokenizer, 
        use_highlight_detection: bool = True,
        prompt_name: str = 'bank_statement_ato_compliance_prompt'
    ) -> Dict[str, Any]:
        """
        Process bank statement with optional highlight detection.
        
        Args:
            image_path: Path to bank statement image
            model: InternVL model
            tokenizer: Model tokenizer  
            use_highlight_detection: Whether to detect highlights
            prompt_name: Prompt template to use
            
        Returns:
            Processed bank statement data with ATO compliance
        """
        try:
            result = {
                'success': False,
                'bank_statement_data': {},
                'highlighted_transactions': [],
                'ato_compliance': {},
                'processing_metadata': {
                    'highlight_detection_used': use_highlight_detection,
                    'highlights_detected': 0,
                    'total_transactions': 0,
                    'prompt_used': prompt_name
                }
            }
            
            # Step 1: Detect highlights if enabled
            highlight_regions = []
            highlight_summary = {}
            if use_highlight_detection:
                highlight_result = detect_bank_statement_highlights(
                    image_path=image_path,
                    extract_text=True,
                    create_visualization=False
                )
                
                if highlight_result['success']:
                    highlight_regions = highlight_result['highlights']
                    highlight_summary = highlight_result['summary']
                    result['processing_metadata']['highlights_detected'] = len(highlight_regions)
                    logger.info(f"Detected {len(highlight_regions)} highlighted regions")
                else:
                    logger.warning(f"Highlight detection failed: {highlight_result.get('error', 'Unknown error')}")
            
            # Step 2: Choose appropriate prompt based on highlights
            if highlight_regions:
                actual_prompt_name = 'bank_statement_highlighted_prompt'
            else:
                actual_prompt_name = prompt_name
            
            # Step 3: Get prompt text from prompts.yaml
            prompt = self._load_prompt_from_yaml(actual_prompt_name, highlight_regions)
            
            # Step 4: Process with InternVL
            from internvl.model.inference import get_raw_prediction
            
            # Generate response
            raw_response = get_raw_prediction(
                image_path=image_path,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                generation_config={"max_new_tokens": 2048, "do_sample": False},
                device="auto"
            )
            
            # Step 5: Parse bank statement response
            logger.info(f"Raw InternVL response length: {len(raw_response)} characters")
            logger.debug(f"Raw InternVL response: {raw_response[:500]}...")  # First 500 chars
            
            # Save raw response for debugging
            try:
                debug_file = Path("/tmp/bank_statement_raw_response.txt")
                debug_file.write_text(raw_response)
                logger.info("Raw response saved to /tmp/bank_statement_raw_response.txt")
            except Exception as e:
                logger.warning(f"Could not save raw response: {e}")
            
            parsed_data = self._parse_bank_statement_response(raw_response)
            
            # Step 6: Enhance with highlight information
            if highlight_regions:
                parsed_data = self._enhance_with_highlights(parsed_data, highlight_regions)
            
            # Step 7: Assess ATO compliance
            ato_assessment = self._assess_ato_compliance(parsed_data)
            
            # Step 8: Categorize transactions for work expenses
            if 'transactions' in parsed_data:
                parsed_data['transactions'] = self._categorize_transactions(parsed_data['transactions'])
            
            result.update({
                'success': True,
                'bank_statement_data': parsed_data,
                'highlighted_transactions': self._extract_highlighted_transactions(parsed_data),
                'ato_compliance': ato_assessment,
                'highlight_summary': highlight_summary,
                'processing_metadata': {
                    **result['processing_metadata'],
                    'total_transactions': len(parsed_data.get('transactions', [])),
                    'work_related_transactions': len([
                        t for t in parsed_data.get('transactions', []) 
                        if t.get('work_related_likelihood', 0) > 0.5
                    ]),
                    'highlighted_work_transactions': len([
                        t for t in parsed_data.get('transactions', [])
                        if t.get('highlight_detected', False)
                    ])
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Bank statement processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'bank_statement_data': {},
                'highlighted_transactions': [],
                'ato_compliance': {},
                'highlight_summary': {}
            }
    
    def _load_prompt_from_yaml(self, prompt_name: str, highlight_regions: List[HighlightRegion]) -> str:
        """Load prompt from prompts.yaml file."""
        
        try:
            # Find prompts.yaml file
            prompts_path = Path("prompts.yaml")
            if not prompts_path.exists():
                # Try relative to current working directory  
                prompts_path = Path.cwd() / "prompts.yaml"
            
            if prompts_path.exists():
                with prompts_path.open("r") as f:
                    prompts = yaml.safe_load(f)
                
                prompt = prompts.get(prompt_name, "")
                if prompt:
                    logger.info(f"Loaded prompt '{prompt_name}' from prompts.yaml")
                    return prompt
                else:
                    logger.warning(f"Prompt '{prompt_name}' not found in prompts.yaml")
            else:
                logger.warning("prompts.yaml not found, using fallback prompt")
                
        except Exception as e:
            logger.error(f"Error loading prompt from YAML: {e}")
        
        # Fallback to original custom prompt
        return self._get_bank_statement_prompt(prompt_name, highlight_regions)
    
    def _get_bank_statement_prompt(self, prompt_name: str, highlight_regions: List[HighlightRegion]) -> str:
        """Get appropriate prompt for bank statement processing (fallback)."""
        
        # Build highlight context
        if highlight_regions:
            highlight_info = f"\nHighlighted regions detected: {len(highlight_regions)} areas marked by user"
            highlight_details = "\n".join([
                f"- {region.color} highlight at ({region.x}, {region.y}): {region.extracted_text[:50]}..."
                for region in highlight_regions[:3]  # Show first 3
                if region.extracted_text
            ])
            highlight_context = f"{highlight_info}\n{highlight_details}"
        else:
            highlight_context = "\nNo highlighted regions detected."
        
        # Return appropriate prompt based on type
        if 'highlighted' in prompt_name:
            return f"""<image>
Analyze this Australian bank statement with user-highlighted transactions for work-related expense claims.

{highlight_context}

PRIORITY: Focus on extracting highlighted/marked transactions first, then process the full statement.

Extract information in this KEY-VALUE format:

BANK: [Financial institution name]
ACCOUNT_HOLDER: [Account holder name]
ACCOUNT_NUMBER: [Account number - mask middle digits]
BSB: [Bank State Branch code if visible]
STATEMENT_PERIOD: [Start date - End date in DD/MM/YYYY format]
OPENING_BALANCE: [Starting balance]
CLOSING_BALANCE: [Ending balance]

TRANSACTIONS:
DATE: [DD/MM/YYYY] | DESCRIPTION: [Transaction description] | DEBIT: [Amount withdrawn] | CREDIT: [Amount deposited] | BALANCE: [Balance after transaction] | HIGHLIGHTED: [Yes/No]

HIGHLIGHTED_TRANSACTIONS:
[List only transactions that appear highlighted or marked]

WORK_EXPENSE_ASSESSMENT:
- Analyze each highlighted transaction for work-related expense potential
- Identify merchant categories (fuel, office supplies, travel, etc.)
- Assess ATO deductibility likelihood
- Note any patterns in highlighted transactions

Focus on transactions that appear highlighted or marked by the user, as these are likely work-related expenses the taxpayer wants to claim."""
        
        else:
            return f"""<image>
Analyze this Australian bank statement for work-related expense transactions.

{highlight_context}

Extract information in this KEY-VALUE format:

BANK: [Financial institution name]
ACCOUNT_HOLDER: [Account holder name]  
ACCOUNT_NUMBER: [Account number - mask middle digits]
BSB: [Bank State Branch code if visible]
STATEMENT_PERIOD: [Start date - End date in DD/MM/YYYY format]
OPENING_BALANCE: [Starting balance]
CLOSING_BALANCE: [Ending balance]

TRANSACTIONS:
DATE: [DD/MM/YYYY] | DESCRIPTION: [Transaction description] | DEBIT: [Amount withdrawn] | CREDIT: [Amount deposited] | BALANCE: [Balance after transaction] | WORK_RELEVANCE: [High/Medium/Low/None]

For work relevance, consider:
- Fuel purchases (BP, Shell, Caltex, etc.)
- Office supplies (Officeworks, Bunnings, etc.) 
- Professional services (accounting, legal, consulting)
- Travel expenses (hotels, flights, car rental)
- Equipment purchases for work
- Training and education expenses

Extract ALL visible transactions in chronological order."""
    
    def _parse_bank_statement_response(self, response: str) -> Dict[str, Any]:
        """Parse InternVL response into structured bank statement data."""
        
        # Initialize data structure
        data = {
            'bank_name': '',
            'account_holder': '',
            'account_number': '',
            'bsb': '',
            'statement_period': '',
            'opening_balance': '',
            'closing_balance': '',
            'transactions': []
        }
        
        try:
            lines = response.strip().split('\n')
            current_section = None
            logger.debug(f"Parsing {len(lines)} lines from response")
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                logger.debug(f"Line {i}: [{current_section}] {line[:100]}...")
                
                # Parse metadata fields
                if line.startswith('BANK:'):
                    data['bank_name'] = line.replace('BANK:', '').strip()
                elif line.startswith('ACCOUNT_HOLDER:'):
                    data['account_holder'] = line.replace('ACCOUNT_HOLDER:', '').strip()
                elif line.startswith('ACCOUNT_NUMBER:'):
                    data['account_number'] = line.replace('ACCOUNT_NUMBER:', '').strip()
                elif line.startswith('BSB:'):
                    data['bsb'] = line.replace('BSB:', '').strip()
                elif line.startswith('STATEMENT_PERIOD:'):
                    data['statement_period'] = line.replace('STATEMENT_PERIOD:', '').strip()
                elif line.startswith('OPENING_BALANCE:'):
                    data['opening_balance'] = line.replace('OPENING_BALANCE:', '').strip()
                elif line.startswith('CLOSING_BALANCE:'):
                    data['closing_balance'] = line.replace('CLOSING_BALANCE:', '').strip()
                
                # Identify sections (handle both formats)
                elif line.startswith('TRANSACTIONS:') or line.startswith('**TRANSACTIONS:**'):
                    current_section = 'transactions'
                elif line.startswith('HIGHLIGHTED_TRANSACTIONS:') or line.startswith('**HIGHLIGHTED_TRANSACTIONS:**'):
                    current_section = 'highlighted'
                elif line.startswith('WORK_EXPENSE_ASSESSMENT:') or line.startswith('**WORK_EXPENSE_ASSESSMENT:**'):
                    current_section = 'assessment'
                
                # Parse transaction lines
                elif current_section == 'transactions' and '|' in line:
                    transaction = self._parse_transaction_line(line)
                    if transaction:
                        data['transactions'].append(transaction)
            
            logger.info(f"Parsed {len(data['transactions'])} transactions from bank statement")
            return data
            
        except Exception as e:
            logger.error(f"Failed to parse bank statement response: {e}")
            return data
    
    def _parse_transaction_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single transaction line from the response."""
        
        try:
            # Handle both formats:
            # 1. Expected: DATE: value | DESCRIPTION: value | DEBIT: value | etc.
            # 2. Actual: - DD/MM/YYYY | DESCRIPTION | AMOUNT | BALANCE | [Highlighted]
            
            transaction = {}
            
            # Check if line starts with dash (actual format)
            if line.strip().startswith('-'):
                # Parse actual format: - 26/04/2016 | INTEREST | 2,534.19 | 640,944.03 | [Highlighted]
                line = line.strip()[1:].strip()  # Remove leading dash
                parts = [part.strip() for part in line.split('|')]
                
                if len(parts) >= 3:
                    # Date is first part
                    transaction['transaction_date'] = parts[0].strip()
                    
                    # Description is second part
                    transaction['description'] = parts[1].strip()
                    
                    # Parse amounts - could be debit or credit
                    if len(parts) >= 4:
                        amount1 = parts[2].strip()
                        balance = parts[3].strip()
                        
                        # If amount1 is not empty, it's either debit or credit
                        if amount1 and amount1 != '':
                            # Check if it's a debit (loan payment, interest) or credit
                            desc_lower = transaction['description'].lower()
                            if any(word in desc_lower for word in ['payment', 'interest', 'debit', 'withdrawal']):
                                transaction['debit_amount'] = amount1
                            else:
                                transaction['credit_amount'] = amount1
                        
                        transaction['balance'] = balance
                    
                    # Check for highlighting in last part
                    if len(parts) >= 5:
                        highlight_part = parts[4].strip()
                        transaction['highlight_detected'] = '[highlighted]' in highlight_part.lower()
                    elif '[highlighted]' in line.lower():
                        transaction['highlight_detected'] = True
                        
            else:
                # Parse expected format: DATE: value | DESCRIPTION: value | etc.
                parts = [part.strip() for part in line.split('|')]
                
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key == 'date':
                            transaction['transaction_date'] = value
                        elif key == 'description':
                            transaction['description'] = value
                        elif key == 'debit':
                            if value and value != '-' and value.lower() != 'none':
                                transaction['debit_amount'] = value
                        elif key == 'credit':
                            if value and value != '-' and value.lower() != 'none':
                                transaction['credit_amount'] = value
                        elif key == 'balance':
                            transaction['balance'] = value
                        elif key == 'highlighted':
                            transaction['highlight_detected'] = value.lower() in ['yes', 'true', '1']
                        elif key == 'work_relevance':
                            # Convert work relevance to likelihood score
                            relevance_map = {
                                'high': 0.9,
                                'medium': 0.6, 
                                'low': 0.3,
                                'none': 0.0
                            }
                            transaction['work_related_likelihood'] = relevance_map.get(value.lower(), 0.0)
            
            # Validate required fields
            if transaction.get('transaction_date') and transaction.get('description'):
                return transaction
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Failed to parse transaction line: {line} - {e}")
            return None
    
    def _enhance_with_highlights(
        self, 
        parsed_data: Dict[str, Any], 
        highlight_regions: List[HighlightRegion]
    ) -> Dict[str, Any]:
        """Enhance parsed data with highlight detection information."""
        
        if not highlight_regions or not parsed_data.get('transactions'):
            return parsed_data
        
        try:
            # Mark transactions that have associated highlights
            # This is a simplified approach - in practice, would need spatial correlation
            for transaction in parsed_data['transactions']:
                # Check if transaction description matches any extracted highlight text
                description = transaction.get('description', '').lower()
                
                for region in highlight_regions:
                    if region.extracted_text:
                        highlight_text = region.extracted_text.lower()
                        # Simple text matching - could be improved with fuzzy matching
                        if any(word in highlight_text for word in description.split() if len(word) > 3):
                            transaction['highlight_detected'] = True
                            transaction['work_related_likelihood'] = max(
                                transaction.get('work_related_likelihood', 0.5), 0.8
                            )
                            break
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Failed to enhance data with highlights: {e}")
            return parsed_data
    
    def _categorize_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Categorize transactions based on work expense keywords."""
        
        for transaction in transactions:
            description = transaction.get('description', '').lower()
            
            # Find best matching category
            best_category = None
            best_score = 0
            
            for category, keywords in self.work_expense_keywords.items():
                score = sum(1 for keyword in keywords if keyword in description)
                if score > best_score:
                    best_score = score
                    best_category = category
            
            if best_category and best_score > 0:
                transaction['expense_category'] = best_category
                # Increase work likelihood if categorized
                current_likelihood = transaction.get('work_related_likelihood', 0.0)
                transaction['work_related_likelihood'] = min(current_likelihood + 0.3, 1.0)
        
        return transactions
    
    def _assess_ato_compliance(self, bank_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess ATO compliance for bank statement transactions."""
        
        assessment = {
            'overall_compliance': 0.0,
            'compliant_transactions': 0,
            'total_transactions': len(bank_data.get('transactions', [])),
            'compliance_issues': [],
            'recommendations': [],
            'work_expense_total': 0.0,
            'highlighted_expense_total': 0.0,
            'ato_ready_transactions': 0
        }
        
        if not bank_data.get('transactions'):
            assessment['compliance_issues'].append('No transactions found in statement')
            return assessment
        
        total_work_expenses = 0.0
        total_highlighted_expenses = 0.0
        
        # Assess each transaction for ATO requirements
        for transaction in bank_data['transactions']:
            compliance_score = self._assess_transaction_compliance(transaction)
            
            if compliance_score >= 0.8:
                assessment['compliant_transactions'] += 1
            
            # Calculate work expense totals
            if transaction.get('work_related_likelihood', 0) > 0.5:
                amount = self._extract_amount(transaction.get('debit_amount', '0'))
                total_work_expenses += amount
                
                if compliance_score >= 0.8:
                    assessment['ato_ready_transactions'] += 1
            
            # Calculate highlighted expense totals
            if transaction.get('highlight_detected', False):
                amount = self._extract_amount(transaction.get('debit_amount', '0'))
                total_highlighted_expenses += amount
        
        # Calculate overall compliance
        if assessment['total_transactions'] > 0:
            assessment['overall_compliance'] = (
                assessment['compliant_transactions'] / assessment['total_transactions'] * 100
            )
        
        assessment['work_expense_total'] = total_work_expenses
        assessment['highlighted_expense_total'] = total_highlighted_expenses
        
        # Generate recommendations
        if assessment['overall_compliance'] < 80:
            assessment['recommendations'].append(
                'Consider adding more transaction details for better ATO compliance'
            )
        
        if total_work_expenses > self.ato_requirements['max_claim_without_receipt']:
            assessment['recommendations'].append(
                f'Work expenses total ${total_work_expenses:.2f} - ensure receipts available for amounts over $82.50'
            )
        
        # Check for BSB validation
        bsb = bank_data.get('bsb', '')
        if bsb and not validate_australian_bsb(bsb):
            assessment['compliance_issues'].append(f'Invalid or unrecognized BSB: {bsb}')
        
        return assessment
    
    def _assess_transaction_compliance(self, transaction: Dict[str, Any]) -> float:
        """Assess ATO compliance for a single transaction."""
        
        score = 0.0
        total_checks = len(self.ato_requirements['required_fields'])
        
        # Check required fields
        for field in self.ato_requirements['required_fields']:
            if transaction.get(field):
                score += 1.0
        
        # Bonus for preferred fields
        bonus_checks = len(self.ato_requirements['preferred_fields'])
        if bonus_checks > 0:
            for field in self.ato_requirements['preferred_fields']:
                if transaction.get(field):
                    score += 0.5
            total_checks += bonus_checks
        
        # Date format validation
        date_value = transaction.get('transaction_date', '')
        if date_value and re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_value):
            score += 0.5
            total_checks += 0.5
        
        return score / total_checks if total_checks > 0 else 0.0
    
    def _extract_amount(self, amount_str: str) -> float:
        """Extract numeric amount from string."""
        if not amount_str:
            return 0.0
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[^\d.,\-]', '', str(amount_str).strip())
        
        try:
            return float(cleaned.replace(',', ''))
        except (ValueError, AttributeError):
            return 0.0
    
    def _extract_highlighted_transactions(self, bank_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract transactions that were highlighted by the user."""
        
        return [
            t for t in bank_data.get('transactions', [])
            if t.get('highlight_detected', False)
        ]


def extract_bank_statement_with_highlights(
    image_path: str, 
    model, 
    tokenizer, 
    detect_highlights: bool = True,
    prompt_name: str = 'bank_statement_ato_compliance_prompt'
) -> Dict[str, Any]:
    """
    Convenience function for bank statement extraction with highlight detection.
    
    Args:
        image_path: Path to bank statement image
        model: InternVL model
        tokenizer: Model tokenizer
        detect_highlights: Whether to detect highlighted regions
        prompt_name: Prompt template to use
        
    Returns:
        Complete bank statement analysis with ATO compliance
    """
    processor = BankStatementProcessor()
    return processor.process_bank_statement(
        image_path=image_path,
        model=model,
        tokenizer=tokenizer,
        use_highlight_detection=detect_highlights,
        prompt_name=prompt_name
    )