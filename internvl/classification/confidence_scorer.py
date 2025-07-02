"""
Confidence Scoring for Document Classification.

This module provides confidence assessment for document classification results,
ensuring high-quality classification decisions with clear failure modes.
"""

import logging
import re
from typing import Dict, List, Tuple

from .document_types import DocumentType, get_classification_keywords

logger = logging.getLogger(__name__)


class ConfidenceLevel:
    """Confidence level thresholds for classification decisions."""
    
    HIGH = 0.8      # Definitive classification - proceed with processing
    MEDIUM = 0.6    # Reasonable confidence - may proceed with caution
    LOW = 0.4       # Uncertain classification - should request manual review
    FAIL = 0.4      # Below this threshold = classification failure


class ClassificationResult:
    """Result of document classification with confidence assessment."""
    
    def __init__(
        self,
        document_type: DocumentType,
        confidence: float,
        classification_reasoning: str,
        processing_prompt: str,
        secondary_type: DocumentType = None,
        evidence: Dict = None
    ):
        self.document_type = document_type
        self.confidence = confidence
        self.classification_reasoning = classification_reasoning
        self.processing_prompt = processing_prompt
        self.secondary_type = secondary_type
        self.evidence = evidence or {}
        
    @property
    def confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.confidence >= ConfidenceLevel.HIGH:
            return "High"
        elif self.confidence >= ConfidenceLevel.MEDIUM:
            return "Medium"
        elif self.confidence >= ConfidenceLevel.LOW:
            return "Low"
        else:
            return "Failed"
    
    @property
    def is_definitive(self) -> bool:
        """Check if classification is definitive enough for processing."""
        return self.confidence >= ConfidenceLevel.HIGH
    
    def to_dict(self) -> Dict:
        """Convert classification result to dictionary."""
        return {
            "document_type": self.document_type.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "classification_reasoning": self.classification_reasoning,
            "processing_prompt": self.processing_prompt,
            "secondary_type": self.secondary_type.value if self.secondary_type else None,
            "evidence": self.evidence,
            "is_definitive": self.is_definitive
        }


class ClassificationFailedException(Exception):
    """Exception raised when document classification fails or is uncertain."""
    
    def __init__(self, message: str, confidence: float = 0.0, evidence: Dict = None):
        super().__init__(message)
        self.confidence = confidence
        self.evidence = evidence or {}


class ConfidenceScorer:
    """Confidence scoring for document classification based on keyword matching and evidence."""
    
    def __init__(self):
        self.keyword_weights = {
            'exact_match': 1.0,      # Exact business name match
            'strong_indicator': 0.8,  # Strong industry indicators
            'medium_indicator': 0.5,  # Medium confidence indicators
            'weak_indicator': 0.3,    # Weak indicators
            'format_match': 0.4,      # Document format indicators
            'logo_match': 0.6         # Logo or branding detection
        }
    
    def score_classification(
        self, 
        response_text: str, 
        candidate_type: DocumentType,
        secondary_candidates: List[DocumentType] = None
    ) -> Tuple[float, Dict]:
        """
        Score classification confidence based on response analysis.
        
        Args:
            response_text: Raw response from classification model
            candidate_type: Primary document type candidate
            secondary_candidates: Alternative document types
            
        Returns:
            Tuple of (confidence_score, evidence_dict)
        """
        
        evidence = {
            'keyword_matches': [],
            'confidence_indicators': [],
            'format_indicators': [],
            'negative_indicators': [],
            'competing_types': []
        }
        
        response_lower = response_text.lower()
        total_score = 0.0
        max_possible_score = 0.0
        
        # Get keywords for primary candidate
        primary_keywords = get_classification_keywords(candidate_type)
        
        # Score keyword matches for primary type
        keyword_score, keyword_evidence = self._score_keyword_matches(
            response_lower, primary_keywords, candidate_type
        )
        total_score += keyword_score
        max_possible_score += len(primary_keywords) * self.keyword_weights['strong_indicator']
        evidence['keyword_matches'].extend(keyword_evidence)
        
        # Check for competing types
        if secondary_candidates:
            for secondary_type in secondary_candidates:
                secondary_keywords = get_classification_keywords(secondary_type)
                competing_score, competing_evidence = self._score_keyword_matches(
                    response_lower, secondary_keywords, secondary_type
                )
                
                if competing_score > keyword_score * 0.7:  # Significant competition
                    evidence['competing_types'].append({
                        'type': secondary_type.value,
                        'score': competing_score,
                        'evidence': competing_evidence
                    })
                    # Reduce confidence if there's strong competition
                    total_score *= 0.8
        
        # Score confidence indicators from response
        confidence_indicators = self._extract_confidence_indicators(response_text)
        evidence['confidence_indicators'] = confidence_indicators
        
        # Add confidence bonus/penalty
        confidence_bonus = self._calculate_confidence_bonus(confidence_indicators)
        total_score += confidence_bonus
        max_possible_score += 1.0  # Max confidence bonus
        
        # Score format indicators
        format_score, format_evidence = self._score_format_indicators(
            response_text, candidate_type
        )
        total_score += format_score
        max_possible_score += 1.0  # Max format score
        evidence['format_indicators'] = format_evidence
        
        # Calculate final confidence (0.0 to 1.0)
        if max_possible_score > 0:
            confidence = min(total_score / max_possible_score, 1.0)
        else:
            confidence = 0.0
        
        # Apply penalty for negative indicators
        negative_penalty = self._calculate_negative_penalty(response_text, candidate_type)
        confidence *= (1.0 - negative_penalty)
        evidence['negative_indicators'] = self._get_negative_indicators(response_text, candidate_type)
        
        logger.debug(f"Classification confidence for {candidate_type.value}: {confidence:.3f}")
        logger.debug(f"Evidence: {evidence}")
        
        return confidence, evidence
    
    def _score_keyword_matches(
        self, 
        text: str, 
        keywords: List[str], 
        document_type: DocumentType  # noqa: ARG002
    ) -> Tuple[float, List[Dict]]:
        """Score keyword matches in text."""
        
        score = 0.0
        evidence = []
        
        for keyword in keywords:
            if keyword.lower() in text:
                # Weight based on keyword specificity
                if len(keyword) > 10:  # Specific business names
                    weight = self.keyword_weights['exact_match']
                elif len(keyword) > 6:   # Industry terms
                    weight = self.keyword_weights['strong_indicator']
                else:                    # General terms
                    weight = self.keyword_weights['medium_indicator']
                
                score += weight
                evidence.append({
                    'keyword': keyword,
                    'weight': weight,
                    'type': 'keyword_match'
                })
        
        return score, evidence
    
    def _extract_confidence_indicators(self, response_text: str) -> List[str]:
        """Extract confidence indicators from classification response."""
        
        indicators = []
        
        # Look for explicit confidence statements
        confidence_patterns = [
            r'confidence:\s*(high|medium|low)',
            r'(high|medium|low)\s*confidence',
            r'(definitely|clearly|obviously|certainly)',
            r'(probably|likely|appears to be|seems to be)',
            r'(unclear|uncertain|difficult to determine|hard to say)'
        ]
        
        for pattern in confidence_patterns:
            matches = re.findall(pattern, response_text.lower())
            indicators.extend(matches)
        
        return indicators
    
    def _calculate_confidence_bonus(self, indicators: List[str]) -> float:
        """Calculate confidence bonus based on indicators."""
        
        bonus = 0.0
        
        for indicator in indicators:
            if indicator in ['high', 'definitely', 'clearly', 'obviously', 'certainly']:
                bonus += 0.3
            elif indicator in ['medium', 'probably', 'likely']:
                bonus += 0.1
            elif indicator in ['low', 'unclear', 'uncertain', 'difficult', 'hard']:
                bonus -= 0.2
        
        return max(-0.5, min(0.5, bonus))  # Cap bonus/penalty
    
    def _score_format_indicators(
        self, 
        response_text: str, 
        document_type: DocumentType
    ) -> Tuple[float, List[str]]:
        """Score document format indicators."""
        
        score = 0.0
        indicators = []
        
        # Format patterns by document type
        format_patterns = {
            DocumentType.TAX_INVOICE: [
                r'tax invoice', r'gst invoice', r'invoice number', r'abn'
            ],
            DocumentType.BANK_STATEMENT: [
                r'account statement', r'transaction history', r'bsb', r'account number'
            ],
            DocumentType.FUEL_RECEIPT: [
                r'fuel', r'petrol', r'diesel', r'litres?', r'pump'
            ],
            DocumentType.BUSINESS_RECEIPT: [
                r'receipt', r'purchase', r'total', r'gst'
            ]
        }
        
        patterns = format_patterns.get(document_type, [])
        
        for pattern in patterns:
            if re.search(pattern, response_text.lower()):
                score += self.keyword_weights['format_match']
                indicators.append(f"Format indicator: {pattern}")
        
        return min(score, 1.0), indicators
    
    def _calculate_negative_penalty(self, response_text: str, document_type: DocumentType) -> float:  # noqa: ARG002
        """Calculate penalty for negative indicators."""
        
        penalty = 0.0
        
        # Negative indicators that reduce confidence
        negative_patterns = [
            r'not (clear|visible|readable)',
            r'cannot (determine|identify|read)',
            r'poor quality',
            r'blurred',
            r'unclear',
            r'difficult to read'
        ]
        
        for pattern in negative_patterns:
            if re.search(pattern, response_text.lower()):
                penalty += 0.1
        
        return min(penalty, 0.5)  # Cap penalty at 50%
    
    def _get_negative_indicators(self, response_text: str, document_type: DocumentType) -> List[str]:  # noqa: ARG002
        """Get list of negative indicators found."""
        
        indicators = []
        negative_patterns = [
            'not clear', 'not visible', 'cannot determine',
            'poor quality', 'blurred', 'unclear'
        ]
        
        for pattern in negative_patterns:
            if pattern in response_text.lower():
                indicators.append(pattern)
        
        return indicators


def create_classification_result(
    document_type: DocumentType,
    confidence: float,
    reasoning: str,
    prompt_name: str,
    secondary_type: DocumentType = None,
    evidence: Dict = None
) -> ClassificationResult:
    """Create a classification result with proper confidence assessment."""
    
    return ClassificationResult(
        document_type=document_type,
        confidence=confidence,
        classification_reasoning=reasoning,
        processing_prompt=prompt_name,
        secondary_type=secondary_type,
        evidence=evidence
    )