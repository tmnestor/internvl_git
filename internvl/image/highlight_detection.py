"""
Highlight Detection for Bank Statement Processing.

This module provides computer vision capabilities to detect highlighted regions
in bank statement images, supporting various highlight colors and extraction
of text from highlighted areas for ATO work expense identification.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from internvl.schemas.bank_statement_schemas import HighlightRegion

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - highlight detection disabled")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pytesseract/PIL not available - text extraction from highlights disabled")


class BankStatementHighlightDetector:
    """Detect highlighted regions in bank statement images."""
    
    def __init__(self):
        """Initialize highlight detector with color ranges and parameters."""
        # HSV color ranges for different highlight colors
        if CV2_AVAILABLE:
            self.color_ranges = {
                'yellow': {
                    'lower': np.array([15, 50, 50]),
                    'upper': np.array([35, 255, 255]),
                    'color_name': 'yellow'
                },
                'pink': {
                    'lower': np.array([160, 50, 50]),
                    'upper': np.array([180, 255, 255]),
                    'color_name': 'pink'
                },
                'green': {
                    'lower': np.array([35, 50, 50]),
                    'upper': np.array([85, 255, 255]),
                    'color_name': 'green'
                },
                'orange': {
                    'lower': np.array([5, 50, 50]),
                    'upper': np.array([15, 255, 255]),
                    'color_name': 'orange'
                }
            }
        else:
            self.color_ranges = {}
        
        # Detection parameters
        self.min_highlight_area = 100  # Minimum pixel area for valid highlight
        self.max_highlight_area = 50000  # Maximum pixel area (avoid full page highlights)
        self.confidence_threshold = 0.6
        self.morphology_kernel_size = 3
        self.overlap_threshold = 0.3
    
    def is_available(self) -> bool:
        """Check if highlight detection is available (requires OpenCV)."""
        return CV2_AVAILABLE
    
    def detect_highlights(self, image_path: str) -> List[HighlightRegion]:
        """
        Detect highlighted regions in bank statement image.
        
        Args:
            image_path: Path to bank statement image
            
        Returns:
            List of detected highlight regions
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - returning empty highlight list")
            return []
            
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            detected_regions = []
            
            # Detect each highlight color
            for color_name, color_config in self.color_ranges.items():
                if color_config['lower'] is not None and color_config['upper'] is not None:
                    regions = self._detect_color_highlights(
                        hsv, image, color_config, color_name
                    )
                    detected_regions.extend(regions)
            
            # Filter and validate regions
            valid_regions = self._filter_valid_highlights(detected_regions)
            
            logger.info(f"Detected {len(valid_regions)} highlight regions in {Path(image_path).name}")
            return valid_regions
            
        except Exception as e:
            logger.error(f"Highlight detection failed for {image_path}: {e}")
            return []
    
    def _detect_color_highlights(
        self, 
        hsv: np.ndarray, 
        _original: np.ndarray, 
        color_config: Dict[str, Any], 
        color_name: str
    ) -> List[HighlightRegion]:
        """Detect highlights of specific color."""
        
        # Create color mask
        mask = cv2.inRange(hsv, color_config['lower'], color_config['upper'])
        
        # Morphological operations to clean up mask
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_highlight_area <= area <= self.max_highlight_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area and shape regularity
                aspect_ratio = w / h if h > 0 else 0
                shape_score = min(aspect_ratio, 1/aspect_ratio) if aspect_ratio > 0 else 0
                area_score = min(area / 1000.0, 1.0)  # Normalize by expected highlight size
                confidence = (shape_score + area_score) / 2.0
                
                if confidence >= self.confidence_threshold:
                    regions.append(HighlightRegion(
                        x=x, y=y, width=w, height=h,
                        color=color_name,
                        confidence=confidence
                    ))
        
        return regions
    
    def _filter_valid_highlights(self, regions: List[HighlightRegion]) -> List[HighlightRegion]:
        """Filter overlapping and invalid highlight regions."""
        
        if not regions:
            return []
        
        # Sort by confidence (highest first)
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        # Remove overlapping regions (keep highest confidence)
        filtered = []
        for region in regions:
            is_overlap = False
            for existing in filtered:
                if region.overlaps_with(existing, self.overlap_threshold):
                    is_overlap = True
                    break
            
            if not is_overlap:
                filtered.append(region)
        
        return filtered
    
    def extract_text_from_highlights(
        self, 
        image_path: str, 
        regions: List[HighlightRegion]
    ) -> List[HighlightRegion]:
        """
        Extract text from highlighted regions using OCR.
        
        Args:
            image_path: Path to source image
            regions: List of highlight regions to process
            
        Returns:
            Updated regions with extracted text
        """
        if not OCR_AVAILABLE:
            logger.warning("OCR not available - skipping text extraction from highlights")
            return regions
        
        try:
            image = Image.open(image_path)
            
            for region in regions:
                # Crop highlighted region with small padding
                padding = 5
                x1 = max(0, region.x - padding)
                y1 = max(0, region.y - padding)
                x2 = min(image.width, region.x + region.width + padding)
                y2 = min(image.height, region.y + region.height + padding)
                
                cropped = image.crop((x1, y1, x2, y2))
                
                # Extract text with OCR optimized for bank statements
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$.,/- '
                text = pytesseract.image_to_string(cropped, config=custom_config)
                region.extracted_text = text.strip()
                
                logger.debug(f"Extracted text from {region.color} highlight: '{region.extracted_text[:50]}...'")
                
            return regions
            
        except Exception as e:
            logger.error(f"Text extraction from highlights failed: {e}")
            return regions
    
    def get_highlight_summary(self, regions: List[HighlightRegion]) -> Dict[str, Any]:
        """
        Generate summary statistics for detected highlights.
        
        Args:
            regions: List of detected highlight regions
            
        Returns:
            Summary dictionary with statistics
        """
        if not regions:
            return {
                'total_highlights': 0,
                'colors_detected': [],
                'average_confidence': 0.0,
                'total_area': 0,
                'has_text_extraction': False
            }
        
        color_counts = {}
        total_area = 0
        total_confidence = 0
        has_extracted_text = False
        
        for region in regions:
            # Count colors
            color_counts[region.color] = color_counts.get(region.color, 0) + 1
            
            # Sum metrics
            total_area += region.area()
            total_confidence += region.confidence
            
            # Check for text extraction
            if region.extracted_text:
                has_extracted_text = True
        
        return {
            'total_highlights': len(regions),
            'colors_detected': list(color_counts.keys()),
            'color_distribution': color_counts,
            'average_confidence': total_confidence / len(regions),
            'total_area': total_area,
            'has_text_extraction': has_extracted_text,
            'top_confidence': max(r.confidence for r in regions),
            'extraction_texts': [r.extracted_text for r in regions if r.extracted_text]
        }
    
    def visualize_highlights(
        self, 
        image_path: str, 
        regions: List[HighlightRegion], 
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create visualization image with highlighted regions marked.
        
        Args:
            image_path: Path to source image
            regions: List of highlight regions to visualize
            output_path: Optional path to save visualization
            
        Returns:
            Path to saved visualization or None if failed
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - cannot create visualization")
            return None
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Define colors for visualization
            viz_colors = {
                'yellow': (0, 255, 255),    # Yellow in BGR
                'pink': (255, 0, 255),      # Magenta in BGR  
                'green': (0, 255, 0),       # Green in BGR
                'orange': (0, 165, 255),    # Orange in BGR
                'default': (255, 255, 255)  # White in BGR
            }
            
            # Draw rectangles around highlights
            for region in regions:
                color = viz_colors.get(region.color, viz_colors['default'])
                
                # Draw rectangle
                cv2.rectangle(
                    image, 
                    (region.x, region.y), 
                    (region.x + region.width, region.y + region.height),
                    color, 
                    3
                )
                
                # Add label
                label = f"{region.color} ({region.confidence:.2f})"
                cv2.putText(
                    image, 
                    label,
                    (region.x, region.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Save visualization
            if output_path:
                output_file = output_path
            else:
                input_path = Path(image_path)
                output_file = str(input_path.parent / f"{input_path.stem}_highlights{input_path.suffix}")
            
            cv2.imwrite(output_file, image)
            logger.info(f"Highlight visualization saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to create highlight visualization: {e}")
            return None


def detect_bank_statement_highlights(
    image_path: str,
    extract_text: bool = True,
    create_visualization: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for bank statement highlight detection.
    
    Args:
        image_path: Path to bank statement image
        extract_text: Whether to extract text from highlighted regions
        create_visualization: Whether to create visualization image
        output_dir: Directory to save outputs
        
    Returns:
        Complete highlight detection results
    """
    detector = BankStatementHighlightDetector()
    
    if not detector.is_available():
        return {
            'success': False,
            'error': 'OpenCV not available for highlight detection',
            'highlights': [],
            'summary': {}
        }
    
    try:
        # Detect highlights
        highlights = detector.detect_highlights(image_path)
        
        # Extract text if requested
        if extract_text and highlights:
            highlights = detector.extract_text_from_highlights(image_path, highlights)
        
        # Create visualization if requested
        visualization_path = None
        if create_visualization and highlights:
            if output_dir:
                viz_path = str(Path(output_dir) / f"{Path(image_path).stem}_highlights.png")
            else:
                viz_path = None
            visualization_path = detector.visualize_highlights(image_path, highlights, viz_path)
        
        # Generate summary
        summary = detector.get_highlight_summary(highlights)
        
        return {
            'success': True,
            'highlights': highlights,
            'summary': summary,
            'visualization_path': visualization_path,
            'processing_metadata': {
                'image_path': image_path,
                'text_extraction_enabled': extract_text,
                'visualization_created': visualization_path is not None
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'highlights': [],
            'summary': {}
        }