from typing import Any, Dict, Set
import difflib
from llm_ocr.evaluators.metrics.base import BaseMetric


class ErrorAnalysisMetric(BaseMetric):
    """Detailed error analysis between ground truth and extracted text."""
    
    def __init__(self, special_chars: Set[str]):
        """
        Initialize with set of special characters.
        
        Args:
            special_chars: Set of special characters to check for errors
        """
        self.special_chars = special_chars
    
    @property
    def name(self) -> str:
        return "error_analysis"
    
    def evaluate(self, ground_truth: str, extracted: str) -> Dict[str, Any]:
        """
        Analyze differences and errors between ground truth and extracted text.
        
        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR
            
        Returns:
            Dictionary with detailed error analysis
        """
        analysis = {
            "substitutions": [],
            "deletions": [],
            "insertions": [],
            "special_char_errors": [],
            "total_errors": 0
        }
    
        # Use difflib for detailed comparison
        matcher = difflib.SequenceMatcher(None, ground_truth, extracted)
    
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Character substitution
                gt_segment = ground_truth[i1:i2]
                ext_segment = extracted[j1:j2]
            
                analysis["substitutions"].append({
                    'ground_truth': gt_segment,
                    'extracted': ext_segment,
                    'position': i1
                })
            
                # Check if special characters were affected
                gt_special_chars = set(c for c in gt_segment if c in self.special_chars)
                ext_special_chars = set(c for c in ext_segment if c in self.special_chars)
                if gt_special_chars or ext_special_chars:
                    analysis["special_char_errors"].append({
                        'ground_truth': gt_segment,
                        'extracted': ext_segment,
                        'special_chars_ground_truth': list(gt_special_chars),
                        'special_chars_extracted': list(ext_special_chars),
                        'position': i1
                    })
                
            elif tag == 'delete':
                # Character deletion
                analysis["deletions"].append({
                    'text': ground_truth[i1:i2],
                    'position': i1
                })
            
            elif tag == 'insert':
                # Character insertion
                analysis["insertions"].append({
                    'text': extracted[j1:j2],
                    'position': i1
                })
    
        # Calculate total errors
        analysis["total_errors"] = (
            len(analysis["substitutions"]) +
            len(analysis["deletions"]) +
            len(analysis["insertions"])
        )
    
        return analysis

