from typing import Set, Dict
from collections import Counter
from llm_ocr.evaluators.metrics.base import BaseMetric


class OldCharPreservationMetric(BaseMetric):
    """
    Comprehensive metric for preservation of old characters that considers
    both frequency and sequence of characters.
    """
    
    def __init__(self, special_chars: Set[str], frequency_weight: float = 0.6, sequence_weight: float = 0.4):
        """
        Initialize with set of special characters.
        
        Args:
            special_chars: Set of special characters to check for preservation
            frequency_weight: Weight for the frequency-based score (default: 0.6)
            sequence_weight: Weight for the sequence-based score (default: 0.4)
        """
        self.special_chars = special_chars
        self.frequency_weight = frequency_weight
        self.sequence_weight = sequence_weight
        self.detailed_report = {}
    
    @property
    def name(self) -> str:
        return "comprehensive_old_char_preservation"
    
    def extract_special_char_sequence(self, text: str) -> str:
        """
        Extract only the special characters from the text, preserving their order.
        """
        return ''.join(c.lower() for c in text if c.lower() in self.special_chars)
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    
    def calculate_frequency_score(self, ground_truth: str, extracted: str) -> float:
        """Calculate the frequency-based preservation score."""
        # Count occurrences of each special character
        gt_counter = Counter(c.lower() for c in ground_truth if c.lower() in self.special_chars)
        ex_counter = Counter(c.lower() for c in extracted if c.lower() in self.special_chars)
        
        # If no special characters in ground truth, return perfect score
        if sum(gt_counter.values()) == 0:
            return 1.0
        
        # Calculate preservation ratio (considering frequency)
        total_gt_chars = sum(gt_counter.values())
        preserved_chars = 0
        
        for char, gt_count in gt_counter.items():
            ex_count = ex_counter.get(char, 0)
            preserved_count = min(gt_count, ex_count)
            preserved_chars += preserved_count
        
        return preserved_chars / total_gt_chars
    
    def calculate_sequence_score(self, ground_truth: str, extracted: str) -> float:
        """Calculate the sequence-based preservation score."""
        # Extract special character sequences
        gt_sequence = self.extract_special_char_sequence(ground_truth)
        ex_sequence = self.extract_special_char_sequence(extracted)
        
        # If no special characters in ground truth, return perfect score
        if not gt_sequence:
            return 1.0
        
        # Calculate Levenshtein distance
        distance = self.levenshtein_distance(gt_sequence, ex_sequence)
        
        # Calculate preservation ratio
        max_possible_distance = max(len(gt_sequence), len(ex_sequence))
        return 1.0 - (distance / max_possible_distance) if max_possible_distance > 0 else 1.0
    
    def evaluate(self, ground_truth: str, extracted: str) -> float:
        """Calculate combined preservation score."""
        # Calculate individual scores
        frequency_score = self.calculate_frequency_score(ground_truth, extracted)
        sequence_score = self.calculate_sequence_score(ground_truth, extracted)
        
        # Calculate weighted combined score
        combined_score = (
            self.frequency_weight * frequency_score + 
            self.sequence_weight * sequence_score
        )
        
        # Prepare detailed report
        self.detailed_report = {
            "frequency_score": frequency_score,
            "sequence_score": sequence_score,
            "combined_score": combined_score,
            "weights": {
                "frequency": self.frequency_weight,
                "sequence": self.sequence_weight
            }
        }
        
        return combined_score
    
    def get_detailed_report(self) -> Dict:
        """Get detailed analysis report of character preservation."""
        return self.detailed_report