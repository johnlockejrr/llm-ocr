from collections import Counter
from typing import Set

from llm_ocr.evaluators.metrics.base import BaseMetric


class OldCharPreservationMetric(BaseMetric):
    """
    Simplified metric for preservation of old characters using frequency-based calculation.
    """

    def __init__(self, special_chars: Set[str]):
        """
        Initialize with set of special characters.

        Args:
            special_chars: Set of special characters to check for preservation
        """
        self.special_chars = special_chars

    @property
    def name(self) -> str:
        return "old_char_preservation"

    def evaluate(self, ground_truth: str, extracted: str) -> float:
        """
        Calculate old character preservation score based on frequency matching.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Preservation score between 0.0 and 1.0
        """
        # Count occurrences of each special character
        gt_counter = Counter(c.lower() for c in ground_truth if c.lower() in self.special_chars)
        ex_counter = Counter(c.lower() for c in extracted if c.lower() in self.special_chars)

        # If no special characters in ground truth, return perfect score
        total_gt_chars = sum(gt_counter.values())
        if total_gt_chars == 0:
            return 1.0

        # Calculate preservation ratio (considering frequency)
        preserved_chars = 0
        for char, gt_count in gt_counter.items():
            ex_count = ex_counter.get(char, 0)
            preserved_count = min(gt_count, ex_count)
            preserved_chars += preserved_count

        return preserved_chars / total_gt_chars
