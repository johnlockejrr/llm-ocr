"""
Metrics Module - Provides modular evaluation metrics for OCR text comparisons.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Base class for all OCR evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""
        pass

    @abstractmethod
    def evaluate(self, ground_truth: str, extracted: str) -> Any:
        """
        Evaluate the metric between ground truth and extracted text.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Evaluation result (score, dictionary, etc.)
        """
        pass
