"""
Metrics Module - Provides modular evaluation metrics for OCR text comparisons.
"""

import logging
from typing import Any, Dict, Optional

from llm_ocr.evaluators.metrics.base import BaseMetric
from llm_ocr.evaluators.metrics.case_accuracy import CaseAccuracyMetric
from llm_ocr.evaluators.metrics.character_accuracy import CharacterAccuracyMetric, SimilarityMetric
from llm_ocr.evaluators.metrics.character_accuracy_without_case import (
    CaseInsensitiveCharacterAccuracyMetric,
)
from llm_ocr.evaluators.metrics.error_analysis import ErrorAnalysisMetric
from llm_ocr.evaluators.metrics.historic_chars import OverHistoricizationMetric
from llm_ocr.evaluators.metrics.old_char_preservation import OldCharPreservationMetric
from llm_ocr.evaluators.metrics.word_accuracy import WordAccuracyMetric
from llm_ocr.evaluators.metrics.word_analysis import WordAnalysisMetric

from ..config import EvaluationConfig
from ..models import OCRMetrics


class OCREvaluator:
    """
    Evaluates OCR text extraction results using configurable metrics.

    This class orchestrates the evaluation process using multiple metrics
    that can be enabled or disabled through configuration.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize OCR evaluator with configuration.

        Args:
            config: Optional evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize metrics
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize metric instances based on configuration."""
        # Convert old_russian_chars string to set for faster lookups
        old_chars_set = set(self.config.old_russian_chars)

        self.metrics: Dict[str, BaseMetric] = {}

        # Core metrics
        self.metrics["char_accuracy"] = CharacterAccuracyMetric()
        self.metrics["char_accuracy_case_insensitive"] = CaseInsensitiveCharacterAccuracyMetric()
        self.metrics["word_accuracy"] = WordAccuracyMetric()
        self.metrics["old_char_preservation"] = OldCharPreservationMetric(old_chars_set)
        self.metrics["case_accuracy"] = CaseAccuracyMetric()
        self.metrics["historicization"] = OverHistoricizationMetric()

        # Analysis metrics
        self.metrics["error_analysis"] = ErrorAnalysisMetric(old_chars_set)
        self.metrics["word_analysis"] = WordAnalysisMetric()
        self.metrics["similarity"] = SimilarityMetric(
            self.config.char_similarity_weight, self.config.word_similarity_weight
        )

    def evaluate_line(self, ground_truth: str, extracted: str) -> OCRMetrics:
        """
        Evaluate single line extraction using configured metrics.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            OCRMetrics with various accuracy measurements
        """
        if not ground_truth or not extracted:
            return self._create_empty_metrics()

        # Initialize metrics with default values
        metrics_values: Dict[str, Any] = {
            "char_accuracy": 0.0,
            "char_accuracy_case_insensitive": 0.0,
            "word_accuracy": 0.0,
            "old_char_preservation": 0.0,
            "case_accuracy": 0.0,
            "case_errors": [],
            "capitalization_analysis": {},
        }

        # Calculate enabled metrics
        for metric_name, metric in self.metrics.items():
            # Check if metric is enabled in config
            config_attr = f"use_{metric_name}"
            if not hasattr(self.config, config_attr) or getattr(self.config, config_attr):
                if isinstance(extracted, list):
                    # If extracted is a list, take the first element
                    extracted = extracted[0]
                result = metric.evaluate(ground_truth, extracted)

                if metric_name == "case_accuracy":
                    # Special handling for case accuracy which returns a dict
                    metrics_values["case_accuracy"] = result["accuracy"]
                    metrics_values["case_errors"] = result["errors"]
                    metrics_values["capitalization_analysis"] = result["analysis"]
                elif metric_name in [
                    "char_accuracy",
                    "char_accuracy_case_insensitive",
                    "word_accuracy",
                    "old_char_preservation",
                ]:
                    metrics_values[metric_name] = result

        # Create and return metrics object
        return OCRMetrics(**metrics_values)

    def _create_empty_metrics(self) -> OCRMetrics:
        """Create empty metrics for failed evaluations."""
        return OCRMetrics(
            char_accuracy=0.0,
            char_accuracy_case_insensitive=0.0,
            word_accuracy=0.0,
            old_char_preservation=0.0,
            case_accuracy=0.0,
            case_errors=[],
            capitalization_analysis={
                "ground_truth_capitals": 0,
                "extracted_capitals": 0,
                "ground_truth_initial_caps": 0,
                "extracted_initial_caps": 0,
                "capital_positions_ground_truth": [],
                "capital_positions_extracted": [],
            },
        )

    def analyze_errors(self, ground_truth: str, extracted: str) -> Dict[str, Any]:
        """
        Analyze differences and errors between ground truth and extracted text.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Dictionary with detailed error analysis
        """
        result = self.metrics["error_analysis"].evaluate(ground_truth, extracted)
        return result if isinstance(result, dict) else {}

    def analyze_words(self, ground_truth: str, extracted: str) -> Dict[str, Any]:
        """
        Analyze word-level differences between ground truth and extracted text.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Dictionary with word-level error analysis
        """
        result = self.metrics["word_analysis"].evaluate(ground_truth, extracted)
        return result if isinstance(result, dict) else {}

    def calculate_similarity(self, ground_truth: str, extracted: str) -> float:
        """
        Calculate weighted similarity score.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Similarity score between 0.0 and 1.0
        """
        result = self.metrics["similarity"].evaluate(ground_truth, extracted)
        return float(result) if isinstance(result, (int, float)) else 0.0


class MetricsComparer:
    """Utility for comparing OCR metrics before and after correction."""

    @staticmethod
    def compare_metrics(
        original_metrics: OCRMetrics, corrected_metrics: OCRMetrics
    ) -> Dict[str, float]:
        """
        Calculate improvement between original and corrected metrics.

        Args:
            original_metrics: Metrics from original OCR output
            corrected_metrics: Metrics after correction

        Returns:
            Dictionary with improvement deltas for each metric
        """
        return {
            "character_accuracy_delta": round(
                corrected_metrics.char_accuracy - original_metrics.char_accuracy, 10
            ),
            "word_accuracy_delta": round(
                corrected_metrics.word_accuracy - original_metrics.word_accuracy, 10
            ),
            "old_char_preservation_delta": round(
                corrected_metrics.old_char_preservation - original_metrics.old_char_preservation, 10
            ),
            "case_accuracy_delta": round(
                corrected_metrics.case_accuracy - original_metrics.case_accuracy, 10
            ),
        }
