from typing import Any, Dict

from llm_ocr.evaluators.metrics.base import BaseMetric


class CaseAccuracyMetric(BaseMetric):
    """Metric for capitalization accuracy."""

    @property
    def name(self) -> str:
        return "case_accuracy"

    def evaluate(self, ground_truth: str, extracted: str) -> Dict[str, Any]:
        """
        Calculate case accuracy and analyze capitalization.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Dictionary with case accuracy, errors, and analysis
        """
        case_errors = []
        case_matches = 0
        total_chars = 0

        for i, (gt_char, ext_char) in enumerate(zip(ground_truth, extracted)):
            if gt_char.isalpha() and ext_char.isalpha():
                total_chars += 1
                if gt_char.isupper() == ext_char.isupper():
                    case_matches += 1
                else:
                    case_errors.append(
                        {"position": i, "ground_truth": gt_char, "extracted": ext_char}
                    )

        case_accuracy = case_matches / total_chars if total_chars > 0 else 0.0

        return {
            "accuracy": case_accuracy,
            "errors": case_errors,
            "analysis": self._analyze_capitalization(ground_truth, extracted),
        }

    def _analyze_capitalization(self, ground_truth: str, extracted: str) -> Dict[str, Any]:
        """
        Analyze capitalization patterns.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Dictionary with capitalization statistics
        """
        return {
            "ground_truth_capitals": sum(1 for c in ground_truth if c.isupper()),
            "extracted_capitals": sum(1 for c in extracted if c.isupper()),
            "ground_truth_initial_caps": sum(
                1 for w in ground_truth.split() if w and w[0].isupper()
            ),
            "extracted_initial_caps": sum(1 for w in extracted.split() if w and w[0].isupper()),
            "capital_positions_ground_truth": [
                i for i, c in enumerate(ground_truth) if c.isupper()
            ],
            "capital_positions_extracted": [i for i, c in enumerate(extracted) if c.isupper()],
        }
