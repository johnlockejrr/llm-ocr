"""
OCR Evaluation Service - Completely standalone and flexible evaluation functionality.
"""

import logging
from collections import Counter, defaultdict
from itertools import zip_longest
from typing import Any
from typing import Counter as CounterType
from typing import Dict, List, Optional

import numpy as np

from llm_ocr.evaluators.metrics.character_accuracy import SimilarityMetric
from llm_ocr.evaluators.metrics.error_analysis import ErrorAnalysisMetric
from llm_ocr.evaluators.metrics.historic_chars import OverHistoricizationMetric

from ..config import EvaluationConfig
from ..models import OCRResult
from .evaluator import OCREvaluator


class OCREvaluationService:
    """Standalone service for evaluating OCR results with flexible comparison options."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize OCR evaluation service with evaluator and optional config.

        Args:
            config: Optional evaluation configuration
        """
        self.evaluator = OCREvaluator()
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_specialized_metrics()

    def _init_specialized_metrics(self) -> None:
        """Initialize specialized metrics for direct use in the evaluation."""
        # Convert old_russian_chars to a set for use in metrics
        old_chars_set = set(self.config.old_russian_chars)

        # Create metrics for direct use
        self.error_analyzer = ErrorAnalysisMetric(old_chars_set)
        self.similarity_metric = SimilarityMetric(
            self.config.char_similarity_weight, self.config.word_similarity_weight
        )
        self.over_historicization_metric = OverHistoricizationMetric()

    def generate_report(
        self, results: List[OCRResult], include_details: bool = False
    ) -> Dict[str, Any]:
        """Generate a report."""
        metrics = self._calculate_model_metrics(results)
        error_analysis = self._analyze_error_patterns(results)

        report = {
            **metrics,
            "error_analysis": error_analysis,
        }

        if include_details:
            report["line_details"] = self._analyze_lines(results)
            del report["error_analysis"][
                "position_errors"
            ]  # Remove position errors from detailed report
            del report["error_analysis"]["word_length_errors"]

        self._log_detailed_report(report)

        return report

    def convert_to_ocr_results(self, raw_results: List[Any]) -> List[OCRResult]:
        """
        Convert raw OCR results to OCRResult objects with metrics.

        Args:
            raw_results: List of lists of raw OCR result dictionaries

        Returns:
            List of OCRResult objects
        """
        results = []

        for result in raw_results:
            metrics = self.evaluator.evaluate_line(
                result.get("ground_truth_text", ""), result.get("extracted_text", "")
            )
            results.append(
                OCRResult(
                    ground_truth_text=result.get("ground_truth_text", ""),
                    extracted_text=result.get("extracted_text", ""),
                    metrics=metrics,
                )
            )

        return results

    def evaluate_ocr_results(
        self, raw_results: List[Any], include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate raw OCR results and generate a comprehensive report.

        Args:
            raw_results: List of raw OCR result dictionaries
            include_details: Whether to include detailed line analysis

        Returns:
            Dictionary with evaluation report data
        """
        # First convert raw results to OCRResult objects with metrics
        logging.info(f"Converting raw results: {raw_results}")
        ocr_results = self.convert_to_ocr_results(raw_results)
        logging.info(f"Converted: {ocr_results}")

        # Then generate report using the OCRResult objects
        return self.generate_report(ocr_results, include_details)

    def _calculate_model_metrics(self, results: List[OCRResult]) -> Dict[str, Any]:
        """
        Calculate aggregated metrics for model results.

        Args:
            results: List of OCRResults

        Returns:
            Dictionary of metric names to values
        """
        try:
            metrics: Dict[str, Any] = {
                "character_accuracy": np.mean(
                    [r.metrics.char_accuracy if r.metrics else 0.0 for r in results]
                ),
                "char_accuracy_case_insensitive": np.mean(
                    [
                        r.metrics.char_accuracy_case_insensitive if r.metrics else 0.0
                        for r in results
                    ]
                ),
                "word_accuracy": np.mean(
                    [r.metrics.word_accuracy if r.metrics else 0.0 for r in results]
                ),
                "old_char_preservation": np.mean(
                    [r.metrics.old_char_preservation if r.metrics else 0.0 for r in results]
                ),
                "case_accuracy": np.mean(
                    [r.metrics.case_accuracy if r.metrics else 0.0 for r in results]
                ),
                "total_lines": len(results),
                "total_case_errors": sum(
                    (
                        len(r.metrics.case_errors)
                        if r.metrics and hasattr(r.metrics, "case_errors")
                        else 0
                    )
                    for r in results
                ),
            }

            # Calculate historicization metrics
            historicization_results = []
            for result in results:
                # Apply the over-historicization metric
                historicization_metric = self.over_historicization_metric.evaluate(
                    result.ground_truth_text, result.extracted_text
                )
                historicization_results.append(historicization_metric)

            # Aggregate historicization statistics
            if historicization_results:
                # Combine all character frequencies
                char_frequencies: CounterType[str] = Counter()
                all_chars = set()
                total_insertions = 0
                affected_lines = 0

                for res in historicization_results:
                    # Count incorrect characters
                    all_chars.update(res.get("incorrect_archaic_chars", []))

                    # Sum up frequencies
                    for char, count in res.get("char_frequencies", {}).items():
                        char_frequencies[char] += count
                        total_insertions += count

                    # Count affected lines
                    if res.get("total_insertions", 0) > 0:
                        affected_lines += 1

                # Calculate average insertion ratio
                avg_insertion_ratio = np.mean(
                    [res.get("insertion_ratio", 0) for res in historicization_results]
                )

                # Collect top contexts for reference
                all_contexts = []
                for res in historicization_results:
                    all_contexts.extend(res.get("insertion_contexts", []))

                # Sort contexts by frequency of character
                # This creates a list of contexts sorted by character frequency
                sorted_contexts = []
                for char, _ in char_frequencies.most_common():
                    contexts_for_char = [ctx for ctx in all_contexts if ctx.get("char") == char][
                        :5
                    ]  # Limit to 5 contexts per character
                    sorted_contexts.extend(contexts_for_char)

                # Include historicization metrics in the output
                metrics["historicization"] = {
                    "incorrect_archaic_chars": list(all_chars),
                    "char_frequencies": dict(char_frequencies.most_common()),
                    "total_insertions": total_insertions,
                    "average_insertion_ratio": avg_insertion_ratio,
                    "affected_lines": affected_lines,
                    "affected_lines_percentage": affected_lines / len(results) if results else 0,
                    "top_contexts": sorted_contexts[:20],  # Limit to top 20 contexts overall
                }

            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "character_accuracy": 0.0,
                "word_accuracy": 0.0,
                "old_char_preservation": 0.0,
                "case_accuracy": 0.0,
                "total_lines": 0,
                "total_case_errors": 0,
                "error": str(e),
            }

    def _analyze_error_patterns(self, results: List[OCRResult]) -> Dict[str, Any]:
        """
        Analyze common error patterns in results.

        Args:
            results: List of OCRResults

        Returns:
            Dictionary with error pattern analysis
        """
        patterns: Dict[str, Dict[str, int]] = {
            "common_char_errors": defaultdict(int),
            "position_errors": defaultdict(int),
            "word_length_errors": defaultdict(int),
            "old_char_error_patterns": defaultdict(int),
            "historicization_patterns": defaultdict(int),
        }

        try:
            for result in results:
                # Use our specialized ErrorAnalysisMetric directly
                error_analysis = self.error_analyzer.evaluate(
                    result.ground_truth_text, result.extracted_text
                )

                for sub in error_analysis["substitutions"]:
                    patterns["common_char_errors"][
                        f"{sub['ground_truth']}->{sub['extracted']}"
                    ] += 1

                # Track position-based errors
                words_gt = result.ground_truth_text.split()
                words_ext = result.extracted_text.split()
                for i, (w1, w2) in enumerate(zip_longest(words_gt, words_ext)):
                    if w1 != w2:
                        patterns["position_errors"][f"word_{i+1}"] += 1

                # Track word length related errors
                for w1, w2 in zip_longest(words_gt, words_ext):
                    if w1 and w2:
                        len_diff = len(w1) - len(w2)
                        if len_diff != 0:
                            patterns["word_length_errors"][str(len_diff)] += 1

                # Track special character errors
                for err in error_analysis.get("special_char_errors", []):
                    gt_chars = err.get("special_chars_ground_truth", [])
                    ext_chars = err.get("special_chars_extracted", [])

                    for gt_char in gt_chars:
                        if ext_chars:
                            for ext_char in ext_chars:
                                patterns["old_char_error_patterns"][f"{gt_char}->{ext_char}"] += 1
                        else:
                            patterns["old_char_error_patterns"][
                                f"{gt_char}->∅"
                            ] += 1  # ∅ means omitted
                # Historicization patterns
                historicization = self.over_historicization_metric.evaluate(
                    result.ground_truth_text, result.extracted_text
                )

                # Track frequency of each inserted archaic character
                for char, count in historicization.get("char_frequencies", {}).items():
                    patterns["historicization_patterns"][f"∅->{char}"] += count

                # Also track specific character substitutions from contexts
                for ctx in historicization.get("insertion_contexts", []):
                    if "original_char" in ctx and "char" in ctx:
                        patterns["historicization_patterns"][
                            f"{ctx['original_char']}->{ctx['char']}"
                        ] += 1

            return {
                k: dict(sorted(v.items(), key=lambda x: x[1], reverse=True))
                for k, v in patterns.items()
            }
        except Exception as e:
            self.logger.error(f"Error analyzing error patterns: {str(e)}")
            return {k: {} for k in patterns.keys()}

    def _analyze_lines(self, results: List[OCRResult]) -> Dict[str, Any]:
        """
        Detailed analysis of line-level patterns.

        Args:
            results: List of OCRResults

        Returns:
            Dictionary with line analysis data
        """
        return {
            "line_length_impact": self._analyze_length_impact(results),
            "position_impact": self._analyze_position_impact(results),
        }

    def _analyze_length_impact(self, results: List[OCRResult]) -> Dict[str, float]:
        """
        Analyze impact of line length on accuracy.

        Args:
            results: List of OCRResults

        Returns:
            Dictionary mapping length ranges to accuracy
        """
        try:
            length_groups = defaultdict(list)
            for r in results:
                if r.metrics:
                    length = len(r.ground_truth_text)
                    length_group = length // 20 * 20  # Group by 20-char intervals
                    length_groups[length_group].append(r.metrics.char_accuracy)

            return {
                f"{k}-{k+19}_chars": float(np.mean(v)) for k, v in sorted(length_groups.items())
            }
        except Exception as e:
            self.logger.error(f"Error analyzing length impact: {str(e)}")
            return {}

    def _analyze_position_impact(self, results: List[OCRResult]) -> Dict[str, float]:
        """
        Analyze impact of line position on accuracy.

        Args:
            results: List of OCRResults

        Returns:
            Dictionary mapping position ranges to accuracy
        """
        try:
            n = len(results)
            if n == 0:
                return {}

            chunk_size = max(1, n // 5)  # Divide into 5 chunks

            return {
                f"position_{i+1}": float(
                    np.mean(
                        [
                            r.metrics.char_accuracy if r.metrics else 0.0
                            for r in results[i : i + chunk_size]
                        ]
                    )
                )
                for i in range(0, n, chunk_size)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing position impact: {str(e)}")
            return {}

    def _log_detailed_report(self, report_data: Dict[str, Any]) -> None:
        """Log detailed report with proper encoding."""
        try:
            # Basic metrics
            self.logger.info(f"Character Accuracy: {report_data.get('character_accuracy', 0):.2%}")
            self.logger.info(f"Word Accuracy: {report_data.get('word_accuracy', 0):.2%}")
            self.logger.info(
                f"Old Character Preservation: {report_data.get('old_char_preservation', 0):.2%}"
            )
            self.logger.info(f"Case Accuracy: {report_data.get('case_accuracy', 0):.2%}")
            self.logger.info(f"Total Lines Processed: {report_data.get('total_lines', 0)}")

        except Exception as e:
            self.logger.error(f"Error generating detailed report: {str(e)}")
