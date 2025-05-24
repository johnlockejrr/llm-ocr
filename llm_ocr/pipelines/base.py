import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from llm_ocr.processors.alto import ALTOLine

from ..config import EvaluationConfig
from ..evaluators.evaluator import (
    ErrorAnalysisMetric,
    OCREvaluator,
    SimilarityMetric,
    WordAnalysisMetric,
)
from ..llm.base import BaseOCRModel
from ..models import Line, OCRResult, ProcessingMode


class BasePipeline(ABC):
    """Base abstract class for OCR evaluation pipelines."""

    def __init__(
        self,
        model: BaseOCRModel,
        evaluator: OCREvaluator,
        config: Optional[EvaluationConfig] = None,
    ):
        self.model = model
        self.evaluator = evaluator
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_specialized_metrics()

    def _init_specialized_metrics(self) -> None:
        """Initialize specialized metrics for direct use in the pipeline."""
        # Convert old_russian_chars to a set for use in metrics
        old_chars_set = set(self.config.old_russian_chars)

        # Create metrics for direct use
        self.error_analyzer = ErrorAnalysisMetric(old_chars_set)
        self.word_analyzer = WordAnalysisMetric()
        self.similarity_metric = SimilarityMetric(
            self.config.char_similarity_weight, self.config.word_similarity_weight
        )

    @abstractmethod
    def ocr_document(
        self,
        lines: List[ALTOLine],
        image_str: str,
        id: str,
        mode: ProcessingMode = ProcessingMode.SINGLE_LINE,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluate OCR on document lines with specified processing mode.

        Args:
            lines: List of ALTOLine objects
            image_path: Optional path to full page image
            mode: Processing mode to use

        Returns:
            Dictionary mapping model names to lists of OCRResults
        """
        pass

    def generate_report(
        self, results: Dict[str, List[OCRResult]], include_details: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate evaluation report with metrics and analysis.

        Args:
            results: Dictionary of model results
            include_details: Whether to include detailed line analysis

        Returns:
            Dictionary with report data by model
        """
        report = {}

        for model_name, model_results in results.items():
            metrics = self._calculate_model_metrics(model_results)
            error_analysis = self._analyze_error_patterns(model_results)

            report[model_name] = {
                **metrics,
                "error_analysis": error_analysis,
            }

            if include_details:
                report[model_name]["line_details"] = self._analyze_lines(model_results)

            self._log_detailed_report(model_name, report[model_name])

        return report

    def _safe_process(
        self, func: Callable[..., Any], *args: Any, fallback: Any = None, **kwargs: Any
    ) -> Any:
        """
        Generic error handling for processing functions.

        Args:
            func: Function to call
            args: Positional arguments
            fallback: Value to return on error
            kwargs: Keyword arguments

        Returns:
            Function result or fallback on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            return fallback

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
                "average_processing_time": np.mean([r.processing_time for r in results]),
                "character_accuracy": np.mean(
                    [r.metrics.char_accuracy if r.metrics else 0.0 for r in results]
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
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "average_processing_time": 0.0,
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

                # Use our specialized WordAnalysisMetric
                word_analysis = self.word_analyzer.evaluate(
                    result.ground_truth_text, result.extracted_text
                )

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

            return {
                k: dict(sorted(v.items(), key=lambda x: x[1], reverse=True))
                for k, v in patterns.items()
            }
        except Exception as e:
            self.logger.error(f"Error analyzing error patterns: {str(e)}")
            return {k: {} for k in patterns.keys()}

    def _create_empty_result(self, line: Line, model_name: str) -> OCRResult:
        """
        Create empty OCRResult for failed processing.

        Args:
            line: Source Line object
            model_name: Name of the model

        Returns:
            OCRResult with empty extracted text
        """
        return OCRResult(
            ground_truth_text=line.text,
            extracted_text="",
            processing_time=0.0,
            model_name=model_name,
            metrics=self.evaluator._create_empty_metrics(),
        )

    def _match_lines(
        self, ground_truth_lines: List[Line], extracted_lines: List[Dict[str, str]]
    ) -> List[Tuple[Line, Optional[Dict[str, str]]]]:
        """
        Match extracted lines with ground truth lines using similarity matching.

        Args:
            ground_truth_lines: List of Line objects
            extracted_lines: List of dictionaries with 'line' key

        Returns:
            List of tuples (ground_truth_line, matched_extracted_line or None)
        """
        matches: List[Tuple[Line, Optional[Dict[str, str]]]] = []
        used_extracted = set()

        # For each ground truth line
        for gt_line in ground_truth_lines:
            best_match = None
            best_score = 0.0
            best_idx = None

            # Find best matching extracted line
            for i, ext_line in enumerate(extracted_lines):
                if i in used_extracted:
                    continue

                if not ext_line or "line" not in ext_line:
                    continue

                # Calculate similarity score
                score = self.similarity_metric.evaluate(gt_line.text, ext_line["line"])

                if score > best_score and score > self.config.match_threshold:
                    best_score = score
                    best_match = ext_line
                    best_idx = i

            if best_match:
                used_extracted.add(best_idx)
                matches.append((gt_line, best_match))
            else:
                matches.append((gt_line, None))

        return matches

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

            return {f"{k}-{k+19}_chars": np.mean(v) for k, v in sorted(length_groups.items())}
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
                f"position_{i+1}": np.mean(
                    [
                        r.metrics.char_accuracy if r.metrics else 0.0
                        for r in results[i : i + chunk_size]
                    ]
                )
                for i in range(0, n, chunk_size)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing position impact: {str(e)}")
            return {}

    def _log_detailed_report(self, model_name: str, report_data: Dict[str, Any]) -> None:
        """Log detailed report with proper encoding."""
        try:
            self.logger.info(f"\nDetailed Analysis for {model_name}")
            self.logger.info("=" * 50)

            # Basic metrics
            self.logger.info("\nBasic Metrics:")
            self.logger.info(f"Character Accuracy: {report_data.get('character_accuracy', 0):.2%}")
            self.logger.info(f"Word Accuracy: {report_data.get('word_accuracy', 0):.2%}")
            self.logger.info(
                f"Old Character Preservation: {report_data.get('old_char_preservation', 0):.2%}"
            )
            self.logger.info(f"Case Accuracy: {report_data.get('case_accuracy', 0):.2%}")
            self.logger.info(f"Total Lines Processed: {report_data.get('total_lines', 0)}")

            # Error patterns
            if (
                "error_analysis" in report_data
                and "common_char_errors" in report_data["error_analysis"]
            ):
                self.logger.info("\nTop Error Patterns:")
                errors = report_data["error_analysis"]["common_char_errors"]
                if errors:
                    for pattern, count in list(errors.items())[:5]:
                        # Convert error pattern to readable format
                        from_char, to_char = pattern.split("->")
                        self.logger.info(f"'{from_char}' → '{to_char}': {count} occurrences")
                else:
                    self.logger.info("No common error patterns detected")

        except Exception as e:
            self.logger.error(f"Error generating detailed report: {str(e)}")
