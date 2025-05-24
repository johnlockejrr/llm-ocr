"""
OCR Correction Pipeline - Module for LLM-based OCR correction.
"""

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from ..config import EvaluationConfig
from ..evaluators.evaluator import MetricsComparer, OCREvaluator
from ..llm.base import BaseOCRModel
from ..models import OCRCorrectionResult


class OCRCorrectionPipeline:
    """Pipeline for correcting OCR text using LLM."""

    def __init__(
        self,
        model: BaseOCRModel,
        evaluator: OCREvaluator,
        config: Optional[EvaluationConfig] = None,
    ):
        """
        Initialize the OCR correction pipeline.

        Args:
            models: List of LLM models to use for correction
            evaluator: OCR evaluator instance
            config: Evaluation configuration (optional)
        """
        self.model = model
        self.evaluator = evaluator
        self.config = config or EvaluationConfig()
        self.metrics_comparer = MetricsComparer()
        self.logger = logging.getLogger(__name__)
        self.results: Dict[str, Any] = {}

    def run_correction(
        self,
        image_str: str,
        ocr_text: str,
        mode: str = "line",
    ) -> Optional[Dict[str, Any]]:
        """
        Run OCR correction using LLM.

        Args:
            image_str: base64 encoded image string
            ocr_text: Original OCR text to correct
            mode: "line" for single line output, "para" for paragraph detection

        Returns:
            Dictionary mapping model names to OCRCorrectionResult objects
        """
        model_name = self.model.__class__.__name__
        self.logger.info(f"Running OCR correction with {model_name}")

        try:
            # Start timing
            start_time = time.time()

            # Perform correction
            corrected_text = self.model.correct_text(ocr_text, image_str, mode=mode)

            # Calculate processing time
            processing_time = time.time() - start_time

            # First create the basic result without evaluation
            correction_result = OCRCorrectionResult(
                extracted_text=ocr_text,
                corrected_text=corrected_text,
                processing_time=processing_time,
                model_name=model_name,
            )
            logging.info(f"Correction completed in {processing_time:.2f} seconds")
            logging.info(f"Correction results: {correction_result}")

            return asdict(correction_result)

        except Exception as e:
            self.logger.error(f"Error processing correction with {model_name}: {str(e)}")
            return None

    def _evaluate_correction(
        self, ground_truth: str, ocr_text: str, corrected_text: str, result: OCRCorrectionResult
    ) -> None:
        """
        Evaluate correction and update the result object.

        Args:
            ground_truth: Ground truth text
            ocr_text: Original OCR text
            corrected_text: Corrected text
            result: OCRCorrectionResult object to update
        """
        try:
            # Evaluate corrected text
            logging.info("Evaluating corrected text")
            logging.info(f"Ground truth: {ground_truth}")
            logging.info(f"Corrected text: {corrected_text}")
            corrected_metrics = self.evaluator.evaluate_line(ground_truth, corrected_text)
            result.metrics = corrected_metrics

            # Evaluate original OCR text for comparison
            ocr_metrics = self.evaluator.evaluate_line(ground_truth, ocr_text)

            # Calculate improvement
            result.improvement = {
                "character_accuracy_delta": round(
                    corrected_metrics.char_accuracy - ocr_metrics.char_accuracy, 10
                ),
                "word_accuracy_delta": round(
                    corrected_metrics.word_accuracy - ocr_metrics.word_accuracy, 10
                ),
                "old_char_preservation_delta": round(
                    corrected_metrics.old_char_preservation - ocr_metrics.old_char_preservation, 10
                ),
                "case_accuracy_delta": round(
                    corrected_metrics.case_accuracy - ocr_metrics.case_accuracy, 10
                ),
            }

            # Add detailed analysis if configured
            if self.config.include_detailed_analysis:
                # Get character-level error analysis
                result.error_analysis = self.evaluator.analyze_errors(ground_truth, corrected_text)
                result.word_analysis = self.evaluator.analyze_words(ground_truth, corrected_text)

                # Add similarity score
                result.similarity = self.evaluator.calculate_similarity(
                    ground_truth, corrected_text
                )

        except Exception as e:
            self.logger.error(f"Error evaluating correction: {str(e)}")
