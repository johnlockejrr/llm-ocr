"""
OCR Correction Pipeline - Module for LLM-based OCR correction.
"""

import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..config import EvaluationConfig
from ..evaluators.evaluator import MetricsComparer, OCREvaluator
from ..llm.base import BaseOCRModel
from ..models import CorrectionMode, LineCorrection, OCRCorrectionResult, ParagraphCorrection
from ..prompts.prompt_builder import PromptBuilder, PromptType, PromptVersion


class OCRCorrectionPipeline:
    """Enhanced pipeline for correcting OCR text using LLM with multiple modes."""

    def __init__(
        self,
        model: BaseOCRModel,
        evaluator: OCREvaluator,
        config: Optional[EvaluationConfig] = None,
        prompt_version: Optional[PromptVersion] = None,
    ):
        self.model = model
        self.evaluator = evaluator
        self.config = config or EvaluationConfig()
        self.metrics_comparer = MetricsComparer()
        self.logger = logging.getLogger(__name__)
        self.results: Dict[str, Any] = {}
        self.prompt_version = prompt_version or PromptVersion.V3
        self.prompt_builder = PromptBuilder()

        # Mode-specific processors
        self.mode_processors: Dict[
            CorrectionMode, Callable[[str, str], Union[LineCorrection, ParagraphCorrection]]
        ] = {
            CorrectionMode.LINE: self._process_line_mode,
            CorrectionMode.PARA: self._process_para_mode,
        }

    def run_correction(
        self,
        image_str: str,
        ocr_text: str,
        mode: str = "line",
        ground_truth: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run OCR correction using LLM with specified mode.

        Args:
            image_str: base64 encoded image string
            ocr_text: Original OCR text to correct
            mode: "line" for single line output, "para" for paragraph detection
            ground_truth: Optional ground truth for evaluation

        Returns:
            Dictionary with correction results
        """
        # Validate mode
        try:
            correction_mode = CorrectionMode(mode)
        except ValueError:
            self.logger.error(f"Invalid correction mode: {mode}")
            raise ValueError(f"Invalid correction mode: {mode}")

        model_name = self.model.__class__.__name__
        self.logger.info(f"Running OCR correction with {model_name} in {mode} mode")

        try:
            # Perform correction with mode-specific handling
            corrected_text = self.mode_processors[correction_mode](ocr_text, image_str)

            # Create result object
            correction_result = OCRCorrectionResult(
                extracted_text=ocr_text,
                correction_mode=correction_mode,
                corrected_text=corrected_text,
                model_name=model_name,
            )

            # Evaluate if ground truth provided
            if ground_truth:
                self._evaluate_correction_by_mode(ground_truth, ocr_text, correction_result)

            self.logger.info("Correction completed")

            return self._serialize_result(correction_result)

        except Exception as e:
            self.logger.error(f"Error processing correction: {str(e)}")
            return None

    def _process_line_mode(self, ocr_text: str, image_str: str) -> LineCorrection:
        """Process correction in line mode."""
        # Build correction prompt
        prompt = self.prompt_builder.build_prompt(
            mode="correction",
            prompt_type=PromptType.SIMPLE,
            version=self.prompt_version,
            text=ocr_text
        )
        
        corrected_text = self.model.correct_text(prompt, ocr_text, image_str, mode="line")

        # Post-process: ensure single line (remove any newlines)
        corrected_text = corrected_text.replace("\n", " ").strip()

        # Optional: Get confidence from model if available
        confidence = getattr(self.model, "get_last_confidence", lambda: None)()

        return LineCorrection(corrected_text=corrected_text, confidence=confidence)

    def _process_para_mode(self, ocr_text: str, image_str: str) -> ParagraphCorrection:
        """Process correction in paragraph mode."""
        # Build correction prompt for paragraph mode
        prompt = self.prompt_builder.build_prompt(
            mode="correction_para",
            prompt_type=PromptType.SIMPLE,
            version=self.prompt_version,
            text=ocr_text
        )
        
        corrected_text = self.model.correct_text(prompt, ocr_text, image_str, mode="para")

        # Parse paragraphs
        paragraphs = self._parse_paragraphs(corrected_text)
        boundaries = self._calculate_boundaries(paragraphs)

        # Optional: Get per-paragraph confidence if model supports it
        confidence_scores = None
        if hasattr(self.model, "get_paragraph_confidences"):
            confidence_scores = self.model.get_paragraph_confidences()

        return ParagraphCorrection(
            paragraphs=paragraphs,
            paragraph_boundaries=boundaries,
            confidence_scores=confidence_scores,
        )

    def _parse_paragraphs(self, text: str) -> List[str]:
        """Parse text into paragraphs."""
        # Split by double newlines or other paragraph markers
        paragraphs = []
        current_para = []

        lines = text.split("\n")
        for line in lines:
            if line.strip():  # Non-empty line
                current_para.append(line)
            elif current_para:  # Empty line and we have content
                paragraphs.append(" ".join(current_para))
                current_para = []

        # Don't forget the last paragraph
        if current_para:
            paragraphs.append(" ".join(current_para))

        return paragraphs

    def _calculate_boundaries(self, paragraphs: List[str]) -> List[int]:
        """Calculate character positions where paragraphs start."""
        boundaries = [0]
        current_pos = 0

        for para in paragraphs[:-1]:
            current_pos += len(para) + 1  # +1 for space/newline between paragraphs
            boundaries.append(current_pos)

        return boundaries

    def _evaluate_correction_by_mode(
        self, ground_truth: str, ocr_text: str, result: OCRCorrectionResult
    ) -> None:
        """Evaluate correction based on mode."""
        if result.correction_mode == CorrectionMode.LINE:
            self._evaluate_line_correction(ground_truth, ocr_text, result)
        elif result.correction_mode == CorrectionMode.PARA:
            self._evaluate_para_correction(ground_truth, ocr_text, result)

    def _evaluate_line_correction(
        self, ground_truth: str, ocr_text: str, result: OCRCorrectionResult
    ) -> None:
        """Evaluate single line correction."""
        if result.correction_mode == CorrectionMode.LINE and isinstance(
            result.corrected_text, LineCorrection
        ):
            corrected_text = result.corrected_text.corrected_text
        else:
            return

        # Standard evaluation
        corrected_metrics = self.evaluator.evaluate_line(ground_truth, corrected_text)
        ocr_metrics = self.evaluator.evaluate_line(ground_truth, ocr_text)

        result.metrics = asdict(corrected_metrics)
        result.improvement = self._calculate_improvement(corrected_metrics, ocr_metrics)

        if self.config.include_detailed_analysis:
            result.error_analysis = self.evaluator.analyze_errors(ground_truth, corrected_text)

    def _evaluate_para_correction(
        self, ground_truth: str, ocr_text: str, result: OCRCorrectionResult
    ) -> None:
        """Evaluate paragraph-based correction."""
        # Join paragraphs for overall metrics
        if result.correction_mode == CorrectionMode.PARA and isinstance(
            result.corrected_text, ParagraphCorrection
        ):
            corrected_text = "\n\n".join(result.corrected_text.paragraphs)
        else:
            return

        # Overall metrics
        corrected_metrics = self.evaluator.evaluate_line(ground_truth, corrected_text)
        ocr_metrics = self.evaluator.evaluate_line(ground_truth, ocr_text)

        # Paragraph-specific metrics
        para_metrics = {
            "overall": asdict(corrected_metrics),
            "paragraph_count": len(result.corrected_text.paragraphs),
            "average_paragraph_length": np.mean([len(p) for p in result.corrected_text.paragraphs]),
        }

        # Check paragraph boundary accuracy if ground truth has paragraphs
        ground_truth_paragraphs = self._parse_paragraphs(ground_truth)
        if len(ground_truth_paragraphs) > 1:
            para_metrics["paragraph_boundary_accuracy"] = self._evaluate_boundaries(
                ground_truth_paragraphs, result.corrected_text.paragraphs
            )

        result.metrics = para_metrics
        result.improvement = self._calculate_improvement(corrected_metrics, ocr_metrics)

    def _evaluate_boundaries(
        self, ground_truth_paras: List[str], corrected_paras: List[str]
    ) -> float:
        """Evaluate how well paragraph boundaries were detected."""
        # Simple approach: check if paragraph count matches
        count_accuracy = 1.0 - abs(len(ground_truth_paras) - len(corrected_paras)) / len(
            ground_truth_paras
        )

        # More sophisticated: check actual boundary positions
        # This would require aligning the texts first

        return max(0.0, count_accuracy)

    def _calculate_improvement(self, corrected_metrics: Any, ocr_metrics: Any) -> Dict[str, float]:
        """Calculate improvement metrics."""
        return {
            "character_accuracy_delta": round(
                corrected_metrics.char_accuracy - ocr_metrics.char_accuracy, 4
            ),
            "word_accuracy_delta": round(
                corrected_metrics.word_accuracy - ocr_metrics.word_accuracy, 4
            ),
            "old_char_preservation_delta": round(
                corrected_metrics.old_char_preservation - ocr_metrics.old_char_preservation, 4
            ),
            "case_accuracy_delta": round(
                corrected_metrics.case_accuracy - ocr_metrics.case_accuracy, 4
            ),
        }

    def _serialize_result(self, result: OCRCorrectionResult) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {
            "extracted_text": result.extracted_text,
            "correction_mode": result.correction_mode.value,
            "model_name": result.model_name,
        }

        if result.correction_mode == CorrectionMode.LINE and isinstance(
            result.corrected_text, LineCorrection
        ):
            serialized["corrected_text"] = result.corrected_text.corrected_text
            serialized["confidence"] = result.corrected_text.confidence
        elif result.correction_mode == CorrectionMode.PARA and isinstance(
            result.corrected_text, ParagraphCorrection
        ):
            serialized["paragraphs"] = result.corrected_text.paragraphs
            serialized["paragraph_boundaries"] = result.corrected_text.paragraph_boundaries
            serialized["corrected_text"] = "\n\n".join(result.corrected_text.paragraphs)
            if result.corrected_text.confidence_scores is not None:
                serialized["confidence_scores"] = result.corrected_text.confidence_scores

        # Add optional fields
        if result.metrics:
            serialized["metrics"] = result.metrics
        if result.improvement:
            serialized["improvement"] = result.improvement
        if result.error_analysis:
            serialized["error_analysis"] = result.error_analysis

        return serialized
