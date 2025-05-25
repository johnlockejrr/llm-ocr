"""
OCR Pipeline Module - Focused only on OCR processing with no evaluation.
"""

import logging
from typing import Any, Dict, List, Optional

from llm_ocr.processors.alto import ALTOLine

from ..config import WINDOW_SIZE, EvaluationConfig
from ..evaluators.evaluation import OCREvaluationService
from ..evaluators.evaluator import OCREvaluator
from ..llm.base import BaseOCRModel
from ..models import ProcessingMode


class OCRPipeline:
    """Pipeline focused solely on OCR processing without evaluation."""

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

        # Use OCREvaluationService for comprehensive evaluation functionality
        self.evaluation_service = OCREvaluationService(self.config)

    def ocr_document(
        self,
        lines: List[ALTOLine],
        image_str: str,
        id: str,
        mode: ProcessingMode = ProcessingMode.SINGLE_LINE,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run OCR on document lines with specified processing mode.

        Args:
            lines: List of Line objects
            image_str: Base64 encoded image string (required for FULL_PAGE mode)
            mode: Processing mode to use

        Returns:
            Dictionary mapping model names to lists of OCR results
        """
        try:
            results = {}

            self.logger.info(f"Processing with model: {self.model.__class__.__name__}")
            # Select processing strategy based on mode
            processor = self._get_processor(mode)
            model_results = processor(self.model, lines, image_str, document_id=id)

            # Store results directly as dictionaries with ground truth and extracted text
            results[self.model.__class__.__name__] = model_results

            return results

        except Exception as e:
            self.logger.error(f"Error running OCR on document: {str(e)}")
            raise

    def _get_processor(self, mode: ProcessingMode) -> Any:
        """
        Get the appropriate processing method based on mode.

        Args:
            mode: Processing mode

        Returns:
            Processing function
        """
        processors = {
            ProcessingMode.SINGLE_LINE: self._process_single_line,
            ProcessingMode.SLIDING_WINDOW: self._process_sliding_window,
            ProcessingMode.FULL_PAGE: self._process_full_page,
        }

        if mode not in processors:
            raise ValueError(f"Unknown processing mode: {mode}")

        return processors[mode]

    def _process_single_line(
        self,
        model: BaseOCRModel,
        lines: List[ALTOLine],
        image_str: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process lines one at a time.

        Args:
            model: OCR model
            lines: List of Line objects

        Returns:
            List of OCR result dictionaries
        """
        results = []

        for line in lines:
            try:
                # Process line
                result = model.process_single_line(line.get_base64_image())

                # Create simple result dictionary
                ocr_result = {
                    "ground_truth_text": line.text,
                    "extracted_text": result["line"],
                }

                results.append(ocr_result)

            except Exception as e:
                self.logger.error(f"Error processing line: {str(e)}")
                # Add empty result when processing fails
                results.append(
                    {
                        "ground_truth_text": line.text,
                        "extracted_text": "",
                        "error": str(e),
                    }
                )

        return results

    def _process_sliding_window(
        self,
        model: BaseOCRModel,
        lines: List[ALTOLine],
        image_str: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process lines using sliding window with context.

        Args:
            model: OCR model
            lines: List of Line objects
            image_str: Unused for sliding window mode

        Returns:
            List of OCR result dictionaries
        """
        results = []
        window_size = WINDOW_SIZE
        half_window = window_size // 2

        # Process all lines in batches
        i = 0
        while i < len(lines):
            try:
                # Calculate window indices
                window_start = max(0, i - half_window)
                window_end = min(len(lines), i + half_window + 1)

                # Get window lines
                window_lines = lines[window_start:window_end]
                window_images = [line.get_base64_image() for line in window_lines]
                # window_texts = [line.text for line in window_lines]

                # Process entire window
                window_result = model.process_sliding_window(window_images)

                if window_result is not None:
                    # Handle both single result and list of results
                    if isinstance(window_result, list):
                        # Multiple results returned
                        for j, line in enumerate(window_lines):
                            if j < len(window_result):
                                ocr_result = {
                                    "ground_truth_text": line.text,
                                    "extracted_text": window_result[j]["line"],
                                }
                                results.append(ocr_result)
                    else:
                        # Single result returned (middle line)
                        target_idx = min(half_window, len(window_lines) - 1)
                        ocr_result = {
                            "ground_truth_text": window_lines[target_idx].text,
                            "extracted_text": window_result["line"],
                        }
                        results.append(ocr_result)
                else:
                    # Handle failed window processing
                    self.logger.warning(f"No result returned for window at position {i}")
                    results.append(
                        {
                            "ground_truth_text": lines[i].text,
                            "extracted_text": "",
                            "error": "No result returned from model",
                        }
                    )

                # Move to next position
                if isinstance(window_result, list):
                    # Move by number of processed lines
                    i += len(window_result)
                else:
                    i += 1  # Move one line at a time for single results

            except Exception as e:
                self.logger.error(f"Error processing window starting at line {i}: {str(e)}")
                results.append(
                    {
                        "ground_truth_text": lines[i].text,
                        "extracted_text": "",
                        "error": str(e),
                    }
                )
                i += 1

        # Ensure we have a result for each input line
        while len(results) < len(lines):
            results.append(
                {
                    "ground_truth_text": lines[len(results)].text,
                    "extracted_text": "",
                    "error": "Missing result",
                }
            )

        return results[: len(lines)]  # Trim to original length

    def _process_full_page(
        self, model: BaseOCRModel, lines: List[ALTOLine], image_str: str, document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Process entire page at once.

        Args:
            model: OCR model
            lines: List of ALTOLine objects
            image_str: Base64 encoded image string
            document_id: Document ID

        Returns:
            List of OCR result dictionaries
        """
        if not image_str:
            raise ValueError("Full page mode requires image in base64 format.")

        texts = "\n".join([line.text for line in lines])

        try:
            extracted_lines = model.process_full_page(image_str, document_id=document_id)

            # Create result dictionary
            return [
                {
                    "ground_truth_text": texts,
                    "extracted_text": extracted_lines,
                }
            ]

        except Exception as e:
            self.logger.error(f"Error processing full page with model: {str(e)}")
            return [
                {
                    "ground_truth_text": texts,
                    "extracted_text": "",
                    "error": str(e),
                }
            ]
