import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from llm_ocr.processors.alto import ALTOLine

from ..config import EvaluationConfig
from ..evaluators.evaluation import OCREvaluationService
from ..evaluators.evaluator import OCREvaluator
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

        # Use OCREvaluationService for comprehensive evaluation functionality
        self.evaluation_service = OCREvaluationService(self.config)

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
            # Use the evaluation service to generate comprehensive reports
            model_report = self.evaluation_service.generate_report(model_results, include_details)
            report[model_name] = model_report

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
