from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class ProcessingMode(Enum):
    """Defines different processing strategies for OCR."""
    SINGLE_LINE = "singleline"
    SLIDING_WINDOW = "slidingwindow"
    FULL_PAGE = "fullpage"


@dataclass
class OCRMetrics:
    """Dataclass for storing OCR evaluation metrics."""
    char_accuracy: float = 0.0
    char_accuracy_case_insensitive: float = 0.0
    word_accuracy: float = 0.0
    old_char_preservation: float = 0.0
    case_accuracy: float = 0.0
    case_errors: List[Dict[str, Any]] = field(default_factory=list)
    capitalization_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Line:
    """Represents a line of text with its image data."""
    text: str
    image_data: Optional[bytes] = None
    base64_image: Optional[str] = None
    line_id: Optional[str] = None
    position: Optional[Dict[str, int]] = None
    
    def get_base64_image(self) -> str:
        """Return base64 encoded image if available."""
        return self.base64_image if self.base64_image else ""


@dataclass
class OCRResult:
    """Dataclass for storing OCR results."""
    ground_truth_text: str
    extracted_text: str
    processing_time: float
    model_name: Optional[str] = None
    metrics: Optional[OCRMetrics] = None
    error_analysis: Optional[Dict[str, Any]] = None
    word_analysis: Optional[Dict[str, Any]] = None


@dataclass
class OCRCorrectionResult:
    """Dataclass for storing OCR correction results."""
    extracted_text: str
    corrected_text: str
    processing_time: float
    model_name: str
    metrics: Optional[OCRMetrics] = None
    improvement: Optional[Dict[str, float]] = None
    error_analysis: Optional[Dict[str, Any]] = None
    word_analysis: Optional[Dict[str, Any]] = None
    similarity: Optional[float] = None