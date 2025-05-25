from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


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
    model_name: Optional[str] = None
    metrics: Optional[OCRMetrics] = None
    error_analysis: Optional[Dict[str, Any]] = None
    word_analysis: Optional[Dict[str, Any]] = None


class CorrectionMode(Enum):
    LINE = "line"
    PARA = "para"


@dataclass
class LineCorrection:
    """Single line correction result."""

    corrected_text: str
    confidence: Optional[float] = None


@dataclass
class ParagraphCorrection:
    """Paragraph-based correction result."""

    paragraphs: List[str]
    paragraph_boundaries: List[int]  # Character positions where paragraphs start
    confidence_scores: Optional[List[float]] = None


@dataclass
class OCRCorrectionResult:
    """Enhanced correction result supporting multiple modes."""

    extracted_text: str
    correction_mode: CorrectionMode
    corrected_text: Union[LineCorrection, ParagraphCorrection]
    model_name: str
    metrics: Optional[Dict[str, Any]] = None
    improvement: Optional[Dict[str, float]] = None
    error_analysis: Optional[Dict[str, Any]] = None
