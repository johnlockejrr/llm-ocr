"""
Unit tests for data models.
"""
from llm_ocr.models import (
    Line, OCRResult, OCRMetrics, OCRCorrectionResult, 
    ProcessingMode
)


class TestProcessingMode:
    """Test ProcessingMode enum."""
    
    def test_processing_mode_values(self):
        """Test that processing modes have correct values."""
        assert ProcessingMode.SINGLE_LINE.value == "singleline"
        assert ProcessingMode.SLIDING_WINDOW.value == "slidingwindow"
        assert ProcessingMode.FULL_PAGE.value == "fullpage"
    
    def test_processing_mode_iteration(self):
        """Test that all processing modes are accessible."""
        modes = list(ProcessingMode)
        assert len(modes) == 3
        assert ProcessingMode.SINGLE_LINE in modes
        assert ProcessingMode.SLIDING_WINDOW in modes
        assert ProcessingMode.FULL_PAGE in modes


class TestOCRMetrics:
    """Test OCRMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test default metric values."""
        metrics = OCRMetrics()
        assert metrics.char_accuracy == 0.0
        assert metrics.char_accuracy_case_insensitive == 0.0
        assert metrics.word_accuracy == 0.0
        assert metrics.old_char_preservation == 0.0
        assert metrics.case_accuracy == 0.0
        assert metrics.case_errors == []
        assert metrics.capitalization_analysis == {}
    
    def test_custom_initialization(self):
        """Test metrics with custom values."""
        case_errors = [{"error": "test"}]
        cap_analysis = {"uppercase": 5}
        
        metrics = OCRMetrics(
            char_accuracy=0.95,
            word_accuracy=0.8,
            old_char_preservation=0.9,
            case_accuracy=0.85,
            case_errors=case_errors,
            capitalization_analysis=cap_analysis
        )
        
        assert metrics.char_accuracy == 0.95
        assert metrics.word_accuracy == 0.8
        assert metrics.old_char_preservation == 0.9
        assert metrics.case_accuracy == 0.85
        assert metrics.case_errors == case_errors
        assert metrics.capitalization_analysis == cap_analysis


class TestLine:
    """Test Line dataclass."""
    
    def test_basic_line_creation(self):
        """Test creating a basic line."""
        line = Line(text="Test line")
        assert line.text == "Test line"
        assert line.image_data is None
        assert line.base64_image is None
        assert line.line_id is None
        assert line.position is None
    
    def test_line_with_all_fields(self):
        """Test line with all fields populated."""
        position = {"x": 100, "y": 200, "width": 300, "height": 20}
        line = Line(
            text="Complete line",
            image_data=b"fake_image_data",
            base64_image="base64_string",
            line_id="line_001",
            position=position
        )
        
        assert line.text == "Complete line"
        assert line.image_data == b"fake_image_data"
        assert line.base64_image == "base64_string"
        assert line.line_id == "line_001"
        assert line.position == position
    
    def test_get_base64_image(self):
        """Test get_base64_image method."""
        # Test with base64_image set
        line = Line(text="test", base64_image="encoded_data")
        assert line.get_base64_image() == "encoded_data"
        
        # Test with no base64_image
        line = Line(text="test")
        assert line.get_base64_image() == ""


class TestOCRResult:
    """Test OCRResult dataclass."""
    
    def test_basic_ocr_result(self):
        """Test creating basic OCR result."""
        result = OCRResult(
            ground_truth_text="original",
            extracted_text="extracted",
            processing_time=1.5
        )
        
        assert result.ground_truth_text == "original"
        assert result.extracted_text == "extracted"
        assert result.processing_time == 1.5
        assert result.model_name is None
        assert result.metrics is None
        assert result.error_analysis is None
        assert result.word_analysis is None
    
    def test_complete_ocr_result(self):
        """Test OCR result with all fields."""
        metrics = OCRMetrics(char_accuracy=0.95)
        error_analysis = {"errors": ["test"]}
        word_analysis = {"words": 5}
        
        result = OCRResult(
            ground_truth_text="original",
            extracted_text="extracted", 
            processing_time=2.0,
            model_name="test-model",
            metrics=metrics,
            error_analysis=error_analysis,
            word_analysis=word_analysis
        )
        
        assert result.model_name == "test-model"
        assert result.metrics == metrics
        assert result.error_analysis == error_analysis
        assert result.word_analysis == word_analysis


class TestOCRCorrectionResult:
    """Test OCRCorrectionResult dataclass."""
    
    def test_basic_correction_result(self):
        """Test creating basic correction result."""
        result = OCRCorrectionResult(
            extracted_text="original",
            corrected_text="corrected",
            processing_time=1.0,
            model_name="corrector-model"
        )
        
        assert result.extracted_text == "original"
        assert result.corrected_text == "corrected"
        assert result.processing_time == 1.0
        assert result.model_name == "corrector-model"
        assert result.metrics is None
        assert result.improvement is None
        assert result.similarity is None
    
    def test_complete_correction_result(self):
        """Test correction result with all fields."""
        metrics = OCRMetrics(char_accuracy=0.98)
        improvement = {"char_accuracy_delta": 0.1}
        
        result = OCRCorrectionResult(
            extracted_text="original",
            corrected_text="corrected",
            processing_time=1.5,
            model_name="corrector-model",
            metrics=metrics,
            improvement=improvement,
            similarity=0.95
        )
        
        assert result.metrics == metrics
        assert result.improvement == improvement
        assert result.similarity == 0.95