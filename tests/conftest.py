"""
Pytest configuration and shared fixtures for LLM OCR package tests.
"""
import pytest
import tempfile
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_ocr.config import ModelConfig, EvaluationConfig
from llm_ocr.models import Line, OCRResult, OCRMetrics, ProcessingMode
from llm_ocr.evaluators.evaluator import OCREvaluator
from llm_ocr.prompts.prompt import PromptVersion, ModelType


# Test data fixtures
@pytest.fixture
def sample_text_pairs():
    """Sample ground truth and extracted text pairs for testing."""
    return [
        {
            "ground_truth": "Hello world",
            "extracted": "Hello world",
            "expected_char_accuracy": 1.0,
            "expected_word_accuracy": 1.0
        },
        {
            "ground_truth": "Hello world",
            "extracted": "Helo world",
            "expected_char_accuracy": 0.909,  # 10/11 chars correct
            "expected_word_accuracy": 0.5      # 1/2 words correct
        },
        {
            "ground_truth": "Testing OCR",
            "extracted": "Testing 0CR",
            "expected_char_accuracy": 0.909,  # 10/11 chars correct
            "expected_word_accuracy": 0.5      # 1/2 words correct
        },
        {
            "ground_truth": "Café résumé",
            "extracted": "Cafe resume",
            "expected_char_accuracy": 0.818,  # 9/11 chars correct (accents lost)
            "expected_word_accuracy": 0.0      # 0/2 words exactly correct
        }
    ]


@pytest.fixture
def sample_old_russian_text():
    """Sample Old Russian text with historical characters."""
    return {
        "ground_truth": "Вѣрую въ единаго Бога Отца",
        "extracted_good": "Вѣрую въ единаго Бога Отца",
        "extracted_bad": "Верую в единаго Бога Отца",
        "old_chars": "ѣъ"
    }


@pytest.fixture
def sample_line_objects():
    """Sample Line objects for testing."""
    return [
        Line(
            text="First line of text",
            line_id="line_001",
            position={"x": 100, "y": 100, "width": 200, "height": 20}
        ),
        Line(
            text="Second line of text",
            line_id="line_002", 
            position={"x": 100, "y": 120, "width": 250, "height": 20}
        ),
        Line(
            text="Third line with special chars: àéîöü",
            line_id="line_003",
            position={"x": 100, "y": 140, "width": 300, "height": 20}
        )
    ]


@pytest.fixture
def sample_base64_image():
    """Sample base64 encoded image data."""
    # Create a minimal PNG image (1x1 black pixel)
    png_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13'
        b'\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0bIDATx\x9cc```'
        b'\x00\x00\x00\x04\x00\x01\x827\x9a\xd1\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    return base64.b64encode(png_data).decode('utf-8')


# Configuration fixtures
@pytest.fixture
def default_model_config():
    """Default model configuration for testing."""
    return ModelConfig(
        max_tokens=1024,
        temperature=0.0,
        sliding_window_size=3,
        batch_size=5,
        model_name="test-model",
        model_type=ModelType.CLAUDE,
        prompt_version=PromptVersion.V1
    )


@pytest.fixture
def default_evaluation_config():
    """Default evaluation configuration for testing."""
    return EvaluationConfig(
        old_russian_chars='ѣѲѳѵѡѠѢѴѶѷѸѹѺѻѼѽѾѿъь',
        include_detailed_analysis=True,
        use_char_accuracy=True,
        use_word_accuracy=True,
        use_old_char_preservation=True,
        use_case_accuracy=True
    )


@pytest.fixture
def ocr_evaluator(default_evaluation_config):
    """OCR evaluator instance for testing."""
    return OCREvaluator(default_evaluation_config)


# Mock fixtures
@pytest.fixture
def mock_llm_model():
    """Mock LLM model for testing."""
    mock_model = MagicMock()
    mock_model.__class__.__name__ = "MockLLMModel"
    mock_model.model_name = "mock-model"
    
    # Configure mock responses
    mock_model.process_single_line.return_value = {
        "line": "Mocked single line response"
    }
    
    mock_model.process_sliding_window.return_value = {
        "lines": [
            {"line": "Mocked window line 1"},
            {"line": "Mocked window line 2"},
            {"line": "Mocked window line 3"}
        ]
    }
    
    mock_model.process_full_page.return_value = "Mocked full page response"
    mock_model.correct_text.return_value = "Mocked corrected text"
    
    return mock_model


@pytest.fixture
def mock_api_responses():
    """Mock API responses for different providers."""
    return {
        "claude": {
            "content": [{"text": '{"lines": [{"line": "Mocked Claude response"}]}'}],
            "usage": {"input_tokens": 100, "output_tokens": 50}
        },
        "openai": {
            "choices": [{"message": {"content": '{"lines": [{"line": "Mocked OpenAI response"}]}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        },
        "gemini": {
            "candidates": [{"content": {"parts": [{"text": '{"lines": [{"line": "Mocked Gemini response"}]}'}]}}]
        }
    }


# Temporary file fixtures
@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_alto_xml(temp_dir):
    """Sample ALTO XML file for testing."""
    alto_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
        <Layout>
            <Page WIDTH="600" HEIGHT="400">
                <TextBlock>
                    <TextLine ID="line_001" HPOS="100" VPOS="100" WIDTH="400" HEIGHT="20">
                        <Polygon POINTS="100,100 500,100 500,120 100,120"/>
                        <String CONTENT="First line of text" HPOS="100" VPOS="100" WIDTH="400" HEIGHT="20"/>
                    </TextLine>
                    <TextLine ID="line_002" HPOS="100" VPOS="140" WIDTH="400" HEIGHT="20">
                        <Polygon POINTS="100,140 500,140 500,160 100,160"/>
                        <String CONTENT="Second line of text" HPOS="100" VPOS="140" WIDTH="400" HEIGHT="20"/>
                    </TextLine>
                </TextBlock>
            </Page>
        </Layout>
    </alto>'''
    
    alto_file = temp_dir / "test_document.xml"
    alto_file.write_text(alto_content)
    return alto_file


@pytest.fixture
def sample_image_file(temp_dir):
    """Sample image file for testing."""
    import cv2
    import numpy as np
    
    # Create a simple white image with some text-like regions
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Add some dark rectangles to simulate text lines
    cv2.rectangle(image, (100, 100), (500, 120), (0, 0, 0), -1)  # First line
    cv2.rectangle(image, (100, 140), (500, 160), (0, 0, 0), -1)  # Second line
    
    image_file = temp_dir / "test_document.jpeg"
    cv2.imwrite(str(image_file), image)
    return image_file


@pytest.fixture
def sample_ground_truth_file(temp_dir):
    """Sample ground truth text file for testing."""
    ground_truth_content = "First line of text\nSecond line of text"
    gt_file = temp_dir / "test_document.txt"
    gt_file.write_text(ground_truth_content)
    return gt_file


# Test data generators
@pytest.fixture
def ocr_result_factory():
    """Factory for creating OCRResult objects."""
    def create_ocr_result(
        ground_truth: str = "test text",
        extracted: str = "test text", 
        model_name: str = "test-model",
        char_accuracy: float = 1.0,
        word_accuracy: float = 1.0
    ) -> OCRResult:
        metrics = OCRMetrics(
            char_accuracy=char_accuracy,
            word_accuracy=word_accuracy,
            old_char_preservation=1.0,
            case_accuracy=1.0
        )
        
        return OCRResult(
            ground_truth_text=ground_truth,
            extracted_text=extracted,
            model_name=model_name,
            metrics=metrics
        )
    
    return create_ocr_result


# Environment variable mocking
@pytest.fixture
def mock_env_vars():
    """Mock environment variables for API keys."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "OPENAI_API_KEY": "test-openai-key",
        "GEMINI_API_KEY": "test-gemini-key",
        "TOGETHER_API_KEY": "test-together-key"
    }
    
    with patch.dict('os.environ', env_vars):
        yield env_vars


# Performance testing fixtures
@pytest.fixture
def large_text_sample():
    """Large text sample for performance testing."""
    return "Lorem ipsum dolor sit amet. " * 1000


@pytest.fixture
def processing_modes():
    """All processing modes for parametrized tests."""
    return [ProcessingMode.SINGLE_LINE, ProcessingMode.SLIDING_WINDOW, ProcessingMode.FULL_PAGE]