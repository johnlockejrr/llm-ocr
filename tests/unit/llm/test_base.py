"""
Unit tests for LLM base module.
"""
import pytest
import json
from unittest.mock import MagicMock, patch

from llm_ocr.llm.base import BaseOCRModel


class MockOCRModel(BaseOCRModel):
    """Mock implementation of BaseOCRModel for testing."""
    
    def process_single_line(self, image_base64: str):
        return {"text": "mock result"}
        
    def process_sliding_window(self, images_base64):
        return {"results": ["mock1", "mock2"]}
        
    def process_full_page(self, page_image_base64: str, document_id: str):
        return "mock full page result"
        
    def correct_text(self, text: str, image_base64: str, mode: str = "line"):
        return f"corrected: {text}"


class TestBaseOCRModel:
    """Tests for BaseOCRModel abstract class."""
    
    def test_mock_model_creation(self):
        """Test creating a mock OCR model."""
        model = MockOCRModel("test-model")
        assert model.model_name == "test-model"
        
    def test_abstract_methods_implemented(self):
        """Test that mock implements all abstract methods."""
        model = MockOCRModel("test-model")
        
        # Test process_single_line
        result = model.process_single_line("image_data")
        assert result == {"text": "mock result"}
        
        # Test process_sliding_window
        result = model.process_sliding_window(["img1", "img2"])
        assert result == {"results": ["mock1", "mock2"]}
        
        # Test process_full_page
        result = model.process_full_page("page_data", "doc_id")
        assert result == "mock full page result"
        
        # Test correct_text
        result = model.correct_text("original text", "image_data")
        assert result == "corrected: original text"
        
    def test_json_extraction_valid_json(self):
        """Test JSON extraction with valid JSON."""
        model = MockOCRModel("test-model")
        
        # Test with valid JSON object
        response = '{"key": "value"}'
        result = model._extract_json_from_response(response)
        assert result == {"key": "value"}
        
        # Test with valid JSON array
        response = '[{"item": 1}, {"item": 2}]'
        result = model._extract_json_from_response(response)
        assert result == [{"item": 1}, {"item": 2}]
        
    def test_json_extraction_markdown_blocks(self):
        """Test JSON extraction from markdown code blocks."""
        model = MockOCRModel("test-model")
        
        # Test with markdown JSON block
        response = '''
        Here is the result:
        ```json
        {"extracted": "text"}
        ```
        '''
        result = model._extract_json_from_response(response)
        assert result == {"extracted": "text"}
        
    def test_json_extraction_embedded_json(self):
        """Test JSON extraction from text with embedded JSON."""
        model = MockOCRModel("test-model")
        
        # Test with embedded JSON object
        response = 'Some text before {"result": "success"} some text after'
        result = model._extract_json_from_response(response)
        assert result == {"result": "success"}
        
        # Test with embedded JSON array
        response = 'Before [{"item": 1}] after'
        result = model._extract_json_from_response(response)
        assert result == [{"item": 1}]
        
    def test_json_extraction_invalid_json(self):
        """Test JSON extraction with invalid JSON."""
        model = MockOCRModel("test-model")
        
        # Test with invalid JSON
        response = 'This is not JSON at all'
        with pytest.raises(ValueError):
            model._extract_json_from_response(response)
            
    def test_model_config_assignment(self):
        """Test that model config is assigned."""
        model = MockOCRModel("test-model")
        assert hasattr(model, 'config')
        
    def test_logger_creation(self):
        """Test that logger is created."""
        model = MockOCRModel("test-model")
        assert hasattr(model, 'logger')
        assert model.logger.name == "MockOCRModel"