"""
Mock LLM providers for testing without making actual API calls.
"""
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional

from llm_ocr.llm.base import BaseOCRModel


class MockClaudeModel(BaseOCRModel):
    """Mock Claude model for testing."""
    
    def __init__(self, model_name: str = "mock-claude"):
        super().__init__(model_name)
        self.call_count = 0
        self.last_request = None
    
    def process_single_line(self, image_base64: str) -> Dict[str, Any]:
        self.call_count += 1
        self.last_request = {"type": "single_line", "image_length": len(image_base64)}
        return {"line": f"Mock Claude single line response {self.call_count}"}
    
    def process_sliding_window(self, images_base64: List[str]) -> Optional[Dict[str, Any]]:
        self.call_count += 1
        self.last_request = {"type": "sliding_window", "num_images": len(images_base64)}
        return {
            "lines": [
                {"line": f"Mock Claude window line {i+1}"} 
                for i in range(len(images_base64))
            ]
        }
    
    def process_full_page(self, page_image_base64: str, id: str) -> str:
        self.call_count += 1
        self.last_request = {"type": "full_page", "id": id, "image_length": len(page_image_base64)}
        return f"Mock Claude full page response for {id}"
    
    def correct_text(self, text: str, image_base64: str, mode: str = "line") -> str:
        self.call_count += 1
        self.last_request = {"type": "correction", "text": text, "mode": mode}
        return f"Corrected: {text}"


class MockOpenAIModel(BaseOCRModel):
    """Mock OpenAI model for testing."""
    
    def __init__(self, model_name: str = "mock-gpt"):
        super().__init__(model_name)
        self.call_count = 0
        self.last_request = None
    
    def process_single_line(self, image_base64: str) -> Dict[str, Any]:
        self.call_count += 1
        self.last_request = {"type": "single_line", "image_length": len(image_base64)}
        return {"line": f"Mock GPT single line response {self.call_count}"}
    
    def process_sliding_window(self, images_base64: List[str]) -> Optional[Dict[str, Any]]:
        self.call_count += 1
        self.last_request = {"type": "sliding_window", "num_images": len(images_base64)}
        return {
            "lines": [
                {"line": f"Mock GPT window line {i+1}"} 
                for i in range(len(images_base64))
            ]
        }
    
    def process_full_page(self, page_image_base64: str, id: str) -> str:
        self.call_count += 1
        self.last_request = {"type": "full_page", "id": id, "image_length": len(page_image_base64)}
        return f"Mock GPT full page response for {id}"
    
    def correct_text(self, text: str, image_base64: str, mode: str = "line") -> str:
        self.call_count += 1
        self.last_request = {"type": "correction", "text": text, "mode": mode}
        return f"GPT corrected: {text}"


class MockFailingModel(BaseOCRModel):
    """Mock model that simulates failures for error testing."""
    
    def __init__(self, model_name: str = "mock-failing"):
        super().__init__(model_name)
        self.should_fail = True
        
    def process_single_line(self, image_base64: str) -> Dict[str, Any]:
        if self.should_fail:
            raise Exception("Mock API failure")
        return {"line": "Success after failure"}
    
    def process_sliding_window(self, images_base64: List[str]) -> Optional[Dict[str, Any]]:
        if self.should_fail:
            raise Exception("Mock API failure")
        return {"lines": [{"line": "Success after failure"}]}
    
    def process_full_page(self, page_image_base64: str, id: str) -> str:
        if self.should_fail:
            raise Exception("Mock API failure")
        return "Success after failure"
    
    def correct_text(self, text: str, image_base64: str, mode: str = "line") -> str:
        if self.should_fail:
            raise Exception("Mock API failure")
        return f"Corrected after failure: {text}"


def mock_anthropic_client():
    """Create a mock anthropic client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"lines": [{"line": "Mock Anthropic response"}]}')]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_response
    return mock_client


def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"lines": [{"line": "Mock OpenAI response"}]}'))
    ]
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def mock_gemini_client():
    """Create a mock Gemini client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.candidates = [
        MagicMock(content=MagicMock(parts=[
            MagicMock(text='{"lines": [{"line": "Mock Gemini response"}]}')
        ]))
    ]
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


class MockProviderFactory:
    """Factory for creating mock providers with different behaviors."""
    
    @staticmethod
    def create_mock_providers() -> Dict[str, BaseOCRModel]:
        """Create a set of mock providers for testing."""
        return {
            "claude": MockClaudeModel("mock-claude"),
            "openai": MockOpenAIModel("mock-gpt"),
            "failing": MockFailingModel("mock-failing")
        }
    
    @staticmethod
    def patch_all_providers():
        """Context manager to patch all LLM provider imports."""
        return patch.multiple(
            'llm_ocr.llm',
            claude=patch('anthropic.Anthropic', return_value=mock_anthropic_client()),
            openai=patch('openai.OpenAI', return_value=mock_openai_client()),
            gemini=patch('google.genai.Client', return_value=mock_gemini_client()),
        )


# Predefined mock responses for different scenarios
MOCK_RESPONSES = {
    "simple_line": {
        "claude": '{"line": "Simple test line"}',
        "openai": '{"line": "Simple test line"}',
        "gemini": '{"line": "Simple test line"}'
    },
    "multiple_lines": {
        "claude": '{"lines": [{"line": "Line 1"}, {"line": "Line 2"}, {"line": "Line 3"}]}',
        "openai": '{"lines": [{"line": "Line 1"}, {"line": "Line 2"}, {"line": "Line 3"}]}',
        "gemini": '{"lines": [{"line": "Line 1"}, {"line": "Line 2"}, {"line": "Line 3"}]}'
    },
    "correction": {
        "claude": "Corrected text output",
        "openai": "Corrected text output", 
        "gemini": "Corrected text output"
    },
    "error_cases": {
        "invalid_json": "This is not valid JSON",
        "empty_response": "",
        "malformed_structure": '{"wrong": "structure"}'
    }
}