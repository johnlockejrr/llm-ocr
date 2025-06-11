"""
Comprehensive tests for the new PromptBuilder system.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from llm_ocr.prompts.prompt_builder import PromptBuilder, PromptType, PromptVersion, get_default_builder, get_prompt


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "components": {
            "base_ocr": "Extract OCR text from 18th century Russian book",
            "orthography": "Preserve ѣ, Ѳ, ѳ, ѵ, ъ characters",
            "json_format": "Respond with JSON: {\"line\": \"text\"}",
            "text_only": "Return only extracted text"
        },
        "context_enrichment": {
            "v1": "",
            "v2": " from {book_year} book \"{book_title}\"",
            "v3": " processing \"{book_title}\" ({book_year})",
            "v4": " обрабатываете \"{book_title}\" {book_year} года"
        },
        "mode_instructions": {
            "single_line": "Process single line",
            "sliding_window": "Process sliding window",
            "full_page": "Process full page",
            "correction": "Process correction"
        },
        "output_formats": {
            "structured": {
                "single_line": "{json_format}",
                "sliding_window": "{json_format}",
                "full_page": "Return JSON array",
                "correction": "Return corrected text"
            },
            "simple": {
                "single_line": "{text_only}",
                "sliding_window": "{text_only}",
                "full_page": "Return text line by line",
                "correction": "Return corrected text"
            }
        }
    }


@pytest.fixture
def mock_metadata():
    """Mock metadata for testing."""
    return {
        "books": [
            {
                "title": "История государства Российского",
                "year": "1767",
                "publication_info": "Санкт-Петербург",
                "image_ids": ["test_doc_1", "test_doc_2"]
            },
            {
                "title": "Другая книга",
                "year": "1785",
                "publication_info": "Москва",
                "image_ids": ["test_doc_3"]
            }
        ]
    }


class TestPromptEnums:
    """Test the prompt enums."""
    
    def test_prompt_type_values(self):
        """Test PromptType enum values."""
        assert PromptType.STRUCTURED.value == "structured"
        assert PromptType.SIMPLE.value == "simple"
        
    def test_prompt_version_values(self):
        """Test PromptVersion enum values."""
        assert PromptVersion.V1.value == "v1"
        assert PromptVersion.V2.value == "v2"
        assert PromptVersion.V3.value == "v3"
        assert PromptVersion.V4.value == "v4"


class TestPromptBuilder:
    """Test the PromptBuilder class."""
    
    @pytest.fixture
    def builder_with_mocks(self, tmp_path, mock_config, mock_metadata):
        """Create PromptBuilder with mocked config and metadata."""
        config_path = tmp_path / "config.json"
        metadata_path = tmp_path / "metadata.json"
        
        with open(config_path, 'w') as f:
            json.dump(mock_config, f)
        with open(metadata_path, 'w') as f:
            json.dump(mock_metadata, f)
        
        builder = PromptBuilder(str(config_path), str(metadata_path))
        return builder
    
    def test_builder_initialization(self, builder_with_mocks):
        """Test PromptBuilder initialization."""
        builder = builder_with_mocks
        assert builder.config_path.exists()
        assert builder.metadata_path.exists()
        assert builder._config is None  # Lazy loading
        assert builder._metadata is None  # Lazy loading
    
    def test_config_lazy_loading(self, builder_with_mocks):
        """Test configuration lazy loading."""
        builder = builder_with_mocks
        
        # First access loads config
        config = builder.config
        assert config is not None
        assert builder._config is not None
        assert "components" in config
        
        # Second access uses cached config
        config2 = builder.config
        assert config2 is config  # Same object
    
    def test_metadata_retrieval_success(self, builder_with_mocks):
        """Test successful metadata retrieval."""
        builder = builder_with_mocks
        
        metadata = builder.get_metadata("test_doc_1")
        assert metadata["book_title"] == "История государства Российского"
        assert metadata["book_year"] == "1767"
        assert metadata["publication_info"] == "Санкт-Петербург"
    
    def test_metadata_retrieval_not_found(self, builder_with_mocks):
        """Test metadata retrieval for non-existent document."""
        builder = builder_with_mocks
        
        metadata = builder.get_metadata("nonexistent_doc")
        assert metadata == {}
    
    def test_basic_prompt_building(self, builder_with_mocks):
        """Test basic prompt building without variables."""
        builder = builder_with_mocks
        
        prompt = builder.build_prompt(
            mode="single_line",
            prompt_type=PromptType.STRUCTURED,
            version=PromptVersion.V1
        )
        
        expected = (
            "Extract OCR text from 18th century Russian book. "
            "Process single line. "
            "Preserve ѣ, Ѳ, ѳ, ѵ, ъ characters. "
            "Respond with JSON: {\"line\": \"text\"}"
        )
        assert prompt == expected
    
    def test_prompt_with_context_enrichment(self, builder_with_mocks):
        """Test prompt building with context enrichment."""
        builder = builder_with_mocks
        
        prompt = builder.build_prompt(
            mode="single_line",
            prompt_type=PromptType.STRUCTURED,
            version=PromptVersion.V2,
            book_title="Test Book",
            book_year="1750"
        )
        
        assert "from 1750 book \"Test Book\"" in prompt
        assert "Process single line" in prompt
        assert "JSON: {\"line\": \"text\"}" in prompt
    
    def test_prompt_with_document_id_metadata(self, builder_with_mocks):
        """Test prompt building with automatic metadata enrichment."""
        builder = builder_with_mocks
        
        prompt = builder.build_prompt(
            mode="full_page",
            prompt_type=PromptType.SIMPLE,
            version=PromptVersion.V2,
            document_id="test_doc_1"
        )
        
        assert "from 1767 book \"История государства Российского\"" in prompt
        assert "Process full page" in prompt
        assert "Return text line by line" in prompt
    
    def test_all_modes_structured(self, builder_with_mocks):
        """Test all processing modes with structured output."""
        builder = builder_with_mocks
        modes = ["single_line", "sliding_window", "full_page", "correction"]
        
        for mode in modes:
            prompt = builder.build_prompt(
                mode=mode,
                prompt_type=PromptType.STRUCTURED,
                version=PromptVersion.V1
            )
            assert mode.replace("_", " ") in prompt.lower()
            assert "Extract OCR text" in prompt
            assert len(prompt) > 50  # Reasonable length check
    
    def test_all_modes_simple(self, builder_with_mocks):
        """Test all processing modes with simple output."""
        builder = builder_with_mocks
        modes = ["single_line", "sliding_window", "full_page", "correction"]
        
        for mode in modes:
            prompt = builder.build_prompt(
                mode=mode,
                prompt_type=PromptType.SIMPLE,
                version=PromptVersion.V1
            )
            assert mode.replace("_", " ") in prompt.lower()
            assert "Return" in prompt  # Simple prompts should have "Return" instruction
    
    def test_all_versions(self, builder_with_mocks):
        """Test all prompt versions."""
        builder = builder_with_mocks
        versions = [PromptVersion.V1, PromptVersion.V2, PromptVersion.V3, PromptVersion.V4]
        
        for version in versions:
            prompt = builder.build_prompt(
                mode="single_line",
                prompt_type=PromptType.STRUCTURED,
                version=version,
                book_title="Test",
                book_year="1750"
            )
            assert len(prompt) > 30
            if version != PromptVersion.V1:
                # V2+ should have context enrichment
                assert "Test" in prompt or "1750" in prompt
    
    def test_russian_v4_prompt(self, builder_with_mocks):
        """Test Russian language V4 prompt."""
        builder = builder_with_mocks
        
        prompt = builder.build_prompt(
            mode="single_line",
            prompt_type=PromptType.STRUCTURED,
            version=PromptVersion.V4,
            book_title="Тестовая книга",
            book_year="1750"
        )
        
        assert "обрабатываете \"Тестовая книга\" 1750 года" in prompt
    
    def test_component_substitution(self, builder_with_mocks):
        """Test that component references are properly substituted."""
        builder = builder_with_mocks
        
        prompt = builder.build_prompt(
            mode="single_line",
            prompt_type=PromptType.STRUCTURED,
            version=PromptVersion.V1
        )
        
        # Should not contain component references like {json_format}
        assert "{json_format}" not in prompt
        assert "{text_only}" not in prompt
        # Should contain actual values
        assert "JSON: {\"line\": \"text\"}" in prompt
    
    def test_missing_template_variable(self, builder_with_mocks):
        """Test handling of missing template variables."""
        builder = builder_with_mocks
        
        # This should not raise an exception, just log a warning
        prompt = builder.build_prompt(
            mode="single_line",
            prompt_type=PromptType.STRUCTURED,
            version=PromptVersion.V2
            # Missing book_title and book_year
        )
        
        # Should still return a prompt, possibly with unfilled templates
        assert len(prompt) > 30
        assert "Extract OCR text" in prompt
    
    def test_invalid_mode(self, builder_with_mocks):
        """Test error handling for invalid mode."""
        builder = builder_with_mocks
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            builder.build_prompt(
                mode="invalid_mode",
                prompt_type=PromptType.STRUCTURED,
                version=PromptVersion.V1
            )
    
    def test_invalid_prompt_type(self, builder_with_mocks):
        """Test error handling for invalid prompt type."""
        builder = builder_with_mocks
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            builder.build_prompt(
                mode="single_line",
                prompt_type="invalid_type",  # Wrong type
                version=PromptVersion.V1
            )
    
    def test_invalid_version(self, builder_with_mocks):
        """Test error handling for invalid version."""
        builder = builder_with_mocks
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            builder.build_prompt(
                mode="single_line",
                prompt_type=PromptType.STRUCTURED,
                version="invalid_version"  # Wrong type
            )
    
    def test_missing_config_file(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            builder = PromptBuilder("nonexistent_config.json")
            builder.config  # Trigger lazy loading
    
    def test_invalid_json_config(self):
        """Test error handling for invalid JSON config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()
            
            builder = PromptBuilder(f.name)
            with pytest.raises(json.JSONDecodeError):
                builder.config


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_default_builder(self):
        """Test get_default_builder function."""
        builder1 = get_default_builder()
        builder2 = get_default_builder()
        
        # Should return the same instance (singleton pattern)
        assert builder1 is builder2
        assert isinstance(builder1, PromptBuilder)
    
    @patch('llm_ocr.prompts.prompt_builder.get_default_builder')
    def test_get_prompt_convenience_function(self, mock_get_builder):
        """Test get_prompt convenience function."""
        mock_builder = mock_get_builder.return_value
        mock_builder.build_prompt.return_value = "test prompt"
        
        result = get_prompt(
            mode="single_line",
            prompt_type=PromptType.STRUCTURED,
            version=PromptVersion.V1,
            test_arg="value"
        )
        
        assert result == "test prompt"
        mock_builder.build_prompt.assert_called_once_with(
            "single_line",
            PromptType.STRUCTURED,
            PromptVersion.V1,
            test_arg="value"
        )


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_kwargs(self, tmp_path, mock_config):
        """Test prompt building with no additional kwargs."""
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(mock_config, f)
        
        builder = PromptBuilder(str(config_path))
        prompt = builder.build_prompt(
            mode="single_line",
            prompt_type=PromptType.STRUCTURED,
            version=PromptVersion.V1
        )
        
        assert len(prompt) > 0
        assert "Extract OCR text" in prompt
    
    def test_missing_metadata_file(self, tmp_path, mock_config):
        """Test behavior when metadata file is missing."""
        config_path = tmp_path / "config.json"
        metadata_path = tmp_path / "nonexistent.json"
        
        with open(config_path, 'w') as f:
            json.dump(mock_config, f)
        
        builder = PromptBuilder(str(config_path), str(metadata_path))
        
        # Should return empty metadata without crashing
        metadata = builder.get_metadata("test_doc")
        assert metadata == {}
    
    def test_case_insensitive_mode(self, tmp_path, mock_config):
        """Test that mode names are case insensitive."""
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(mock_config, f)
        
        builder = PromptBuilder(str(config_path))
        
        # Test both cases
        prompt1 = builder.build_prompt("single_line", PromptType.STRUCTURED, PromptVersion.V1)
        prompt2 = builder.build_prompt("SINGLE_LINE", PromptType.STRUCTURED, PromptVersion.V1)
        
        assert prompt1 == prompt2