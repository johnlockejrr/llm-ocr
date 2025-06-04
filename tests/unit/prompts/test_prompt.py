"""
Unit tests for prompt module.
"""

from llm_ocr.prompts.prompt import ModelType, PromptVersion


class TestPromptVersion:
    """Tests for PromptVersion enum."""

    def test_prompt_version_values(self):
        """Test PromptVersion enum values."""
        # Values are auto-generated integers, just check they exist
        assert PromptVersion.V1 is not None
        assert PromptVersion.V2 is not None

    def test_prompt_version_iteration(self):
        """Test iterating over PromptVersion values."""
        versions = list(PromptVersion)
        assert len(versions) >= 2
        assert PromptVersion.V1 in versions
        assert PromptVersion.V2 in versions


class TestModelType:
    """Tests for ModelType enum."""

    def test_model_type_values(self):
        """Test ModelType enum values."""
        # Values are auto-generated integers, just check they exist
        assert ModelType.CLAUDE is not None
        assert ModelType.GPT is not None
        assert ModelType.GEMINI is not None
        assert ModelType.TOGETHER is not None

    def test_model_type_iteration(self):
        """Test iterating over ModelType values."""
        types = list(ModelType)
        assert len(types) >= 4
        assert ModelType.CLAUDE in types
        assert ModelType.GPT in types
        assert ModelType.GEMINI in types
        assert ModelType.TOGETHER in types


class TestPromptFunctions:
    """Tests for prompt utility functions."""

    def test_prompt_functions_exist(self):
        """Test that prompt functions are available."""
        # Import the module to check for functions
        import llm_ocr.prompts.prompt as prompt_module

        # Check that the module has some callable functions/methods
        module_contents = dir(prompt_module)

        # Should have enums at minimum
        assert "PromptVersion" in module_contents
        assert "ModelType" in module_contents
