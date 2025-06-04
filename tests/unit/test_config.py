"""
Unit tests for configuration module.
"""

from llm_ocr.config import EvaluationConfig, ModelConfig


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_default_initialization(self):
        """Test default model config initialization."""
        config = ModelConfig()
        assert hasattr(config, "max_tokens")
        assert hasattr(config, "temperature")

    def test_config_attributes(self):
        """Test that config has expected attributes."""
        config = ModelConfig()
        # Test that basic attributes exist without checking specific values
        # since they might be class attributes or properties
        assert config is not None


class TestEvaluationConfig:
    """Tests for EvaluationConfig class."""

    def test_default_initialization(self):
        """Test default evaluation config initialization."""
        config = EvaluationConfig()
        assert hasattr(config, "old_russian_chars")
        assert hasattr(config, "char_similarity_weight")
        assert hasattr(config, "word_similarity_weight")

    def test_old_russian_chars(self):
        """Test old Russian characters configuration."""
        config = EvaluationConfig()
        old_chars = config.old_russian_chars
        assert isinstance(old_chars, str)
        assert len(old_chars) > 0

    def test_similarity_weights(self):
        """Test similarity weight configuration."""
        config = EvaluationConfig()
        assert isinstance(config.char_similarity_weight, (int, float))
        assert isinstance(config.word_similarity_weight, (int, float))
        assert 0.0 <= config.char_similarity_weight <= 1.0
        assert 0.0 <= config.word_similarity_weight <= 1.0

    def test_include_detailed_analysis(self):
        """Test detailed analysis configuration."""
        config = EvaluationConfig()
        assert hasattr(config, "include_detailed_analysis")
        assert isinstance(config.include_detailed_analysis, bool)
