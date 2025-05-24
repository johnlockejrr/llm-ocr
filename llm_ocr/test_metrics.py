"""
Tests for the metrics module.
"""
import unittest
from unittest.mock import patch, MagicMock
import difflib
from typing import Set

from llm_ocr.evaluators.evaluator import (
    BaseMetric,
    CharacterAccuracyMetric, 
    WordAccuracyMetric,
    OldCharPreservationMetric,
    CaseAccuracyMetric,
    ErrorAnalysisMetric,
    WordAnalysisMetric,
    SimilarityMetric,
    OCREvaluator,
    MetricsComparer
)
from llm_ocr.models import OCRMetrics
from llm_ocr.config import EvaluationConfig


class TestCharacterAccuracyMetric(unittest.TestCase):
    """Tests for CharacterAccuracyMetric."""
    
    def setUp(self):
        self.metric = CharacterAccuracyMetric()
    
    def test_name(self):
        """Test the name property."""
        self.assertEqual(self.metric.name, "char_accuracy")
    
    def test_perfect_match(self):
        """Test when texts match perfectly."""
        ground_truth = "This is a test."
        extracted = "This is a test."
        self.assertEqual(self.metric.evaluate(ground_truth, extracted), 1.0)
    
    def test_empty_strings(self):
        """Test with empty strings."""
        self.assertEqual(self.metric.evaluate("", ""), 1.0)
        self.assertEqual(self.metric.evaluate("text", ""), 0.0)
        self.assertEqual(self.metric.evaluate("", "text"), 0.0)
    
    def test_partial_match(self):
        """Test with partially matching texts."""
        ground_truth = "This is a test."
        extracted = "This is test."
        # The 'a ' is missing (2 characters out of 15)
        expected = 1 - (2 / 15)  # ~0.867
        self.assertAlmostEqual(self.metric.evaluate(ground_truth, extracted), expected, places=3)
    
    def test_no_match(self):
        """Test with completely different texts."""
        ground_truth = "This is a test."
        extracted = "Something else."
        # Using Levenshtein distance
        distance = self.metric._levenshtein_distance(ground_truth, extracted)
        max_length = max(len(ground_truth), len(extracted))
        expected = 1 - (distance / max_length)
        self.assertAlmostEqual(self.metric.evaluate(ground_truth, extracted), expected, places=3)
        # This should be close to 0
        self.assertLess(self.metric.evaluate(ground_truth, extracted), 0.3)
    
    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        self.assertEqual(self.metric._levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(self.metric._levenshtein_distance("saturday", "sunday"), 3)
        self.assertEqual(self.metric._levenshtein_distance("", ""), 0)
        self.assertEqual(self.metric._levenshtein_distance("", "abc"), 3)
        self.assertEqual(self.metric._levenshtein_distance("abc", ""), 3)


class TestWordAccuracyMetric(unittest.TestCase):
    """Tests for WordAccuracyMetric."""
    
    def setUp(self):
        self.metric = WordAccuracyMetric()
    
    def test_name(self):
        """Test the name property."""
        self.assertEqual(self.metric.name, "word_accuracy")
    
    def test_perfect_match(self):
        """Test when texts match perfectly."""
        ground_truth = "This is a test."
        extracted = "This is a test."
        self.assertEqual(self.metric.evaluate(ground_truth, extracted), 1.0)
    
    def test_empty_strings(self):
        """Test with empty strings."""
        self.assertEqual(self.metric.evaluate("", ""), 1.0)
        self.assertEqual(self.metric.evaluate("text", ""), 0.0)
        self.assertEqual(self.metric.evaluate("", "text"), 0.0)
    
    def test_partial_match(self):
        """Test with partially matching texts."""
        ground_truth = "This is a test."
        extracted = "This is test."
        # 2 matching words out of 4 words
        expected = 2 / 4
        self.assertAlmostEqual(self.metric.evaluate(ground_truth, extracted), expected, places=3)
    
    def test_word_order(self):
        """Test when word order matters."""
        ground_truth = "This is a test."
        extracted = "A test this is."
        # 0 matching words in the same position
        self.assertEqual(self.metric.evaluate(ground_truth, extracted), 0.0)


class TestOldCharPreservationMetric(unittest.TestCase):
    """Tests for OldCharPreservationMetric."""
    
    def setUp(self):
        self.special_chars = {'Ѣ', 'ѣ', 'І', 'і'}
        self.metric = OldCharPreservationMetric(self.special_chars)
    
    def test_name(self):
        """Test the name property."""
        self.assertEqual(self.metric.name, "old_char_preservation")
    
    def test_perfect_preservation(self):
        """Test when all special chars are preserved."""
        ground_truth = "This Ѣ has і some І special ѣ chars."
        extracted = "This Ѣ has і some І special ѣ chars."
        self.assertEqual(self.metric.evaluate(ground_truth, extracted), 1.0)
    
    def test_no_special_chars(self):
        """Test when no special chars are in the text."""
        ground_truth = "This has no special chars."
        extracted = "This has no special chars."
        self.assertEqual(self.metric.evaluate(ground_truth, extracted), 1.0)
    
    def test_missing_some_special_chars(self):
        """Test when some special chars are missing."""
        ground_truth = "This Ѣ has і some І special ѣ chars."
        extracted = "This Ѣ has i some I special ѣ chars."
        # 2 out of 4 special chars preserved
        self.assertEqual(self.metric.evaluate(ground_truth, extracted), 0.5)
    
    def test_all_special_chars_missing(self):
        """Test when all special chars are missing."""
        ground_truth = "This Ѣ has і some І special ѣ chars."
        extracted = "This E has i some I special e chars."
        self.assertEqual(self.metric.evaluate(ground_truth, extracted), 0.0)


class TestCaseAccuracyMetric(unittest.TestCase):
    """Tests for CaseAccuracyMetric."""
    
    def setUp(self):
        self.metric = CaseAccuracyMetric()
    
    def test_name(self):
        """Test the name property."""
        self.assertEqual(self.metric.name, "case_accuracy")
    
    def test_perfect_case_match(self):
        """Test when all case is preserved."""
        ground_truth = "This Is A Test."
        extracted = "This Is A Test."
        result = self.metric.evaluate(ground_truth, extracted)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(len(result["errors"]), 0)
    
    def test_case_errors(self):
        """Test when some case is incorrect."""
        ground_truth = "This Is A Test."
        extracted = "this is a Test."
        result = self.metric.evaluate(ground_truth, extracted)
        # 3 case errors out of 11 characters
        expected_accuracy = (11 - 3) / 11
        self.assertAlmostEqual(result["accuracy"], expected_accuracy, places=3)

    def test_capitalization_analysis(self):
        """Test capitalization analysis."""
        ground_truth = "This Is A Test."
        extracted = "this is a Test."
        result = self.metric.evaluate(ground_truth, extracted)
        analysis = result["analysis"]
        
        self.assertEqual(analysis["ground_truth_capitals"], 4)
        self.assertEqual(analysis["extracted_capitals"], 1)
        self.assertEqual(analysis["ground_truth_initial_caps"], 4)
        self.assertEqual(analysis["extracted_initial_caps"], 1)
        
        # Check positions
        self.assertEqual(analysis["capital_positions_ground_truth"], [0, 5, 8, 10])
        self.assertEqual(analysis["capital_positions_extracted"], [10])


class TestErrorAnalysisMetric(unittest.TestCase):
    """Tests for ErrorAnalysisMetric."""
    
    def setUp(self):
        self.special_chars = {'Ѣ', 'ѣ', 'І', 'і'}
        self.metric = ErrorAnalysisMetric(self.special_chars)
    
    def test_name(self):
        """Test the name property."""
        self.assertEqual(self.metric.name, "error_analysis")
    
    def test_no_errors(self):
        """Test when texts match perfectly."""
        ground_truth = "This is a test."
        extracted = "This is a test."
        result = self.metric.evaluate(ground_truth, extracted)
        self.assertEqual(result["total_errors"], 0)
        self.assertEqual(len(result["substitutions"]), 0)
        self.assertEqual(len(result["deletions"]), 0)
        self.assertEqual(len(result["insertions"]), 0)
    
    def test_substitutions(self):
        """Test character substitutions."""
        ground_truth = "This is a test."
        extracted = "That is a text."
        result = self.metric.evaluate(ground_truth, extracted)
        # The difflib SequenceMatcher finds these substitutions:
        # 'is' in 'This' -> 'at' in 'That'
        # 's' in 'test' -> 'x' in 'text'
        self.assertEqual(len(result["substitutions"]), 2)
        self.assertEqual(result["substitutions"][0]["ground_truth"], "is")
        self.assertEqual(result["substitutions"][0]["extracted"], "at")
        self.assertEqual(result["substitutions"][1]["ground_truth"], "s")
        self.assertEqual(result["substitutions"][1]["extracted"], "x")
    
    def test_deletions(self):
        """Test character deletions."""
        ground_truth = "This is a test."
        extracted = "This is test."
        result = self.metric.evaluate(ground_truth, extracted)
        # 1 deletion: 'a '
        self.assertEqual(len(result["deletions"]), 1)
        self.assertEqual(result["deletions"][0]["text"], "a ")
    
    def test_insertions(self):
        """Test character insertions."""
        ground_truth = "This is test."
        extracted = "This is a test."
        result = self.metric.evaluate(ground_truth, extracted)
        # 1 insertion: 'a '
        self.assertEqual(len(result["insertions"]), 1)
        self.assertEqual(result["insertions"][0]["text"], "a ")
    
    def test_special_char_errors(self):
        """Test special character errors."""
        ground_truth = "This Ѣ test with і."
        extracted = "This E test with i."
        result = self.metric.evaluate(ground_truth, extracted)
        # 2 special char errors
        self.assertEqual(len(result["special_char_errors"]), 2)
        self.assertIn('Ѣ', result["special_char_errors"][0]["special_chars_ground_truth"])
        self.assertIn('і', result["special_char_errors"][1]["special_chars_ground_truth"])


class TestWordAnalysisMetric(unittest.TestCase):
    """Tests for WordAnalysisMetric."""
    
    def setUp(self):
        self.metric = WordAnalysisMetric()
    
    def test_name(self):
        """Test the name property."""
        self.assertEqual(self.metric.name, "word_analysis")
    
    def test_no_word_errors(self):
        """Test when texts match perfectly."""
        ground_truth = "This is a test."
        extracted = "This is a test."
        result = self.metric.evaluate(ground_truth, extracted)
        self.assertEqual(result["total_word_errors"], 0)
        self.assertEqual(len(result["word_substitutions"]), 0)
        self.assertEqual(len(result["word_deletions"]), 0)
        self.assertEqual(len(result["word_insertions"]), 0)
    
    def test_word_substitutions(self):
        """Test word substitutions."""
        ground_truth = "This is a test."
        extracted = "That is a text."
        result = self.metric.evaluate(ground_truth, extracted)
        # 2 substitutions: 'This'->'That' and 'test'->'text'
        self.assertEqual(len(result["word_substitutions"]), 2)
        self.assertEqual(result["word_substitutions"][0]["ground_truth"], ["This"])
        self.assertEqual(result["word_substitutions"][0]["extracted"], ["That"])
        self.assertEqual(result["word_substitutions"][1]["ground_truth"], ["test."])
        self.assertEqual(result["word_substitutions"][1]["extracted"], ["text."])
    
    def test_word_deletions(self):
        """Test word deletions."""
        ground_truth = "The quick brown fox jumps."
        extracted = "The fox jumps."
        result = self.metric.evaluate(ground_truth, extracted)
        # 1 deletion: 'a test'
        self.assertEqual(len(result["word_deletions"]), 1)
        self.assertEqual(result["word_deletions"][0]["words"], ['quick', 'brown'])
    
    def test_word_insertions(self):
        """Test word insertions."""
        ground_truth = "The fox jumps."
        extracted = "The quick brown fox jumps."
        result = self.metric.evaluate(ground_truth, extracted)
    
        # difflib's SequenceMatcher typically groups consecutive insertions together
        # rather than treating them as separate insertions
        self.assertEqual(len(result["word_insertions"]), 1)
        self.assertEqual(result["word_insertions"][0]["words"], ["quick", "brown"])
        self.assertEqual(result["word_insertions"][0]["position"], 1)  # Position after "The"


class TestSimilarityMetric(unittest.TestCase):
    """Tests for SimilarityMetric."""
    
    def setUp(self):
        self.metric = SimilarityMetric(char_weight=0.7, word_weight=0.3)
    
    def test_name(self):
        """Test the name property."""
        self.assertEqual(self.metric.name, "similarity")
    
    def test_perfect_match(self):
        """Test when texts match perfectly."""
        ground_truth = "This is a test."
        extracted = "This is a test."
        self.assertEqual(self.metric.evaluate(ground_truth, extracted), 1.0)
    
    def test_partial_match(self):
        """Test with partially matching texts."""
        ground_truth = "This is a test."
        extracted = "This is test."
        
        # Character similarity (using Levenshtein)
        char_metric = CharacterAccuracyMetric()
        char_sim = char_metric.evaluate(ground_truth, extracted)
        
        # Word overlap
        ground_truth_words = set(ground_truth.split())
        extracted_words = set(extracted.split())
        word_overlap = len(ground_truth_words.intersection(extracted_words)) / max(
            len(ground_truth_words), len(extracted_words)
        )
        
        # Combined similarity
        expected = (0.7 * char_sim + 0.3 * word_overlap) / 1.0
        self.assertAlmostEqual(self.metric.evaluate(ground_truth, extracted), expected, places=3)
    
    def test_weights(self):
        """Test that weights are applied correctly."""
        ground_truth = "This is a test."
        extracted = "This is test."
        
        # Try with different weights
        metric1 = SimilarityMetric(char_weight=0.5, word_weight=0.5)
        metric2 = SimilarityMetric(char_weight=0.9, word_weight=0.1)
        
        # Results should be different
        self.assertNotEqual(metric1.evaluate(ground_truth, extracted), 
                          metric2.evaluate(ground_truth, extracted))


class TestOCREvaluator(unittest.TestCase):
    """Tests for OCREvaluator."""
    
    def setUp(self):
        # Create EvaluationConfig with old_russian_chars
        self.config = EvaluationConfig(
            old_russian_chars='ѣѲѳѵ',
            use_char_accuracy=True,
            use_word_accuracy=True,
            use_old_char_preservation=True,
            use_case_accuracy=True,
            char_similarity_weight=0.7,
            word_similarity_weight=0.3
        )
        self.evaluator = OCREvaluator(self.config)
    
    def test_init_metrics(self):
        """Test metric initialization."""
        self.evaluator._init_metrics()
        self.assertIn("char_accuracy", self.evaluator.metrics)
        self.assertIn("word_accuracy", self.evaluator.metrics)
        self.assertIn("old_char_preservation", self.evaluator.metrics)
        self.assertIn("case_accuracy", self.evaluator.metrics)
        self.assertIn("error_analysis", self.evaluator.metrics)
        self.assertIn("word_analysis", self.evaluator.metrics)
        self.assertIn("similarity", self.evaluator.metrics)
    
    def test_evaluate_line_empty(self):
        """Test evaluation with empty strings."""
        metrics = self.evaluator.evaluate_line("", "")
        self.assertEqual(metrics.char_accuracy, 0.0)
        self.assertEqual(metrics.word_accuracy, 0.0)
        self.assertEqual(metrics.old_char_preservation, 0.0)
        self.assertEqual(metrics.case_accuracy, 0.0)
        self.assertEqual(len(metrics.case_errors), 0)
    
    def test_evaluate_line(self):
        """Test line evaluation."""
        ground_truth = "This is a test."
        extracted = "This is a test."
        metrics = self.evaluator.evaluate_line(ground_truth, extracted)
        self.assertEqual(metrics.char_accuracy, 1.0)
        self.assertEqual(metrics.word_accuracy, 1.0)
        self.assertEqual(metrics.old_char_preservation, 1.0)
        self.assertEqual(metrics.case_accuracy, 1.0)
    
    def test_disabled_metrics(self):
        """Test when some metrics are disabled."""
        # Create config with disabled metrics
        config = EvaluationConfig(
            old_russian_chars='ѣѲѳѵ',
            use_char_accuracy=False
        )
        evaluator = OCREvaluator(config)
        
        # Override the char_accuracy metric with a mock
        evaluator.metrics["char_accuracy"] = MagicMock()
        evaluator.metrics["char_accuracy"].name = "char_accuracy"
        
        ground_truth = "This is a test."
        extracted = "This is a test."
        metrics = evaluator.evaluate_line(ground_truth, extracted)
        
        # Verify char_accuracy metric wasn't called
        evaluator.metrics["char_accuracy"].evaluate.assert_not_called()
    
    def test_analyze_errors(self):
        """Test error analysis."""
        ground_truth = "This is a test."
        extracted = "This is test."
        
        # Mock the error analysis metric
        self.evaluator.metrics["error_analysis"] = MagicMock()
        self.evaluator.metrics["error_analysis"].name = "error_analysis"
        self.evaluator.metrics["error_analysis"].evaluate.return_value = {"total_errors": 1}
        
        result = self.evaluator.analyze_errors(ground_truth, extracted)
        self.assertEqual(result["total_errors"], 1)
        self.evaluator.metrics["error_analysis"].evaluate.assert_called_once_with(ground_truth, extracted)
    
    def test_analyze_words(self):
        """Test word analysis."""
        ground_truth = "This is a test."
        extracted = "This is test."
        
        # Mock the word analysis metric
        self.evaluator.metrics["word_analysis"] = MagicMock()
        self.evaluator.metrics["word_analysis"].name = "word_analysis"
        self.evaluator.metrics["word_analysis"].evaluate.return_value = {"total_word_errors": 1}
        
        result = self.evaluator.analyze_words(ground_truth, extracted)
        self.assertEqual(result["total_word_errors"], 1)
        self.evaluator.metrics["word_analysis"].evaluate.assert_called_once_with(ground_truth, extracted)
    
    def test_calculate_similarity(self):
        """Test similarity calculation."""
        ground_truth = "This is a test."
        extracted = "This is test."
        
        # Mock the similarity metric
        self.evaluator.metrics["similarity"] = MagicMock()
        self.evaluator.metrics["similarity"].name = "similarity"
        self.evaluator.metrics["similarity"].evaluate.return_value = 0.8
        
        result = self.evaluator.calculate_similarity(ground_truth, extracted)
        self.assertEqual(result, 0.8)
        self.evaluator.metrics["similarity"].evaluate.assert_called_once_with(ground_truth, extracted)


class TestMetricsComparer(unittest.TestCase):
    """Tests for MetricsComparer."""
    
    def test_compare_metrics(self):
        """Test metrics comparison."""
        original_metrics = OCRMetrics(
            char_accuracy=0.8,
            word_accuracy=0.7,
            old_char_preservation=0.9,
            case_accuracy=0.8
        )
        
        corrected_metrics = OCRMetrics(
            char_accuracy=0.9,
            word_accuracy=0.85,
            old_char_preservation=0.95,
            case_accuracy=0.9
        )
        
        result = MetricsComparer.compare_metrics(original_metrics, corrected_metrics)
        
        self.assertEqual(result["character_accuracy_delta"], 0.1)
        self.assertEqual(result["word_accuracy_delta"], 0.15)
        self.assertEqual(result["old_char_preservation_delta"], 0.05)
        self.assertEqual(result["case_accuracy_delta"], 0.1)


if __name__ == '__main__':
    unittest.main()