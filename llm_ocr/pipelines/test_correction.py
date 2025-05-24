"""
Tests for the OCR Correction Pipeline module.
"""
import unittest
from unittest.mock import MagicMock, patch
import time
import logging

from llm_ocr.pipelines.correction import OCRCorrectionPipeline
from llm_ocr.models import OCRMetrics
from llm_ocr.config import EvaluationConfig

class TestOCRCorrectionPipeline(unittest.TestCase):
    """Test cases for OCRCorrectionPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock models
        self.mock_model1 = MagicMock()
        self.mock_model1.__class__.__name__ = "MockModel1"
        self.mock_model1.correct_text.return_value = "Corrected text 1"
        
        self.mock_model2 = MagicMock()
        self.mock_model2.__class__.__name__ = "MockModel2"
        self.mock_model2.correct_text.return_value = "Corrected text 2"
        
        # Create mock evaluator
        self.mock_evaluator = MagicMock()
        
        # Create mock metrics
        self.mock_metrics = OCRMetrics(
            char_accuracy=0.95,
            word_accuracy=0.85,
            old_char_preservation=0.9,
            case_accuracy=0.98,
            case_errors=[],
            capitalization_analysis={}
        )
        
        self.mock_ocr_metrics = OCRMetrics(
            char_accuracy=0.75,
            word_accuracy=0.65,
            old_char_preservation=0.7,
            case_accuracy=0.85,
            case_errors=[],
            capitalization_analysis={}
        )
        
        self.mock_evaluator.evaluate_line.side_effect = [
            self.mock_metrics,    # First call (corrected text)
            self.mock_ocr_metrics # Second call (original OCR text)
        ]
        
        # Configure mock error analysis
        self.mock_evaluator.analyze_errors.return_value = {
            "substitutions": [
                {"ground_truth": "e", "extracted": "c"},
                {"ground_truth": "a", "extracted": "o"}
            ],
            "deletions": [],
            "insertions": [],
            "old_char_errors": [
                {"ground_truth": "ѣ", "extracted": "e"}
            ]
        }
        
        # Configure mock word analysis
        self.mock_evaluator.analyze_words.return_value = {
            "word_substitutions": [{"ground_truth": ["word1"], "extracted": ["werd1"]}],
            "word_deletions": [],
            "word_insertions": []
        }
        
        # Configure mock similarity score
        self.mock_evaluator.calculate_similarity.return_value = 0.92
        
        # Create pipeline instance
        self.pipeline = OCRCorrectionPipeline(
            models=[self.mock_model1, self.mock_model2],
            evaluator=self.mock_evaluator
        )
        
        # Test data
        self.image_str = "base64_encoded_image_data"
        self.ocr_text = "Original OCR text with errors"
        self.ground_truth = "Ground truth text for comparison"

    def test_initialization(self):
        """Test proper initialization of the pipeline."""
        self.assertEqual(self.pipeline.models, [self.mock_model1, self.mock_model2])
        self.assertEqual(self.pipeline.evaluator, self.mock_evaluator)
        self.assertIsInstance(self.pipeline.config, EvaluationConfig)

    def test_run_correction_without_ground_truth(self):
        """Test running correction without ground truth."""
        # Patch time.time to return predictable values
        with patch('time.time', side_effect=[0, 1, 2, 3]):
            results = self.pipeline.run_correction(
                image_str=self.image_str,
                ocr_text=self.ocr_text
            )
        
        # Check results structure and content
        self.assertIn("MockModel1", results)
        self.assertIn("MockModel2", results)
        
        # Check model 1 results
        model1_result = results["MockModel1"]
        self.assertEqual(model1_result["processing_time"], 1)  # 1 - 0
        self.assertEqual(model1_result["extracted_text"], self.ocr_text)
        self.assertEqual(model1_result["corrected_text"], "Corrected text 1")
        
        # Check model 2 results
        model2_result = results["MockModel2"]
        self.assertEqual(model2_result["processing_time"], 1)  # 3 - 2
        self.assertEqual(model2_result["extracted_text"], self.ocr_text)
        self.assertEqual(model2_result["corrected_text"], "Corrected text 2")
        
        # Verify evaluator was not called
        self.mock_evaluator.evaluate_line.assert_not_called()

    def test_run_correction_with_ground_truth(self):
        """Test running correction with ground truth."""
        # Reset side_effect for multiple calls
        self.mock_evaluator.evaluate_line.side_effect = [
            self.mock_metrics,      # Model1 corrected text
            self.mock_ocr_metrics,  # Original OCR text for Model1
            self.mock_metrics,      # Model2 corrected text
            self.mock_ocr_metrics   # Original OCR text for Model2
        ]
        
        # Run correction with ground truth
        with patch('time.time', side_effect=[0, 1, 2, 3]):
            results = self.pipeline.run_correction(
                image_str=self.image_str,
                ocr_text=self.ocr_text,
                ground_truth=self.ground_truth
            )
        
        # Check metrics were added to results
        for model_name in ["MockModel1", "MockModel2"]:
            model_result = results[model_name]
            
            # Check basic metrics
            self.assertEqual(model_result["character_accuracy"], 0.95)
            self.assertEqual(model_result["word_accuracy"], 0.85)
            self.assertEqual(model_result["old_char_preservation"], 0.9)
            self.assertEqual(model_result["case_accuracy"], 0.98)
            
            # Check improvement deltas
            self.assertEqual(model_result["improvement"]["character_accuracy_delta"], 0.2)  # 0.95 - 0.75
            self.assertEqual(model_result["improvement"]["word_accuracy_delta"], 0.2)  # 0.85 - 0.65
            self.assertEqual(model_result["improvement"]["old_char_preservation_delta"], 0.2)  # 0.9 - 0.7
            self.assertEqual(model_result["improvement"]["case_accuracy_delta"], 0.13)  # 0.98 - 0.85

    def test_detailed_analysis(self):
        """Test detailed analysis is included when configured."""
        # Enable detailed analysis
        self.pipeline.config.include_detailed_analysis = True
        
        # Reset side_effect for evaluate_line
        self.mock_evaluator.evaluate_line.side_effect = [
            self.mock_metrics,      # Model1 corrected text
            self.mock_ocr_metrics   # Original OCR text for Model1
        ]
        
        # Run correction with ground truth
        results = self.pipeline.run_correction(
            image_str=self.image_str,
            ocr_text=self.ocr_text,
            ground_truth=self.ground_truth
        )
        
        # Check error analysis was added
        model_result = results["MockModel1"]
        self.assertIn("error_analysis", model_result)
        
        # Check error patterns
        self.assertIn("common_char_errors", model_result["error_analysis"])
        self.assertEqual(model_result["error_analysis"]["common_char_errors"], {"e → c": 1, "a → o": 1})
        
        # Check old character errors
        self.assertIn("old_char_errors", model_result["error_analysis"])
        self.assertEqual(model_result["error_analysis"]["old_char_errors"], {"ѣ → e": 1})
        
        # Check word errors
        self.assertIn("word_errors", model_result["error_analysis"])
        self.assertEqual(model_result["error_analysis"]["word_errors"]["substitutions"], 1)
        self.assertEqual(model_result["error_analysis"]["word_errors"]["deletions"], 0)
        self.assertEqual(model_result["error_analysis"]["word_errors"]["insertions"], 0)
        
        # Check similarity score
        self.assertIn("similarity", model_result)
        self.assertEqual(model_result["similarity"], 0.92)
        
        # Verify correct method calls
        self.mock_evaluator.analyze_errors.assert_called_once_with(self.ground_truth, "Corrected text 1")
        self.mock_evaluator.analyze_words.assert_called_once_with(self.ground_truth, "Corrected text 1")
        self.mock_evaluator.calculate_similarity.assert_called_once_with(self.ground_truth, "Corrected text 1")

    def test_error_handling(self):
        """Test error handling when model correction fails."""
        # Make first model raise an exception
        self.mock_model1.correct_text.side_effect = Exception("Model error")
        
        # Run correction
        with self.assertLogs(level=logging.ERROR) as log:
            results = self.pipeline.run_correction(
                image_str=self.image_str,
                ocr_text=self.ocr_text
            )
            
        # Check logs for error
        self.assertIn("Error processing correction with MockModel1", log.output[0])
        
        # Check results - should only have Model2
        self.assertNotIn("MockModel1", results)
        self.assertIn("MockModel2", results)
        
    def test_empty_models_list(self):
        """Test running pipeline with no models."""
        pipeline = OCRCorrectionPipeline(models=[], evaluator=self.mock_evaluator)
        
        results = pipeline.run_correction(
            image_str=self.image_str,
            ocr_text=self.ocr_text
        )
        
        # Results should be an empty dict
        self.assertEqual(results, {})

if __name__ == "__main__":
    unittest.main()