from llm_ocr.evaluators.metrics.base import BaseMetric
from llm_ocr.evaluators.metrics.utils import character_accuracy


class CharacterAccuracyMetric(BaseMetric):
    """Metric for character-level accuracy using Levenshtein distance."""

    def __init__(self, case_sensitive: bool = True):
        """
        Initialize character accuracy metric.

        Args:
            case_sensitive: Whether to consider case when comparing characters
        """
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "char_accuracy" if self.case_sensitive else "char_accuracy_case_insensitive"

    def evaluate(self, ground_truth: str, extracted: str) -> float:
        """
        Calculate character-level accuracy using Levenshtein distance.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        return character_accuracy(ground_truth, extracted, self.case_sensitive)


class SimilarityMetric(BaseMetric):
    """Overall similarity metric that combines character and word similarities."""

    def __init__(self, char_weight: float = 0.7, word_weight: float = 0.3):
        """
        Initialize with weights.

        Args:
            char_weight: Weight for character similarity
            word_weight: Weight for word similarity
        """
        self.char_weight = char_weight
        self.word_weight = word_weight

    @property
    def name(self) -> str:
        return "similarity"

    def evaluate(self, ground_truth: str, extracted: str) -> float:
        """
        Calculate weighted similarity score.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Use character-level similarity
        char_similarity = character_accuracy(ground_truth, extracted)

        # Also consider word overlap
        ground_truth_words = set(ground_truth.split())
        extracted_words = set(extracted.split())

        word_overlap = 0.0
        if ground_truth_words or extracted_words:
            word_overlap = len(ground_truth_words.intersection(extracted_words)) / max(
                len(ground_truth_words), len(extracted_words)
            )

        # Combine scores with weights
        return (self.char_weight * char_similarity + self.word_weight * word_overlap) / (
            self.char_weight + self.word_weight
        )
