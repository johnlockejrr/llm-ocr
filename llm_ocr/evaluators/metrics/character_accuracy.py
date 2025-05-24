from llm_ocr.evaluators.metrics.base import BaseMetric



class CharacterAccuracyMetric(BaseMetric):
    """Metric for character-level accuracy using Levenshtein distance."""
    
    @property
    def name(self) -> str:
        return "char_accuracy"
    
    def evaluate(self, ground_truth: str, extracted: str) -> float:
        """
        Calculate character-level accuracy using Levenshtein distance.
        
        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR
            
        Returns:
            Accuracy score between 0.0 and 1.0
        """
        if not ground_truth and not extracted:
            return 1.0
        if not ground_truth or not extracted:
            return 0.0
            
        distance = self._levenshtein_distance(ground_truth, extracted)
        max_length = max(len(ground_truth), len(extracted))
        return 1 - (distance / max_length) if max_length > 0 else 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Integer representing the edit distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


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
        self.char_metric = CharacterAccuracyMetric()
    
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
        char_similarity = self.char_metric.evaluate(ground_truth, extracted)
    
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

