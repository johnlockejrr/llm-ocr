from llm_ocr.evaluators.metrics.base import BaseMetric


class CaseInsensitiveCharacterAccuracyMetric(BaseMetric):
    """Metric for character-level accuracy using Levenshtein distance, ignoring case differences."""

    @property
    def name(self) -> str:
        return "char_accuracy_case_insensitive"

    def evaluate(self, ground_truth: str, extracted: str) -> float:
        """
        Calculate character-level accuracy using Levenshtein distance, ignoring case.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        # Convert both strings to lowercase to ignore case
        ground_truth_lower = ground_truth.lower()
        extracted_lower = extracted.lower()

        if not ground_truth_lower and not extracted_lower:
            return 1.0
        if not ground_truth_lower or not extracted_lower:
            return 0.0

        distance = self._levenshtein_distance(ground_truth_lower, extracted_lower)
        max_length = max(len(ground_truth_lower), len(extracted_lower))
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

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
