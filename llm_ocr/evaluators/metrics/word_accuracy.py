import re

from Levenshtein import distance as levenshtein_distance

from llm_ocr.evaluators.metrics.base import BaseMetric


class WordAccuracyMetric(BaseMetric):
    """Metric for word-level accuracy with improved handling of historical texts."""

    @property
    def name(self) -> str:
        return "word_accuracy"

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better comparison."""
        # Replace line breaks with spaces
        text = re.sub(r"\n", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Trim whitespace
        return text.strip()

    def _tokenize(self, text: str) -> list:
        """Tokenize text into words, handling punctuation."""
        # Normalize text first
        text = self._normalize_text(text)
        # Match words, considering hyphenated words as single tokens
        # Include Unicode characters for historical texts
        tokens = re.findall(r"[\w\u0400-\u04FF\-]+|[^\w\u0400-\u04FF\s]", text)
        return tokens

    def _handle_hyphenation(self, tokens: list) -> list:
        """Handle hyphenated words that are split across lines."""
        result = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i].endswith("-") and tokens[i + 1] != "-":
                # Remove the hyphen and join with the next token
                combined = tokens[i][:-1] + tokens[i + 1]
                result.append(combined)
                i += 2  # Skip the next token as we've combined it
            else:
                result.append(tokens[i])
                i += 1

        return result

    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words based on Levenshtein distance."""
        max_len = max(len(word1), len(word2))
        if max_len == 0:
            return 1.0

        distance: int = levenshtein_distance(word1, word2)
        return 1.0 - (distance / max_len)

    def _align_tokens(
        self, ground_truth_tokens: list, extracted_tokens: list, partial_match_threshold: float
    ) -> float:
        """
        Align tokens from ground truth and extracted text to find best matches.
        Uses a greedy algorithm to maximize total similarity.
        """
        n = len(ground_truth_tokens)
        m = len(extracted_tokens)

        # Create a similarity matrix
        similarity_matrix = [[0.0 for _ in range(m)] for _ in range(n)]

        # Fill the similarity matrix
        for i in range(n):
            for j in range(m):
                if ground_truth_tokens[i] == extracted_tokens[j]:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity = self._calculate_similarity(
                        ground_truth_tokens[i], extracted_tokens[j]
                    )
                    similarity_matrix[i][j] = (
                        similarity if similarity >= partial_match_threshold else 0.0
                    )

        # Align tokens greedily to maximize total similarity
        matches = 0.0
        matched = set()

        for i in range(n):
            best_match = -1
            best_score = 0.0

            for j in range(m):
                if j not in matched and similarity_matrix[i][j] > best_score:
                    best_score = similarity_matrix[i][j]
                    best_match = j

            if best_match >= 0 and best_score > 0:
                matches += best_score
                matched.add(best_match)

        return matches

    def evaluate(
        self,
        ground_truth: str,
        extracted: str,
        partial_match_threshold: float = 0.75,
        case_sensitive: bool = True,
    ) -> float:
        """
        Calculate word-level accuracy with partial matching support.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR
            partial_match_threshold: Similarity threshold for partial matches (0.0-1.0)
            case_sensitive: Whether to perform case-sensitive comparison

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        if not case_sensitive:
            ground_truth = ground_truth.lower()
            extracted = extracted.lower()

        ground_truth_tokens = self._tokenize(ground_truth)
        extracted_tokens = self._tokenize(extracted)

        # Apply hyphenation handling
        ground_truth_tokens = self._handle_hyphenation(ground_truth_tokens)
        extracted_tokens = self._handle_hyphenation(extracted_tokens)

        if not ground_truth_tokens and not extracted_tokens:
            return 1.0
        if not ground_truth_tokens or not extracted_tokens:
            return 0.0

        # Use token alignment for better matching
        matches = self._align_tokens(ground_truth_tokens, extracted_tokens, partial_match_threshold)
        total = max(len(ground_truth_tokens), len(extracted_tokens))

        return matches / total
