"""
Shared utilities for metrics calculations.
"""

try:
    import Levenshtein

    _LEVENSHTEIN_AVAILABLE = True
except ImportError:
    _LEVENSHTEIN_AVAILABLE = False


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings using the fastest available method.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Integer representing the edit distance
    """
    if _LEVENSHTEIN_AVAILABLE:
        return Levenshtein.distance(s1, s2)
    else:
        # Fallback to custom implementation
        return _levenshtein_distance_fallback(s1, s2)


def _levenshtein_distance_fallback(s1: str, s2: str) -> int:
    """
    Fallback Levenshtein distance implementation when external library is not available.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Integer representing the edit distance
    """
    if len(s1) < len(s2):
        return _levenshtein_distance_fallback(s2, s1)

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


def character_accuracy(ground_truth: str, extracted: str, case_sensitive: bool = True) -> float:
    """
    Calculate character-level accuracy using Levenshtein distance.

    Args:
        ground_truth: The known correct text
        extracted: The text extracted by OCR
        case_sensitive: Whether to consider case when comparing characters

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if not ground_truth and not extracted:
        return 1.0
    if not ground_truth or not extracted:
        return 0.0

    # Apply case normalization if needed
    if not case_sensitive:
        ground_truth = ground_truth.lower()
        extracted = extracted.lower()

    distance = levenshtein_distance(ground_truth, extracted)
    max_length = max(len(ground_truth), len(extracted))
    return 1 - (distance / max_length) if max_length > 0 else 0.0
