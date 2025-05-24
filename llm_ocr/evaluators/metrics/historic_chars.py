from collections import Counter
from typing import Any
from typing import Counter as CounterType
from typing import Dict, List

from llm_ocr.evaluators.metrics.base import BaseMetric

# Characters not commonly used in 18th century Russian texts
ARCHAIC_CHARS = {
    "ѧ",  # 'Little yus' - replaced by 'я' in later Russian
    "ѫ",  # 'Big yus' - obsolete by 18th century
    "ѩ",  # 'Iotated little yus' - obsolete by 18th century
    "ѭ",  # 'Iotated big yus' - obsolete by 18th century
    "ѱ",  # 'Psi' - mainly used in Church Slavonic
    "ѯ",  # 'Ksi' - mainly used in Church Slavonic
    # 'ѳ',  # 'Fita' - eventually replaced with 'ф' or 'т'
    # 'ѵ',  # 'Izhitsa' - eventually replaced with 'и' or 'в'
    "ѡ",  # 'Omega' - replaced by 'о' in civil script
    "ѿ",  # 'Ot' - obsolete ligature
    # 'ѣ',  # 'Yat' - actually used in 18th century but could be incorrectly inserted
    # 'і',   # 'Decimal i' - used in 18th century but sometimes incorrectly inserted
    # 'ѕ',  # 'Zelo' - replaced by 'з' in civil script
    "ѯ",  # 'Ksi' - obsolete by 18th century
    "ѱ",  # 'Psi' - obsolete by 18th century
    "ꙗ",  # 'Iotated a' - obsolete by 18th century
    "ѥ",  # 'Iotated e' - obsolete by 18th century
    # Add others based on your specific historical period
}


class OverHistoricizationMetric(BaseMetric):
    """
    Metric that detects when OCR incorrectly adds archaic characters not present in the original text.
    This measures the tendency of models to "over-historicize" by inserting very ancient Russian
    characters that weren't typically used in the 18th century.
    """

    @property
    def name(self) -> str:
        return "over_historicization"

    def evaluate(self, ground_truth: str, extracted: str) -> Dict[str, Any]:
        """
        Calculate metrics for incorrect insertion of archaic characters.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Dictionary with over-historicization metrics
        """
        # Find archaic characters in both texts
        gt_archaic = {c for c in ground_truth if c in ARCHAIC_CHARS}
        ext_archaic = {c for c in extracted if c in ARCHAIC_CHARS}

        # Find characters that were incorrectly added (in extracted but not in ground truth)
        incorrectly_added = ext_archaic - gt_archaic

        # Count occurrences of each incorrectly added character
        char_counts: CounterType[str] = Counter()
        for c in extracted:
            if c in incorrectly_added:
                char_counts[c] += 1

        # Calculate insertion ratio (# of incorrectly added archaic chars / text length)
        total_insertions = sum(char_counts.values())
        insertion_ratio = total_insertions / len(extracted) if extracted else 0.0

        # Find contexts where insertions occurred
        insertion_contexts = self._find_insertion_contexts(ground_truth, extracted)

        return {
            "incorrect_archaic_chars": list(incorrectly_added),
            "char_frequencies": dict(char_counts),
            "total_insertions": total_insertions,
            "insertion_ratio": insertion_ratio,
            "insertion_contexts": insertion_contexts,
        }

    def _find_insertion_contexts(
        self, ground_truth: str, extracted: str, context_size: int = 2
    ) -> List[Dict[str, str]]:
        """
        Find contexts where archaic characters were incorrectly inserted.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR
            context_size: Number of characters before/after to include for context

        Returns:
            List of dictionaries with character and context info
        """
        try:
            import Levenshtein

            # Get edit operations
            ops = Levenshtein.editops(ground_truth, extracted)

            contexts = []
            for op, src_pos, dest_pos in ops:
                if op in ("insert", "replace"):
                    # Character was inserted or replaced in extracted text
                    if dest_pos < len(extracted):
                        inserted_char = extracted[dest_pos]

                        if inserted_char in ARCHAIC_CHARS:  # Use global constant here
                            # If it's a replacement, check if it's actually new (not in ground truth)
                            if op == "replace" and src_pos < len(ground_truth):
                                if ground_truth[src_pos] == inserted_char:
                                    continue  # Not a true insertion, same character

                            # Get context (careful with boundaries)
                            before = extracted[max(0, dest_pos - context_size) : dest_pos]
                            after = extracted[
                                dest_pos + 1 : min(len(extracted), dest_pos + context_size + 1)
                            ]

                            # What was in the ground truth at this position?
                            original = (
                                ground_truth[src_pos]
                                if op == "replace" and src_pos < len(ground_truth)
                                else "∅"
                            )

                            contexts.append(
                                {
                                    "char": inserted_char,
                                    "position": dest_pos,
                                    "context_before": before,
                                    "context_after": after,
                                    "original_char": original,
                                }
                            )

            return contexts

        except ImportError:
            # Fallback if Levenshtein not available
            return []
