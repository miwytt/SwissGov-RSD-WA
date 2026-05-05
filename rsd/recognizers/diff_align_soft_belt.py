from typing import List

import torch

from rsd.recognizers.feature_based import FeatureExtractionRecognizer
from rsd.recognizers.utils import DifferenceSample, cos_sim


class DiffAlignSoftBelt(FeatureExtractionRecognizer):
    """
    DiffAlign variant that applies a Gaussian soft-belt mask to the similarity matrix
    before computing max similarities. Penalises cross-lingual matches that are far
    from the expected diagonal position (i/(m-1) ≈ j/(n-1)), so tokens whose only
    similar counterpart lies far from the diagonal receive a higher hallucination label.

    k (float): tolerance in absolute subword tokens on the longer side. The Gaussian
               sigma in normalised space is k / max(m, n). Larger k = wider belt =
               less penalisation of off-diagonal matches.
    """

    def __init__(self, *args, k: float = 150.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def __str__(self):
        model_name = getattr(getattr(self.pipeline, "model", None), "name_or_path", None)
        if model_name is None:
            model_name = str(self.model_name_or_path)
        return f"DiffAlignSoftBelt(model={model_name}, layer={self.layer}, k={self.k})"

    def _soft_belt_weights(self, m: int, n: int, device) -> torch.Tensor:
        """
        Gaussian weight matrix: weights[i, j] = exp(-0.5 * ((i/(m-1) - j/(n-1)) / sigma)^2).
        Positions on the main diagonal receive weight 1; off-diagonal positions are
        down-weighted proportionally to their distance from the diagonal.
        Uses (m-1) and (n-1) denominators so both endpoints map to exactly 0 and 1,
        ensuring the diagonal spans both corners of the matrix.
        """
        i_idx = torch.arange(m, dtype=torch.float32, device=device)[:, None] / max(m - 1, 1)
        j_idx = torch.arange(n, dtype=torch.float32, device=device)[None, :] / max(n - 1, 1)
        sigma = self.k / max(m, n)
        return torch.exp(-0.5 * ((i_idx - j_idx) / sigma) ** 2)

    @torch.no_grad()
    def _predict_all(self,
                     a: List[str],
                     b: List[str],
                     **kwargs,
                     ) -> List[DifferenceSample]:
        outputs_a = self.encode_batch(a, **kwargs)
        outputs_b = self.encode_batch(b, **kwargs)
        subwords_by_words_a = [self._get_subwords_by_word(s) for s in a]
        subwords_by_words_b = [self._get_subwords_by_word(s) for s in b]

        samples = []
        for i in range(len(a)):
            sim = cos_sim(outputs_a[i], outputs_b[i])          # (m, n)
            m, n = sim.shape
            weights = self._soft_belt_weights(m, n, sim.device)
            masked = sim * weights

            max_a = masked.max(dim=1).values                   # best belt-weighted target match per source token
            max_b = masked.max(dim=0).values                   # best belt-weighted source match per target token

            labels_a = self._subword_labels_to_word_labels(1 - max_a, subwords_by_words_a[i])
            labels_b = self._subword_labels_to_word_labels(1 - max_b, subwords_by_words_b[i])

            samples.append(DifferenceSample(
                tokens_a=tuple(a[i].split()),
                tokens_b=tuple(b[i].split()),
                labels_a=tuple(labels_a),
                labels_b=tuple(labels_b),
            ))
        return samples
