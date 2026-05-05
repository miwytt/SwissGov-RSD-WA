from typing import List

import torch
import torch.nn.functional as F

from rsd.recognizers.feature_based import FeatureExtractionRecognizer
from rsd.recognizers.utils import DifferenceSample, cos_sim


class DiffAlignPyramidHard(FeatureExtractionRecognizer):
    """
    DiffAlign variant that applies coarse-to-fine pyramid masking to the similarity
    matrix before computing max similarities. Starting from a 2×2 coarse grid, each
    level aligns via bidirectional argmax intersection and zeroes out blocks not
    covered by any aligned pair. The remaining (unmasked) matrix is used to compute
    per-token hallucination labels: tokens whose only possible matches were zeroed
    out receive max_sim ≈ 0 and therefore label ≈ 1.

    width (int): number of neighbouring coarse blocks to keep around each aligned
                 pair at every pyramid level. width=0 keeps only the exactly aligned
                 blocks; higher values add tolerance for coarse alignment errors
                 (analogous to VecAlign's width_over2 parameter).

    Note: this method can cascade-fail for models with weak diagonal signal (e.g.
    masked LMs at document level) — a wrong coarse alignment zeroes out the entire
    correct diagonal region.
    """

    def __init__(self, *args, width: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width

    def __str__(self):
        model_name = getattr(getattr(self.pipeline, "model", None), "name_or_path", None)
        if model_name is None:
            model_name = str(self.model_name_or_path)
        return f"DiffAlignPyramidHard(model={model_name}, layer={self.layer}, width={self.width})"

    def _pyramid_mask(self, sim: torch.Tensor) -> torch.Tensor:
        """
        Iteratively coarsen the similarity matrix, align at each coarse level via
        bidirectional argmax intersection, zero out unaligned blocks, then halve the
        block size. Returns the progressively masked similarity matrix.

        Tokens in zeroed-out regions have no valid counterpart within the masked
        region and will yield max_sim = 0 → hallucination label = 1.
        """
        m, n = sim.shape
        block_h = max(m // 2, 1)
        block_w = max(n // 2, 1)

        while block_h >= 1 and block_w >= 1:
            n_blocks_h = (m + block_h - 1) // block_h
            n_blocks_w = (n + block_w - 1) // block_w

            coarse = F.avg_pool2d(
                sim[None, None].float(),
                kernel_size=(block_h, block_w),
                stride=(block_h, block_w),
                ceil_mode=True,
                count_include_pad=False,
            ).squeeze(0, 1)                                      # (n_blocks_h, n_blocks_w)

            # Bidirectional argmax intersection
            src_to_tgt = coarse.argmax(dim=1)
            tgt_to_src = coarse.argmax(dim=0)
            coarse_alignment = [
                (i, src_to_tgt[i].item())
                for i in range(n_blocks_h)
                if tgt_to_src[src_to_tgt[i]] == i
            ]

            if block_h == 1 and block_w == 1:
                # Token-level resolution reached: return the masked matrix as-is.
                # Tokens in zeroed-out regions yield max_sim = 0 → label = 1.
                return sim

            coarse_mask = torch.zeros(n_blocks_h, n_blocks_w, device=sim.device)
            for ci, cj in coarse_alignment:
                r0 = max(ci - self.width, 0)
                r1 = min(ci + self.width + 1, n_blocks_h)
                c0 = max(cj - self.width, 0)
                c1 = min(cj + self.width + 1, n_blocks_w)
                coarse_mask[r0:r1, c0:c1] = 1.0

            mask = F.interpolate(
                coarse_mask[None, None],
                size=(n_blocks_h * block_h, n_blocks_w * block_w),
                mode='nearest',
            ).squeeze(0, 1)[:m, :n]
            sim = sim * mask

            block_h = max(block_h // 2, 1)
            block_w = max(block_w // 2, 1)

        return sim

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
            sim = cos_sim(outputs_a[i], outputs_b[i])            # (m, n)
            masked = self._pyramid_mask(sim)

            max_a = masked.max(dim=1).values                     # best masked match per source token
            max_b = masked.max(dim=0).values                     # best masked match per target token

            labels_a = self._subword_labels_to_word_labels(1 - max_a, subwords_by_words_a[i])
            labels_b = self._subword_labels_to_word_labels(1 - max_b, subwords_by_words_b[i])

            samples.append(DifferenceSample(
                tokens_a=tuple(a[i].split()),
                tokens_b=tuple(b[i].split()),
                labels_a=tuple(labels_a),
                labels_b=tuple(labels_b),
            ))
        return samples
