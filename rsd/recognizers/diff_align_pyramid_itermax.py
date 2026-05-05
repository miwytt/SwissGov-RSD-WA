from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from rsd.recognizers.feature_based import FeatureExtractionRecognizer
from rsd.recognizers.utils import DifferenceSample, cos_sim


def _iter_max(sim: np.ndarray, max_count: int = 2) -> List[tuple]:
    """SimAlign itermax (Jalili-Sabet et al., 2020)."""
    alpha_ratio = 0.9
    m, n = sim.shape
    forward = np.eye(n)[sim.argmax(axis=1)]
    backward = np.eye(m)[sim.argmax(axis=0)]
    inter = forward * backward.T

    if min(m, n) <= 2:
        rows, cols = np.where(inter > 0)
        return list(zip(rows.tolist(), cols.tolist()))

    new_inter = np.zeros((m, n))
    count = 1
    while count < max_count:
        mask_x = 1.0 - np.tile(inter.sum(1)[:, None], (1, n)).clip(0.0, 1.0)
        mask_y = 1.0 - np.tile(inter.sum(0)[None, :], (m, 1)).clip(0.0, 1.0)
        mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
        mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
        if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
            mask *= 0.0
            mask_zeros *= 0.0
        new_sim = sim * mask
        fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
        bac = np.eye(m)[new_sim.argmax(axis=0)].T * mask_zeros
        new_inter = fwd * bac
        if np.array_equal(inter + new_inter, inter):
            break
        inter = inter + new_inter
        count += 1

    rows, cols = np.where(inter > 0)
    return list(zip(rows.tolist(), cols.tolist()))


class DiffAlignPyramidItermax(FeatureExtractionRecognizer):
    """
    DiffAlign variant that applies coarse-to-fine pyramid masking using itermax
    alignment at each coarse level. Compared to DiffAlignPyramidHard (which uses
    argmax), itermax recovers more alignments per coarse level (higher recall),
    which can reduce the number of tokens incorrectly zeroed out.

    width (int):     neighbouring blocks to keep around each aligned pair (same as
                     DiffAlignPyramidHard).
    max_count (int): itermax iteration count; higher = more recall at each level.
    """

    def __init__(self, *args, width: int = 10, max_count: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.max_count = max_count

    def __str__(self):
        model_name = getattr(getattr(self.pipeline, "model", None), "name_or_path", None)
        if model_name is None:
            model_name = str(self.model_name_or_path)
        return f"DiffAlignPyramidItermax(model={model_name}, layer={self.layer}, width={self.width}, max_count={self.max_count})"

    def _pyramid_mask(self, sim: torch.Tensor) -> torch.Tensor:
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
            ).squeeze(0, 1)

            coarse_alignment = _iter_max(coarse.cpu().numpy(), max_count=self.max_count)

            if block_h == 1 and block_w == 1:
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
            sim = cos_sim(outputs_a[i], outputs_b[i])
            masked = self._pyramid_mask(sim)

            max_a = masked.max(dim=1).values
            max_b = masked.max(dim=0).values

            labels_a = self._subword_labels_to_word_labels(1 - max_a, subwords_by_words_a[i])
            labels_b = self._subword_labels_to_word_labels(1 - max_b, subwords_by_words_b[i])

            samples.append(DifferenceSample(
                tokens_a=tuple(a[i].split()),
                tokens_b=tuple(b[i].split()),
                labels_a=tuple(labels_a),
                labels_b=tuple(labels_b),
            ))
        return samples
