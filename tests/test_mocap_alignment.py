from __future__ import annotations

import unittest

import numpy as np

from mocap_alignment import apply_row_alignment, compose_row_alignment, kabsch_rows


class MocapAlignmentTests(unittest.TestCase):
    def test_kabsch_rows_recovers_row_vector_transform(self) -> None:
        moving = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.asarray([10.0, 20.0, 30.0])
        reference = moving @ rotation + translation

        recovered_rotation, recovered_translation = kabsch_rows(reference, moving)

        np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(recovered_translation, translation, atol=1e-12)
        np.testing.assert_allclose(
            apply_row_alignment(moving, recovered_rotation, recovered_translation),
            reference,
            atol=1e-12,
        )

    def test_compose_row_alignment_matches_sequential_transforms(self) -> None:
        first_rotation = np.asarray(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        first_translation = np.asarray([10.0, 20.0, 30.0])
        second_rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
        )
        second_translation = np.asarray([-5.0, 6.0, 7.0])
        points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        rotation, translation = compose_row_alignment(
            first_rotation, first_translation, second_rotation, second_translation
        )

        sequential = (
            points @ first_rotation + first_translation
        ) @ second_rotation + second_translation
        composed = points @ rotation + translation
        np.testing.assert_allclose(composed, sequential)


if __name__ == "__main__":
    unittest.main()
