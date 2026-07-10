from __future__ import annotations

import unittest

from mocap_units import point_unit_scale_to_m, point_unit_scale_to_mm


class MocapUnitsTests(unittest.TestCase):
    def test_point_unit_scale_to_mm_handles_common_c3d_units(self) -> None:
        self.assertEqual(point_unit_scale_to_mm("mm"), 1.0)
        self.assertEqual(point_unit_scale_to_mm("millimetres"), 1.0)
        self.assertEqual(point_unit_scale_to_mm("cm"), 10.0)
        self.assertEqual(point_unit_scale_to_mm("centimeters"), 10.0)
        self.assertEqual(point_unit_scale_to_mm("m"), 1000.0)
        self.assertEqual(point_unit_scale_to_mm("metres"), 1000.0)

    def test_point_unit_scale_to_mm_defaults_to_millimetres(self) -> None:
        self.assertEqual(point_unit_scale_to_mm(""), 1.0)
        self.assertEqual(point_unit_scale_to_mm("unknown"), 1.0)

    def test_point_unit_scale_to_m_uses_same_normalization(self) -> None:
        self.assertEqual(point_unit_scale_to_m("mm"), 0.001)
        self.assertEqual(point_unit_scale_to_m("cm"), 0.01)
        self.assertEqual(point_unit_scale_to_m("m"), 1.0)
        self.assertEqual(point_unit_scale_to_m("unknown"), 0.001)


if __name__ == "__main__":
    unittest.main()
