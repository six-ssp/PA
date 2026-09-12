"""曲面插值和联合网格的快速检查，不运行 460 个长时工况。"""

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sensitivity_surface import OFFSETS, SCALES, interpolate, matrices, source_matches


class SurfaceTests(unittest.TestCase):
    def test_source_check_requires_complete_verified_fingerprint(self):
        original = {"model": "original", "attachment": "unchanged"}
        verified = {"model": "published", "attachment": "unchanged"}
        metadata = {"source_sha256": original, "verified_compatible_source_sha256": [verified]}
        for current, expected in [(original, True), (verified, True),
                                  ({"model": "published", "attachment": "changed"}, False),
                                  ({"model": "unverified", "attachment": "unchanged"}, False)]:
            with self.subTest(current=current), patch("sensitivity_surface.hashes", return_value=current):
                self.assertEqual(source_matches(metadata), expected)

    def test_source_check_without_compatibility_record_remains_strict(self):
        with patch("sensitivity_surface.hashes", return_value={"model": "other"}):
            self.assertFalse(source_matches({"source_sha256": {"model": "original"}}))

    def test_interpolation_preserves_real_nodes(self):
        z = np.exp(-.04 * OFFSETS[:, None]) * SCALES[None, :] ** 1.8
        np.testing.assert_allclose(interpolate(z, OFFSETS, SCALES), z, atol=1e-12)

    def test_linear_additive_response_has_zero_interaction(self):
        z = 50 - 2 * OFFSETS[:, None] + 90 * (SCALES[None, :] - 1)
        t, s = np.array([-.75, .25]), np.array([.8125, 1.0125])
        np.testing.assert_allclose(interpolate(z, t, s), 50 - 2 * t[:, None] + 90 * (s[None, :] - 1), atol=1e-12)
        np.testing.assert_allclose(z - z[:, [8]] - z[[6], :] + z[6, 8], 0, atol=1e-12)

    def test_incomplete_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            matrices([])


if __name__ == "__main__":
    unittest.main()
