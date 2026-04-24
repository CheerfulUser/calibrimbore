"""
Tests for calibrimbore/R_load.py

Includes a regression test for the bug where Rg0 and Rr0 were both assigned
the same value (coeff[1] of the target band), making the extinction colour
correction always evaluate to zero.
"""

import numpy as np
import pytest
from calibrimbore.R_load import line, R_val, R


# ---------------------------------------------------------------------------
# line()
# ---------------------------------------------------------------------------

class TestLine:
    def test_zero_input(self):
        assert line(0, 5, 2) == pytest.approx(5)

    def test_positive_slope(self):
        assert line(1, 5, 2) == pytest.approx(7)

    def test_negative_slope(self):
        assert line(0.5, 3, -1) == pytest.approx(2.5)

    def test_array_input(self):
        x = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(line(x, 1, 2), [1, 3, 5])

    def test_identity_when_slope_zero(self):
        assert line(100, 7, 0) == pytest.approx(7)


# ---------------------------------------------------------------------------
# R_val()
# ---------------------------------------------------------------------------

class TestRVal:
    def test_gr_zero_returns_intercept(self):
        # At gr=0, R_val returns the intercept of the linear fit: coeff[0]
        Rb, Rb_e = R_val("g", "ps1", gr=0.0)
        assert Rb == pytest.approx(R["ps1"]["g"]["coeff"][0])
        assert Rb_e == pytest.approx(R["ps1"]["g"]["std"])

    def test_with_gr_ps1_g(self):
        gr = 0.5
        Rb, Rb_e = R_val("g", "ps1", gr=gr)
        coeff = R["ps1"]["g"]["coeff"]
        assert Rb == pytest.approx(line(gr, coeff[0], coeff[1]))
        assert Rb_e == pytest.approx(R["ps1"]["g"]["std"])

    def test_gr_from_g_and_r_kwargs(self):
        g, r = 0.6, 0.2
        Rb_direct, _ = R_val("g", "ps1", gr=g - r)
        Rb_kwargs, _ = R_val("g", "ps1", g=g, r=r)
        assert Rb_direct == pytest.approx(Rb_kwargs)

    def test_all_ps1_bands(self):
        for band in ("g", "r", "i", "z", "y"):
            Rb, Rb_e = R_val(band, "ps1", gr=0.5)
            assert np.isfinite(Rb), f"R_val({band}, ps1) is not finite"
            assert Rb_e > 0, f"R_val({band}, ps1) std is not positive"

    def test_all_supported_systems(self):
        cases = [
            ("g", "ps1"), ("r", "ps1"),
            ("g", "decam"), ("r", "decam"),
            ("g", "skymapper"), ("r", "skymapper"),
            ("r", "gaia"),   # Gaia 'r' key maps to the G-band
        ]
        for band, system in cases:
            Rb, _ = R_val(band, system, gr=0.5)
            assert np.isfinite(Rb), f"R_val({band}, {system}) is not finite"

    # -----------------------------------------------------------------------
    # Bug-fix regression: extinction colour correction (Rg0 != Rr0)
    # -----------------------------------------------------------------------

    def test_rg0_rr0_are_different_bands(self):
        """
        After the fix, Rg0 uses R['ps1']['g']['coeff'][0] and Rr0 uses
        R['ps1']['r']['coeff'][0]. They must differ so the colour correction
        is non-trivial.
        """
        Rg0 = R["ps1"]["g"]["coeff"][0]
        Rr0 = R["ps1"]["r"]["coeff"][0]
        assert Rg0 != pytest.approx(Rr0)

    def test_extinction_correction_changes_result(self):
        """
        R_val with ext > 0 should return a different value than ext = 0 because
        the intrinsic g-r colour is shifted by ext * (Rg0 - Rr0).
        Before the fix both Rg0 and Rr0 were the same value, so ext had no
        effect and this test would fail.
        """
        gr, ext = 0.5, 0.3
        Rb_no_ext, _ = R_val("i", "ps1", gr=gr, ext=0)
        Rb_with_ext, _ = R_val("i", "ps1", gr=gr, ext=ext)
        assert Rb_with_ext != pytest.approx(Rb_no_ext)

    def test_extinction_corrected_value_is_correct(self):
        """Check the numerically expected result after the colour correction."""
        gr, ext = 0.5, 0.3
        Rg0 = R["ps1"]["g"]["coeff"][0]
        Rr0 = R["ps1"]["r"]["coeff"][0]
        gr_int = gr - ext * (Rg0 - Rr0)
        coeff_i = R["ps1"]["i"]["coeff"]
        expected = line(gr_int, coeff_i[0], coeff_i[1])
        Rb, _ = R_val("i", "ps1", gr=gr, ext=ext)
        assert Rb == pytest.approx(expected)
