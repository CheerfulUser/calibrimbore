"""
Tests for the sauron class in calibrimbore/calibrimbore.py.

Most tests bypass __init__ via object.__new__ and set attributes directly,
avoiding the need for pysynphot and network catalog access.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_check_system_valid(self):
        from calibrimbore.calibrimbore import sauron
        s = object.__new__(sauron)
        for system in ("ps1", "decam", "skymapper", "lsst", "gaia"):
            s.system = system
            s._check_system()  # must not raise

    def test_check_system_invalid_raises(self):
        from calibrimbore.calibrimbore import sauron
        s = object.__new__(sauron)
        s.system = "hubble"
        with pytest.raises(ValueError, match="not supported"):
            s._check_system()

    def test_check_system_uppercase_raises(self):
        # __init__ lower-cases the value; calling _check_system with mixed case
        # documents that the method itself requires lowercase input.
        from calibrimbore.calibrimbore import sauron
        s = object.__new__(sauron)
        s.system = "PS1"
        with pytest.raises(ValueError):
            s._check_system()

    def test_check_spec_model_valid(self):
        from calibrimbore.calibrimbore import sauron
        s = object.__new__(sauron)
        for model in ("calspec", "ckmodel"):
            s.spec_model = model
            s._check_spec_model()

    def test_check_spec_model_invalid_raises(self):
        from calibrimbore.calibrimbore import sauron
        s = object.__new__(sauron)
        s.spec_model = "phoenix"
        with pytest.raises(ValueError, match="not supported"):
            s._check_spec_model()


# ---------------------------------------------------------------------------
# cubic_correction
# ---------------------------------------------------------------------------

class TestCubicCorrection:
    def test_value_at_zero(self, sauron_instance):
        result = sauron_instance.cubic_correction(x=np.array([0.0]))
        assert result[0] == pytest.approx(sauron_instance.cubic_coeff[0])

    def test_value_at_one(self, sauron_instance):
        a = sauron_instance.cubic_coeff
        expected = a[0] + a[1] + a[2] + a[3]
        result = sauron_instance.cubic_correction(x=np.array([1.0]))
        assert result[0] == pytest.approx(expected)

    def test_default_x_uses_gr(self, sauron_instance):
        result_none = sauron_instance.cubic_correction(x=None)
        result_gr = sauron_instance.cubic_correction(x=sauron_instance.gr)
        np.testing.assert_array_almost_equal(result_none, result_gr)

    def test_array_input_shape_preserved(self, sauron_instance):
        x = np.linspace(0, 1, 10)
        result = sauron_instance.cubic_correction(x=x)
        assert result.shape == x.shape

    def test_polynomial_evaluated_correctly(self, sauron_instance):
        a = sauron_instance.cubic_coeff
        x = np.array([0.3, 0.7])
        expected = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3
        np.testing.assert_allclose(
            sauron_instance.cubic_correction(x=x), expected
        )


# ---------------------------------------------------------------------------
# R_vector
# ---------------------------------------------------------------------------

class TestRVector:
    def test_at_zero(self, sauron_instance):
        assert sauron_instance.R_vector(x=0.0) == pytest.approx(
            sauron_instance.R_coeff[0]
        )

    def test_at_one(self, sauron_instance):
        c = sauron_instance.R_coeff
        assert sauron_instance.R_vector(x=1.0) == pytest.approx(c[0] + c[1])

    def test_linear_array(self, sauron_instance):
        c = sauron_instance.R_coeff
        x = np.array([0.0, 0.5, 1.0])
        expected = c[0] + c[1] * x
        np.testing.assert_allclose(sauron_instance.R_vector(x=x), expected)


# ---------------------------------------------------------------------------
# ascii output methods
# ---------------------------------------------------------------------------

class TestAsciiComp:
    def test_returns_string(self, sauron_instance):
        assert isinstance(sauron_instance.ascii_comp(), str)

    def test_starts_with_f_comp(self, sauron_instance):
        assert sauron_instance.ascii_comp().startswith("f_(comp)=")

    def test_contains_active_filters(self, sauron_instance):
        eqn = sauron_instance.ascii_comp()
        # sys_filters='gri', coeff[0..2] > 0.001 — g, r, i terms must appear
        assert "f_g" in eqn
        assert "f_r" in eqn
        assert "f_i" in eqn

    def test_contains_color_correction_term(self, sauron_instance):
        eqn = sauron_instance.ascii_comp()
        assert "f_g/f_i" in eqn

    def test_no_color_correction_omits_power_term(self, sauron_instance):
        sauron_instance.color_correction = False
        sauron_instance.coeff = np.array([0.4, 0.4, 0.2, 0.0, 0.0])
        eqn = sauron_instance.ascii_comp()
        assert "f_g/f_i" not in eqn


class TestAsciiR:
    def test_returns_string(self, sauron_instance):
        assert isinstance(sauron_instance.ascii_R(), str)

    def test_starts_with_R_equals(self, sauron_instance):
        assert sauron_instance.ascii_R().startswith("R=")

    def test_contains_gr_int(self, sauron_instance):
        assert "g-r" in sauron_instance.ascii_R()

    def test_returns_none_when_no_coeff(self, sauron_instance):
        sauron_instance.R_coeff = None
        assert sauron_instance.ascii_R() is None


class TestAsciiCubicCorrection:
    def test_returns_string(self, sauron_instance):
        assert isinstance(sauron_instance.ascii_cubic_correction(), str)

    def test_starts_with_mc(self, sauron_instance):
        assert sauron_instance.ascii_cubic_correction().startswith("m_c=")

    def test_contains_all_powers(self, sauron_instance):
        eqn = sauron_instance.ascii_cubic_correction()
        assert "(g-r)" in eqn
        assert "(g-r)^2" in eqn
        assert "(g-r)^3" in eqn


# ---------------------------------------------------------------------------
# make_composite
# ---------------------------------------------------------------------------

class TestMakeComposite:
    def _mags_dict(self, n=30):
        rng = np.random.default_rng(99)
        return {
            "g": rng.normal(18.5, 0.2, n),
            "r": rng.normal(18.0, 0.2, n),
            "i": rng.normal(17.8, 0.2, n),
            "z": rng.normal(17.6, 0.2, n),
            "y": rng.normal(17.5, 0.2, n),
        }

    def test_explicit_mags_returns_array(self, sauron_instance):
        sauron_instance.color_correction = False
        sauron_instance.coeff = np.array([0.4, 0.4, 0.2, 0.0, 0.0])
        result = sauron_instance.make_composite(mags=self._mags_dict())
        assert result is not None
        assert result.shape == (30,)

    def test_explicit_mags_all_finite(self, sauron_instance):
        sauron_instance.color_correction = False
        sauron_instance.coeff = np.array([0.4, 0.4, 0.2, 0.0, 0.0])
        result = sauron_instance.make_composite(mags=self._mags_dict())
        assert np.all(np.isfinite(result))

    def test_self_mags_returns_none_sets_comp(self, sauron_instance):
        sauron_instance.color_correction = False
        sauron_instance.coeff = np.array([0.4, 0.4, 0.2, 0.0, 0.0])
        result = sauron_instance.make_composite(mags=None)
        assert result is None
        assert sauron_instance.comp is not None

    def test_comp_length_matches_sys_mags(self, sauron_instance):
        sauron_instance.color_correction = False
        sauron_instance.coeff = np.array([0.4, 0.4, 0.2, 0.0, 0.0])
        sauron_instance.make_composite(mags=None)
        assert len(sauron_instance.comp) == len(sauron_instance.sys_mags["g"])

    def test_magnitudes_in_plausible_range(self, sauron_instance):
        sauron_instance.color_correction = False
        sauron_instance.coeff = np.array([0.4, 0.4, 0.2, 0.0, 0.0])
        result = sauron_instance.make_composite(mags=self._mags_dict())
        assert np.nanmin(result) > 10
        assert np.nanmax(result) < 30

    def test_with_color_correction(self, sauron_instance):
        sauron_instance.color_correction = True
        result = sauron_instance.make_composite(mags=self._mags_dict())
        assert result is not None
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# _make_c0 and _make_bds
# ---------------------------------------------------------------------------

class TestMakeC0:
    def test_length_with_color_correction(self, sauron_instance):
        c0 = sauron_instance._make_c0()
        assert len(c0) == 6  # 5 filters + power term

    def test_length_without_color_correction(self, sauron_instance):
        sauron_instance.color_correction = False
        c0 = sauron_instance._make_c0()
        assert len(c0) == 5

    def test_active_filters_get_nonzero_initial(self, sauron_instance):
        sauron_instance.sys_filters = "gr"
        c0 = sauron_instance._make_c0()
        assert c0[0] > 0  # g
        assert c0[1] > 0  # r
        assert c0[2] == 0  # i not in filters

    def test_inactive_filters_are_zero(self, sauron_instance):
        sauron_instance.sys_filters = "gi"
        c0 = sauron_instance._make_c0()
        assert c0[1] == 0  # r not in filters


class TestMakeBds:
    def test_length_with_color_correction(self, sauron_instance):
        bds = sauron_instance._make_bds()
        assert len(bds) == 6

    def test_active_filter_has_wide_bounds(self, sauron_instance):
        sauron_instance.sys_filters = "gr"
        bds = sauron_instance._make_bds()
        assert bds[0] == (0, 2)  # g active
        assert bds[1] == (0, 2)  # r active

    def test_inactive_filter_has_tight_bounds(self, sauron_instance):
        sauron_instance.sys_filters = "gr"
        bds = sauron_instance._make_bds()
        assert bds[2][1] < 1e-5  # i inactive


# ---------------------------------------------------------------------------
# save_state / _read_load_state round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadState:
    def test_round_trip_preserves_system(self, sauron_instance):
        from calibrimbore.calibrimbore import sauron

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state")
            sauron_instance.band = np.column_stack(
                [np.linspace(5000, 7000, 50), np.ones(50)]
            )
            sauron_instance.save_state(path)

            s2 = object.__new__(sauron)
            s2._read_load_state(path + ".npy")
            assert s2.system == sauron_instance.system

    def test_round_trip_preserves_coeff(self, sauron_instance):
        from calibrimbore.calibrimbore import sauron

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state")
            sauron_instance.band = np.column_stack(
                [np.linspace(5000, 7000, 50), np.ones(50)]
            )
            sauron_instance.save_state(path)

            s2 = object.__new__(sauron)
            s2._read_load_state(path + ".npy")
            np.testing.assert_array_almost_equal(s2.coeff, sauron_instance.coeff)

    def test_round_trip_preserves_R_coeff(self, sauron_instance):
        from calibrimbore.calibrimbore import sauron

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state")
            sauron_instance.band = np.column_stack(
                [np.linspace(5000, 7000, 50), np.ones(50)]
            )
            sauron_instance.save_state(path)

            s2 = object.__new__(sauron)
            s2._read_load_state(path + ".npy")
            np.testing.assert_array_almost_equal(
                s2.R_coeff, sauron_instance.R_coeff
            )

    def test_round_trip_preserves_cubic_coeff(self, sauron_instance):
        from calibrimbore.calibrimbore import sauron

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state")
            sauron_instance.band = np.column_stack(
                [np.linspace(5000, 7000, 50), np.ones(50)]
            )
            sauron_instance.save_state(path)

            s2 = object.__new__(sauron)
            s2._read_load_state(path + ".npy")
            np.testing.assert_array_almost_equal(
                s2.cubic_coeff, sauron_instance.cubic_coeff
            )

    def test_file_is_created(self, sauron_instance):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state")
            sauron_instance.band = np.column_stack(
                [np.linspace(5000, 7000, 50), np.ones(50)]
            )
            sauron_instance.save_state(path)
            assert os.path.exists(path + ".npy")


# ---------------------------------------------------------------------------
# _load_band bug-fix regression (ndarray path)
# ---------------------------------------------------------------------------

class TestLoadBand:
    """
    Before the fix, _load_band raised NameError when self.band was a numpy
    array because the local variable `band` was referenced before assignment
    in the elif branch. These tests verify the corrected behaviour.
    """

    def _make_band_sauron(self, band_array):
        from calibrimbore.calibrimbore import sauron
        s = object.__new__(sauron)
        s.band = band_array
        s.mag_system = "ab"
        s.name = "test"
        return s

    def test_portrait_array_does_not_raise(self):
        wave = np.linspace(5000, 7000, 100)
        tp = np.exp(-0.5 * ((wave - 6000) / 500) ** 2)
        band = np.column_stack([wave, tp])  # shape (100, 2) — portrait

        s = self._make_band_sauron(band)
        with patch("calibrimbore.calibrimbore.get_pb_zpt", return_value=25.0):
            s._load_band()  # must not raise NameError

    def test_landscape_array_is_transposed(self):
        wave = np.linspace(5000, 7000, 100)
        tp = np.exp(-0.5 * ((wave - 6000) / 500) ** 2)
        band = np.row_stack([wave, tp])  # shape (2, 100) — landscape

        s = self._make_band_sauron(band)

        captured = {}

        def fake_bandpass(w, t, **kw):
            captured["wave"] = w
            captured["throughput"] = t
            bp = MagicMock()
            bp.wave = w
            bp.throughput = t
            return bp

        import sys
        sys.modules["pysynphot"].ArrayBandpass.side_effect = fake_bandpass

        with patch("calibrimbore.calibrimbore.get_pb_zpt", return_value=25.0):
            s._load_band()

        # After transposing, first column should be the wavelength array
        assert len(captured["wave"]) == 100
        np.testing.assert_array_equal(captured["wave"], wave)

    def test_band_attribute_is_replaced(self):
        wave = np.linspace(5000, 7000, 50)
        tp = np.ones(50) * 0.5
        band = np.column_stack([wave, tp])

        s = self._make_band_sauron(band)
        with patch("calibrimbore.calibrimbore.get_pb_zpt", return_value=25.0):
            s._load_band()

        # self.band must have been replaced with the pysynphot bandpass object
        assert not isinstance(s.band, np.ndarray)

    def test_zp_is_set(self):
        wave = np.linspace(5000, 7000, 50)
        tp = np.ones(50) * 0.5
        band = np.column_stack([wave, tp])

        s = self._make_band_sauron(band)
        with patch("calibrimbore.calibrimbore.get_pb_zpt", return_value=24.5):
            s._load_band()

        assert s.zp == pytest.approx(24.5)


# ---------------------------------------------------------------------------
# estimate_mag — catalog.lower() bug-fix regression
# ---------------------------------------------------------------------------

class TestEstimateMagCatalogLower:
    """
    Before the fix, `catalog.lower` was a method reference (always truthy),
    so the 'casjobs' branch was never entered.  After the fix it is
    `catalog.lower()` (a call).  We verify the fixed routing by checking
    that `catalog='vizier'` invokes `_get_catalog` and that passing an
    uppercase variant is handled without AttributeError.
    """

    def test_vizier_catalog_calls_get_catalog(self, sauron_instance):
        ra = np.array([10.0, 10.1])
        dec = np.array([-20.0, -20.1])

        fake_mags = {
            "g": np.array([18.0, 18.5]),
            "r": np.array([17.5, 18.0]),
            "i": np.array([17.3, 17.8]),
            "z": np.array([17.1, 17.6]),
            "y": np.array([17.0, 17.5]),
            "ra": pd.Series([10.0, 10.1]),
            "dec": pd.Series([-20.0, -20.1]),
        }

        sauron_instance.color_correction = False
        sauron_instance.coeff = np.array([0.4, 0.4, 0.2, 0.0, 0.0])
        sauron_instance.cubic_corr = False
        sauron_instance.gr_lims = None
        sauron_instance.gi_lims = None
        sauron_instance.R_coeff = np.array([2.85, -0.09])

        with patch.object(
            sauron_instance, "_get_catalog", return_value=fake_mags
        ) as mock_get:
            with patch.object(sauron_instance, "get_extinctions", return_value=np.zeros(2)):
                sauron_instance.estimate_mag(
                    ra=ra, dec=dec, catalog="vizier", extinction=False
                )
            mock_get.assert_called_once()

    def test_catalog_case_insensitive_vizier(self, sauron_instance):
        """catalog='VIZIER' should be treated identically to 'vizier'."""
        ra = np.array([10.0])
        dec = np.array([-20.0])

        fake_mags = {
            "g": np.array([18.0]),
            "r": np.array([17.5]),
            "i": np.array([17.3]),
            "z": np.array([17.1]),
            "y": np.array([17.0]),
            "ra": pd.Series([10.0]),
            "dec": pd.Series([-20.0]),
        }

        sauron_instance.color_correction = False
        sauron_instance.coeff = np.array([0.4, 0.4, 0.2, 0.0, 0.0])
        sauron_instance.cubic_corr = False
        sauron_instance.gr_lims = None
        sauron_instance.gi_lims = None
        sauron_instance.R_coeff = np.array([2.85, -0.09])

        with patch.object(
            sauron_instance, "_get_catalog", return_value=fake_mags
        ):
            with patch.object(sauron_instance, "get_extinctions", return_value=np.zeros(1)):
                # Should not raise AttributeError (the pre-fix bug)
                sauron_instance.estimate_mag(
                    ra=ra, dec=dec, catalog="VIZIER", extinction=False
                )
