"""
Mock heavy optional dependencies before any calibrimbore module is imported.

pysynphot requires a large external CDBS data installation; we replace it with
a lightweight mock that preserves the array-handling interface used by bill.py
and calibrimbore.py. The mock is installed into sys.modules before collection,
so every test file in this suite sees the mocked version automatically.
"""

import sys
import numpy as np
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helper factories for realistic mock objects
# ---------------------------------------------------------------------------

def _fake_bandpass(wave, throughput, **kwargs):
    bp = MagicMock()
    bp.wave = np.asarray(wave, dtype=float)
    bp.throughput = np.asarray(throughput, dtype=float)
    return bp


def _fake_spectrum(wave, flux, **kwargs):
    sp = MagicMock()
    sp.wave = np.asarray(wave, dtype=float)
    sp.flux = np.asarray(flux, dtype=float)
    return sp


# Build the top-level mock with the behaviour calibrimbore expects
_mock_S = MagicMock()
_mock_S.ArrayBandpass.side_effect = _fake_bandpass
_mock_S.ArraySpectrum.side_effect = _fake_spectrum

_ref_wave = np.linspace(1000, 25000, 2000)
_ref_flux = np.ones(2000) * 3.63e-9  # rough AB f_lambda zeropoint

_mock_S.ABMag.return_value = _fake_spectrum(_ref_wave, _ref_flux)
_mock_S.FlatSpectrum.return_value = _fake_spectrum(_ref_wave, _ref_flux)

sys.modules["pysynphot"] = _mock_S

# Mock network / optional catalog dependencies
sys.modules["dl"] = MagicMock()
sys.modules["dl.queryClient"] = MagicMock()
sys.modules["mastcasjobs"] = MagicMock()

# Mock IPython display — calibrimbore.py imports it at module level but it is
# not a hard runtime requirement (only used in print_* methods).
try:
    import IPython  # noqa: F401
except ModuleNotFoundError:
    _mock_ipython = MagicMock()
    sys.modules["IPython"] = _mock_ipython
    sys.modules["IPython.display"] = _mock_ipython.display


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sauron_instance():
    """
    Return a sauron object with a pre-set calibrated state, bypassing __init__.

    All attributes are set to realistic values so individual methods can be
    exercised without running the full fitting pipeline.
    """
    from calibrimbore.calibrimbore import sauron

    s = object.__new__(sauron)
    rng = np.random.default_rng(42)
    n = 60

    s.system = "ps1"
    s.mag_system = "ab"
    s.spec_model = "calspec"
    s.name = "test_band"
    s.savename = None
    s.color_correction = True
    s.cubic_corr = True
    s.gr_lims = None
    s.gi_lims = None
    s.sys_filters = "gri"
    s.spline = None
    s.comp = None
    s.R = None

    s.coeff = np.array([0.3, 0.5, 0.2, 0.0, 0.0, 0.15])
    s.R_coeff = np.array([2.85, -0.09])
    s.cubic_coeff = np.array([0.005, -0.002, 0.0003, -0.00001])

    s.gr = rng.uniform(-0.2, 1.5, n)
    s._grind = np.ones(n, dtype=bool)
    s.mask = np.ones(n, dtype=bool)
    s.diff = rng.normal(0, 0.01, n)
    s.mags = rng.normal(18.0, 0.5, n)

    s.sys_mags = {
        "g": rng.normal(18.5, 0.5, n),
        "r": rng.normal(18.0, 0.5, n),
        "i": rng.normal(17.8, 0.5, n),
        "z": rng.normal(17.6, 0.5, n),
        "y": rng.normal(17.5, 0.5, n),
    }
    return s
