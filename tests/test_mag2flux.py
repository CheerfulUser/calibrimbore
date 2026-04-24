import numpy as np
import pytest
from calibrimbore.calibrimbore import mag2flux


def test_flux_equals_one_at_zeropoint():
    assert mag2flux(25, zp=25) == pytest.approx(1.0)


def test_flux_at_mag_zero():
    # mag=0, zp=25 → 10^(2/5 * 25) = 10^10
    assert mag2flux(0, zp=25) == pytest.approx(1e10)


def test_default_zp_is_25():
    assert mag2flux(25) == pytest.approx(mag2flux(25, zp=25))


def test_flux_decreases_with_magnitude():
    mags = np.linspace(15, 25, 20)
    fluxes = mag2flux(mags, zp=25)
    assert np.all(np.diff(fluxes) < 0)


def test_array_input_length_preserved():
    mags = np.array([20.0, 22.0, 24.0])
    result = mag2flux(mags, zp=25)
    assert result.shape == mags.shape


def test_round_trip():
    mag = 21.3
    flux = mag2flux(mag, zp=25)
    mag_back = -2.5 * np.log10(flux) + 25
    assert mag_back == pytest.approx(mag)
