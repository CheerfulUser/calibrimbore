"""
Tests for pure-logic functions in calibrimbore/bill.py that do not require
network access or pysynphot (the latter is mocked in conftest.py).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# isolate_stars
# ---------------------------------------------------------------------------

class TestIsolateStars:
    def _base_cat(self):
        return pd.DataFrame({
            "gQfPerfect": [0.95, 0.95, 0.70],
            "rQfPerfect": [0.95, 0.95, 0.70],
            "iQfPerfect": [0.95, 0.95, 0.70],
            "zQfPerfect": [0.95, 0.95, 0.70],
            "rMeanPSFMag":  [18.0, 18.5, 17.0],
            "rMeanKronMag": [18.02, 18.53, 16.5],  # last entry is a galaxy
        })

    def test_star_column_added(self):
        from calibrimbore.bill import isolate_stars
        result = isolate_stars(self._base_cat())
        assert "star" in result.columns

    def test_good_stars_flagged_one(self):
        from calibrimbore.bill import isolate_stars
        result = isolate_stars(self._base_cat())
        assert result.iloc[0]["star"] == 1
        assert result.iloc[1]["star"] == 1

    def test_galaxy_flagged_zero(self):
        from calibrimbore.bill import isolate_stars
        result = isolate_stars(self._base_cat())
        assert result.iloc[2]["star"] == 0

    def test_only_stars_filters_rows(self):
        from calibrimbore.bill import isolate_stars
        result = isolate_stars(self._base_cat(), only_stars=True)
        assert len(result) == 2
        assert (result["star"] == 1).all()

    def test_low_quality_flag_excluded(self):
        from calibrimbore.bill import isolate_stars
        cat = self._base_cat()
        cat.loc[0, "rQfPerfect"] = 0.50  # below threshold
        result = isolate_stars(cat)
        assert result.iloc[0]["star"] == 0

    def test_psfkron_threshold(self):
        from calibrimbore.bill import isolate_stars
        cat = self._base_cat()
        cat.loc[0, "rMeanKronMag"] = 17.90  # PSF-Kron diff = 0.10 > 0.05
        result = isolate_stars(cat)
        assert result.iloc[0]["star"] == 0


# ---------------------------------------------------------------------------
# cut_bad_detections
# ---------------------------------------------------------------------------

class TestCutBadDetections:
    def test_valid_row_passes(self):
        from calibrimbore.bill import cut_bad_detections
        cat = pd.DataFrame({
            "rMeanPSFMag": [18.0],
            "iMeanPSFMag": [17.5],
            "zMeanPSFMag": [17.0],
        })
        result = cut_bad_detections(cat)
        assert len(result) == 1

    def test_negative_r_rejected(self):
        from calibrimbore.bill import cut_bad_detections
        cat = pd.DataFrame({
            "rMeanPSFMag": [-999.0, 18.0],
            "iMeanPSFMag": [17.5, 17.5],
            "zMeanPSFMag": [17.0, 17.0],
        })
        result = cut_bad_detections(cat)
        assert len(result) == 1
        assert result.iloc[0]["rMeanPSFMag"] == 18.0

    def test_negative_i_rejected(self):
        from calibrimbore.bill import cut_bad_detections
        cat = pd.DataFrame({
            "rMeanPSFMag": [18.0, 19.0],
            "iMeanPSFMag": [-999.0, 17.5],
            "zMeanPSFMag": [17.0, 17.0],
        })
        result = cut_bad_detections(cat)
        assert len(result) == 1

    def test_all_bad_returns_empty(self):
        from calibrimbore.bill import cut_bad_detections
        cat = pd.DataFrame({
            "rMeanPSFMag": [-999.0],
            "iMeanPSFMag": [-999.0],
            "zMeanPSFMag": [-999.0],
        })
        result = cut_bad_detections(cat)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# sigma_mask
# ---------------------------------------------------------------------------

class TestSigmaMask:
    def test_returns_boolean_array(self):
        from calibrimbore.bill import sigma_mask
        data = np.linspace(0, 10, 30)
        clipped = sigma_mask(data, sigma=3)
        assert clipped.dtype in (bool, np.bool_)

    def test_no_outliers_nothing_clipped(self):
        from calibrimbore.bill import sigma_mask
        rng = np.random.default_rng(1)
        data = rng.normal(5, 0.1, 50)
        clipped = sigma_mask(data, sigma=5)
        assert not np.any(clipped)

    def test_outlier_is_clipped(self):
        from calibrimbore.bill import sigma_mask
        rng = np.random.default_rng(5)
        data = np.concatenate([rng.normal(0, 0.1, 50), [50.0]])
        clipped = sigma_mask(data, sigma=3)
        assert clipped[-1]

    def test_inliers_not_clipped(self):
        from calibrimbore.bill import sigma_mask
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        clipped = sigma_mask(data, sigma=3)
        assert not clipped[0]
        assert not clipped[2]

    def test_output_length_matches_input(self):
        from calibrimbore.bill import sigma_mask
        data = np.arange(20, dtype=float)
        clipped = sigma_mask(data, sigma=3)
        assert len(clipped) == len(data)
