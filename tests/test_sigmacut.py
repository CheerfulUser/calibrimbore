import numpy as np
import pytest
from calibrimbore.sigmacut import calcaverageclass


@pytest.fixture
def calc():
    return calcaverageclass()


class TestBasicStatistics:
    def test_mean_of_uniform_data(self, calc):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        calc.calcaverage_sigmacutloop(data, Nsigma=3)
        assert calc.mean == pytest.approx(3.0)
        assert calc.Nused == 5

    def test_all_data_used_when_no_outliers(self, calc):
        rng = np.random.default_rng(0)
        data = rng.normal(10, 1, 50)
        calc.calcaverage_sigmacutloop(data, Nsigma=5)
        assert calc.Nused == 50

    def test_stdev_close_to_one_for_unit_normal(self, calc):
        rng = np.random.default_rng(7)
        data = rng.normal(0, 1, 500)
        calc.calcaverage_sigmacutloop(data, Nsigma=4)
        assert 0.85 < calc.stdev < 1.15


class TestSigmaClipping:
    def test_outlier_excluded(self, calc):
        # Use a tight cluster with a clear outlier far from the group mean.
        rng = np.random.default_rng(5)
        data = np.concatenate([rng.normal(0, 0.1, 50), [50.0]])
        calc.calcaverage_sigmacutloop(data, Nsigma=3)
        assert calc.mean == pytest.approx(0.0, abs=0.1)
        assert calc.Nused == 50

    def test_clipped_array_marks_outlier(self, calc):
        rng = np.random.default_rng(5)
        data = np.concatenate([rng.normal(0, 0.1, 50), [50.0]])
        calc.calcaverage_sigmacutloop(data, Nsigma=3, saveused=True)
        assert calc.clipped[-1]        # outlier clipped
        assert not calc.clipped[0]     # inlier not clipped

    def test_saveused_false_no_clipped_attr(self, calc):
        data = np.array([1.0, 2.0, 3.0])
        calc.calcaverage_sigmacutloop(data, Nsigma=3, saveused=False)
        # clipped may not be set; if it is it should match the data length
        if hasattr(calc, "clipped") and calc.clipped is not None:
            assert len(calc.clipped) == len(data)

    def test_fixed_mean_respected(self, calc):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        calc.calcaverage_sigmacutloop(data, Nsigma=3, fixmean=3.0)
        assert calc.mean == pytest.approx(3.0)


class TestConvergence:
    def test_converges_on_clean_data(self, calc):
        rng = np.random.default_rng(42)
        data = rng.normal(10, 1, 100)
        calc.calcaverage_sigmacutloop(data, Nsigma=3)
        assert calc.converged

    def test_ntot_equals_data_length(self, calc):
        data = np.arange(10, dtype=float)
        calc.calcaverage_sigmacutloop(data, Nsigma=3)
        assert calc.Ntot == len(data)

    def test_nused_plus_nskipped_equals_ntot(self, calc):
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        calc.calcaverage_sigmacutloop(data, Nsigma=3)
        assert calc.Nused + calc.Nskipped == calc.Ntot


class TestC4BiasCorrection:
    def test_c4_between_zero_and_one(self, calc):
        for n in range(2, 10):
            assert 0 < calc.c4(n) <= 1

    def test_c4_approaches_one_for_large_n(self, calc):
        assert calc.c4(200) == pytest.approx(1.0, abs=0.005)

    def test_c4_increases_with_n(self, calc):
        values = [calc.c4(n) for n in range(2, 15)]
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))
