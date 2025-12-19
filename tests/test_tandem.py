"""
tests/test_tandem.py
Unit tests for the tandem-solar package.
Run: python -m pytest tests/ -v
"""

import numpy as np
import pytest
from tandem import SolarCell, TandemModule, CTMLossAnalyser, ShadingAnalysis
from tandem.cell_model import (
    CellParameters, SILICON_PARAMS, PEROVSKITE_PARAMS,
    fit_iv_parameters,
)
from tandem.ctm_loss import CTMLossFactors


# ── Cell model ────────────────────────────────────────────────────────────────

class TestSolarCell:
    def test_iv_curve_shape(self):
        cell = SolarCell(SILICON_PARAMS)
        V, I, P = cell.iv_curve(n_points=100)
        assert len(V) == len(I) == len(P) == 100

    def test_isc_at_zero_voltage(self):
        cell = SolarCell(SILICON_PARAMS)
        I0 = cell.current(0.0)
        assert I0 > SILICON_PARAMS.Isc * 0.5   # significant current at V=0

    def test_current_zero_at_voc(self):
        cell = SolarCell(SILICON_PARAMS)
        I_sc  = cell.current(0.0)
        I_voc = cell.current(SILICON_PARAMS.Voc)
        assert I_voc < I_sc * 0.5   # current substantially lower near Voc

    def test_power_positive(self):
        cell = SolarCell(SILICON_PARAMS)
        _, _, P = cell.iv_curve()
        assert P.max() > 0

    def test_mpp_within_bounds(self):
        cell = SolarCell(SILICON_PARAMS)
        V, I, P = cell.iv_curve()
        Vmpp = float(V[P.argmax()])
        assert 0 < Vmpp < SILICON_PARAMS.Voc * 1.05

    def test_irradiance_scaling(self):
        """Power should scale approximately linearly with irradiance."""
        cell = SolarCell(SILICON_PARAMS)
        _, _, P1000 = cell.iv_curve(G=1000)
        _, _, P500  = cell.iv_curve(G=500)
        ratio = P1000.max() / P500.max()
        assert 1.8 < ratio < 2.2

    def test_perovskite_higher_voc(self):
        """Perovskite cell should have higher Voc than silicon."""
        si = SolarCell(SILICON_PARAMS)
        pe = SolarCell(PEROVSKITE_PARAMS)
        assert pe.p.Voc > si.p.Voc

    def test_fill_factor_reasonable(self):
        cell = SolarCell(SILICON_PARAMS)
        ff = cell.p.FF
        assert 0.5 < ff < 0.95

    def test_temperature_coefficients(self):
        cell = SolarCell(SILICON_PARAMS)
        tc = cell.temperature_coefficients()
        assert tc["beta_Voc"] < 0    # Voc decreases with T
        assert tc["alpha_Isc"] > 0   # Isc increases with T

    def test_fit_iv_parameters(self):
        """Fitted parameters should be physically reasonable."""
        cell = SolarCell(SILICON_PARAMS)
        V, I, _ = cell.iv_curve()
        params = fit_iv_parameters(V, I, cell_type="silicon")
        assert params.Voc > 0
        assert params.Isc > 0
        assert params.Rs >= 0


# ── Tandem module ─────────────────────────────────────────────────────────────

class TestTandemModule:
    def setup_method(self):
        self.mod = TandemModule(n_cells=11)

    def test_2T_iv_returns_arrays(self):
        V, I, P = self.mod.iv_2T()
        assert len(V) == len(I) == len(P)

    def test_2T_power_positive(self):
        _, _, P = self.mod.iv_2T()
        assert P.max() > 0

    def test_2T_current_limited_by_weaker_subcell(self):
        """2T string current cannot exceed the lower Isc sub-cell."""
        V, I, P = self.mod.iv_2T()
        Isc_min = min(self.mod.top.p.Isc, self.mod.bottom.p.Isc)
        assert I.max() <= Isc_min * 1.05

    def test_4T_total_higher_than_2T(self):
        """4T should produce more power than 2T (no matching loss)."""
        mpp_2T = self.mod.mpp_2T()
        mpp_4T = self.mod.mpp_4T()
        assert mpp_4T["Pmpp_total"] >= mpp_2T["Pmpp"]

    def test_3T_end_losses_smaller_r_than_s(self):
        """3T-r has fewer end losses than 3T-s (McMahon et al.)."""
        r3 = self.mod.iv_3T(config="r")
        s3 = self.mod.iv_3T(config="s")
        assert r3["end_loss_fraction"] < s3["end_loss_fraction"]

    def test_configuration_comparison_4T_is_reference(self):
        comp = self.mod.compare_configurations()
        assert comp["4T"]["rel_eff"] == 1.0

    def test_configuration_ordering(self):
        """Expected: 4T ≥ 3T-r ≥ 2T (approximately)."""
        comp = self.mod.compare_configurations()
        assert comp["4T"]["Pmpp"] >= comp["2T"]["Pmpp"]

    def test_irradiance_reduces_power(self):
        mpp_high = self.mod.mpp_2T(G=1000)
        mpp_low  = self.mod.mpp_2T(G=200)
        assert mpp_high["Pmpp"] > mpp_low["Pmpp"]


# ── CTM loss ──────────────────────────────────────────────────────────────────

class TestCTMLossAnalyser:
    def setup_method(self):
        self.ctm = CTMLossAnalyser(n_cells_series=11)

    def test_ctm_ratio_below_1(self):
        summary = self.ctm.loss_summary()
        assert summary["ctm_ratio"] < 1.0

    def test_ctm_ratio_above_0(self):
        summary = self.ctm.loss_summary()
        assert summary["ctm_ratio"] > 0.0

    def test_module_power_less_than_cell_sum(self):
        summary = self.ctm.loss_summary()
        assert summary["P_module_W"] < summary["P_cell_sum_W"]

    def test_loss_factors_sum_correctly(self):
        lf = CTMLossFactors()
        assert abs(lf.total_loss - (
            lf.optical_reflection + lf.optical_encapsulant +
            lf.optical_interconnect + lf.resistive_interconnect +
            lf.resistive_busbar + lf.mismatch_isc + lf.mismatch_voc
            - lf.optical_gain
        )) < 1e-9

    def test_mismatch_sensitivity_decreases_ctm(self):
        result = self.ctm.mismatch_sensitivity()
        # More spread → lower CTM ratio
        assert result["ctm_ratio"][0] >= result["ctm_ratio"][-1]

    def test_cell_pmax_sum_positive(self):
        cells = self.ctm.cell_pmax_sum()
        assert cells["P_total_sum"] > 0

    def test_irradiance_sensitivity_shape(self):
        result = self.ctm.irradiance_sensitivity()
        assert len(result["P_module"]) == len(result["G"])


# ── Shading ───────────────────────────────────────────────────────────────────

class TestShadingAnalysis:
    def setup_method(self):
        self.sa = ShadingAnalysis(n_cells=22)

    def test_stc_power_positive(self):
        assert self.sa.stc_power() > 0

    def test_shading_reduces_power(self):
        result = self.sa.shade_single_cell(n_cells_under_diode=0)
        assert result["P_shaded_no_bpd"] < result["P_stc"]

    def test_bpd_recovers_power(self):
        result = self.sa.shade_single_cell(n_cells_under_diode=9)
        assert result["P_bpd"] >= result["P_shaded_no_bpd"]

    def test_bypass_sweep_correct_shape(self):
        sweep = self.sa.bypass_diode_sweep()
        assert len(sweep["P_mpp"]) == len(sweep["n_cells_bpd"])

    def test_breakdown_voltage_analysis(self):
        result = self.sa.breakdown_voltage_analysis()
        assert len(result["Vbd_Si"]) == len(result["is_pero_protected"])

    def test_9_cells_less_power_than_1_cell_bpd(self):
        """More cells under BPD → more cells bypassed → LESS power (per Devoto et al. Fig. 8)."""
        r1 = self.sa.shade_single_cell(n_cells_under_diode=1)
        r9 = self.sa.shade_single_cell(n_cells_under_diode=9)
        assert r1["P_bpd"] >= r9["P_bpd"]   # fewer bypassed = more power
