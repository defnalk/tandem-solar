"""
tandem/ctm_loss.py
------------------
Cell-to-module (CTM) loss analysis for perovskite-silicon tandem modules.

CTM losses quantify the performance gap between the sum of individual cell
Pmpp values and the measured module output. They arise from:

  1. Optical losses   — reflection, absorption in encapsulant/glass, shading
                        by interconnects and busbars
  2. Resistive losses — series resistance of interconnects, busbar, cell Rs
  3. Mismatch losses  — non-uniform Isc or Voc between cells (sorting, shading)
  4. Recombination    — increased in module environment vs. bare cell

CTM ratio = P_module / Σ P_cell  (ideal = 1.0; real < 1.0 due to losses)

For perovskite-silicon tandems, mismatch is a critical concern because:
  - Perovskite degrades non-uniformly → cell-to-cell Isc variation
  - Current mismatch in 2T is especially penalising

Reference: SERIS NUS internship project — cell-to-module loss evaluation
           for next-generation tandem solar technology (Summer 2025)
"""

from dataclasses import dataclass, field

import numpy as np

from .cell_model import PEROVSKITE_PARAMS, SILICON_PARAMS, CellParameters, SolarCell


@dataclass
class CTMLossFactors:
    """
    Collection of CTM loss factors (fractional, relative to cell Pmax).

    Each factor represents the fraction of power lost to that mechanism.
    Total CTM ratio = 1 - sum(losses) + optical_gain

    Attributes
    ----------
    optical_reflection   : float  Reflection loss at glass/air interface
    optical_encapsulant  : float  Absorption in encapsulant (EVA/POE)
    optical_interconnect : float  Shading by cell interconnects and busbars
    resistive_interconnect: float Series resistance of interconnect ribbons
    resistive_busbar     : float  Busbar resistance contribution
    mismatch_isc         : float  Isc mismatch between cells in string
    mismatch_voc         : float  Voc mismatch between cells
    """
    optical_reflection:    float = 0.020   # 2.0%  — glass AR coating typical
    optical_encapsulant:   float = 0.015   # 1.5%  — EVA/POE absorption
    optical_interconnect:  float = 0.025   # 2.5%  — ribbon/busbar shading
    resistive_interconnect:float = 0.010   # 1.0%  — interconnect Rs
    resistive_busbar:      float = 0.008   # 0.8%  — busbar Rs
    mismatch_isc:          float = 0.015   # 1.5%  — Isc spread between cells
    mismatch_voc:          float = 0.005   # 0.5%  — Voc spread between cells

    # Optical gain from light trapping / texturing
    optical_gain: float = 0.010   # 1.0%  — light trapping benefit

    @property
    def total_loss(self) -> float:
        """Total fractional CTM loss."""
        return (
            self.optical_reflection
            + self.optical_encapsulant
            + self.optical_interconnect
            + self.resistive_interconnect
            + self.resistive_busbar
            + self.mismatch_isc
            + self.mismatch_voc
            - self.optical_gain
        )

    @property
    def ctm_ratio(self) -> float:
        """CTM ratio (1.0 = no losses)."""
        return 1.0 - self.total_loss


class CTMLossAnalyser:
    """
    Cell-to-module (CTM) loss analyser for tandem solar modules.

    Computes the power gap between bare cells and the encapsulated module,
    broken down by loss mechanism. Supports sensitivity analysis to identify
    the dominant loss contributions.

    Parameters
    ----------
    n_cells_series : int
        Number of cells connected in series per string. Default 11.
    n_strings : int
        Number of strings in parallel. Default 2.
    top_params : CellParameters
        Perovskite sub-cell parameters.
    bottom_params : CellParameters
        Silicon sub-cell parameters.
    losses : CTMLossFactors or None
        Custom loss factors. If None, uses typical literature values.

    Examples
    --------
    >>> ctm = CTMLossAnalyser(n_cells_series=11)
    >>> summary = ctm.loss_summary()
    >>> print(f"CTM ratio: {summary['ctm_ratio']:.3f}")
    """

    def __init__(
        self,
        n_cells_series: int = 11,
        n_strings: int = 2,
        top_params: CellParameters = PEROVSKITE_PARAMS,
        bottom_params: CellParameters = SILICON_PARAMS,
        losses: CTMLossFactors | None = None,
    ):
        self.n_cells_series = n_cells_series
        self.n_strings      = n_strings
        self.n_total        = n_cells_series * n_strings
        self.top    = SolarCell(top_params)
        self.bottom = SolarCell(bottom_params)
        self.losses = losses or CTMLossFactors()

    def cell_pmax_sum(self, G: float = 1000.0) -> dict[str, float]:
        """
        Sum of individual cell Pmax values — the upper bound before CTM losses.

        Parameters
        ----------
        G : float
            Irradiance (W/m²).

        Returns
        -------
        dict with: 'P_top', 'P_bottom', 'P_tandem_cell', 'P_total_sum'
        """
        _, _, Pt = self.top.iv_curve(G=G)
        _, _, Pb = self.bottom.iv_curve(G=G)

        P_top_cell    = float(Pt.max())
        P_bottom_cell = float(Pb.max())
        P_tandem_cell = P_top_cell + P_bottom_cell

        return {
            "P_top_cell":      P_top_cell,
            "P_bottom_cell":   P_bottom_cell,
            "P_tandem_cell":   P_tandem_cell,
            "P_total_sum":     P_tandem_cell * self.n_total,
        }

    def module_pmax(self, G: float = 1000.0) -> float:
        """
        Estimated module Pmax after applying all CTM loss factors.

        Returns
        -------
        float
            Module power output (W).
        """
        cell_sum = self.cell_pmax_sum(G=G)
        return cell_sum["P_total_sum"] * self.losses.ctm_ratio

    def loss_summary(self, G: float = 1000.0) -> dict:
        """
        Full CTM loss breakdown — absolute (W) and fractional (%).

        Returns
        -------
        dict with loss categories, CTM ratio, and a waterfall table.
        """
        cell_sum = self.cell_pmax_sum(G=G)
        P_cell   = cell_sum["P_total_sum"]
        lf = self.losses

        losses_abs = {
            "Optical: reflection":    lf.optical_reflection    * P_cell,
            "Optical: encapsulant":   lf.optical_encapsulant   * P_cell,
            "Optical: interconnect":  lf.optical_interconnect  * P_cell,
            "Resistive: interconnect":lf.resistive_interconnect * P_cell,
            "Resistive: busbar":      lf.resistive_busbar      * P_cell,
            "Mismatch: Isc":          lf.mismatch_isc          * P_cell,
            "Mismatch: Voc":          lf.mismatch_voc          * P_cell,
            "Optical gain (texturing)": -lf.optical_gain       * P_cell,
        }

        return {
            "P_cell_sum_W":     round(P_cell, 4),
            "P_module_W":       round(self.module_pmax(G=G), 4),
            "ctm_ratio":        round(lf.ctm_ratio, 4),
            "total_loss_pct":   round(lf.total_loss * 100, 2),
            "losses_W":         {k: round(v, 4) for k, v in losses_abs.items()},
            "losses_pct":       {k: round(v / P_cell * 100, 2) for k, v in losses_abs.items()},
            "n_cells":          self.n_total,
        }

    def mismatch_sensitivity(
        self,
        sigma_Isc_range: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Sensitivity of CTM ratio to Isc mismatch (spread between cells).

        Simulates a string where cells have normally distributed Isc values.
        In a 2T string, the minimum Isc cell limits the entire string current.

        Parameters
        ----------
        sigma_Isc_range : np.ndarray or None
            Range of Isc standard deviation as fraction of mean.
            Default: np.linspace(0, 0.10, 30) — 0% to 10% spread.

        Returns
        -------
        dict with 'sigma_Isc', 'ctm_ratio', 'mismatch_loss_pct'
        """
        if sigma_Isc_range is None:
            sigma_Isc_range = np.linspace(0, 0.10, 30)

        ctm_ratios = []
        Isc_mean = self.bottom.p.Isc

        # Seed once outside the loop. Re-creating the RNG with the same seed
        # for every sigma re-uses an identical underlying draw, so the only
        # thing varying across the sweep is the scale factor — that masks
        # the genuine MC noise and biases the worst-case minimum estimator.
        rng = np.random.default_rng(42)
        n_mc = 1000
        for sigma in sigma_Isc_range:
            Isc_cells = rng.normal(Isc_mean, sigma * Isc_mean,
                                   size=(n_mc, self.n_cells_series))
            # 2T string current = min(Isc) in series string
            I_string = Isc_cells.min(axis=1)
            # CTM ratio from mismatch only
            ctm = float((I_string / Isc_mean).mean())
            ctm_ratios.append(ctm)

        ctm_arr = np.array(ctm_ratios)
        return {
            "sigma_Isc_frac":   sigma_Isc_range,
            "ctm_ratio":        ctm_arr,
            "mismatch_loss_pct": (1 - ctm_arr) * 100,
        }

    def irradiance_sensitivity(
        self,
        G_range: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Module power and CTM ratio across irradiance conditions.

        Parameters
        ----------
        G_range : np.ndarray or None
            Irradiance values (W/m²). Default: np.linspace(100, 1000, 20).

        Returns
        -------
        dict with 'G', 'P_module', 'ctm_ratio', 'efficiency'
        """
        if G_range is None:
            G_range = np.linspace(100, 1000, 20)

        # cell_pmax_sum solves the single-diode model 200× per call via
        # brentq, so the original loop did ~20 × 400 root-finds for what is,
        # in this first-order model, a strictly linear scaling in G. Compute
        # the STC reference once and broadcast across G_range. The CTM ratio
        # is treated as constant elsewhere in the package, so this is
        # equivalent to the previous behaviour, just N× faster.
        G_range = np.asarray(G_range, dtype=float)
        ctm_ratio = self.losses.ctm_ratio
        P_total_stc = self.cell_pmax_sum(G=1000.0)["P_total_sum"]
        P_module = P_total_stc * (G_range / 1000.0) * ctm_ratio

        return {
            "G":        G_range,
            "P_module": P_module,
            "ctm_ratio": np.full_like(G_range, ctm_ratio),
        }
