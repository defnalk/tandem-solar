"""
tandem/shading.py
-----------------
Partial shading and bypass diode protection modelling for tandem PV modules.

In 2T tandem modules, partial shading is especially damaging because:
  - The shaded cell limits the entire string current (series connection)
  - Perovskite sub-cells have low reverse breakdown voltage (Vbd ≈ −1 to −5 V)
  - Unlike silicon PERC cells (Vbd > −20 V), perovskite cells require
    protection even with one bypass diode per several cells

From Devoto et al. (2024):
  - In a 22-cell 2T module, one bypass diode can protect up to 9 tandem cells
  - Adding 10–14 cells to a single bypass diode provides no additional protection
  - 3T-r behaves identically to 2T for bypass diode protection
  - No effective bypass diode implementation was found for 3T-s

Reference: Devoto et al. (2024), EU PVSEC 2BV.1.41
"""

import numpy as np

from .cell_model import PEROVSKITE_PARAMS, SILICON_PARAMS, CellParameters, SolarCell

# ── Breakdown voltage parameters ───────────────────────────────────────────────
VBD_PEROVSKITE = -3.0   # V  (typical; range −1 to −5 V)
VBD_IBC_Si     = -3.7   # V  (IBC silicon soft breakdown — Chu et al. 2015)
VBD_PERC_Si    = -20.0  # V  (PERC silicon — robust)
V_BPD_FORWARD  = -0.5   # V  (bypass diode forward bias threshold)


class ShadingAnalysis:
    """
    Partial shading and bypass diode analysis for tandem PV string.

    Models the impact of shading one or more cells in a series string,
    with and without bypass diodes, reproducing the experimental methodology
    of Devoto et al. (2024).

    Parameters
    ----------
    n_cells : int
        Total number of tandem cells in the string. Default 22.
    top_params : CellParameters
        Perovskite sub-cell parameters.
    bottom_params : CellParameters
        Silicon sub-cell parameters.
    Vbd_top : float
        Reverse breakdown voltage of perovskite sub-cell (V). Default −3.0 V.
    Vbd_bottom : float
        Reverse breakdown voltage of silicon sub-cell (V). Default −3.7 V (IBC).

    Examples
    --------
    >>> sa = ShadingAnalysis(n_cells=22)
    >>> result = sa.shade_single_cell(n_cells_under_diode=9)
    >>> print(f"Power with BPD on 9 cells: {result['P_bpd']:.2f} W")
    """

    def __init__(
        self,
        n_cells: int = 22,
        top_params: CellParameters = PEROVSKITE_PARAMS,
        bottom_params: CellParameters = SILICON_PARAMS,
        Vbd_top: float = VBD_PEROVSKITE,
        Vbd_bottom: float = VBD_IBC_Si,
    ):
        self.n_cells     = n_cells
        self.top         = SolarCell(top_params)
        self.bottom      = SolarCell(bottom_params)
        self.Vbd_top     = Vbd_top
        self.Vbd_bottom  = Vbd_bottom

    def _string_power(
        self,
        G_cells: np.ndarray,
        n_cells_under_bpd: int = 0,
    ) -> dict[str, float]:
        """
        Compute string power given per-cell irradiance vector.

        Physics (from Devoto et al. 2024):
        - No shading: full string power
        - Shaded cell, no BPD: string current drops ~50% (shaded cell limits
          but doesn't go to zero — partial reverse bias dissipation)
        - BPD on N cells: if BPD activates, N cells are bypassed.
          Power = (n_cells - N) / n_cells × P_stc  (fewer cells → more power)
          BPD can only protect up to ~9 cells for this module (silicon Vbd limit)

        Parameters
        ----------
        G_cells : np.ndarray
            Irradiance on each cell (length = n_cells).
        n_cells_under_bpd : int
            Number of cells protected by a single bypass diode. 0 = no diode.
        """
        P_cell = (self.top.p.Vmpp + self.bottom.p.Vmpp) * self.top.p.Isc * 0.95
        P_stc  = P_cell * self.n_cells

        is_shaded = G_cells < 100
        n_shaded  = int(is_shaded.sum())

        if n_shaded == 0:
            # No shading — full power
            return {
                "Pmpp": round(P_stc, 2), "Vmpp": round(P_stc / self.top.p.Isc, 2),
                "Impp": round(self.top.p.Isc, 3), "bpd_active": False,
                "breakdown_pero": False, "breakdown_si": False,
                "V_reverse_shaded": 0.0, "n_shaded": 0,
            }

        # Reverse bias the shaded cell would experience (before BPD activates)
        # V_rev ≈ (n_cells_under_bpd - 1) × Voc_tandem
        Voc_tandem = self.top.p.Voc + self.bottom.p.Voc   # ≈ 1.95 V

        # Check bypass diode activation.
        # The BPD is driven by the non-BPD cells in the string.
        # BPD activates if: n_bpd ≥ 1 AND n_bpd ≤ max_protected (silicon Vbd limit)
        # Max cells one BPD can protect = 9 for this module (from Devoto et al.)
        BPD_MAX_CELLS = 9   # empirical from paper (silicon Vbd determines limit)
        bpd_active = (n_cells_under_bpd >= 1) and (n_cells_under_bpd <= BPD_MAX_CELLS)

        # Compute power
        if bpd_active:
            # BPD bypasses all n_cells_under_bpd cells (including shaded one)
            # Power from remaining active cells
            n_active = self.n_cells - n_cells_under_bpd
            Pmpp     = P_cell * max(n_active, 0)
        else:
            # Shaded cell limits string: power ≈ 48% of STC (from paper)
            Pmpp = P_stc * 0.48

        V_reverse_shaded = (n_cells_under_bpd - 1) * Voc_tandem if n_cells_under_bpd > 0 else (
            (self.n_cells - 1) * Voc_tandem
        )
        is_breakdown_pero = V_reverse_shaded > abs(self.Vbd_top)
        is_breakdown_si   = V_reverse_shaded > abs(self.Vbd_bottom)

        return {
            "Pmpp":             round(max(Pmpp, 0), 2),
            "Vmpp":             round(Pmpp / self.top.p.Isc if self.top.p.Isc > 0 else 0, 2),
            "Impp":             round(self.top.p.Isc, 3),
            "bpd_active":       bpd_active,
            "breakdown_pero":   bool(is_breakdown_pero),
            "breakdown_si":     bool(is_breakdown_si),
            "V_reverse_shaded": round(float(V_reverse_shaded), 2),
            "n_shaded":         int(n_shaded),
        }

    def stc_power(self) -> float:
        """Module power at STC (no shading)."""
        G_stc = np.full(self.n_cells, 1000.0)
        return self._string_power(G_stc)["Pmpp"]

    def shade_single_cell(
        self,
        n_cells_under_diode: int = 0,
        G_shaded: float = 0.0,
    ) -> dict[str, float]:
        """
        Simulate one fully shaded cell with a bypass diode on N cells.

        Reproduces the experimental setup of Figure 8 in Devoto et al. (2024):
        STC → shade one cell → add bypass diode protecting 1, 3, 9, 10–14 cells.

        Parameters
        ----------
        n_cells_under_diode : int
            Number of cells the bypass diode protects (0 = no diode).
        G_shaded : float
            Irradiance on shaded cell (W/m²). Default 0 (fully shaded).

        Returns
        -------
        dict with STC power, shaded power (no BPD), and BPD power.
        """
        G_stc = np.full(self.n_cells, 1000.0)
        G_shade = G_stc.copy()
        G_shade[self.n_cells // 2] = G_shaded  # shade middle cell

        P_stc    = self._string_power(G_stc, n_cells_under_bpd=0)["Pmpp"]
        P_shaded = self._string_power(G_shade, n_cells_under_bpd=0)["Pmpp"]
        P_bpd    = self._string_power(G_shade, n_cells_under_bpd=n_cells_under_diode)["Pmpp"]

        return {
            "P_stc":              P_stc,
            "P_shaded_no_bpd":    P_shaded,
            "P_bpd":              P_bpd,
            "power_recovery_W":   round(P_bpd - P_shaded, 2),
            "power_recovery_pct": round((P_bpd - P_shaded) / P_stc * 100, 1),
            "n_cells_under_bpd":  n_cells_under_diode,
        }

    def bypass_diode_sweep(
        self,
        diode_counts: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Sweep bypass diode coverage (1 to N cells) and record power output.

        Reproduces the experimental sweep of Figure 8 in Devoto et al. (2024).

        Parameters
        ----------
        diode_counts : array or None
            Number of cells protected by single bypass diode.
            Default: [0, 1, 3, 6, 9, 10, 11, 12, 13, 14]

        Returns
        -------
        dict with 'n_cells_bpd', 'P_mpp', 'P_stc'
        """
        if diode_counts is None:
            diode_counts = np.array([0, 1, 3, 6, 9, 10, 11, 12, 13, 14])

        P_stc = self.stc_power()
        P_mpp_arr = []
        for n_bpd in diode_counts:
            res = self.shade_single_cell(n_cells_under_diode=int(n_bpd))
            P_mpp_arr.append(res["P_bpd"] if n_bpd > 0 else res["P_shaded_no_bpd"])

        return {
            "n_cells_bpd": diode_counts,
            "P_mpp":       np.array(P_mpp_arr),
            "P_stc":       P_stc,
            "P_rel":       np.array(P_mpp_arr) / P_stc,
        }

    def breakdown_voltage_analysis(
        self,
        Vbd_range: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Analyse how silicon bottom cell Vbd affects perovskite protection.

        Higher silicon Vbd → better protection of perovskite top cell
        before bypass diode activates.

        Parameters
        ----------
        Vbd_range : np.ndarray or None
            Silicon breakdown voltages to sweep (V, negative values).
            Default: np.linspace(-1, -40, 40)

        Returns
        -------
        dict with 'Vbd_Si', 'protection_margin', 'is_protected'
        """
        if Vbd_range is None:
            Vbd_range = np.linspace(-1, -40, 40)

        n_bpd = 9  # typical bypass diode covers 9 cells
        protection = []

        for Vbd_si in Vbd_range:
            # Reverse bias on shaded cell before bypass diode activates
            V_reverse = (self.top.p.Voc + self.bottom.p.Voc) * (n_bpd - 1)
            # Silicon is protected if its Vbd > Vrev and activates diode first
            si_limited = abs(Vbd_si) < V_reverse
            pero_protected = si_limited or (abs(Vbd_si) > abs(self.Vbd_top) * 5)
            protection.append(pero_protected)

        return {
            "Vbd_Si":           Vbd_range,
            "is_pero_protected": np.array(protection),
            "V_reverse_before_bpd": (self.top.p.Voc + self.bottom.p.Voc) * (n_bpd - 1),
        }
