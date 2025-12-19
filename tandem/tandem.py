"""
tandem/tandem.py
----------------
Tandem module I-V modelling for 2T, 3T, and 4T configurations.

Terminal configurations (from Devoto et al. 2024, EU PVSEC):

  2T (monolithic, series):
      Top (perovskite) and bottom (silicon) sub-cells connected in series.
      Single output terminal pair. Current-matched condition required.
      I_module = I_top = I_bottom  (current-matched)
      V_module = V_top + V_bottom

  3T (three-terminal):
      Three electrical terminals. Allows independent current extraction
      from top and bottom sub-cells via voltage-matching condition.
      Two variants: 3T-r (recombination) and 3T-s (series).
      End losses are intrinsic to voltage-matched (vm) strings.

  4T (mechanically stacked, independent):
      Top and bottom cells operate independently with separate terminals.
      No current or voltage matching required.
      P_total = P_top + P_bottom

The test module used in Devoto et al. used IBC silicon cells (Voc ≈ 0.65 V)
with two cells in series to simulate a "perovskite" sub-cell (Voc ≈ 1.30 V).
"""

import numpy as np
from .cell_model import SolarCell, CellParameters, SILICON_PARAMS, PEROVSKITE_PARAMS


class TandemModule:
    """
    Tandem solar module model for 2T, 3T, and 4T configurations.

    Models a string of n_cells tandem cells (each = one perovskite + one silicon
    sub-cell) connected in the chosen terminal configuration.

    Parameters
    ----------
    n_cells : int
        Number of tandem cells in the string. Default 11 (half of 22-cell module).
    top_params : CellParameters
        Perovskite sub-cell parameters.
    bottom_params : CellParameters
        Silicon sub-cell parameters.

    Examples
    --------
    >>> mod = TandemModule(n_cells=11)
    >>> V2T, I2T, P2T = mod.iv_2T()
    >>> V4T, I4T, P4T = mod.iv_4T()
    """

    def __init__(
        self,
        n_cells: int = 11,
        top_params: CellParameters = PEROVSKITE_PARAMS,
        bottom_params: CellParameters = SILICON_PARAMS,
    ):
        self.n_cells = n_cells
        self.top    = SolarCell(top_params)
        self.bottom = SolarCell(bottom_params)

    # ── 2T: Monolithic series connection ─────────────────────────────────────

    def iv_2T(
        self,
        G: float = 1000.0,
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2T tandem string I-V curve (current-matched, series-connected).

        The module current is limited by the sub-cell with lower Isc.
        Voltages add; the string operates at the intersection of the two
        individual I-V curves plotted as V(I).

        Parameters
        ----------
        G : float
            Irradiance (W/m²).
        n_points : int
            Resolution of the I-V curve.

        Returns
        -------
        V, I, P : np.ndarray
            Module voltage (V), current (A), power (W).
        """
        Vt_arr, It_arr, _ = self.top.iv_curve(n_points=n_points, G=G)
        Vb_arr, Ib_arr, _ = self.bottom.iv_curve(n_points=n_points, G=G)

        # Limit current to the minimum of both sub-cells' Isc
        I_common = np.linspace(0, min(self.top.p.Isc, self.bottom.p.Isc) * (G / 1000), n_points)

        V_top    = np.interp(I_common, It_arr[::-1], Vt_arr[::-1], left=0, right=0)
        V_bottom = np.interp(I_common, Ib_arr[::-1], Vb_arr[::-1], left=0, right=0)

        V_string = (V_top + V_bottom) * self.n_cells
        I_string = I_common  # same current through all series cells
        P_string = V_string * I_string

        return V_string, I_string, P_string

    def mpp_2T(self, G: float = 1000.0) -> dict[str, float]:
        """Maximum power point for 2T configuration."""
        V, I, P = self.iv_2T(G=G)
        idx = P.argmax()
        return {
            "Vmpp": float(V[idx]),
            "Impp": float(I[idx]),
            "Pmpp": float(P[idx]),
            "Voc":  float(V[I > 0.001][0]) if (I > 0.001).any() else 0,
            "Isc":  float(I[0]),
            "FF":   float(P[idx] / (V[I < 0.001][-1] * I[0])) if I[0] > 0 else 0,
        }

    # ── 4T: Mechanically stacked, independent sub-cells ──────────────────────

    def iv_4T(
        self,
        G: float = 1000.0,
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        4T tandem: independent top and bottom I-V curves.

        Returns separate curves for top (perovskite) and bottom (silicon).
        Total power = P_top + P_bottom.

        Returns
        -------
        Vt, It, Pt, Vb, Ib, Pb : np.ndarray
            Top and bottom cell voltage, current, power arrays (per cell).
        """
        Vt, It, Pt = self.top.iv_curve(n_points=n_points, G=G)
        Vb, Ib, Pb = self.bottom.iv_curve(n_points=n_points, G=G)

        # Scale to string (n_cells in series)
        return (Vt * self.n_cells, It, Pt * self.n_cells,
                Vb * self.n_cells, Ib, Pb * self.n_cells)

    def mpp_4T(self, G: float = 1000.0) -> dict[str, float]:
        """Maximum power point for 4T configuration (sum of both sub-cells)."""
        Vt, It, Pt, Vb, Ib, Pb = self.iv_4T(G=G)
        Pmpp_top    = float(Pt.max())
        Pmpp_bottom = float(Pb.max())
        return {
            "Pmpp_top":    Pmpp_top,
            "Pmpp_bottom": Pmpp_bottom,
            "Pmpp_total":  Pmpp_top + Pmpp_bottom,
            "Vmpp_top":    float(Vt[Pt.argmax()]),
            "Vmpp_bottom": float(Vb[Pb.argmax()]),
        }

    # ── 3T: Three-terminal ────────────────────────────────────────────────────

    def iv_3T(
        self,
        G: float = 1000.0,
        config: str = "r",
        n_points: int = 200,
    ) -> dict[str, np.ndarray]:
        """
        3T tandem (recombination or series type) I-V characteristics.

        In 3T, voltage-matching is applied: the top string and bottom string
        are operated at a common load voltage with a 2:1 voltage ratio for
        current-matched sub-cells. End losses are intrinsic.

        Parameters
        ----------
        G : float
            Irradiance (W/m²).
        config : str
            '3T-r' (recombination) or '3T-s' (series). Default 'r'.
        n_points : int
            Resolution.

        Returns
        -------
        dict with: 'V_load', 'I_total', 'P_total', 'end_loss_fraction'
        """
        Vt, It, Pt = self.top.iv_curve(n_points=n_points, G=G)
        Vb, Ib, Pb = self.bottom.iv_curve(n_points=n_points, G=G)

        # Voltage matching: V_top_string = 2 × V_bottom_string (for 2:1 ratio)
        # For 3T-r: end losses from cells at string boundaries
        # Approximate end loss: 3 disconnected cells in 3T-s, 1 in 3T-r
        n_end_losses = 3 if config.lower() in ("3t-s", "s") else 1
        active_fraction = (self.n_cells - n_end_losses) / self.n_cells

        Vt_scaled, It_scaled, Pt_scaled = (
            Vt * self.n_cells * active_fraction, It, Pt * self.n_cells * active_fraction
        )
        Vb_scaled, Ib_scaled, Pb_scaled = (
            Vb * self.n_cells * active_fraction, Ib, Pb * self.n_cells * active_fraction
        )

        # Total power: top + bottom operating at vm condition
        I_common = np.linspace(0, min(It.max(), Ib.max()), n_points)
        Vt_vm = np.interp(I_common, It[::-1], Vt_scaled[::-1], left=0, right=0)
        Vb_vm = np.interp(I_common, Ib[::-1], Vb_scaled[::-1], left=0, right=0)

        P_total = (Vt_vm + Vb_vm) * I_common

        return {
            "V_load":             Vt_vm + Vb_vm,
            "I_total":            I_common,
            "P_total":            P_total,
            "end_loss_fraction":  n_end_losses / self.n_cells,
            "Pmpp":               float(P_total.max()),
        }

    # ── Comparative analysis ──────────────────────────────────────────────────

    def compare_configurations(self, G: float = 1000.0) -> dict[str, dict]:
        """
        Compare Pmpp across 2T, 3T-r, 3T-s, and 4T configurations.

        Returns
        -------
        dict with keys '2T', '3T-r', '3T-s', '4T', each containing
        a dict with Pmpp and relative efficiency.
        """
        mpp_2T   = self.mpp_2T(G=G)
        mpp_4T   = self.mpp_4T(G=G)
        mpp_3Tr  = self.iv_3T(G=G, config="r")
        mpp_3Ts  = self.iv_3T(G=G, config="s")

        P_ref = mpp_4T["Pmpp_total"]  # 4T is the reference (no matching loss)

        return {
            "2T":  {"Pmpp": mpp_2T["Pmpp"],          "rel_eff": mpp_2T["Pmpp"] / P_ref},
            "3T-r":{"Pmpp": mpp_3Tr["Pmpp"],         "rel_eff": mpp_3Tr["Pmpp"] / P_ref},
            "3T-s":{"Pmpp": mpp_3Ts["Pmpp"],         "rel_eff": mpp_3Ts["Pmpp"] / P_ref},
            "4T":  {"Pmpp": mpp_4T["Pmpp_total"],    "rel_eff": 1.0},
        }
