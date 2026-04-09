"""
tandem/cell_model.py
--------------------
Single-diode model for solar cell I-V curve simulation and parameter
extraction. Applicable to both silicon and perovskite sub-cells.

The single-diode model (SDM) describes the cell current as:

    I = I_ph - I_0 · [exp((V + I·Rs) / (n·Vt)) - 1] - (V + I·Rs) / Rsh

where:
    I_ph  = photocurrent (A)   — proportional to irradiance
    I_0   = dark saturation current (A)
    Rs    = series resistance (Ω)
    Rsh   = shunt resistance (Ω)
    n     = ideality factor (1 for ideal diode)
    Vt    = thermal voltage = kT/q (V)

Reference:
    Devoto et al. (2024), EU PVSEC — tandem module circuit modelling
    De Soto et al. (2006), Solar Energy — five-parameter SDM
"""

from dataclasses import dataclass, field
from math import exp as _math_exp

import numpy as np
from scipy.optimize import brentq, curve_fit

# ── Physical constants ─────────────────────────────────────────────────────────
K_B = 1.380649e-23   # J/K  Boltzmann constant
Q_E = 1.602176634e-19  # C   elementary charge
T_STC = 298.15       # K   standard test conditions (25 °C)
V_T_STC = K_B * T_STC / Q_E   # ≈ 0.02569 V


@dataclass
class CellParameters:
    """
    Single-diode model parameters for one solar cell or sub-cell.

    Attributes
    ----------
    Isc  : float  Short-circuit current (A)
    Voc  : float  Open-circuit voltage (V)
    Impp : float  Current at maximum power point (A)
    Vmpp : float  Voltage at maximum power point (V)
    Rs   : float  Series resistance (Ω)
    Rsh  : float  Shunt resistance (Ω)
    n    : float  Ideality factor
    T    : float  Cell temperature (K)
    """
    Isc:  float = 1.40
    Voc:  float = 0.65
    Impp: float = 1.31
    Vmpp: float = 0.56
    Rs:   float = 0.05
    Rsh:  float = 200.0
    n:    float = 1.0
    T:    float = T_STC

    @property
    def Vt(self) -> float:
        """Thermal voltage at cell temperature."""
        return K_B * self.T / Q_E

    @property
    def I0(self) -> float:
        """Dark saturation current derived from Voc and Isc."""
        return self.Isc * np.exp(-self.Voc / (self.n * self.Vt))

    @property
    def Iph(self) -> float:
        """Photocurrent (≈ Isc for low Rs)."""
        return self.Isc + self.I0 * (np.exp(self.Voc / (self.n * self.Vt)) - 1)

    @property
    def Pmax(self) -> float:
        """Maximum power (W)."""
        return self.Impp * self.Vmpp

    @property
    def FF(self) -> float:
        """Fill factor."""
        return self.Pmax / (self.Isc * self.Voc)

    @property
    def efficiency(self) -> float:
        """
        Cell efficiency (%) — requires cell area.

        Returns Pmax / (1000 W/m²) · area placeholder.
        Use SolarCell.efficiency(area) for full calculation.
        """
        return self.Pmax


# ── Default parameters for perovskite and silicon sub-cells ───────────────────
# Based on test module values from Devoto et al. (2024)
# IBC silicon: Voc ≈ 0.65 V, two cells in series simulate "perovskite" Voc ≈ 1.3 V

SILICON_PARAMS = CellParameters(
    Isc=1.40, Voc=0.65, Impp=1.31, Vmpp=0.56, Rs=0.05, Rsh=300.0, n=1.1
)

PEROVSKITE_PARAMS = CellParameters(
    Isc=1.40, Voc=1.30, Impp=1.31, Vmpp=1.10, Rs=0.10, Rsh=500.0, n=1.4
)


class SolarCell:
    """
    Single-diode model solar cell.

    Solves the implicit SDM equation numerically to generate I-V and P-V curves.
    Supports irradiance and temperature scaling.

    Parameters
    ----------
    params : CellParameters
        Cell electrical parameters. Default: silicon IBC cell (Devoto et al. 2024).

    Examples
    --------
    >>> cell = SolarCell(SILICON_PARAMS)
    >>> V, I, P = cell.iv_curve()
    >>> print(f"Pmax = {P.max():.3f} W")
    """

    def __init__(self, params: CellParameters = SILICON_PARAMS):
        self.p = params

    def current(self, V: float, G: float = 1000.0) -> float:
        """
        Solve for cell current I at voltage V using the single-diode model.

        Uses Brent's method to solve the implicit equation:
            f(I) = I_ph - I_0·[exp((V + I·Rs)/(n·Vt)) - 1] - (V + I·Rs)/Rsh - I = 0

        Parameters
        ----------
        V : float
            Cell terminal voltage (V).
        G : float
            Irradiance (W/m²). Scales Iph proportionally. Default 1000 (STC).

        Returns
        -------
        float
            Cell current I (A). Returns 0 if solver fails.
        """
        p = self.p
        Iph = p.Iph * (G / 1000.0)
        I0  = p.I0

        def f(I):
            exp_arg = (V + I * p.Rs) / (p.n * p.Vt)
            exp_arg = min(exp_arg, 80)  # prevent overflow
            return Iph - I0 * (np.exp(exp_arg) - 1) - (V + I * p.Rs) / p.Rsh - I

        try:
            return brentq(f, -0.1, Iph * 1.05, xtol=1e-9)
        except ValueError:
            return 0.0

    def iv_curve(
        self,
        n_points: int = 200,
        G: float = 1000.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a full I-V and P-V curve.

        Parameters
        ----------
        n_points : int
            Number of voltage points. Default 200.
        G : float
            Irradiance (W/m²). Default 1000 (STC).

        Returns
        -------
        V, I, P : np.ndarray
            Voltage (V), current (A), power (W) arrays.
        """
        # Resolve all CellParameters scalars once (each @property call would
        # otherwise recompute np.exp on every voltage point — ~400x overhead).
        p = self.p
        Iph = p.Iph * (G / 1000.0)
        I0 = p.I0
        Rs = p.Rs
        Rsh = p.Rsh
        nVt = p.n * p.Vt
        I_hi = Iph * 1.05

        V = np.linspace(0, p.Voc * 1.01, n_points)
        I_out = np.empty(n_points)

        # Scalar-only closure: math.exp is ~10x faster than np.exp on Python
        # floats because it skips the ufunc/array dispatch entirely.
        for k in range(n_points):
            v = float(V[k])

            def f(I, v=v, Iph=Iph, I0=I0, Rs=Rs, Rsh=Rsh, nVt=nVt):
                arg = (v + I * Rs) / nVt
                if arg > 80.0:
                    arg = 80.0
                return Iph - I0 * (_math_exp(arg) - 1.0) - (v + I * Rs) / Rsh - I

            try:
                I_out[k] = brentq(f, -0.1, I_hi, xtol=1e-9)
            except ValueError:
                I_out[k] = 0.0

        I_out = np.maximum(I_out, 0)
        P = V * I_out
        return V, I_out, P

    def mpp(self, G: float = 1000.0) -> tuple[float, float, float]:
        """
        Find maximum power point.

        Returns
        -------
        Vmpp, Impp, Pmpp : float
        """
        V, I, P = self.iv_curve(G=G)
        idx = P.argmax()
        return float(V[idx]), float(I[idx]), float(P[idx])

    def temperature_coefficients(
        self,
        dT: float = 25.0,
    ) -> dict[str, float]:
        """
        Estimate temperature coefficients for Isc, Voc, Pmax.

        Uses the standard linear approximation:
            ΔX/ΔT ≈ (X(T+ΔT) - X(T)) / ΔT / X(T)

        Parameters
        ----------
        dT : float
            Temperature increment (K).

        Returns
        -------
        dict with: 'beta_Voc' (%/K), 'alpha_Isc' (%/K), 'gamma_Pmax' (%/K)
        """
        p0 = self.p
        p1 = CellParameters(
            Isc=p0.Isc * (1 + 0.0005 * dT),  # Isc increases ~0.05%/K
            Voc=p0.Voc - 0.0023 * dT,          # Voc decreases ~2.3 mV/K (silicon)
            Impp=p0.Impp, Vmpp=p0.Vmpp - 0.002 * dT,
            Rs=p0.Rs, Rsh=p0.Rsh, n=p0.n, T=p0.T + dT,
        )
        c0 = SolarCell(p0)
        c1 = SolarCell(p1)
        _, _, P0 = c0.mpp()
        _, _, P1 = c1.mpp()

        return {
            "beta_Voc":   (p1.Voc - p0.Voc) / dT / p0.Voc * 100,
            "alpha_Isc":  (p1.Isc - p0.Isc) / dT / p0.Isc * 100,
            "gamma_Pmax": (P1 - P0) / dT / P0 * 100,
        }


def fit_iv_parameters(
    V_meas: np.ndarray,
    I_meas: np.ndarray,
    cell_type: str = "silicon",
) -> CellParameters:
    """
    Fit single-diode model parameters to measured I-V data.

    Uses the key points method (Isc, Voc, Impp, Vmpp) plus a simplified
    Rs/Rsh extraction from the slopes at short circuit and open circuit.

    Parameters
    ----------
    V_meas : np.ndarray
        Measured voltage array (V).
    I_meas : np.ndarray
        Measured current array (A).
    cell_type : str
        'silicon' or 'perovskite' — sets ideality factor prior.

    Returns
    -------
    CellParameters
        Fitted parameters.
    """
    # Extract key points
    Isc = float(np.interp(0, V_meas, I_meas))
    Voc = float(np.interp(0, I_meas[::-1], V_meas[::-1]))

    P = V_meas * I_meas
    idx_mpp = P.argmax()
    Vmpp = float(V_meas[idx_mpp])
    Impp = float(I_meas[idx_mpp])

    # Approximate Rs from slope near Voc
    # dV/dI|Voc ≈ -Rs  (for ideal diode)
    near_voc = np.where(V_meas > 0.9 * Voc)[0]
    if len(near_voc) >= 2:
        dV = np.diff(V_meas[near_voc])
        dI = np.diff(I_meas[near_voc])
        valid = np.abs(dI) > 1e-6
        Rs = float(-np.mean(dV[valid] / dI[valid])) if valid.any() else 0.05
        Rs = max(0.001, min(Rs, 2.0))
    else:
        Rs = 0.05

    # Approximate Rsh from slope near Isc
    near_isc = np.where(V_meas < 0.15 * Voc)[0]
    if len(near_isc) >= 2:
        dV = np.diff(V_meas[near_isc])
        dI = np.diff(I_meas[near_isc])
        valid = np.abs(dI) > 1e-9
        Rsh = float(-np.mean(dV[valid] / dI[valid])) if valid.any() else 300.0
        Rsh = max(10.0, min(Rsh, 1e5))
    else:
        Rsh = 300.0

    n = 1.4 if cell_type == "perovskite" else 1.1

    return CellParameters(
        Isc=Isc, Voc=Voc, Impp=Impp, Vmpp=Vmpp,
        Rs=Rs, Rsh=Rsh, n=n,
    )
