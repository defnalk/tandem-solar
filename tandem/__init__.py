"""
tandem-solar
------------
Python simulation toolkit for perovskite-silicon tandem solar cell modules.

Covers:
  - Single-diode model I-V curve generation and parameter extraction
  - Tandem cell configurations: 2T (monolithic), 3T, 4T (mechanically stacked)
  - Cell-to-module (CTM) loss analysis
  - Partial shading and bypass diode protection modelling

Based on work at the Solar Energy Research Institute of Singapore (SERIS),
National University of Singapore — internship project on cell-to-module loss
evaluation for next-generation tandem solar technology (Summer 2025).

Related publication:
  Devoto et al. (2024). "Modelling the effects of tandem module circuit
  configurations." 41st EU PVSEC. doi: 10.4229/EUPVSEC2024/2BV.1.41

Modules
-------
cell_model    : Single-diode model for solar cells (I-V, P-V, parameter fit)
tandem        : 2T, 3T, 4T tandem configurations and string models
ctm_loss      : Cell-to-module loss analysis (optical, resistive, mismatch)
shading       : Partial shading and bypass diode response
"""

import logging

from .cell_model import SolarCell, fit_iv_parameters
from .ctm_loss import CTMLossAnalyser
from .shading import ShadingAnalysis
from .tandem import TandemModule

# Library convention: attach a NullHandler so importing this package never
# emits unconfigured log records. Applications configure handlers themselves.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SolarCell",
    "fit_iv_parameters",
    "TandemModule",
    "CTMLossAnalyser",
    "ShadingAnalysis",
    "health_check",
]


def health_check() -> int:
    """
    Self-test that exercises core models with reference inputs.

    Returns
    -------
    int
        0 on success, 1 on failure. Suitable for CI smoke tests and the
        ``python -m tandem --health-check`` CLI entry point.
    """
    log = logging.getLogger(__name__)
    log.info("Running tandem-solar health check")
    try:
        cell = SolarCell()
        v_mpp, i_mpp, p_mpp = cell.mpp()
        assert p_mpp > 0, f"non-positive Pmax: {p_mpp}"
        log.info("Health check OK (Pmax=%.3f W)", p_mpp)
    except Exception as exc:
        log.error("Health check FAILED: %s", exc)
        return 1
    return 0
