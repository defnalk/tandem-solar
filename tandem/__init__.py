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

from .cell_model import SolarCell, fit_iv_parameters
from .tandem     import TandemModule
from .ctm_loss   import CTMLossAnalyser
from .shading    import ShadingAnalysis

__all__ = [
    "SolarCell",
    "fit_iv_parameters",
    "TandemModule",
    "CTMLossAnalyser",
    "ShadingAnalysis",
]
