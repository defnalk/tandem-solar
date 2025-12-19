"""
examples/full_simulation.py
---------------------------
Full tandem module simulation:
  A. I-V curves: single cell + 2T, 4T string comparison
  B. Configuration comparison (2T / 3T-r / 3T-s / 4T) — Pmpp bar chart
  C. CTM loss waterfall — quantifying loss mechanisms
  D. Bypass diode sweep — reproducing Devoto et al. Fig. 8

Run from project root:
    python examples/full_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tandem import SolarCell, TandemModule, CTMLossAnalyser, ShadingAnalysis
from tandem.cell_model import SILICON_PARAMS, PEROVSKITE_PARAMS

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

BLUE   = "#1d3557"
RED    = "#e63946"
GREEN  = "#2d6a4f"
ORANGE = "#f4a261"
PURPLE = "#7b2d8b"
GREY   = "#6b7280"


print("Building tandem module models...")

mod = TandemModule(n_cells=11)
ctm = CTMLossAnalyser(n_cells_series=11)
sa  = ShadingAnalysis(n_cells=22)


# ════════════════════════════════════════════════════════════════════════════
# Panel A: I-V curves — sub-cells and 2T/4T tandem string
# ════════════════════════════════════════════════════════════════════════════
print("Generating I-V curves...")

si   = SolarCell(SILICON_PARAMS)
pero = SolarCell(PEROVSKITE_PARAMS)

Vsi, Isi, Psi   = si.iv_curve()
Vp,  Ip,  Pp    = pero.iv_curve()
V2T, I2T, P2T   = mod.iv_2T()
_, _, _, Vb, Ib, Pb = mod.iv_4T()

mpp_2T = mod.mpp_2T()
mpp_4T = mod.mpp_4T()

print(f"  2T Pmpp = {mpp_2T['Pmpp']:.2f} W  (11-cell string)")
print(f"  4T Pmpp = {mpp_4T['Pmpp_total']:.2f} W  (top + bottom)")


# ════════════════════════════════════════════════════════════════════════════
# Panel B: Configuration comparison
# ════════════════════════════════════════════════════════════════════════════
print("Comparing configurations...")

comp = mod.compare_configurations()
configs = ["2T", "3T-r", "3T-s", "4T"]
Pmpp_vals = [comp[c]["Pmpp"] for c in configs]
rel_effs  = [comp[c]["rel_eff"] * 100 for c in configs]

configs_str = ', '.join(c + '=' + str(round(comp[c]['Pmpp'],1)) + 'W' for c in configs)
print(f'  Config comparison: {configs_str}')


# ════════════════════════════════════════════════════════════════════════════
# Panel C: CTM loss waterfall
# ════════════════════════════════════════════════════════════════════════════
print("Computing CTM losses...")

summary = ctm.loss_summary()
print(f"  CTM ratio: {summary['ctm_ratio']:.3f}  "
      f"(total loss: {summary['total_loss_pct']:.1f}%)")
print(f"  Cell sum: {summary['P_cell_sum_W']:.2f} W  →  Module: {summary['P_module_W']:.2f} W")

loss_names = list(summary["losses_pct"].keys())
loss_vals  = list(summary["losses_pct"].values())

# Mismatch sensitivity
mismatch = ctm.mismatch_sensitivity()


# ════════════════════════════════════════════════════════════════════════════
# Panel D: Bypass diode sweep (Devoto et al. Fig. 8 reproduction)
# ════════════════════════════════════════════════════════════════════════════
print("Running bypass diode sweep...")

bpd_sweep = sa.bypass_diode_sweep()
P_stc     = sa.stc_power()
P_shaded_no_bpd = sa.shade_single_cell(n_cells_under_diode=0)["P_shaded_no_bpd"]

print(f"  STC power:          {P_stc:.1f} W")
print(f"  1 cell shaded (no BPD): {P_shaded_no_bpd:.1f} W  "
      f"({P_shaded_no_bpd/P_stc*100:.0f}% of STC)")
print(f"  1 BPD on 9 cells:   {bpd_sweep['P_mpp'][4]:.1f} W")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

# ── Panel A: I-V curves ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1b = ax1.twinx()

ax1.plot(Vsi / SILICON_PARAMS.Voc, Isi, color=BLUE,   lw=2,   label="Si bottom (STC)")
ax1.plot(Vp  / PEROVSKITE_PARAMS.Voc, Ip, color=ORANGE, lw=2, label="Perovskite top (STC)")

ax1.axhline(SILICON_PARAMS.Isc, ls=":", color=GREY, lw=1, alpha=0.5)
ax1.set_xlabel("Normalised voltage  V / Voc")
ax1.set_ylabel("Current (A)", color=BLUE)
ax1.tick_params(axis="y", colors=BLUE)

# P-V on right axis (single cell)
ax1b.plot(Vsi / SILICON_PARAMS.Voc, Psi, color=BLUE,   lw=1.5, ls="--", alpha=0.6)
ax1b.plot(Vp  / PEROVSKITE_PARAMS.Voc, Pp, color=ORANGE, lw=1.5, ls="--", alpha=0.6)
ax1b.set_ylabel("Power (W)", color=GREY)
ax1b.tick_params(axis="y", colors=GREY)

ax1.set_title("A  |  Sub-cell I-V and P-V Curves (STC)", fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_xlim(0, 1.05)
ax1.set_ylim(bottom=0)

# ── Panel B: Configuration Pmpp comparison ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
colours_bar = [BLUE, GREEN, PURPLE, RED]
bars = ax2.bar(configs, Pmpp_vals, color=colours_bar, width=0.5, alpha=0.85)
ax2.bar_label(bars, labels=[f"{p:.1f} W\n({r:.0f}%)" for p, r in zip(Pmpp_vals, rel_effs)],
              padding=4, fontsize=8)
ax2.axhline(Pmpp_vals[-1], ls="--", color=RED, lw=1.2, alpha=0.7, label="4T reference")
ax2.set_ylabel("Maximum power Pmpp (W)")
ax2.set_title("B  |  Configuration Comparison (11-cell string)", fontweight="bold")
ax2.set_ylim(0, max(Pmpp_vals) * 1.25)
ax2.legend(fontsize=8)

# ── Panel C: CTM loss breakdown ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
colours_loss = [ORANGE if v > 0 else GREEN for v in loss_vals]
bars3 = ax3.barh(loss_names, loss_vals, color=colours_loss, alpha=0.85)
ax3.axvline(0, color="black", lw=0.8)
ax3.bar_label(bars3, labels=[f"{v:+.2f}%" for v in loss_vals], padding=3, fontsize=7.5)
ax3.set_xlabel("Power loss / gain (%)")
ax3.set_title(f"C  |  CTM Loss Waterfall  (CTM ratio = {summary['ctm_ratio']:.3f})",
              fontweight="bold")
ax3.set_xlim(-0.5, max(loss_vals) * 1.5)

# ── Panel D: Bypass diode sweep ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
diode_labels = [str(int(n)) if n > 0 else "No BPD" for n in bpd_sweep["n_cells_bpd"]]
P_plot = np.concatenate([[P_shaded_no_bpd], bpd_sweep["P_mpp"][1:]])
colours_bpd = [RED if p < P_stc * 0.6 else ORANGE if p < P_stc * 0.85 else GREEN
               for p in P_plot]
ax4.bar(range(len(diode_labels)), P_plot, color=colours_bpd, alpha=0.85)
ax4.axhline(P_stc, ls="--", color=BLUE, lw=1.5, label=f"STC ({P_stc:.0f} W)")
ax4.set_xticks(range(len(diode_labels)))
ax4.set_xticklabels(diode_labels, rotation=30, ha="right", fontsize=8)
ax4.set_ylabel("Module Pmpp (W)")
ax4.set_title("D  |  1-Cell Shading + Bypass Diode Sweep\n(after Devoto et al. 2024, Fig. 8)",
              fontweight="bold")
ax4.legend(fontsize=8)

fig.suptitle(
    "Perovskite-Silicon Tandem Module Simulation\n"
    "SERIS NUS — Cell-to-Module Loss Evaluation (Summer 2025)",
    fontsize=12, fontweight="bold", y=1.01,
)

plt.savefig("examples/tandem_simulation_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Figure saved → examples/tandem_simulation_results.png")
