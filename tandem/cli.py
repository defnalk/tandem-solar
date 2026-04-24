"""
tandem.cli — config-driven entry point for reproducible tandem-module runs.

Usage:
    python -m tandem.cli --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

from tandem import SolarCell, TandemModule, CTMLossAnalyser, ShadingAnalysis
from tandem.cell_model import SILICON_PARAMS, PEROVSKITE_PARAMS


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run(config_path: Path) -> dict:
    cfg = load_config(config_path)
    seed = cfg["output"].get("random_seed", 0)
    random.seed(seed)
    np.random.seed(seed)

    n_cells = cfg["module"]["n_cells"]
    n_ctm = cfg["module"]["n_cells_ctm"]
    n_shade = cfg["module"]["n_cells_shading"]

    mod = TandemModule(n_cells=n_cells)
    ctm = CTMLossAnalyser(n_cells_series=n_ctm)
    sa = ShadingAnalysis(n_cells=n_shade)

    metrics: dict = {"config": str(config_path), "module": cfg["module"]}

    if cfg["run"]["iv_curves"]:
        mpp_2T = mod.mpp_2T()
        mpp_4T = mod.mpp_4T()
        metrics["iv"] = {
            "Pmpp_2T_W": float(mpp_2T["Pmpp"]),
            "Pmpp_4T_W": float(mpp_4T["Pmpp_total"]),
        }

    if cfg["run"]["config_comparison"]:
        comp = mod.compare_configurations()
        metrics["config_comparison"] = {
            c: {"Pmpp_W": float(comp[c]["Pmpp"]), "rel_eff": float(comp[c]["rel_eff"])}
            for c in ["2T", "3T-r", "3T-s", "4T"]
        }

    if cfg["run"]["ctm_loss"]:
        summary = ctm.loss_summary()
        metrics["ctm"] = {
            "ctm_ratio": float(summary["ctm_ratio"]),
            "total_loss_pct": float(summary["total_loss_pct"]),
            "P_cell_sum_W": float(summary["P_cell_sum_W"]),
            "P_module_W": float(summary["P_module_W"]),
            "losses_pct": {k: float(v) for k, v in summary["losses_pct"].items()},
        }

    if cfg["run"]["bypass_diode_sweep"]:
        bpd = sa.bypass_diode_sweep()
        P_stc = sa.stc_power()
        P_no = sa.shade_single_cell(n_cells_under_diode=0)["P_shaded_no_bpd"]
        metrics["shading"] = {
            "P_stc_W": float(P_stc),
            "P_shaded_no_bpd_W": float(P_no),
            "P_mpp_per_bpd_W": [float(p) for p in bpd["P_mpp"]],
            "n_cells_bpd": [int(n) for n in bpd["n_cells_bpd"]],
        }

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / cfg["output"]["metrics_name"]
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    _render_figure(cfg, mod, ctm, sa, out_dir / cfg["output"]["figure_name"])

    print(f"✓ metrics → {metrics_path}")
    print(f"✓ figure  → {out_dir / cfg['output']['figure_name']}")
    return metrics


def _render_figure(cfg, mod, ctm, sa, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

    si = SolarCell(SILICON_PARAMS)
    pero = SolarCell(PEROVSKITE_PARAMS)
    Vsi, Isi, Psi = si.iv_curve()
    Vp, Ip, Pp = pero.iv_curve()

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(Vsi / SILICON_PARAMS.Voc, Isi, lw=2, label="Si bottom")
    ax1.plot(Vp / PEROVSKITE_PARAMS.Voc, Ip, lw=2, label="Perovskite top")
    ax1.set_xlabel("V / Voc")
    ax1.set_ylabel("Current (A)")
    ax1.set_title("A | Sub-cell IV")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    comp = mod.compare_configurations()
    configs = ["2T", "3T-r", "3T-s", "4T"]
    Pmpp_vals = [comp[c]["Pmpp"] for c in configs]
    ax2.bar(configs, Pmpp_vals)
    ax2.set_ylabel("Pmpp (W)")
    ax2.set_title("B | Configuration comparison")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    summary = ctm.loss_summary()
    names = list(summary["losses_pct"].keys())
    vals = list(summary["losses_pct"].values())
    ax3.barh(names, vals)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_xlabel("Loss (%)")
    ax3.set_title(f"C | CTM losses (ratio={summary['ctm_ratio']:.3f})")

    ax4 = fig.add_subplot(gs[1, 1])
    bpd = sa.bypass_diode_sweep()
    P_stc = sa.stc_power()
    labels = [str(int(n)) if n > 0 else "None" for n in bpd["n_cells_bpd"]]
    ax4.bar(labels, bpd["P_mpp"])
    ax4.axhline(P_stc, ls="--", label=f"STC ({P_stc:.0f} W)")
    ax4.set_ylabel("Pmpp (W)")
    ax4.set_title("D | Bypass diode sweep")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    fig.suptitle("Tandem-solar — reproducible run", fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tandem.cli",
        description="Config-driven tandem-module simulation entry point.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
