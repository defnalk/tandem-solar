"""
Benchmark for the single-diode I-V curve hot path.

The dominant cost in tandem simulations is SolarCell.iv_curve, which calls
scipy.optimize.brentq once per voltage point. Every brentq call re-evaluates
CellParameters.Iph/I0 via @property (each doing an np.exp on a scalar), then
calls np.exp on a scalar inside the root-finder closure. This benchmark
exercises both sub-cells across 200 voltage points, repeated.

# Workload: 200 iv_curve calls across both sub-cells (100 irradiance levels each),
# 200 voltage points per curve = 40,000 brentq solves per run.
#
# BASELINE (pre-optimization):
#   mean wall time over 5 runs: 0.5112 s
#   cProfile (cumulative):
#     SolarCell.iv_curve          0.722 s / 200 calls
#     SolarCell.current           0.700 s / 40,000 calls
#     brentq                      0.572 s / 40,000 calls
#     closure `f`                 0.289 s self / 234,324 calls
#     CellParameters.Iph (@prop)  0.074 s / 40,000 calls  ← np.exp per call
#     CellParameters.I0  (@prop)  0.066 s / 80,000 calls  ← np.exp per call
#     CellParameters.Vt  (@prop)  0.026 s / 354,324 calls
#
# AFTER optimization (resolve Iph/I0/Vt/Rs/Rsh once per curve + math.exp in closure):
#   mean wall time over 5 runs: 0.3962 s
#   speedup: 1.29x
#   cProfile (cumulative):
#     SolarCell.iv_curve          0.473 s / 200 calls
#     brentq                      0.437 s / 40,000 calls
#     closure `f`                 0.106 s self / 234,324 calls  (was 0.289 s, ~2.7x)
#     math.exp                    0.019 s / 234,324 calls       (scalar C call)
#     Iph/I0 properties           now 0.001 s total (hoisted out of loop)
#
# Remaining time is dominated by scipy's brentq scaffolding (f_raise wrapper),
# which is inherent to the Python-level root finder; eliminating it would require
# switching to an analytical Lambert-W solution and is out of scope for this pass.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

from tandem.cell_model import PEROVSKITE_PARAMS, SILICON_PARAMS, SolarCell

N_RUNS = 5
N_POINTS = 200


def workload() -> None:
    top = SolarCell(PEROVSKITE_PARAMS)
    bot = SolarCell(SILICON_PARAMS)
    # 100 curves per sub-cell at varying irradiance — mimics a shading sweep.
    for i in range(100):
        G = 200.0 + 8.0 * i
        top.iv_curve(n_points=N_POINTS, G=G)
        bot.iv_curve(n_points=N_POINTS, G=G)


def main() -> None:
    # Warm up (JIT-like caches in scipy / numpy).
    workload()

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        workload()
        times.append(time.perf_counter() - t0)

    mean = sum(times) / len(times)
    print(f"mean wall time over {N_RUNS} runs: {mean:.4f} s")
    print(f"individual runs: {['%.4f' % t for t in times]}")

    # cProfile one representative run for hotspot analysis.
    prof = cProfile.Profile()
    prof.enable()
    workload()
    prof.disable()
    s = StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(15)
    print("\ncProfile (top 15 by cumulative time):")
    print(s.getvalue())


if __name__ == "__main__":
    main()
