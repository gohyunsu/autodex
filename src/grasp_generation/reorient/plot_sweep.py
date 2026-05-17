"""
Render a heatmap of sweep_reset results: which (pickup_x, pickup_tz) cells are
resetable. Optional fail-reason annotation.

Usage:
    python src/grasp_generation/reorient/plot_sweep.py \
        --sweep_dir outputs/reset_cache/inspire_left/attached_container/reorient_0/0_16 \
        --out outputs/reset_cache/.../heatmap.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True)
    p.add_argument("--out", default=None, help="output PNG path (default: <sweep_dir>/heatmap.png)")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    with open(sweep_dir / "sweep_summary.json") as f:
        summary = json.load(f)

    xs = summary["x_values"]
    tzs = summary["tz_values"]
    nx, ntz = len(xs), len(tzs)

    grid = np.zeros((ntz, nx), dtype=np.int8)  # 0=fail, 1=ok
    for c in summary["cells"]:
        ix = xs.index(c["x"])
        itz = tzs.index(c["tz"])
        grid[itz, ix] = 1 if c["status"] == "ok" else 0

    n_ok = int(grid.sum())
    n_tot = grid.size
    title = (f"{summary['obj_name']}  reorient_{summary['h_cm']}  "
             f"{summary['i']}→{summary['j']}  "
             f"place=({summary['place_x']:.2f}, {summary['place_y']:.2f})  "
             f"hand={summary['hand']}  ok={n_ok}/{n_tot}")

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * nx + 2), max(4, 0.4 * ntz + 1.5)))
    cmap = plt.get_cmap("RdYlGn")
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower",
              extent=[min(xs) - 0.025, max(xs) + 0.025,
                      min(tzs) - 15, max(tzs) + 15])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.2f}" for x in xs], rotation=45)
    ax.set_yticks(tzs)
    ax.set_yticklabels([f"{int(t)}°" for t in tzs])
    ax.set_xlabel("pickup x (m)")
    ax.set_ylabel("pickup θ_z (deg)")
    ax.set_title(title)

    for c in summary["cells"]:
        ix = xs.index(c["x"]); itz = tzs.index(c["tz"])
        mark = "✓" if c["status"] == "ok" else "✗"
        ax.text(c["x"], c["tz"], mark, ha="center", va="center",
                color="white" if c["status"] != "ok" else "black", fontsize=10)

    plt.tight_layout()
    out_path = Path(args.out) if args.out else (sweep_dir / "heatmap.png")
    plt.savefig(out_path, dpi=120)
    print(f"[plot] saved -> {out_path}  (ok={n_ok}/{n_tot})")


if __name__ == "__main__":
    main()
