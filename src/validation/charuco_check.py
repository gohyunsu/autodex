#!/usr/bin/env python3
"""Overlay charuco detections on each image in a folder. Useful to sanity-check
how many corners are detected (and which ones get missed) — e.g. for using
"all corners visible" as a post-trial label heuristic.

Usage:
    python src/validation/charuco_check.py <image_dir> [--out <out_dir>]

Default --out: <image_dir>/../images_charuco/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paradex.image.aruco import (
    detect_charuco, _charuco_board_cache, boardinfo_dict,
)


def expected_total_per_board() -> dict:
    """Total interior corners = (numX-1) * (numY-1) per board."""
    return {
        bid: (cfg["numX"] - 1) * (cfg["numY"] - 1)
        for bid, cfg in boardinfo_dict.items()
    }


def overlay(img: np.ndarray, det: dict, totals: dict) -> tuple[np.ndarray, dict]:
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    summary = {}
    palette = [(0, 255, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (255, 0, 0)]

    for i, (b_id, info) in enumerate(det.items()):
        color = palette[i % len(palette)]
        corners = info["checkerCorner"]  # (N, 2)
        ids = info["checkerIDs"]         # (N,)
        for (x, y), cid in zip(corners, ids):
            cv2.circle(vis, (int(round(x)), int(round(y))), 4, color, -1)
            cv2.putText(vis, str(int(cid)),
                        (int(round(x)) + 5, int(round(y)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        summary[b_id] = (len(corners), totals.get(b_id, -1))

    # legend
    y0 = 20
    for b_id, (n, tot) in summary.items():
        txt = f"{b_id}: {n}/{tot} corners"
        cv2.putText(vis, txt, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y0 += 22
    if not summary:
        cv2.putText(vis, "NO charuco detected", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return vis, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_dir", type=str)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    src = Path(args.image_dir).expanduser()
    out = Path(args.out).expanduser() if args.out else src.parent / "images_charuco"
    out.mkdir(parents=True, exist_ok=True)

    totals = expected_total_per_board()
    print(f"Expected corners per board: {totals}")
    print(f"Reading {src}  ->  Writing {out}\n")

    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        sys.exit(f"no images in {src}")

    for fp in paths:
        img = cv2.imread(str(fp))
        if img is None:
            print(f"  [skip] {fp.name}: imread failed")
            continue
        det = detect_charuco(img)
        vis, summary = overlay(img, det, totals)
        cv2.imwrite(str(out / fp.name), vis)
        if summary:
            per_board = "  ".join(f"{b}:{n}/{t}" for b, (n, t) in summary.items())
            print(f"  {fp.name}: {per_board}")
        else:
            print(f"  {fp.name}: NO charuco")

    print(f"\nDone. Overlays saved under {out}")


if __name__ == "__main__":
    main()
