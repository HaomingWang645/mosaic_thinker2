"""Render the GT cognitive map for any ARKitScenes scene given its scene_id.

Usage: python scripts/render_gt_any.py <scene_id>
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


PILOT = Path("/home/haoming/mosaicthinker2/pilot")
FIG = PILOT / "figures"; FIG.mkdir(exist_ok=True)


def parse_ply_xyz(path, max_pts=200_000):
    with open(path, "rb") as f:
        line = b""; binary = False; n_vert = 0
        in_vertex = False; vertex_props = []
        while True:
            c = f.read(1)
            line += c
            if c == b"\n":
                ls = line.decode("ascii", errors="ignore").strip()
                if ls.startswith("format binary"): binary = True
                elif ls.startswith("element vertex"):
                    n_vert = int(ls.split()[-1]); in_vertex = True
                elif ls.startswith("element"):
                    in_vertex = False
                elif ls.startswith("property") and in_vertex:
                    parts = ls.split()
                    if parts[1] != "list":
                        vertex_props.append((parts[-1], parts[1]))
                elif ls == "end_header":
                    break
                line = b""
        body = f.read()
    tmap = {"float": "<f4", "uchar": "u1", "ushort": "<u2",
            "int": "<i4", "double": "<f8", "short": "<i2", "char": "i1"}
    dt = np.dtype([(n, tmap[t]) for n, t in vertex_props])
    rec = np.frombuffer(body, dtype=dt, count=n_vert)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=-1).astype(np.float32)
    if len(xyz) > max_pts:
        idx = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
        xyz = xyz[idx]
    return xyz


def obb_bev(obb):
    """`obbAligned` → BEV rotated rectangle. Z is up."""
    c = np.asarray(obb["centroid"], dtype=float)
    L = np.asarray(obb["axesLengths"], dtype=float)
    R = np.asarray(obb["normalizedAxes"], dtype=float).reshape(3, 3)
    h = L / 2
    x = R[[0, 1], 0]; y = R[[0, 1], 1]
    cx = c[[0, 1]]; hx, hy = h[0], h[1]
    return np.array([cx + hx * x + hy * y, cx + hx * x - hy * y,
                     cx - hx * x - hy * y, cx - hx * x + hy * y])


def main(scene_id):
    sd = PILOT / f"outputs/gt_{scene_id}"
    ann = json.loads((sd / f"{scene_id}_3dod_annotation.json").read_text())
    mesh = parse_ply_xyz(sd / f"{scene_id}_3dod_mesh.ply")
    objs = ann["data"]
    print(f"Scene {scene_id}: {len(objs)} objects, "
          f"{ {x['label']: sum(1 for y in objs if y['label']==x['label']) for x in objs} }")

    z_floor = float(np.percentile(mesh[:, 2], 2))
    floor_band = mesh[mesh[:, 2] < z_floor + 0.30]
    floor_xy = floor_band[:, [0, 1]] if len(floor_band) > 5000 else mesh[:, [0, 1]]
    rng = np.random.default_rng(0)
    if len(floor_xy) > 30_000:
        floor_xy = floor_xy[rng.choice(len(floor_xy), 30_000, replace=False)]

    labels = [o["label"] for o in objs]
    rects = [obb_bev(o["segments"]["obbAligned"]) for o in objs]
    cmap = plt.cm.tab20
    color_of = {ln: cmap(i % 20) for i, ln in enumerate(sorted(set(labels)))}

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(floor_xy[:, 0], floor_xy[:, 1], s=0.25, c="0.6", alpha=0.5)
    seen = set()
    for rect, ln in zip(rects, labels):
        leg = ln if ln not in seen else None; seen.add(ln)
        clr = color_of[ln]
        ax.add_patch(Polygon(rect, closed=True, facecolor=clr, edgecolor=clr,
                             alpha=0.35, linewidth=2.0, label=leg))
        cx, cy = float(rect[:, 0].mean()), float(rect[:, 1].mean())
        ax.text(cx, cy, ln, fontsize=8, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=clr, lw=0.8, alpha=0.9))
    ax.set_aspect("equal"); ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    fp = (floor_band[:, 0].max() - floor_band[:, 0].min()) * (floor_band[:, 1].max() - floor_band[:, 1].min())
    ax.set_title(f"GT cognitive map — ARKitScenes scene {scene_id}\n"
                 f"{len(objs)} oriented bboxes, floor footprint ≈{fp:.1f} m²", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig.tight_layout()
    out = FIG / f"gt_cognitive_map_{scene_id}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "41069043")
