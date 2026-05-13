"""Render the GT cognitive map for any ScanNet scene from EmbodiedScan annotations.

EmbodiedScan stores instances as 9-DoF boxes:
  bbox_3d = [cx, cy, cz, dx, dy, dz, roll, pitch, yaw]
in the scene's mesh-aligned frame after applying `axis_align_matrix` to the
raw ScanNet world (the matrix is already pre-applied: bboxes live in the
axis-aligned frame). Z is up.

We project to X-Y BEV.
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


PILOT = Path("/home/haoming/mosaicthinker2/pilot")
FIG = PILOT / "figures"; FIG.mkdir(exist_ok=True)


def euler_zyx_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def bbox_bev(bbox_3d):
    cx, cy, cz, dx, dy, dz, roll, pitch, yaw = bbox_3d
    R = euler_zyx_to_R(roll, pitch, yaw)
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    # BEV: local X and Y axes projected to world (X, Y).
    x_axis = R[[0, 1], 0]
    y_axis = R[[0, 1], 1]
    cxy = np.array([cx, cy])
    return np.array([
        cxy + hx * x_axis + hy * y_axis,
        cxy + hx * x_axis - hy * y_axis,
        cxy - hx * x_axis - hy * y_axis,
        cxy - hx * x_axis + hy * y_axis,
    ]), cz, dz


def main(json_path, out_png="gt_cognitive_map_scannet.png"):
    d = json.loads(Path(json_path).read_text())
    instances = d["instances"]
    sample_idx = d["sample_idx"]
    print(f"Scene: {sample_idx}, {len(instances)} instances")

    # Drop walls / ceiling / floor / generic 'object' for a cleaner cognitive map
    excluded = {"wall", "ceiling", "floor", "object", "structure"}
    visible = [i for i in instances if i["label"] not in excluded]
    print(f"  after filter: {len(visible)} furniture-level instances")

    rects = []; labels = []; heights = []
    for inst in visible:
        rect, cz, dz = bbox_bev(inst["bbox_3d"])
        # Drop instances obviously above the floor
        rects.append(rect); labels.append(inst["label"]); heights.append((cz, dz))

    unique_labels = sorted(set(labels))
    cmap = plt.cm.tab20
    color_of = {ln: cmap(i % 20) for i, ln in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(11, 9))
    seen = set()
    # Draw floor footprint from union of all rect bounds for context
    all_xy = np.concatenate([r for r in rects])
    x_lo, y_lo = all_xy.min(0)
    x_hi, y_hi = all_xy.max(0)
    pad = 0.5
    # Faint background to show the room region
    ax.fill([x_lo - pad, x_hi + pad, x_hi + pad, x_lo - pad],
            [y_lo - pad, y_lo - pad, y_hi + pad, y_hi + pad],
            color="0.96", zorder=0)

    for rect, ln, (cz, dz) in zip(rects, labels, heights):
        leg = ln if ln not in seen else None; seen.add(ln)
        clr = color_of[ln]
        ax.add_patch(Polygon(rect, closed=True, facecolor=clr, edgecolor=clr,
                             alpha=0.35, linewidth=1.5, label=leg, zorder=3))
        cx, cy = float(rect[:, 0].mean()), float(rect[:, 1].mean())
        ax.text(cx, cy, ln, fontsize=7, ha="center", va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=clr, lw=0.6, alpha=0.85),
                zorder=4)

    fp_x = x_hi - x_lo; fp_y = y_hi - y_lo
    ax.set_aspect("equal"); ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlabel("X (m)  — ScanNet axis-aligned world frame, Z is up (dropped)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"GROUND-TRUTH cognitive map  —  ScanNet {sample_idx.split('/')[-1]}  "
                 f"(EmbodiedScan 9-DoF bboxes)\n"
                 f"{len(visible)} furniture instances after filtering walls/ceiling/object  ·  "
                 f"room footprint ≈{fp_x*fp_y:.1f} m²",
                 fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True,
              borderaxespad=0, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {FIG / out_png}")


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/gt_scannet/scannet_gt.json"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "gt_cognitive_map_scannet.png"
    main(json_path, out_png)
