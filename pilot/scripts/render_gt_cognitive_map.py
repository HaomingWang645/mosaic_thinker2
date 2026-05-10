"""
Render the GROUND-TRUTH cognitive map for ARKitScenes scene 41069025
from the official `41069025_3dod_annotation.json` (oriented 3D bboxes per
object) and the `41069025_3dod_mesh.ply` (room geometry).

ARKitScenes convention used here:
  - Object OBBs in `data[*].segments.obb` are in CENTIMETRES, world frame.
  - Y is the UP axis (height ranges over [48, 217] cm; X/Z range over 4–5 m).
  - BEV plane = (X, Z); we drop Y.

Outputs
  pilot/figures/gt_cognitive_map.png             — GT only
  pilot/figures/cognitive_map_pred_vs_gt.png     — side-by-side prediction vs GT
"""
import json, re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


PILOT = Path("/home/haoming/mosaicthinker2/pilot")
FIG = PILOT / "figures"; FIG.mkdir(exist_ok=True)
GT_DIR = PILOT / "outputs/gt_41069025"
ANN = GT_DIR / "41069025_3dod_annotation.json"
MESH = GT_DIR / "41069025_3dod_mesh.ply"


def parse_ply_xyz(path, max_pts=200_000):
    """Read just the (x,y,z) columns from a small PLY (ASCII or binary little-endian)."""
    with open(path, "rb") as f:
        line = b""; binary = False; n_vert = 0
        in_vertex_block = False; vertex_props = []
        while True:
            c = f.read(1)
            line += c
            if c == b"\n":
                ls = line.decode("ascii", errors="ignore").strip()
                if ls.startswith("format binary"): binary = True
                elif ls.startswith("element vertex"):
                    n_vert = int(ls.split()[-1]); in_vertex_block = True
                elif ls.startswith("element"):
                    in_vertex_block = False         # any other element resets
                elif ls.startswith("property") and in_vertex_block:
                    parts = ls.split()
                    if parts[1] == "list":
                        in_vertex_block = False     # vertex shouldn't have list props
                    else:
                        vertex_props.append((parts[-1], parts[1]))
                elif ls == "end_header":
                    break
                line = b""
        body = f.read()
    if binary:
        tmap = {"float": "<f4", "uchar": "u1", "ushort": "<u2",
                "int": "<i4", "double": "<f8", "short": "<i2", "char": "i1"}
        dt_fields = [(name, tmap[t]) for name, t in vertex_props]
        rec = np.frombuffer(body, dtype=np.dtype(dt_fields), count=n_vert)
        xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=-1).astype(np.float32)
    else:
        # ASCII fallback
        arr = np.loadtxt(body.decode().splitlines()[:n_vert],
                         usecols=(0, 1, 2), dtype=np.float32)
        xyz = arr
    if len(xyz) > max_pts:
        idx = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
        xyz = xyz[idx]
    return xyz


def obb_bev_polygon(obb):
    """Project an oriented 3D bbox onto the X-Z (BEV) plane and return 4 (x,z)
    corners of the resulting rotated rectangle. Pass the bbox dict directly;
    units come from whichever field caller used (we use `obbAligned` which is
    in METRES in the mesh frame)."""
    c = np.asarray(obb["centroid"], dtype=float)
    L = np.asarray(obb["axesLengths"], dtype=float)
    R = np.asarray(obb["normalizedAxes"], dtype=float).reshape(3, 3)
    h = L / 2
    # Build BEV rotated rectangle: project local X and Z axes to (X, Z) plane.
    x_axis_bev = R[[0, 2], 0]
    z_axis_bev = R[[0, 2], 2]
    cx_bev = c[[0, 2]]
    hx, hz = h[0], h[2]
    rect = np.array([
        cx_bev + hx * x_axis_bev + hz * z_axis_bev,
        cx_bev + hx * x_axis_bev - hz * z_axis_bev,
        cx_bev - hx * x_axis_bev - hz * z_axis_bev,
        cx_bev - hx * x_axis_bev + hz * z_axis_bev,
    ])
    return rect


def main():
    # ---- Load annotations ----
    ann = json.loads(ANN.read_text())
    objs = ann["data"]
    print(f"GT scene 41069025: {len(objs)} objects")
    # Group counts by label for display
    from collections import Counter
    counts = Counter(o["label"] for o in objs)
    print(" ", dict(counts))

    # ---- Load mesh for room outline ----
    mesh_xyz = parse_ply_xyz(MESH)
    # Mesh is in METRES (ARKitScenes world frame). `obbAligned` is in the SAME
    # frame, also metres. (`obb` is a different frame in cm — don't mix it.)
    print(f"Mesh: {len(mesh_xyz):,} vertices  "
          f"(X [{mesh_xyz[:,0].min():.2f},{mesh_xyz[:,0].max():.2f}] m)")

    # ---- Render ----
    rects = []; labels = []
    for o in objs:
        rect = obb_bev_polygon(o["segments"]["obbAligned"])
        rects.append(rect); labels.append(o["label"])

    fig, ax = plt.subplots(figsize=(11.5, 8))

    # Floor outline = mesh BEV scatter (thin gray) — already in metres.
    # Filter to floor band only, so we don't see the entire ceiling/wall blob.
    rng = np.random.default_rng(0)
    floor_band = mesh_xyz[(mesh_xyz[:, 1] > -1.5) & (mesh_xyz[:, 1] < 0.05)]
    floor_xy = floor_band[:, [0, 2]] if len(floor_band) > 5000 else mesh_xyz[:, [0, 2]]
    if len(floor_xy) > 30_000:
        floor_xy = floor_xy[rng.choice(len(floor_xy), 30_000, replace=False)]
    ax.scatter(floor_xy[:, 0], floor_xy[:, 1], s=0.25, c="0.6", alpha=0.5, zorder=1)

    canon_order = sorted(set(labels))
    cmap = plt.cm.tab20
    color_of = {ln: cmap(i % 20) for i, ln in enumerate(canon_order)}

    seen_labels = set()
    for rect, ln in zip(rects, labels):
        rect_m = rect              # already in metres (obbAligned)
        clr = color_of[ln]
        leg = ln if ln not in seen_labels else None
        seen_labels.add(ln)
        poly = Polygon(rect_m, closed=True, facecolor=clr, edgecolor=clr,
                       alpha=0.35, linewidth=2.0, zorder=3, label=leg)
        ax.add_patch(poly)
        cx = float(rect_m[:, 0].mean()); cz = float(rect_m[:, 1].mean())
        ax.text(cx, cz, ln, fontsize=9, ha="center", va="center", weight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=clr, lw=1, alpha=0.9),
                zorder=4)

    ax.set_aspect("equal"); ax.grid(alpha=0.4, linestyle=":")
    ax.set_xlabel("X (m)  — ARKitScenes world frame, Y is up")
    ax.set_ylabel("Z (m)")
    ax.set_title("GROUND-TRUTH cognitive map — ARKitScenes scene 41069025\n"
                 "Oriented 3D bboxes from `_3dod_annotation.json`, projected to X-Z BEV; "
                 f"{len(objs)} GT objects",
                 fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9, markerscale=1.0, frameon=True, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(FIG / "gt_cognitive_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'gt_cognitive_map.png'}")

    # ---- Side-by-side: prediction vs GT ----
    pred = json.loads((PILOT / "outputs/vggt_22frames/cognitive_map.json").read_text())
    aligned = np.load(PILOT / "outputs/vggt_22frames/scene_aligned_pts.npz",
                      allow_pickle=True)
    pred_cams = aligned["cams"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # --- Predicted (left) ---
    ax = axes[0]
    sem = aligned["semantic"]
    sem_lbl = aligned["sem_labels"]
    sem_names = list(aligned["sem_label_names"])
    base_vocab = ["chair", "couch", "table", "bed", "tv", "lamp", "pillow",
                  "blanket", "cabinet", "bookshelf", "rug", "window", "door",
                  "plant", "stove", "refrigerator", "oven", "microwave",
                  "sink", "toilet", "desk"]
    def canonicalize(name):
        words = re.findall(r"[a-z]+", name.lower())
        if "sofa" in words: return "couch"
        for w in words:
            if w in base_vocab: return w
        return None
    canon_pts = {}
    for li, ln in enumerate(sem_names):
        c = canonicalize(ln)
        if c is None: continue
        m = sem_lbl == li
        if m.sum() == 0: continue
        canon_pts.setdefault(c, []).append(sem[m])
    canon_pts = {k: np.concatenate(v) for k, v in canon_pts.items()}
    canon_pts = {k: v[(v[:, 2] > -0.05) & (v[:, 2] < 2.5)] for k, v in canon_pts.items()}
    canon_pts = {k: v for k, v in canon_pts.items() if len(v) > 80}
    pcmap = {k: cmap(i % 20) for i, k in enumerate(sorted(canon_pts.keys()))}
    for k, pts in sorted(canon_pts.items(), key=lambda kv: -len(kv[1])):
        xy = pts[:, :2]
        ax.scatter(xy[:, 0], xy[:, 1], s=4, color=pcmap[k], alpha=0.7)
        c = xy.mean(0)
        ax.text(c[0], c[1], k, fontsize=8, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=pcmap[k], lw=1, alpha=0.9))
    ax.plot(pred_cams[:, 0], pred_cams[:, 1], "k.-", lw=0.8, ms=3, alpha=0.55)
    ax.scatter([pred_cams[-1, 0]], [pred_cams[-1, 1]], s=120, marker="*", c="red",
               edgecolor="black", zorder=5)
    ax.set_aspect("equal"); ax.grid(alpha=0.4, linestyle=":")
    pred_extent = (pred_cams[:, 0].max() - pred_cams[:, 0].min(),
                   pred_cams[:, 1].max() - pred_cams[:, 1].min())
    ax.set_title(f"Predicted cognitive map  (VGGT + GroundingDINO+SAM, 22 frames)\n"
                 f"BEV bbox covers ≈4.3 m²,  unanchored similarity scale", fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # --- GT (right) — both mesh and obbAligned in METRES, mesh frame ---
    ax = axes[1]
    floor_band = mesh_xyz[(mesh_xyz[:, 1] > -1.5) & (mesh_xyz[:, 1] < 0.05)]
    floor_xy = floor_band[:, [0, 2]] if len(floor_band) > 5000 else mesh_xyz[:, [0, 2]]
    if len(floor_xy) > 30_000:
        floor_xy = floor_xy[rng.choice(len(floor_xy), 30_000, replace=False)]
    ax.scatter(floor_xy[:, 0], floor_xy[:, 1], s=0.25, c="0.6", alpha=0.45, zorder=1)
    for rect, ln in zip(rects, labels):
        rect_m = rect      # already metres
        clr = color_of[ln]
        poly = Polygon(rect_m, closed=True, facecolor=clr, edgecolor=clr,
                       alpha=0.35, linewidth=2.0, zorder=3)
        ax.add_patch(poly)
        cx = float(rect_m[:, 0].mean()); cz = float(rect_m[:, 1].mean())
        ax.text(cx, cz, ln, fontsize=8, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=clr, lw=1, alpha=0.9),
                zorder=4)
    ax.set_aspect("equal"); ax.grid(alpha=0.4, linestyle=":")
    gt_extent_m2 = ((mesh_xyz[:, 0].max() - mesh_xyz[:, 0].min()) *
                    (mesh_xyz[:, 2].max() - mesh_xyz[:, 2].min()))
    ax.set_title(f"GROUND TRUTH cognitive map  (ARKitScenes scene 41069025)\n"
                 f"{len(objs)} oriented 3D bboxes (obbAligned),  mesh footprint ≈{gt_extent_m2:.1f} m²",
                 fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")

    fig.suptitle("Predicted vs GROUND TRUTH cognitive map — same scene", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "cognitive_map_pred_vs_gt.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'cognitive_map_pred_vs_gt.png'}")


if __name__ == "__main__":
    main()
