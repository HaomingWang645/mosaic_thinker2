"""
Convert the VGGT reconstruction into a cognitive (semantic) map in the style
of MosaicThinker Figure 9: a top-down BEV grid with labelled colored bounding
boxes per object, the camera trajectory drawn on top, and a JSON text
description of every object's BEV coordinates.

Pipeline:
  1. Load VGGT per-pixel world points + extrinsics.
  2. RANSAC-fit the floor plane on the bottom-quartile-height points.
  3. Build a rigid rotation R that sends the floor normal to +Z; translate
     so the floor plane sits at z = 0.
  4. Transform the merged point cloud and the lifted semantic points into
     the gravity-aligned frame.
  5. For every (lifted) object label, fit a 2-D oriented bounding box on the
     XY plane after dropping outliers via per-label DBSCAN.
  6. Render the cognitive map (BEV grid + boxes + cam trajectory) as PNG and
     emit a structured JSON description.

Outputs:
  pilot/figures/cognitive_map.png
  pilot/outputs/vggt_22frames/cognitive_map.json
  pilot/outputs/vggt_22frames/scene_aligned_pts.npz
"""
import json, re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
from scipy.spatial import cKDTree


METHOD = Path("outputs/vggt_22frames")
FRAMES = Path("/home/haoming/mosaic_thinker/frames")
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)


# -------- 1. Load VGGT outputs --------
def load_vggt():
    pts = np.load(METHOD / "points.npy")             # (N, H, W, 3)
    conf = np.load(METHOD / "points_conf.npy")        # (N, H, W)
    extr = np.load(METHOD / "extrinsic.npy")          # (N, 3, 4) world->cam
    images = np.load(METHOD / "images.npy")           # (N, 3, H, W) in [0,1]
    names = json.loads((METHOD / "frame_names.json").read_text())
    sem = np.load(METHOD / "semantic_points.npz", allow_pickle=True)
    return dict(pts=pts, conf=conf, extr=extr, imgs=images, names=names,
                sem_pts=sem["points"], sem_lbl=sem["labels"],
                sem_names=list(sem["label_names"]),
                sem_conf=sem["confidences"], sem_frame=sem["frame_idx"])


# -------- 2. RANSAC floor fit --------
def fit_plane(points, n_iter=600, tol=0.03, rng_seed=0):
    """Fit a plane ax+by+cz+d=0 by RANSAC; return (normal, d, inlier_mask)."""
    rng = np.random.default_rng(rng_seed)
    N = len(points)
    best_inl = None; best_normal = None; best_d = None
    for _ in range(n_iter):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9: continue
        n = n / n_norm
        d = -n.dot(p0)
        dist = np.abs(points @ n + d)
        inl = dist < tol
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl; best_normal = n; best_d = d
    # Re-fit on inliers via SVD
    P = points[best_inl]
    c = P.mean(0)
    cov = np.cov((P - c).T)
    _, evec = np.linalg.eigh(cov)
    n_refined = evec[:, 0]            # smallest eigenvector
    d_refined = -n_refined.dot(c)
    return n_refined, d_refined, best_inl


def gravity_from_cameras(cam_centers, cam_extr):
    """Estimate world gravity (= floor normal) from the camera trajectory.

    Two complementary signals:

    A) PCA on camera centers. In a handheld indoor video the operator mostly
       walks/turns in a roughly horizontal plane; the camera positions span
       a thin pancake whose *thinnest* direction is gravity.

    B) The OpenCV camera frame's `+Y` is image-down. When the camera is held
       roughly upright, `R.T @ [0,1,0]` (in world coords) is approximately
       gravity. Averaging across all frames cancels per-frame tilt.

    We blend the two by averaging their normalized vectors with sign aligned.
    Returns a unit vector pointing in the +up direction of the world.
    """
    # ---- Signal A: PCA on camera centers ----
    centered = cam_centers - cam_centers.mean(0)
    cov = (centered.T @ centered) / max(1, len(centered) - 1)
    evals, evecs = np.linalg.eigh(cov)              # ascending
    pca_up = evecs[:, 0]                            # smallest variance direction
    # ---- Signal B: average of cam_+Y in world ----
    ys_world = []
    for ext in cam_extr:
        R = ext[:, :3]                              # world->cam
        ys_world.append(R.T @ np.array([0.0, 1.0, 0.0]))
    cam_y_avg = np.mean(ys_world, axis=0)
    cam_y_avg = cam_y_avg / (np.linalg.norm(cam_y_avg) + 1e-9)
    # Image-+Y is *down*, so world-up ≈ -cam_y_avg
    cam_up = -cam_y_avg
    # ---- Align signs and blend ----
    if pca_up @ cam_up < 0:
        pca_up = -pca_up
    up = (pca_up + cam_up) / 2
    up = up / (np.linalg.norm(up) + 1e-9)
    print(f"  gravity from PCA(cam centers): {pca_up.round(3)}  "
          f"(eigvals: {evals.round(3)})")
    print(f"  gravity from cam +Y avg:       {cam_up.round(3)}")
    print(f"  blended +up world direction:   {up.round(3)}  "
          f"(consistency: {pca_up @ cam_up:+.3f})")
    return up


def refine_floor_z(pts_aligned, sample=20000, tol=0.05):
    """After rough alignment, the floor should sit at the LOWEST mode of the
    z-distribution. Find a horizontal-ish (small |dx|, |dy| residual after
    plane fit) thin band near the bottom and report its median z."""
    rng = np.random.default_rng(0)
    if len(pts_aligned) > sample:
        pts_aligned = pts_aligned[rng.choice(len(pts_aligned), sample, replace=False)]
    z = pts_aligned[:, 2]
    z_floor_guess = np.percentile(z, 5)
    band = pts_aligned[np.abs(z - z_floor_guess) < 0.30]
    if len(band) > 100:
        return float(np.median(band[:, 2]))
    return float(z_floor_guess)


# -------- 3. Build rigid alignment --------
def alignment_from_floor(normal):
    """Rotation R sending `normal` to (0,0,1)."""
    z = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    if np.allclose(n, z):
        return np.eye(3)
    if np.allclose(n, -z):
        # 180° rotation about X
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c = n.dot(z)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R


# -------- 4. Segment per-label clusters --------
def label_clusters(pts_xy, labels, label_names, eps=0.25, min_pts=80):
    """Per-label DBSCAN using a kd-tree (avoids sklearn dependency)."""
    out = {}
    for li, ln in enumerate(label_names):
        m = labels == li
        if m.sum() < min_pts: continue
        sub = pts_xy[m]
        # DBSCAN by-hand (small N).
        tree = cKDTree(sub)
        labels_dbs = -np.ones(len(sub), dtype=int)
        cid = 0
        visited = np.zeros(len(sub), dtype=bool)
        for i in range(len(sub)):
            if visited[i]: continue
            visited[i] = True
            nb = tree.query_ball_point(sub[i], eps)
            if len(nb) < min_pts: continue
            labels_dbs[i] = cid
            queue = list(nb)
            while queue:
                j = queue.pop()
                if not visited[j]:
                    visited[j] = True
                    nb2 = tree.query_ball_point(sub[j], eps)
                    if len(nb2) >= min_pts:
                        queue.extend(nb2)
                if labels_dbs[j] == -1:
                    labels_dbs[j] = cid
            cid += 1
        # Pick largest cluster
        if cid == 0: continue
        sizes = [(labels_dbs == c).sum() for c in range(cid)]
        kept = int(np.argmax(sizes))
        keep = labels_dbs == kept
        cluster_pts = sub[keep]
        out[ln] = cluster_pts
    return out


def axis_aligned_bbox(pts):
    return pts.min(0), pts.max(0)


# -------- 5. Render --------
def merge_label_aliases(label_names):
    """GroundingDINO sometimes returns merged labels like 'cabinet bookshelf'
    or token-fragment garbage ('tablehelf', 'cabinethel'). Map each text
    return to a single canonical label drawn from a fixed vocab; drop garbage."""
    canon = {}
    base_vocab = {"chair", "couch", "sofa", "table", "bed", "tv", "lamp", "pillow",
                  "blanket", "cabinet", "bookshelf", "rug", "window", "door",
                  "plant", "stove", "refrigerator", "oven", "microwave",
                  "sink", "toilet", "desk"}
    alias = {"sofa": "couch"}
    for name in label_names:
        words = re.findall(r"[a-z]+", name.lower())
        chosen = next((w for w in words if w in base_vocab), None)
        if chosen is None:
            canon[name] = None    # garbage / unknown
            continue
        canon[name] = alias.get(chosen, chosen)
    return canon


def main():
    d = load_vggt()

    # 1. Build merged scene cloud (top-50% confidence) for floor fitting +
    #    rendering.
    pts_all = []; col_all = []; conf_all = []
    for i in range(len(d["names"])):
        p = d["pts"][i].reshape(-1, 3)
        rgb = d["imgs"][i].transpose(1, 2, 0).reshape(-1, 3)
        c = d["conf"][i].reshape(-1)
        ok = np.isfinite(p).all(-1)
        pts_all.append(p[ok]); col_all.append(np.clip(rgb[ok], 0, 1)); conf_all.append(c[ok])
    pts_all = np.concatenate(pts_all)
    col_all = np.concatenate(col_all)
    conf_all = np.concatenate(conf_all)
    keep = conf_all > np.percentile(conf_all, 50)
    pts_all = pts_all[keep]; col_all = col_all[keep]
    print(f"Scene cloud: {len(pts_all):,} pts after top-50%-conf filter")

    # 2. Gravity direction. Estimate from cameras (more robust than RANSAC
    #    on a noisy point cloud where walls can masquerade as the floor).
    print("Estimating gravity from camera trajectory + camera-frame +Y…")
    cams_world = []
    for i in range(len(d["names"])):
        Rwc = d["extr"][i, :, :3]; t = d["extr"][i, :, 3]
        c = -Rwc.T @ t
        cams_world.append(c)
    cams_world = np.array(cams_world)
    up_world = gravity_from_cameras(cams_world, d["extr"])

    # 3. Alignment: rotate so up_world -> +Z.
    R = alignment_from_floor(up_world)
    pts_rough = (R @ pts_all.T).T
    cams_rough = (R @ cams_world.T).T
    # Refine z=0 to the floor: lowest stable z-mode of the rotated cloud.
    z_floor = refine_floor_z(pts_rough)
    print(f"  z-offset (floor median z) after rotation: {z_floor:+.3f} m")
    pts_aligned = pts_rough - np.array([0, 0, z_floor])
    cams_aligned = cams_rough - np.array([0, 0, z_floor])

    # Semantic points.
    sem = (R @ d["sem_pts"].T).T - np.array([0, 0, z_floor])

    np.savez(METHOD / "scene_aligned_pts.npz",
             scene=pts_aligned, scene_rgb=col_all,
             cams=cams_aligned, semantic=sem,
             sem_labels=d["sem_lbl"], sem_label_names=np.array(d["sem_names"]))

    # 4. Cluster each label in BEV (XY plane, height=z dropped).
    canon = merge_label_aliases(d["sem_names"])
    # Re-aggregate semantic points by canonical label.
    canon_to_pts = {}
    for li, ln in enumerate(d["sem_names"]):
        cl = canon[ln]
        if cl is None: continue
        m = d["sem_lbl"] == li
        if m.sum() == 0: continue
        canon_to_pts.setdefault(cl, []).append(sem[m])
    canon_pts = {k: np.concatenate(v) for k, v in canon_to_pts.items()}

    # Drop labels too high above the floor (likely ceiling / spurious lifting),
    # keep only those between floor and 2.5 m.
    canon_pts = {k: v[(v[:, 2] > -0.05) & (v[:, 2] < 2.5)] for k, v in canon_pts.items()}
    canon_pts = {k: v for k, v in canon_pts.items() if len(v) > 80}

    print(f"Canonical labels with >=80 BEV points: {list(canon_pts.keys())}")

    # Per-label cleanup: try sklearn DBSCAN to keep largest cluster, then
    # percentile-trim the BEV. Falls back to percentile-only if sklearn missing.
    try:
        from sklearn.cluster import DBSCAN
        have_sk = True
    except ImportError:
        have_sk = False
    clusters = {}
    for k, pts in canon_pts.items():
        xy = pts[:, :2]
        if have_sk and len(xy) > 100:
            db = DBSCAN(eps=0.25, min_samples=40, n_jobs=-1).fit(xy)
            lbs = db.labels_
            uniq, cts = np.unique(lbs[lbs >= 0], return_counts=True)
            if len(uniq):
                keep_id = uniq[np.argmax(cts)]
                pts_keep = pts[lbs == keep_id]
            else:
                pts_keep = pts
        else:
            pts_keep = pts
        # Percentile trim 5-95% on each axis to drop tail noise / partial views.
        xy = pts_keep[:, :2]
        if len(xy) < 5:
            continue
        lo, hi = np.percentile(xy, [5, 95], axis=0)
        in_box = ((xy[:, 0] > lo[0]) & (xy[:, 0] < hi[0]) &
                  (xy[:, 1] > lo[1]) & (xy[:, 1] < hi[1]))
        pts_keep = pts_keep[in_box]
        if len(pts_keep) >= 80:
            clusters[k] = pts_keep
    # Drop labels that ended up impossibly large after cleanup (full-room wall).
    # 4 m diagonal allows a sofa or large table; reject anything bigger.
    for k in list(clusters):
        xy = clusters[k][:, :2]
        diag = np.linalg.norm(xy.max(0) - xy.min(0))
        if diag > 4.0:
            print(f"  drop label {k!r}: diag {diag:.2f} m too large (over-seg)")
            del clusters[k]

    # 5. Render the cognitive map.
    fig, ax = plt.subplots(figsize=(9, 9))

    # Background scene cloud (thin, gray).
    rng = np.random.default_rng(0)
    bg = pts_aligned
    bg_rgb = col_all
    if len(bg) > 80000:
        sel = rng.choice(len(bg), 80000, replace=False)
        bg = bg[sel]; bg_rgb = bg_rgb[sel]
    # Drop ceiling-y points to keep the BEV uncluttered
    keep = (bg[:, 2] > -0.1) & (bg[:, 2] < 2.5)
    ax.scatter(bg[keep, 0], bg[keep, 1], s=0.6, c=bg_rgb[keep], alpha=0.15, zorder=1)

    # Object boxes.
    cmap = plt.cm.tab10
    color_map = {k: cmap(i % 10) for i, k in enumerate(sorted(clusters.keys()))}
    json_objs = []
    for k, pts in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        xy = pts[:, :2]
        mn = xy.min(0); mx = xy.max(0)
        cx, cy = float((mn[0] + mx[0]) / 2), float((mn[1] + mx[1]) / 2)
        w, h = float(mx[0] - mn[0]), float(mx[1] - mn[1])
        clr = color_map[k]
        rect = Rectangle(mn, w, h, linewidth=2.0, edgecolor=clr,
                         facecolor=clr, alpha=0.18, zorder=3)
        ax.add_patch(rect)
        ax.scatter(xy[:, 0], xy[:, 1], s=2.5, color=clr, alpha=0.7, zorder=2,
                   label=f"{k}  ({len(pts)} pts)")
        ax.text(cx, cy, k, fontsize=10, ha="center", va="center", weight="bold",
                color="black", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=clr, lw=1, alpha=0.85))
        json_objs.append({
            "label": k,
            "center_xy_m": [round(cx, 3), round(cy, 3)],
            "bbox_xy_m": {
                "x_min": round(float(mn[0]), 3), "x_max": round(float(mx[0]), 3),
                "y_min": round(float(mn[1]), 3), "y_max": round(float(mx[1]), 3),
            },
            "size_m": [round(w, 3), round(h, 3)],
            "n_points": int(len(pts)),
        })

    # Camera trajectory.
    ax.plot(cams_aligned[:, 0], cams_aligned[:, 1], "k.-", lw=1.2, ms=4, zorder=5,
            label="camera trajectory")
    # Mark first and last camera with arrows showing the in-plane heading.
    for k in [0, len(cams_aligned) // 2, len(cams_aligned) - 1]:
        c = cams_aligned[k]
        # Heading: -R^T @ z_cam expressed in world, then aligned.
        Rcw = d["extr"][k, :, :3]
        z_cam_world = (Rcw.T @ np.array([0.0, 0.0, 1.0]))   # cam +Z direction in world
        z_cam_aligned = R @ z_cam_world
        head_xy = z_cam_aligned[:2]
        head_xy = head_xy / (np.linalg.norm(head_xy) + 1e-6) * 0.25
        ax.add_patch(FancyArrowPatch(c[:2], c[:2] + head_xy,
                                     arrowstyle="->", mutation_scale=14,
                                     color="red", linewidth=1.6, zorder=6))
        ax.text(c[0], c[1] + 0.05, d["names"][k].replace("frame_", "f").replace(".png", ""),
                fontsize=7, color="red", zorder=6)

    # Floor reference grid.
    ax.set_aspect("equal")
    ax.grid(alpha=0.4, linestyle=":", zorder=0)
    margin = 0.4
    xs = np.concatenate([cams_aligned[:, 0]] + [pts[:, 0] for pts in clusters.values()])
    ys = np.concatenate([cams_aligned[:, 1]] + [pts[:, 1] for pts in clusters.values()])
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)
    ax.set_xlabel("X (m)  — gravity-aligned, floor at z=0")
    ax.set_ylabel("Y (m)")
    ax.set_title("Cognitive (BEV) semantic map\nfrom VGGT-reconstructed scene + GroundingDINO/SAM lift",
                 fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, markerscale=2, frameon=True, borderaxespad=0.0,
              ncol=1).set_zorder(7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cognitive_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6a. Paper-style "Visual Prompt" 3-panel figure
    #     (Aligned 3D point cloud BEV → semantic map → text descriptions).
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                              gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]})
    # Panel 1: aligned point cloud BEV (RGB)
    ax = axes[0]
    keep_bg = (pts_aligned[:, 2] > -0.1) & (pts_aligned[:, 2] < 2.5)
    bg = pts_aligned[keep_bg]; bg_rgb = col_all[keep_bg]
    if len(bg) > 100000:
        sel = rng.choice(len(bg), 100000, replace=False)
        bg = bg[sel]; bg_rgb = bg_rgb[sel]
    ax.scatter(bg[:, 0], bg[:, 1], s=0.7, c=bg_rgb, alpha=0.45)
    ax.plot(cams_aligned[:, 0], cams_aligned[:, 1], "ro-", ms=3, lw=0.8, alpha=0.7,
            label="cameras")
    ax.set_aspect("equal"); ax.grid(alpha=0.3, linestyle=":")
    ax.set_title("Aligned 3D point cloud (BEV view)", fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: clean semantic map — colored boxes only, no scene cloud
    ax = axes[1]
    for k, pts in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        xy = pts[:, :2]
        mn = xy.min(0); mx = xy.max(0)
        cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
        clr = color_map.get(k, "gray")
        rect = Rectangle(mn, mx[0] - mn[0], mx[1] - mn[1],
                         linewidth=2.0, edgecolor=clr, facecolor=clr, alpha=0.3)
        ax.add_patch(rect)
        ax.text(cx, cy, k, fontsize=10, ha="center", va="center", weight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=clr, lw=1, alpha=0.9))
    # Camera trajectory and current view marker
    ax.plot(cams_aligned[:, 0], cams_aligned[:, 1], "k.-", lw=1.0, ms=3, alpha=0.7,
            zorder=4)
    cur = cams_aligned[-1]
    ax.scatter([cur[0]], [cur[1]], s=120, marker="*", c="red", edgecolor="black",
               zorder=5, label="current camera")
    ax.set_aspect("equal"); ax.grid(alpha=0.4, linestyle=":")
    ax.set_xlim(*axes[0].get_xlim()); ax.set_ylim(*axes[0].get_ylim())
    ax.set_title("Semantic map (cognitive map)", fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3: text descriptions in JSON-ish format (rendered as monospace text)
    ax = axes[2]; ax.axis("off")
    lines = ["{", '  "objects": ['] + [
        f'    {{"label": "{o["label"]}", "x": {o["center_xy_m"][0]:+.2f}, '
        f'"y": {o["center_xy_m"][1]:+.2f}, "size": [{o["size_m"][0]:.2f}, '
        f'{o["size_m"][1]:.2f}]}}' + ("," if i < len(json_objs := []) - 1 else "")
        for i, o in enumerate([])  # placeholder; rebuilt below
    ]
    # Re-render with the actual JSON list
    # Use the json_objs from earlier (built in the first render block).
    # Re-derive json_objs here from clusters in case order changed.
    text_objs = []
    for k, pts in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        xy = pts[:, :2]
        mn = xy.min(0); mx = xy.max(0)
        cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
        text_objs.append((k, float(cx), float(cy), float(mx[0] - mn[0]), float(mx[1] - mn[1])))
    text = ["{", '  "current_camera_xy": [{:.2f}, {:.2f}],'.format(cur[0], cur[1]),
            '  "objects": [']
    for i, (k, cx, cy, w, h) in enumerate(text_objs):
        comma = "," if i < len(text_objs) - 1 else ""
        text.append(f'    {{"label": "{k}", "x": {cx:+.2f}, "y": {cy:+.2f}, '
                    f'"size": [{w:.2f}, {h:.2f}]}}{comma}')
    text += ["  ]", "}"]
    ax.text(0.0, 0.98, "\n".join(text), fontsize=9.5, ha="left", va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="#fafaf2", ec="black", lw=0.8))
    ax.set_title("Text description (passed to VLM with the map)", fontsize=11)

    fig2.suptitle("Visual prompt for the VLM (paper Fig. 9 style)\n"
                  "Aligned 3D cloud  →  cognitive map  →  text", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "visual_prompt_three_panel.png", dpi=140)
    plt.close(fig2)
    print(f"Saved {FIG_DIR / 'visual_prompt_three_panel.png'}")

    # 6. JSON description (the text companion the paper uses next to the image).
    description = {
        "scene_extent_m": {
            "x": [round(float(xs.min()), 3), round(float(xs.max()), 3)],
            "y": [round(float(ys.min()), 3), round(float(ys.max()), 3)],
        },
        "up_world_direction": [round(float(x), 4) for x in up_world],
        "z_offset_m": round(z_floor, 3),
        "camera_trajectory_xy_m": [
            [round(float(c[0]), 3), round(float(c[1]), 3)] for c in cams_aligned
        ],
        "n_frames": int(len(cams_aligned)),
        "objects": json_objs,
    }
    out_json = METHOD / "cognitive_map.json"
    out_json.write_text(json.dumps(description, indent=2))
    print(f"Saved {FIG_DIR / 'cognitive_map.png'}")
    print(f"Saved {out_json}")
    print(f"\nObjects detected ({len(json_objs)}):")
    for o in json_objs:
        c = o['center_xy_m']
        s = o['size_m']
        print(f"  {o['label']:10s}  center=({c[0]:+.2f}, {c[1]:+.2f})  "
              f"size=({s[0]:.2f}m × {s[1]:.2f}m)  n={o['n_points']}")


if __name__ == "__main__":
    main()
