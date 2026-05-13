"""
For each of 3 ScanNet scenes, build:
  1) gravity-aligned predicted point cloud + camera trajectory
  2) predicted cognitive map (POINTS-ONLY version) — colored per-class scatter
  3) predicted cognitive map (BOUNDING-BOX version)   — clean rotated rectangles
  4) ground-truth cognitive map from Holi-Spatial AABBs
Render the three panels (2, 3, 4) into a single figure per scene.

Also export `scene_semantic.ply` per scene.
"""
import json, re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

try:
    from sklearn.cluster import DBSCAN
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


PILOT = Path("/home/haoming/mosaicthinker2/pilot")
FRAMES_ROOT = PILOT / "outputs/scannet_frames"
PRED_ROOT = PILOT / "outputs/scannet_pred"
GT_BBOX_DIR = PILOT / "outputs/gt_scannet/bbox_extract/output_scannet_new_aabb"
GT_BEV_DIR = PILOT / "outputs/gt_scannet/scannet_bevs"
FIG_DIR = PILOT / "figures/scannet_compare"; FIG_DIR.mkdir(parents=True, exist_ok=True)
PLY_ROOT = PRED_ROOT


BASE_VOCAB = ["chair", "couch", "table", "bed", "tv", "lamp", "pillow",
              "blanket", "cabinet", "bookshelf", "rug", "window", "door",
              "plant", "stove", "refrigerator", "oven", "microwave",
              "sink", "toilet", "desk", "drawer", "countertop", "shelf",
              "trash can", "television"]
VOCAB_SET = set(BASE_VOCAB)


def canonicalize(name):
    n = name.lower()
    if "trash" in n: return "trash can"
    if "tv" in n.split() or "television" in n.split(): return "tv"
    if "sofa" in n.split(): return "couch"
    words = re.findall(r"[a-z]+", n)
    for w in words:
        if w in VOCAB_SET: return w
    # also allow multi-word vocab items
    for vw in BASE_VOCAB:
        if vw in n:
            return vw
    return None


def gravity_from_cameras(cam_centers, cam_extr):
    centered = cam_centers - cam_centers.mean(0)
    cov = (centered.T @ centered) / max(1, len(centered) - 1)
    _, evecs = np.linalg.eigh(cov)
    pca_up = evecs[:, 0]
    ys = []
    for ext in cam_extr:
        R = ext[:, :3]
        ys.append(R.T @ np.array([0.0, 1.0, 0.0]))
    cam_up = -np.mean(ys, axis=0)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-9)
    if pca_up @ cam_up < 0: pca_up = -pca_up
    up = (pca_up + cam_up) / 2
    return up / (np.linalg.norm(up) + 1e-9)


def alignment_from_up(up):
    z = np.array([0.0, 0.0, 1.0])
    n = up / np.linalg.norm(up)
    if np.allclose(n, z): return np.eye(3)
    if np.allclose(n, -z): return np.diag([1.0, -1.0, -1.0])
    v = np.cross(n, z); s = np.linalg.norm(v); c = n.dot(z)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def cluster_per_label(sem_pts, sem_lbl, sem_names):
    canon_pts = {}
    for li, ln in enumerate(sem_names):
        c = canonicalize(ln)
        if c is None: continue
        m = sem_lbl == li
        if m.sum() == 0: continue
        canon_pts.setdefault(c, []).append(sem_pts[m])
    canon_pts = {k: np.concatenate(v) for k, v in canon_pts.items()}
    # Height filter and minimum count
    canon_pts = {k: v[(v[:, 2] > -0.05) & (v[:, 2] < 2.5)]
                 for k, v in canon_pts.items()}
    canon_pts = {k: v for k, v in canon_pts.items() if len(v) > 80}

    clusters = {}
    for k, pts in canon_pts.items():
        xy = pts[:, :2]
        if HAVE_SK and len(xy) > 100:
            db = DBSCAN(eps=0.25, min_samples=40, n_jobs=-1).fit(xy)
            lbs = db.labels_
            uniq, cts = np.unique(lbs[lbs >= 0], return_counts=True)
            if len(uniq):
                pts = pts[lbs == uniq[np.argmax(cts)]]
        xy = pts[:, :2]
        if len(xy) < 5: continue
        lo, hi = np.percentile(xy, [5, 95], axis=0)
        m = ((xy[:, 0] > lo[0]) & (xy[:, 0] < hi[0]) &
             (xy[:, 1] > lo[1]) & (xy[:, 1] < hi[1]))
        pts = pts[m]
        if len(pts) < 80: continue
        diag = float(np.linalg.norm(pts[:, :2].max(0) - pts[:, :2].min(0)))
        if diag > 4.0: continue
        clusters[k] = pts
    return clusters


def gt_bboxes_bev(json_path, drop_set=None):
    """Holi-Spatial AABB: per object 8 corner vertices in metres."""
    objs = json.loads(Path(json_path).read_text())
    rects = []; labels = []
    drop_set = drop_set or {"wall", "floor", "ceiling"}
    for o in objs:
        if o["label"] in drop_set:
            continue
        verts = np.array([[v["x"], v["y"], v["z"]] for v in o["bounding_box"]])
        # BEV = drop Z
        xy = verts[:, :2]
        # 8-corner AABB-like box's BEV = min/max in x and y
        xs = xy[:, 0]; ys = xy[:, 1]
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        rects.append(rect); labels.append(o["label"])
    return rects, labels


def render_scene(scene, ax_points, ax_bbox, ax_gt):
    pred_dir = PRED_ROOT / scene
    pts_arr = np.load(pred_dir / "points.npy")
    conf = np.load(pred_dir / "points_conf.npy")
    extr = np.load(pred_dir / "extrinsic.npy")
    images = np.load(pred_dir / "images.npy")
    sem = np.load(pred_dir / "semantic_points.npz", allow_pickle=True)
    sem_pts_raw = sem["points"]
    sem_lbl = sem["labels"]
    sem_names = list(sem["label_names"])

    # Build scene cloud
    scene_pts = pts_arr.reshape(-1, 3)
    scene_rgb = images.transpose(0, 2, 3, 1).reshape(-1, 3)
    cf = conf.reshape(-1)
    ok = np.isfinite(scene_pts).all(-1)
    scene_pts = scene_pts[ok]; scene_rgb = scene_rgb[ok]; cf = cf[ok]
    keep = cf > np.percentile(cf, 50)
    scene_pts = scene_pts[keep]; scene_rgb = scene_rgb[keep]

    # Cameras + gravity
    cams = []
    for ext in extr:
        R = ext[:, :3]; t = ext[:, 3]
        cams.append(-R.T @ t)
    cams = np.array(cams)
    up = gravity_from_cameras(cams, extr)
    R_align = alignment_from_up(up)
    scene_pts_a = (R_align @ scene_pts.T).T
    cams_a = (R_align @ cams.T).T
    sem_pts_a = (R_align @ sem_pts_raw.T).T
    z_floor = float(np.percentile(scene_pts_a[:, 2], 5))
    scene_pts_a -= [0, 0, z_floor]
    cams_a -= [0, 0, z_floor]
    sem_pts_a -= [0, 0, z_floor]

    clusters = cluster_per_label(sem_pts_a, sem_lbl, sem_names)

    pred_labels = sorted(clusters.keys())
    gt_rects, gt_labels = gt_bboxes_bev(GT_BBOX_DIR / f"{scene}.json")
    all_labels = sorted(set(pred_labels) | set(gt_labels))
    cmap = plt.cm.tab20
    color_of = {ln: cmap(i % 20) for i, ln in enumerate(all_labels)}

    # Per-panel limits (pred and GT live in different world frames + scales).
    pred_xs = []; pred_ys = []
    for pts in clusters.values():
        pred_xs.append(pts[:, 0]); pred_ys.append(pts[:, 1])
    pred_xs.extend([cams_a[:, 0]]); pred_ys.extend([cams_a[:, 1]])
    pred_xs = np.concatenate(pred_xs); pred_ys = np.concatenate(pred_ys)
    pad_p = 0.15
    xlim = (pred_xs.min() - pad_p, pred_xs.max() + pad_p)
    ylim = (pred_ys.min() - pad_p, pred_ys.max() + pad_p)

    # ---- Panel A: Points-only cognitive map ----
    rng = np.random.default_rng(0)
    bg_mask = (scene_pts_a[:, 2] > -0.1) & (scene_pts_a[:, 2] < 2.5)
    bg = scene_pts_a[bg_mask]; bg_rgb = scene_rgb[bg_mask]
    if len(bg) > 60000:
        sel = rng.choice(len(bg), 60000, replace=False)
        bg = bg[sel]; bg_rgb = bg_rgb[sel]
    ax_points.scatter(bg[:, 0], bg[:, 1], s=0.4, c=bg_rgb, alpha=0.18, zorder=1)
    for k, pts in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        xy = pts[:, :2]
        ax_points.scatter(xy[:, 0], xy[:, 1], s=4, color=color_of[k], alpha=0.75, zorder=2)
    for k, pts in clusters.items():
        c = pts[:, :2].mean(0)
        ax_points.text(c[0], c[1], k, fontsize=7, ha="center", va="center", weight="bold",
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color_of[k], lw=0.8,
                                 alpha=0.9), zorder=4)
    ax_points.plot(cams_a[:, 0], cams_a[:, 1], "k.-", lw=0.9, ms=3, alpha=0.55, zorder=3)
    ax_points.scatter([cams_a[-1, 0]], [cams_a[-1, 1]], s=110, marker="*",
                      c="red", edgecolor="black", zorder=5)
    ax_points.set_aspect("equal"); ax_points.grid(alpha=0.3, linestyle=":")
    ax_points.set_xlim(*xlim); ax_points.set_ylim(*ylim)
    ax_points.set_title(f"Predicted — points-only\n"
                        f"{sum(len(v) for v in clusters.values()):,} pts, "
                        f"{len(clusters)} labels", fontsize=10)
    ax_points.set_xlabel("X (m)"); ax_points.set_ylabel("Y (m)")

    # ---- Panel B: Bounding-box cognitive map ----
    for k, pts in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        xy = pts[:, :2]
        x0, x1 = xy[:, 0].min(), xy[:, 0].max()
        y0, y1 = xy[:, 1].min(), xy[:, 1].max()
        rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        clr = color_of[k]
        ax_bbox.add_patch(Polygon(rect, closed=True, facecolor=clr, edgecolor=clr,
                                  alpha=0.35, linewidth=1.6, zorder=3))
        cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
        ax_bbox.text(cx, cy, k, fontsize=8, ha="center", va="center", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=clr, lw=0.8,
                               alpha=0.9), zorder=4)
    ax_bbox.plot(cams_a[:, 0], cams_a[:, 1], "k.-", lw=0.9, ms=3, alpha=0.55, zorder=3)
    ax_bbox.scatter([cams_a[-1, 0]], [cams_a[-1, 1]], s=110, marker="*",
                    c="red", edgecolor="black", zorder=5)
    ax_bbox.set_aspect("equal"); ax_bbox.grid(alpha=0.3, linestyle=":")
    ax_bbox.set_xlim(*xlim); ax_bbox.set_ylim(*ylim)
    ax_bbox.set_title(f"Predicted — bounding boxes\n"
                      f"{len(clusters)} labelled objects", fontsize=10)
    ax_bbox.set_xlabel("X (m)"); ax_bbox.set_ylabel("Y (m)")

    # ---- Panel C: Ground truth ----
    seen = set()
    for rect, ln in zip(gt_rects, gt_labels):
        clr = color_of.get(ln, "0.5")
        leg = ln if ln not in seen else None; seen.add(ln)
        ax_gt.add_patch(Polygon(rect, closed=True, facecolor=clr, edgecolor=clr,
                                alpha=0.35, linewidth=1.5, zorder=3, label=leg))
        cx = float(rect[:, 0].mean()); cy = float(rect[:, 1].mean())
        ax_gt.text(cx, cy, ln, fontsize=7, ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=clr, lw=0.6, alpha=0.85),
                   zorder=4)
    ax_gt.set_aspect("equal"); ax_gt.grid(alpha=0.3, linestyle=":")
    # GT is in ScanNet axis-aligned frame; align extent visually by using same axis range
    # The GT frame and pred frame are NOT in the same world — show GT in its own frame
    gt_xs = np.concatenate([r[:, 0] for r in gt_rects])
    gt_ys = np.concatenate([r[:, 1] for r in gt_rects])
    pad2 = 0.5
    ax_gt.set_xlim(gt_xs.min() - pad2, gt_xs.max() + pad2)
    ax_gt.set_ylim(gt_ys.min() - pad2, gt_ys.max() + pad2)
    ax_gt.set_title(f"Ground truth (ScanNet, Holi-Spatial AABBs)\n"
                    f"{len(gt_rects)} GT objects, BEV X-Y", fontsize=10)
    ax_gt.set_xlabel("X (m)"); ax_gt.set_ylabel("Y (m)")

    # Save the PLY (semantic point cloud, gravity-aligned, canonical labels)
    save_semantic_ply(scene, sem_pts_a, sem_lbl, sem_names, scene_pts_a, scene_rgb)


def save_semantic_ply(scene, sem_pts_a, sem_lbl, sem_names, scene_pts_a, scene_rgb):
    ply_dir = PLY_ROOT / scene / "ply"; ply_dir.mkdir(parents=True, exist_ok=True)
    UNLABELED = 0xFFFF
    canon_ids = np.full(len(sem_lbl), UNLABELED, dtype=np.uint16)
    name_to_id = {n: i for i, n in enumerate(BASE_VOCAB)}
    for li, ln in enumerate(sem_names):
        c = canonicalize(ln)
        if c is None: continue
        canon_ids[sem_lbl == li] = name_to_id[c]
    valid = canon_ids != UNLABELED
    sem_pts = sem_pts_a[valid]
    sem_lbls = canon_ids[valid]
    cmap = plt.cm.tab20
    palette = np.array([np.array(cmap(i % 20)[:3]) * 255 for i in range(len(BASE_VOCAB))],
                       dtype=np.uint8)
    sem_color = palette[sem_lbls]

    rgb_u8 = (np.clip(scene_rgb, 0, 1) * 255).astype(np.uint8)

    _write_ply(ply_dir / "scene_full.ply",
               vertices=scene_pts_a.astype(np.float32), colors=rgb_u8)
    _write_ply(ply_dir / "scene_semantic.ply",
               vertices=sem_pts.astype(np.float32), colors=sem_color,
               extra=[("label_id", "uint16", sem_lbls)])
    # Combined
    all_pts = np.concatenate([scene_pts_a, sem_pts]).astype(np.float32)
    all_rgb = np.concatenate([rgb_u8, sem_color]).astype(np.uint8)
    is_sem = np.concatenate([np.zeros(len(scene_pts_a), dtype=np.uint8),
                              np.ones(len(sem_pts), dtype=np.uint8)])
    label_id_all = np.concatenate([np.full(len(scene_pts_a), UNLABELED, dtype=np.uint16),
                                    sem_lbls.astype(np.uint16)])
    _write_ply(ply_dir / "scene_combined.ply",
               vertices=all_pts, colors=all_rgb,
               extra=[("is_semantic", "uint8", is_sem),
                      ("label_id", "uint16", label_id_all)])
    (ply_dir / "labels.json").write_text(json.dumps({
        "id_to_label": {i: n for n, i in name_to_id.items()},
        "label_to_id": name_to_id,
        "unlabeled_id": UNLABELED,
        "coordinate_frame": "gravity-aligned, z=up, floor at z=0",
    }, indent=2))
    print(f"  PLYs -> {ply_dir}/")


def _write_ply(path, *, vertices, colors=None, extra=None):
    N = len(vertices); extra = extra or []
    type_pretty = {"uint8": "uchar", "uint16": "ushort", "float32": "float"}
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {N}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    for n, dt, _ in extra:
        header.append(f"property {type_pretty[dt]} {n}")
    header.append("end_header")
    rec_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        rec_fields += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    np_to_dt = {"uint8": "u1", "uint16": "<u2", "float32": "<f4"}
    for n, dt, _ in extra:
        rec_fields.append((n, np_to_dt[dt]))
    rec = np.zeros(N, dtype=rec_fields)
    rec["x"] = vertices[:, 0].astype(np.float32)
    rec["y"] = vertices[:, 1].astype(np.float32)
    rec["z"] = vertices[:, 2].astype(np.float32)
    if colors is not None:
        rec["r"] = colors[:, 0].astype(np.uint8)
        rec["g"] = colors[:, 1].astype(np.uint8)
        rec["b"] = colors[:, 2].astype(np.uint8)
    for n, dt, arr in extra:
        rec[n] = arr.astype({"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}[dt])
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(rec.tobytes())


def main():
    SCENES = ["scene0011_00", "scene0050_00", "scene0046_00"]
    for s in SCENES:
        print(f"\n=== Rendering {s} ===")
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        render_scene(s, axes[0], axes[1], axes[2])
        fig.suptitle(f"ScanNet {s} — predicted (points + bbox) vs ground truth",
                     fontsize=13)
        fig.tight_layout()
        out = FIG_DIR / f"{s}_compare.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
