"""
Deeper VGGT reconstruction-quality check beyond multi-view consistency.

Without external ground truth, we use four reference-free probes:

  1. Camera trajectory smoothness — real cameras move smoothly. Jitter in
     consecutive-frame translations / rotations is a noise indicator.
  2. VGGT's own per-point confidence (`world_points_conf`) — head was trained
     to predict its own reliability; low average confidence flags a poor scene.
  3. Surface thickness — for points lying on a small patch of a flat surface
     (e.g. floor, wall), compute median residual to a locally-fit plane.
     Surface noise lower-bounds reconstruction precision.
  4. Visual quality — render the merged point cloud as a 3-view (BEV / side /
     front) projection to look for ghosting / drift / streaks.

Saves quality_metrics.json + four PNG figures.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    method_dir = Path("outputs/vggt_22frames")
    out = Path("figures"); out.mkdir(exist_ok=True)

    pts = np.load(method_dir / "points.npy")          # (N, H, W, 3)
    conf = np.load(method_dir / "points_conf.npy")    # (N, H, W)
    extr = np.load(method_dir / "extrinsic.npy")      # (N, 3, 4) world->cam
    intr = np.load(method_dir / "intrinsic.npy")      # (N, 3, 3)
    images = np.load(method_dir / "images.npy")       # (N, 3, H, W)
    names = json.loads((method_dir / "frame_names.json").read_text())
    N = len(names)

    metrics = {}

    # ---- 1. Camera trajectory smoothness ----
    # extr is world->cam, so cam center = -R^T @ t
    centers = []
    rotmats = []
    for i in range(N):
        R = extr[i, :, :3]; t = extr[i, :, 3]
        c = -R.T @ t
        centers.append(c); rotmats.append(R)
    centers = np.array(centers)
    deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)  # (N-1,)
    # Smoothness = ratio of step-to-step jitter to mean step
    rotmag = []
    for i in range(N - 1):
        Rrel = rotmats[i + 1] @ rotmats[i].T
        cos = (np.trace(Rrel) - 1) / 2
        cos = np.clip(cos, -1, 1)
        rotmag.append(np.degrees(np.arccos(cos)))
    rotmag = np.array(rotmag)
    metrics["trajectory"] = {
        "n_frames": int(N),
        "translation_step_m": {
            "mean": float(deltas.mean()), "median": float(np.median(deltas)),
            "max": float(deltas.max()), "p90": float(np.percentile(deltas, 90)),
        },
        "rotation_step_deg": {
            "mean": float(rotmag.mean()), "median": float(np.median(rotmag)),
            "max": float(rotmag.max()), "p90": float(np.percentile(rotmag, 90)),
        },
        "trajectory_total_length_m": float(deltas.sum()),
    }
    print("Camera trajectory:")
    print(f"  per-step trans (m): mean={deltas.mean():.3f}, median={np.median(deltas):.3f}, max={deltas.max():.3f}")
    print(f"  per-step rot (deg): mean={rotmag.mean():.1f}, median={np.median(rotmag):.1f}, max={rotmag.max():.1f}")
    print(f"  total path length: {deltas.sum():.2f} m")

    # ---- 2. VGGT confidence ----
    flat_conf = conf.flatten()
    flat_conf = flat_conf[np.isfinite(flat_conf)]
    metrics["confidence"] = {
        "mean": float(flat_conf.mean()),
        "median": float(np.median(flat_conf)),
        "p10": float(np.percentile(flat_conf, 10)),
        "p90": float(np.percentile(flat_conf, 90)),
        "frac_above_0.5": float((flat_conf > 0.5).mean()),
        "frac_above_1.0": float((flat_conf > 1.0).mean()),
    }
    print(f"\nVGGT self-confidence: mean={flat_conf.mean():.3f}, "
          f"median={np.median(flat_conf):.3f}, "
          f"frac > 1.0 = {(flat_conf > 1.0).mean()*100:.1f}%")

    # ---- 3. Surface thickness via local-plane residuals ----
    # Pool all high-confidence points across frames, then sample patches.
    rng = np.random.default_rng(0)
    thresh = float(np.percentile(flat_conf, 80))
    high_pts = []
    for i in range(N):
        c = conf[i]; p = pts[i]
        mask = (c > thresh) & np.isfinite(p).all(-1)
        s = p[mask]
        if len(s) > 0:
            high_pts.append(s)
    high_pts = np.concatenate(high_pts)
    print(f"\nHigh-confidence points (top-20%): {len(high_pts):,} (conf > {thresh:.3f})")

    # Sample 200 random anchor points; fit a local plane to k nearest neighbors;
    # report residual.
    from scipy.spatial import cKDTree
    tree = cKDTree(high_pts)
    n_anchors = min(200, len(high_pts) // 100)
    anchor_idx = rng.choice(len(high_pts), n_anchors, replace=False)
    residuals = []
    K = 32
    for ai in anchor_idx:
        _, idx = tree.query(high_pts[ai], k=K)
        nb = high_pts[idx]
        if len(nb) < 5: continue
        c = nb.mean(0)
        cov = np.cov((nb - c).T)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]   # smallest eigenvalue direction
        d = (nb - c) @ normal
        residuals.append(float(np.std(d)))
    metrics["local_plane_residual_m"] = {
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "p90": float(np.percentile(residuals, 90)),
        "n_patches": len(residuals),
    }
    print(f"Local-plane residual std (surface thickness):")
    print(f"  mean={np.mean(residuals):.4f} m, median={np.median(residuals):.4f} m, "
          f"p90={np.percentile(residuals, 90):.4f} m  (over {len(residuals)} patches)")

    # ---- 4. Visualizations ----
    # 4a. Camera trajectory + per-frame extents.
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "3d"})
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "b.-", lw=1, ms=4)
    for i, n in enumerate(names):
        if i % 4 == 0:
            ax.text(centers[i, 0], centers[i, 1], centers[i, 2],
                    n.replace("frame_", "").replace(".png", ""), fontsize=6)
    ax.set_title(f"Predicted camera trajectory (VGGT)\n"
                 f"path length {deltas.sum():.2f} m, "
                 f"per-step median {np.median(deltas)*100:.1f} cm")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    fig.tight_layout(); fig.savefig(out / "camera_trajectory.png", dpi=130)
    print(f"\nSaved {out / 'camera_trajectory.png'}")

    # 4b. Three orthographic projections of the merged high-confidence cloud,
    # subsampled and colored by RGB.
    # Build a colored point cloud
    rgb_imgs = images.transpose(0, 2, 3, 1)   # (N, H, W, 3) in [0,1]
    sample = rng.choice(len(high_pts), min(80_000, len(high_pts)), replace=False)
    high_pts_s = high_pts[sample]
    # Reconstruct colors by re-traversing
    colors_full = []
    for i in range(N):
        c = conf[i]
        mask = (c > thresh) & np.isfinite(pts[i]).all(-1)
        rgb = rgb_imgs[i][mask]
        colors_full.append(rgb)
    colors_full = np.concatenate(colors_full)
    colors_s = np.clip(colors_full[sample], 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    titles = ["Top-down (XY)", "Front (XZ)", "Side (YZ)"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (a, b), title in zip(axes, pairs, titles):
        ax.scatter(high_pts_s[:, a], high_pts_s[:, b], s=0.5, c=colors_s, alpha=0.5)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.set_xlabel(f"axis {a} (m)"); ax.set_ylabel(f"axis {b} (m)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"VGGT point cloud, top-20% confidence, RGB-colored "
                 f"({len(high_pts_s):,} pts shown)")
    fig.tight_layout(); fig.savefig(out / "point_cloud_three_views.png", dpi=130)
    print(f"Saved {out / 'point_cloud_three_views.png'}")

    # 4c. Confidence histogram + per-frame mean confidence
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(np.log10(np.clip(flat_conf, 1e-3, None)), bins=80, color="C0")
    axes[0].set_title("VGGT confidence distribution (log10)")
    axes[0].set_xlabel("log10(confidence)"); axes[0].set_ylabel("# pixels")
    per_frame_conf = [float(conf[i].mean()) for i in range(N)]
    axes[1].bar(range(N), per_frame_conf, color="C0")
    axes[1].set_xticks(range(N))
    axes[1].set_xticklabels([n.replace("frame_", "").replace(".png", "")
                             for n in names], rotation=45, fontsize=7)
    axes[1].set_title("Per-frame mean confidence")
    axes[1].set_ylabel("mean confidence")
    fig.tight_layout(); fig.savefig(out / "vggt_confidence.png", dpi=130)
    print(f"Saved {out / 'vggt_confidence.png'}")

    # 4d. Local-plane residual histogram
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.array(residuals) * 1000, bins=40, color="C2")
    ax.set_xlabel("Surface thickness (mm)")
    ax.set_ylabel("# patches")
    ax.set_title(f"Local-plane fit residual on top-20%-conf points\n"
                 f"median {np.median(residuals)*1000:.1f} mm, p90 {np.percentile(residuals, 90)*1000:.1f} mm")
    fig.tight_layout(); fig.savefig(out / "surface_thickness.png", dpi=130)
    print(f"Saved {out / 'surface_thickness.png'}")

    Path("outputs/vggt_22frames/quality_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved outputs/vggt_22frames/quality_metrics.json")


if __name__ == "__main__":
    main()
