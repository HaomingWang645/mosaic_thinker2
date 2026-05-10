"""
Generate visualization-heavy figures for the reconstruction-quality report.

Outputs (all saved to figures/quality/):
  A_inputs.png            — 6 representative input frames
  B_depthmaps.png         — VGGT predicted depth for the same 6 frames
  C_per_frame_clouds.png  — per-frame 3D point clouds (RGB), 3 views each, 4 frames
  D_merged_bev.png        — RGB-colored merged BEV (high res)
  E_confidence_overlay.png — input frame + confidence heatmap + masked-low-conf
  F_consistency_overlay.png — two frames' matched-pixel 3D points, color-coded
  G_baseline_vs_vggt_bev.png — BEV side-by-side at the same scale
  H_plane_residuals.png   — patches colored by local-plane residual
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from scipy.spatial import cKDTree


FIG_DIR = Path("figures/quality"); FIG_DIR.mkdir(parents=True, exist_ok=True)
METHOD = Path("outputs/vggt_22frames")
BASELINE = Path("outputs/baseline_22frames")
FRAMES = Path("/home/haoming/mosaic_thinker/frames")


def load(method):
    return {
        "pts": np.load(method / "points.npy"),                 # (N, H, W, 3)
        "depth": np.load(method / "depth.npy"),                # (N, H, W)
        "names": json.loads((method / "frame_names.json").read_text()),
        "imgs": np.load(method / "images.npy") if (method / "images.npy").exists() else None,
        "conf": np.load(method / "points_conf.npy") if (method / "points_conf.npy").exists() else None,
    }


def repr_frame_idx(names, k=6):
    """Pick k frames evenly spread across the sequence."""
    N = len(names)
    return [int(round(i * (N - 1) / (k - 1))) for i in range(k)]


def figA_inputs():
    names = json.loads((METHOD / "frame_names.json").read_text())
    idx = repr_frame_idx(names, 6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, i in zip(axes.flat, idx):
        img = Image.open(FRAMES / names[i])
        ax.imshow(img); ax.set_title(names[i].replace(".png", ""), fontsize=10)
        ax.axis("off")
    fig.suptitle("Input frames (6 of 22) from the test scene", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR / "A_inputs.png", dpi=130)
    plt.close(fig)
    print("A_inputs.png")


def figB_depthmaps():
    d = load(METHOD)
    idx = repr_frame_idx(d["names"], 6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    vmax = float(np.percentile(d["depth"][np.isfinite(d["depth"])], 95))
    for ax, i in zip(axes.flat, idx):
        im = ax.imshow(d["depth"][i], cmap="turbo", vmin=0, vmax=vmax)
        ax.set_title(d["names"][i].replace(".png", ""), fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="depth (m)")
    fig.suptitle("VGGT predicted depth maps (matched to figure A)", fontsize=12)
    fig.savefig(FIG_DIR / "B_depthmaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("B_depthmaps.png")


def per_frame_rgb(pts_frame, img_chw=None, name=None):
    """Return (M, 3) points and (M, 3) colors for valid pixels in this frame."""
    H, W, _ = pts_frame.shape
    if img_chw is not None:
        rgb = img_chw.transpose(1, 2, 0).reshape(-1, 3)
    else:
        # Load from disk and resize to (W, H)
        img = Image.open(FRAMES / name).convert("RGB").resize((W, H))
        rgb = (np.asarray(img).astype(np.float32) / 255.0).reshape(-1, 3)
    flat_pts = pts_frame.reshape(-1, 3)
    ok = np.isfinite(flat_pts).all(-1)
    return flat_pts[ok], np.clip(rgb[ok], 0, 1)


def figC_per_frame_clouds():
    d = load(METHOD)
    pick = repr_frame_idx(d["names"], 4)
    fig, axes = plt.subplots(len(pick), 3, figsize=(12, 3.4 * len(pick)))
    for r, i in enumerate(pick):
        p, c = per_frame_rgb(d["pts"][i], d["imgs"][i])
        # subsample
        rng = np.random.default_rng(i)
        if len(p) > 30000:
            sel = rng.choice(len(p), 30000, replace=False)
            p = p[sel]; c = c[sel]
        for k, (a, b, title) in enumerate([(0, 1, "Top (XY)"), (0, 2, "Front (XZ)"), (1, 2, "Side (YZ)")]):
            ax = axes[r, k]
            ax.scatter(p[:, a], p[:, b], s=0.8, c=c, alpha=0.5)
            ax.set_aspect("equal"); ax.grid(alpha=0.3)
            if r == 0: ax.set_title(title, fontsize=10)
            if k == 0:
                ax.set_ylabel(d["names"][i].replace(".png", ""), fontsize=10)
    fig.suptitle("Per-frame 3D point cloud from VGGT (RGB-colored, 30 k pts/frame)\n"
                 "Each row is one frame, columns are top / front / side views", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR / "C_per_frame_clouds.png", dpi=130)
    plt.close(fig)
    print("C_per_frame_clouds.png")


def figD_merged_bev(method=METHOD, name="D_merged_bev.png", title=None,
                    confidence_filter=True):
    d = load(method)
    rng = np.random.default_rng(0)
    pts_all = []; col_all = []; conf_all = []
    for i in range(len(d["names"])):
        p, c = per_frame_rgb(d["pts"][i], d["imgs"][i])
        if d["conf"] is not None and confidence_filter:
            cf = d["conf"][i].flatten()
            valid = np.isfinite(d["pts"][i].reshape(-1, 3)).all(-1)
            cf = cf[valid]
        else:
            cf = np.ones(len(p))
        pts_all.append(p); col_all.append(c); conf_all.append(cf)
    pts = np.concatenate(pts_all)
    col = np.concatenate(col_all)
    conf = np.concatenate(conf_all)

    if confidence_filter and len(conf) > 0:
        thresh = float(np.percentile(conf, 50))   # keep top 50%
        keep = conf > thresh
        pts = pts[keep]; col = col[keep]
    sel = rng.choice(len(pts), min(150000, len(pts)), replace=False)
    pts = pts[sel]; col = col[sel]

    # Determine 'height' axis as the one with smallest variance after light filtering
    var = pts.var(0); h = int(var.argmin())
    a, b = [x for x in [0, 1, 2] if x != h]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(pts[:, a], pts[:, b], s=1.5, c=col, alpha=0.55)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel(f"axis {a} (m)"); ax.set_ylabel(f"axis {b} (m)")
    ax.set_title(title or f"BEV — merged RGB point cloud ({method.name}, top-50% conf, "
                          f"{len(pts):,} pts shown)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR / name, dpi=140)
    plt.close(fig)
    print(name)


def figE_confidence_overlay():
    d = load(METHOD)
    idx = repr_frame_idx(d["names"], 3)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for r, i in enumerate(idx):
        img = Image.open(FRAMES / d["names"][i]).convert("RGB")
        Hp, Wp = d["depth"].shape[1:]
        img_resized = img.resize((Wp, Hp))
        axes[r, 0].imshow(img_resized); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"Input — {d['names'][i].replace('.png', '')}", fontsize=10)

        cf = d["conf"][i]
        im = axes[r, 1].imshow(cf, cmap="viridis", vmin=0, vmax=10)
        axes[r, 1].axis("off")
        axes[r, 1].set_title("VGGT confidence", fontsize=10)
        plt.colorbar(im, ax=axes[r, 1], fraction=0.045)

        # Mask low-confidence pixels in red
        cf_thresh = np.percentile(cf, 20)
        mask = (cf < cf_thresh).astype(float)
        overlay = np.array(img_resized) / 255.0
        overlay[..., 0] = overlay[..., 0] * (1 - mask) + 1.0 * mask
        overlay[..., 1:] = overlay[..., 1:] * (1 - mask[..., None])
        axes[r, 2].imshow(overlay); axes[r, 2].axis("off")
        axes[r, 2].set_title(f"Bottom-20%-conf pixels (red)", fontsize=10)
    fig.suptitle("VGGT per-pixel confidence — high everywhere, low on featureless walls / windows", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR / "E_confidence_overlay.png", dpi=130)
    plt.close(fig)
    print("E_confidence_overlay.png")


def figF_consistency_overlay():
    """Pick two frames, lift their RoMa-matched points into 3D under each
    method, draw the matched 3D points and the connecting segments to
    visualize cross-view alignment quality."""
    pair_names = ("frame_0067.png", "frame_0117.png")  # broadly different views
    d_v = load(METHOD)
    d_b = load(BASELINE)

    i_v = d_v["names"].index(pair_names[0]); j_v = d_v["names"].index(pair_names[1])
    i_b = d_b["names"].index(pair_names[0]); j_b = d_b["names"].index(pair_names[1])

    img_i = Image.open(FRAMES / pair_names[0])
    img_j = Image.open(FRAMES / pair_names[1])

    # Load pre-cached RoMa matches (run separately on GPU).
    cached = np.load("outputs/cached_matches_67_117.npz")
    kpi = cached["kpi"]; kpj = cached["kpj"]; cert = cached["cert"]
    keep = cert > 0.5
    kpi = kpi[keep]; kpj = kpj[keep]

    def lift(method, fi_idx, fj_idx, kpi, kpj):
        Hp, Wp = method["pts"].shape[1:3]
        Hi, Wi = img_i.height, img_i.width
        Hj, Wj = img_j.height, img_j.width
        sxi, syi = Wp / Wi, Hp / Hi
        sxj, syj = Wp / Wj, Hp / Hj
        xi = np.clip(np.rint(kpi[:, 0] * sxi).astype(int), 0, Wp - 1)
        yi = np.clip(np.rint(kpi[:, 1] * syi).astype(int), 0, Hp - 1)
        xj = np.clip(np.rint(kpj[:, 0] * sxj).astype(int), 0, Wp - 1)
        yj = np.clip(np.rint(kpj[:, 1] * syj).astype(int), 0, Hp - 1)
        pi = method["pts"][fi_idx][yi, xi]
        pj = method["pts"][fj_idx][yj, xj]
        ok = np.isfinite(pi).all(-1) & np.isfinite(pj).all(-1)
        return pi[ok], pj[ok]

    pi_v, pj_v = lift(d_v, i_v, j_v, kpi, kpj)
    pi_b, pj_b = lift(d_b, i_b, j_b, kpi, kpj)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (pi, pj), title in [
        (axes[0], (pi_v, pj_v), f"VGGT  (median Δ = {np.median(np.linalg.norm(pi_v-pj_v, axis=1))*1000:.1f} mm)"),
        (axes[1], (pi_b, pj_b), f"Baseline  (median Δ = {np.median(np.linalg.norm(pi_b-pj_b, axis=1))*1000:.1f} mm)"),
    ]:
        # height axis = lowest variance among union
        u = np.vstack([pi, pj])
        h = int(u.var(0).argmin())
        a, b = [x for x in [0, 1, 2] if x != h]
        for k in range(min(80, len(pi))):
            ax.plot([pi[k, a], pj[k, a]], [pi[k, b], pj[k, b]], "k-", lw=0.5, alpha=0.3)
        ax.scatter(pi[:, a], pi[:, b], s=12, c="C0", label=pair_names[0], alpha=0.7)
        ax.scatter(pj[:, a], pj[:, b], s=12, c="C3", label=pair_names[1], alpha=0.7)
        ax.set_aspect("equal"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"axis {a} (m)"); ax.set_ylabel(f"axis {b} (m)")
    fig.suptitle("Cross-view alignment of 400 RoMa-matched points\n"
                 "Each black line connects a pair of pixels that should map to the same world point", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR / "F_consistency_overlay.png", dpi=130)
    plt.close(fig)
    print("F_consistency_overlay.png")


def figG_baseline_vs_vggt_bev():
    """Plot both methods' merged clouds at COMPATIBLE scales, side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    rng = np.random.default_rng(0)
    for ax, (method, label) in zip(axes, [(METHOD, "VGGT"), (BASELINE, "Baseline (DepthPro+RoMa+MST)")]):
        d = load(method)
        pts_all, col_all, conf_all = [], [], []
        for i in range(len(d["names"])):
            img_chw = d["imgs"][i] if d["imgs"] is not None else None
            p, c = per_frame_rgb(d["pts"][i], img_chw, name=d["names"][i])
            pts_all.append(p); col_all.append(c)
            if d["conf"] is not None:
                cf = d["conf"][i].flatten()
                valid = np.isfinite(d["pts"][i].reshape(-1, 3)).all(-1)
                conf_all.append(cf[valid])
            else:
                conf_all.append(np.ones(len(p)))
        pts = np.concatenate(pts_all); col = np.concatenate(col_all); conf = np.concatenate(conf_all)
        if d["conf"] is not None:
            thr = float(np.percentile(conf, 50))
            keep = conf > thr
            pts = pts[keep]; col = col[keep]
        sel = rng.choice(len(pts), min(120000, len(pts)), replace=False)
        pts = pts[sel]; col = col[sel]
        var = pts.var(0); h = int(var.argmin())
        a, b = [x for x in [0, 1, 2] if x != h]
        ax.scatter(pts[:, a], pts[:, b], s=1.2, c=col, alpha=0.5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        bb_a = pts[:, a].max() - pts[:, a].min()
        bb_b = pts[:, b].max() - pts[:, b].min()
        ax.set_title(f"{label}\nBEV bbox: {bb_a:.1f} m × {bb_b:.1f} m, {len(pts):,} pts shown", fontsize=11)
        ax.set_xlabel(f"axis {a} (m)"); ax.set_ylabel(f"axis {b} (m)")
    fig.suptitle("BEV merged RGB point clouds — same 22 frames, two reconstruction methods", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG_DIR / "G_baseline_vs_vggt_bev.png", dpi=140)
    plt.close(fig)
    print("G_baseline_vs_vggt_bev.png")


def figH_plane_residuals():
    """Color the high-conf cloud by local-plane residual."""
    d = load(METHOD)
    flat_conf = d["conf"].flatten(); flat_conf = flat_conf[np.isfinite(flat_conf)]
    thr = float(np.percentile(flat_conf, 80))
    pts_all, col_all = [], []
    for i in range(len(d["names"])):
        c = d["conf"][i]; p = d["pts"][i]
        mask = (c > thr) & np.isfinite(p).all(-1)
        s = p[mask]
        rgb = d["imgs"][i].transpose(1, 2, 0)[mask]
        if len(s):
            pts_all.append(s); col_all.append(rgb)
    pts = np.concatenate(pts_all); col = np.clip(np.concatenate(col_all), 0, 1)

    rng = np.random.default_rng(0)
    sel = rng.choice(len(pts), min(60000, len(pts)), replace=False)
    pts_sub = pts[sel]; col_sub = col[sel]
    tree = cKDTree(pts)
    K = 32
    res = []
    for q in pts_sub:
        _, idx = tree.query(q, k=K)
        nb = pts[idx]
        c = nb.mean(0)
        cov = np.cov((nb - c).T)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        d_ = (nb - c) @ normal
        res.append(np.std(d_))
    res = np.array(res)
    var = pts_sub.var(0); h = int(var.argmin())
    a, b = [x for x in [0, 1, 2] if x != h]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    sc = axes[0].scatter(pts_sub[:, a], pts_sub[:, b], s=1.5, c=res * 1000,
                         cmap="turbo", vmin=0, vmax=5, alpha=0.7)
    axes[0].set_aspect("equal"); axes[0].grid(alpha=0.3)
    axes[0].set_title("Local-plane residual (mm) per point", fontsize=11)
    axes[0].set_xlabel(f"axis {a} (m)"); axes[0].set_ylabel(f"axis {b} (m)")
    plt.colorbar(sc, ax=axes[0], label="residual (mm)")

    axes[1].scatter(pts_sub[:, a], pts_sub[:, b], s=1.5, c=col_sub, alpha=0.6)
    axes[1].set_aspect("equal"); axes[1].grid(alpha=0.3)
    axes[1].set_title("Same points, RGB-colored", fontsize=11)
    axes[1].set_xlabel(f"axis {a} (m)"); axes[1].set_ylabel(f"axis {b} (m)")

    fig.suptitle(f"Surface noise across the scene — median {np.median(res)*1000:.1f} mm, "
                 f"p90 {np.percentile(res, 90)*1000:.1f} mm  (top-20%-conf points)", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG_DIR / "H_plane_residuals.png", dpi=140)
    plt.close(fig)
    print("H_plane_residuals.png")


if __name__ == "__main__":
    figA_inputs()
    figB_depthmaps()
    figC_per_frame_clouds()
    figD_merged_bev()
    figE_confidence_overlay()
    figF_consistency_overlay()
    figG_baseline_vs_vggt_bev()
    figH_plane_residuals()
    print("\nAll figures saved to", FIG_DIR)
