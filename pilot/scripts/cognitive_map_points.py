"""
Render a points-only cognitive map (no bounding boxes) — just projected
semantic points colored by label, with one floating label per cluster.

Outputs:
  pilot/figures/cognitive_map_points.png            standalone
  pilot/figures/visual_prompt_three_panel_points.png   paper-Fig.9 style with
                                                       a points-only middle panel
"""
import json, re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METHOD = Path("outputs/vggt_22frames")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)


def canonicalize(name, vocab):
    words = re.findall(r"[a-z]+", name.lower())
    if "sofa" in words: return "couch"
    for w in words:
        if w in vocab: return w
    return None


def main():
    aligned = np.load(METHOD / "scene_aligned_pts.npz", allow_pickle=True)
    scene_pts = aligned["scene"]
    scene_rgb = aligned["scene_rgb"]
    cams = aligned["cams"]
    sem = aligned["semantic"]
    sem_lbl = aligned["sem_labels"]
    sem_names = list(aligned["sem_label_names"])

    base_vocab = ["chair", "couch", "table", "bed", "tv", "lamp", "pillow",
                  "blanket", "cabinet", "bookshelf", "rug", "window", "door",
                  "plant", "stove", "refrigerator", "oven", "microwave",
                  "sink", "toilet", "desk"]
    vocab_set = set(base_vocab)

    # Aggregate semantic points by canonical label, height-filtered to room (z in [-0.05, 2.5])
    canon = {ln: canonicalize(ln, vocab_set) for ln in sem_names}
    canon_pts = {}
    for li, ln in enumerate(sem_names):
        c = canon[ln]
        if c is None: continue
        m = sem_lbl == li
        if m.sum() == 0: continue
        canon_pts.setdefault(c, []).append(sem[m])
    canon_pts = {k: np.concatenate(v) for k, v in canon_pts.items()}
    canon_pts = {k: v[(v[:, 2] > -0.05) & (v[:, 2] < 2.5)]
                 for k, v in canon_pts.items()}
    canon_pts = {k: v for k, v in canon_pts.items() if len(v) > 80}

    # Per-label DBSCAN largest cluster + percentile trim (matches cognitive_map.py)
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

    cmap = plt.cm.tab20
    color_map = {k: cmap(i % 20) for i, k in enumerate(sorted(clusters.keys()))}

    # ---- Standalone: points-only cognitive map ----
    # Wider figure so the legend sits OUTSIDE the BEV plot to the right.
    fig, ax = plt.subplots(figsize=(11.5, 8))
    for k, pts in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        xy = pts[:, :2]
        ax.scatter(xy[:, 0], xy[:, 1], s=4, color=color_map[k], alpha=0.7,
                   label=f"{k} ({len(pts)})", zorder=2)
    # camera trajectory + current camera marker
    ax.plot(cams[:, 0], cams[:, 1], "k.-", lw=0.9, ms=3, alpha=0.55, zorder=3,
            label="camera trajectory")
    ax.scatter([cams[-1, 0]], [cams[-1, 1]], s=140, marker="*", c="red",
               edgecolor="black", zorder=4, label="current camera")
    # one label per cluster, placed at centroid
    for k, pts in clusters.items():
        c = pts[:, :2].mean(0)
        ax.text(c[0], c[1], k, fontsize=10, ha="center", va="center",
                weight="bold", color="black", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color_map[k], lw=1, alpha=0.85))
    ax.set_aspect("equal"); ax.grid(alpha=0.4, linestyle=":")
    ax.set_xlabel("X (m)  — gravity-aligned, floor at z=0")
    ax.set_ylabel("Y (m)")
    ax.set_title("Cognitive (BEV) semantic map — points only\n"
                 "Gravity-aligned, GroundingDINO+SAM lifted into VGGT 3D", fontsize=12)
    # Legend OUTSIDE the axes (upper-left of the figure-relative space at x=1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9, markerscale=2, frameon=True, borderaxespad=0.0)
    fig.tight_layout()
    out = FIG / "cognitive_map_points.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")

    # ---- Paper-Fig.9-style 3-panel with points-only middle ----
    rng = np.random.default_rng(0)
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                              gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]})

    # Panel 1: aligned point cloud BEV (RGB)
    ax = axes[0]
    keep_bg = (scene_pts[:, 2] > -0.1) & (scene_pts[:, 2] < 2.5)
    bg = scene_pts[keep_bg]; bg_rgb = scene_rgb[keep_bg]
    if len(bg) > 100000:
        sel = rng.choice(len(bg), 100000, replace=False)
        bg = bg[sel]; bg_rgb = bg_rgb[sel]
    ax.scatter(bg[:, 0], bg[:, 1], s=0.7, c=bg_rgb, alpha=0.45)
    ax.plot(cams[:, 0], cams[:, 1], "ro-", ms=3, lw=0.8, alpha=0.7, label="cameras")
    ax.set_aspect("equal"); ax.grid(alpha=0.3, linestyle=":")
    ax.set_title("Aligned 3D point cloud (BEV view)", fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.legend(loc="upper right", fontsize=8)
    xlim = ax.get_xlim(); ylim = ax.get_ylim()

    # Panel 2: semantic map — points only (no rectangles)
    ax = axes[1]
    for k, pts in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        xy = pts[:, :2]
        ax.scatter(xy[:, 0], xy[:, 1], s=4, color=color_map[k], alpha=0.75)
    for k, pts in clusters.items():
        c = pts[:, :2].mean(0)
        ax.text(c[0], c[1], k, fontsize=9, ha="center", va="center",
                weight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color_map[k], lw=1, alpha=0.9))
    ax.plot(cams[:, 0], cams[:, 1], "k.-", lw=0.9, ms=3, alpha=0.6)
    ax.scatter([cams[-1, 0]], [cams[-1, 1]], s=120, marker="*", c="red",
               edgecolor="black", label="current camera")
    ax.set_aspect("equal"); ax.grid(alpha=0.4, linestyle=":")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Semantic map — points only", fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3: text JSON
    ax = axes[2]; ax.axis("off")
    text_objs = []
    for k, pts in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        xy = pts[:, :2]
        c = xy.mean(0)
        text_objs.append((k, float(c[0]), float(c[1]),
                          float(xy[:, 0].max() - xy[:, 0].min()),
                          float(xy[:, 1].max() - xy[:, 1].min())))
    text = ["{", '  "current_camera_xy": [{:.2f}, {:.2f}],'.format(cams[-1, 0], cams[-1, 1]),
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

    fig2.suptitle("Visual prompt — points-only variant\n"
                  "Aligned 3D cloud  →  semantic map (points)  →  text",
                  fontsize=13)
    fig2.tight_layout()
    out = FIG / "visual_prompt_three_panel_points.png"
    fig2.savefig(out, dpi=140); plt.close(fig2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
