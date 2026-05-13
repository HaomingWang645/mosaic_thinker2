"""Side-by-side qualitative + quantitative figures for the report."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_semantic(method_dir: Path):
    d = np.load(method_dir / "semantic_points.npz", allow_pickle=True)
    return d["points"], d["labels"], list(d["label_names"])


def main():
    out = Path("figures"); out.mkdir(exist_ok=True)

    vggt = load_semantic(Path("outputs/vggt_22frames"))
    bsl = load_semantic(Path("outputs/baseline_22frames"))

    # Pick top-6 most populous labels common to both, by VGGT count.
    counts = {ln: int((vggt[1] == i).sum()) for i, ln in enumerate(vggt[2])}
    top = [ln for ln, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:6]]
    print(f"Top labels: {top}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cmap = plt.cm.tab10
    for ax, (pts, lbls, lns), title in [
        (axes[0], vggt, "VGGT pipeline (proposed)"),
        (axes[1], bsl, "DepthPro + RoMa + MST baseline"),
    ]:
        # Pick height axis = lowest variance
        var = pts.var(0); h = int(var.argmin())
        plane = [a for a in [0, 1, 2] if a != h]
        for ci, ln in enumerate(top):
            if ln not in lns: continue
            li = lns.index(ln)
            m = lbls == li
            if m.sum() == 0: continue
            ax.scatter(pts[m][:, plane[0]], pts[m][:, plane[1]], s=1.5, alpha=0.4,
                       color=cmap(ci), label=ln)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(f"axis {plane[0]} (m)"); ax.set_ylabel(f"axis {plane[1]} (m)")
        ax.legend(loc="upper right", markerscale=4, fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("BEV semantic map from the same 22-frame scene\n"
                 "(GroundingDINO + SAM, points lifted via each method's per-pixel 3D)")
    fig.tight_layout()
    fig.savefig(out / "bev_side_by_side.png", dpi=140)
    print("Saved", out / "bev_side_by_side.png")

    # Per-frame extents bar chart
    fig, ax = plt.subplots(figsize=(11, 4.2))
    n = 22
    x = np.arange(n)
    for tag, color, label in [("vggt_22frames", "C0", "VGGT"),
                              ("baseline_22frames", "C3", "Baseline")]:
        pts = np.load(f"outputs/{tag}/points.npy")
        ext = []
        for i in range(n):
            p = pts[i].reshape(-1, 3)
            ok = np.isfinite(p).all(-1)
            p = p[ok]
            ext.append(np.linalg.norm(p.max(0) - p.min(0)) if len(p) else 0.0)
        offset = -0.2 if tag.startswith("vggt") else 0.2
        ax.bar(x + offset, ext, 0.4, color=color, label=label)
    names = json.loads(open("outputs/vggt_22frames/frame_names.json").read())
    ax.set_xticks(x); ax.set_xticklabels([n.replace("frame_", "").replace(".png", "") for n in names], rotation=45, fontsize=7)
    ax.set_ylabel("Per-frame point-cloud diagonal (m)")
    ax.set_xlabel("Frame index")
    ax.set_title("Per-frame reconstruction extent — baseline collapses on most frames "
                 "due to monocular-depth scale ambiguity")
    ax.axhline(0.1, color="k", ls=":", lw=0.8, label="collapse threshold (0.1 m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "per_frame_extents.png", dpi=140)
    print("Saved", out / "per_frame_extents.png")

    # Latency comparison
    bsl_t = json.loads(open("outputs/baseline_22frames/timing.json").read())
    vggt_t = json.loads(open("outputs/vggt_22frames/timing.json").read())
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = ["VGGT (one fwd pass)", "Baseline:\nDepthPro+RoMa+MST"]
    stages_v = {"VGGT forward": vggt_t["forward_s"]}
    stages_b = {
        "DepthPro": bsl_t["depth_total_s"],
        "RoMa pairwise": bsl_t["roma_total_s"],
        "Other (CLIP, lift, ...)": bsl_t.get("clip_s", 0) + bsl_t.get("chain_and_lift_s", 0),
    }
    bottom = [0, 0]
    cmap2 = plt.cm.Set2
    ci = 0
    for stage, t in stages_v.items():
        ax.bar([methods[0]], [t], bottom=[bottom[0]], color=cmap2(ci), label=stage)
        bottom[0] += t; ci += 1
    for stage, t in stages_b.items():
        ax.bar([methods[1]], [t], bottom=[bottom[1]], color=cmap2(ci), label=stage)
        bottom[1] += t; ci += 1
    ax.set_ylabel("Reconstruction latency (s)")
    ax.set_title(f"22-frame scene, single H100 — VGGT: {vggt_t['forward_s']:.2f}s vs "
                 f"baseline: {bsl_t['total_s']:.1f}s")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "latency_breakdown.png", dpi=140)
    print("Saved", out / "latency_breakdown.png")


if __name__ == "__main__":
    main()
