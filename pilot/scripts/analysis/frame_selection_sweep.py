"""
Test how VGGT reacts to different frame-selection strategies.

Configurations:
  A) FULL-22:    all 22 frames
  B) UNIFORM-12: every 2nd frame (~ 12 frames)
  C) UNIFORM-6:  every 4th frame (~ 6 frames)
  D) UNIFORM-3:  3 frames
  E) MST-leaves: drop the 5 frames whose CLIP-similarity-aggregated centrality
                 is lowest (i.e. visually-isolated frames the paper's MST
                 strategy would push to leaves and treat as low-priority)
  F) MST-anchor: keep only the 3 frames the paper's strategy ranks most-central
                 (the "anchor"-style tight selection)

For each subset we record:
  - VGGT forward latency
  - per-frame extent stats (sanity)
  - multi-view consistency on RoMa-matched pairs that exist within the subset

Saves: outputs/frame_sweep/sweep_results.json + plots
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def load_clip_sim(frames_dir, names, device):
    images = [Image.open(frames_dir / n).convert("RGB") for n in names]
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inp = clip_proc(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = F.normalize(clip_model.get_image_features(**inp), dim=-1).cpu().numpy()
    sim = feat @ feat.T
    return sim


def run_vggt(frames_dir, subset_names, device):
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    paths = [str(frames_dir / n) for n in subset_names]
    if not hasattr(run_vggt, "_model"):
        run_vggt._model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    model = run_vggt._model
    images = load_and_preprocess_images(paths).to(device)
    dtype = torch.bfloat16
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        preds = model(images)
    torch.cuda.synchronize()
    fwd = time.time() - t0
    pts = preds["world_points"][0].cpu().numpy()  # (N, H, W, 3)
    return pts, fwd


def per_frame_extents(pts):
    extents = []
    for i in range(pts.shape[0]):
        p = pts[i].reshape(-1, 3)
        ok = np.isfinite(p).all(-1)
        p = p[ok]
        if len(p) == 0:
            extents.append(0.0); continue
        extents.append(float(np.linalg.norm(p.max(0) - p.min(0))))
    return np.array(extents)


def consistency_on_pairs(pts, subset_names, frames_dir, device, n_pairs=8, samples=128):
    from romatch import roma_indoor
    if not hasattr(consistency_on_pairs, "_roma"):
        consistency_on_pairs._roma = roma_indoor(device="cuda" if device.type == "cuda" else "cpu")
    roma = consistency_on_pairs._roma
    N = len(subset_names)
    if N < 2:
        return None
    rng = np.random.default_rng(0)
    pairs = []
    for k in range(N - 1):
        pairs.append((k, k + 1))
    while len(pairs) < n_pairs:
        a, b = rng.integers(0, N, 2)
        if a != b:
            pairs.append((min(a, b), max(a, b)))
    pairs = pairs[:n_pairs]

    Hp, Wp = pts.shape[1], pts.shape[2]
    medians = []
    for i, j in pairs:
        path_i = frames_dir / subset_names[i]
        path_j = frames_dir / subset_names[j]
        with torch.no_grad():
            warp, cert = roma.match(str(path_i), str(path_j), device=device)
            matches, cert = roma.sample(warp, cert, num=samples)
            img_i = Image.open(path_i); img_j = Image.open(path_j)
            kpi, kpj = roma.to_pixel_coordinates(
                matches, img_i.height, img_i.width, img_j.height, img_j.width
            )
            kpi = kpi.cpu().numpy(); kpj = kpj.cpu().numpy()
            cert = cert.cpu().numpy()
        keep = cert > 0.5
        kpi = kpi[keep]; kpj = kpj[keep]
        if len(kpi) < 5: continue
        sx_i = Wp / img_i.width; sy_i = Hp / img_i.height
        sx_j = Wp / img_j.width; sy_j = Hp / img_j.height
        xi = np.clip(np.rint(kpi[:, 0] * sx_i).astype(int), 0, Wp - 1)
        yi = np.clip(np.rint(kpi[:, 1] * sy_i).astype(int), 0, Hp - 1)
        xj = np.clip(np.rint(kpj[:, 0] * sx_j).astype(int), 0, Wp - 1)
        yj = np.clip(np.rint(kpj[:, 1] * sy_j).astype(int), 0, Hp - 1)
        pi = pts[i][yi, xi]; pj = pts[j][yj, xj]
        ok = np.isfinite(pi).all(-1) & np.isfinite(pj).all(-1)
        pi = pi[ok]; pj = pj[ok]
        if len(pi) < 5: continue
        d = np.linalg.norm(pi - pj, axis=1)
        medians.append(float(np.median(d)))
    return float(np.median(medians)) if medians else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="/home/haoming/mosaic_thinker/frames")
    ap.add_argument("--out-dir", default="outputs/frame_sweep")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    frames_dir = Path(args.frames_dir)
    all_names = sorted([p.name for p in frames_dir.glob("frame_*.png")])
    device = torch.device(args.device)

    sim = load_clip_sim(frames_dir, all_names, device)
    centrality = sim.sum(axis=1)
    most_central = list(np.argsort(-centrality))
    least_central = list(np.argsort(centrality))

    configs = {}
    configs["FULL-22"] = list(all_names)
    configs["UNIFORM-12"] = all_names[::2]
    configs["UNIFORM-6"] = all_names[::4]
    configs["UNIFORM-3"] = [all_names[0], all_names[len(all_names)//2], all_names[-1]]
    configs["MST-drop-leaves-5"] = [all_names[i] for i in sorted(most_central[:17])]
    configs["MST-anchor-3"] = [all_names[i] for i in sorted(most_central[:3])]
    # NEW: VGGT-friendly diversity selection — pick frames that are MAX dissimilar
    # via greedy farthest-point on CLIP similarity (no overlap requirement)
    chosen = [int(np.argmax(centrality))]
    while len(chosen) < 6:
        # for each remaining, max similarity to chosen; pick the one whose max-sim is lowest
        rem = [i for i in range(len(all_names)) if i not in chosen]
        scores = [-max(sim[c, r] for c in chosen) for r in rem]
        chosen.append(rem[int(np.argmax(scores))])
    configs["FPS-DIVERSE-6"] = [all_names[i] for i in sorted(chosen)]

    results = {}
    for name, subset in configs.items():
        print(f"\n=== {name} ({len(subset)} frames) ===")
        pts, fwd = run_vggt(frames_dir, subset, device)
        ext = per_frame_extents(pts)
        cons = consistency_on_pairs(pts, subset, frames_dir, device)
        ent = {
            "n_frames": len(subset),
            "frame_names": subset,
            "vggt_forward_s": fwd,
            "vggt_per_frame_ms": 1000 * fwd / len(subset),
            "extent_median_m": float(np.median(ext)),
            "extent_min_m": float(ext.min()),
            "extent_max_m": float(ext.max()),
            "n_collapsed_under_0.1m": int((ext < 0.1).sum()),
            "consistency_median_m": cons,
        }
        print(f"  fwd: {fwd:.2f}s ({1000*fwd/len(subset):.0f}ms/frame)")
        print(f"  extent: median={np.median(ext):.2f}m, min={ext.min():.2f}m, max={ext.max():.2f}m, "
              f"collapsed={(ext < 0.1).sum()}")
        print(f"  consistency: {cons}")
        results[name] = ent

    (out / "sweep_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved", out / "sweep_results.json")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    nframes = [r["n_frames"] for r in results.values()]
    labels = list(results.keys())
    fwds = [r["vggt_forward_s"] for r in results.values()]
    cons = [r["consistency_median_m"] for r in results.values()]
    coll = [r["n_collapsed_under_0.1m"] for r in results.values()]

    axes[0].bar(labels, fwds); axes[0].set_ylabel("VGGT forward (s)"); axes[0].tick_params(axis='x', rotation=30)
    axes[1].bar(labels, [c if c else 0 for c in cons]); axes[1].set_ylabel("Multi-view consistency (median, m)"); axes[1].tick_params(axis='x', rotation=30)
    axes[2].bar(labels, coll); axes[2].set_ylabel("# collapsed frames (extent<0.1m)"); axes[2].tick_params(axis='x', rotation=30)
    fig.suptitle("VGGT under different frame-selection strategies")
    fig.tight_layout()
    fig.savefig(out / "sweep_plot.png", dpi=120)
    print("Saved", out / "sweep_plot.png")


if __name__ == "__main__":
    main()
