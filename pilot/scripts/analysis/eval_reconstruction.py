"""
Evaluate reconstruction quality of a method's per-frame world-coord point map.

Without a ground-truth point cloud, we use **multi-view consistency** as a
reconstruction-quality proxy:

  For every pair of frames (i, j) with overlapping FoV, RoMa establishes
  pixel matches. Each method maps those pixels into a global frame. The
  Euclidean distance between the *globally aligned* lifted points from
  the two views is the reconstruction inconsistency. Lower = better.

Reported metrics (over MST edges and over a random-pair sample):
  - median pairwise L2 (m)
  - mean pairwise L2 (m)
  - 90th-percentile L2 (m)
  - per-frame point-cloud bbox extents (sanity: scale collapse detection)
  - global-cloud size (number of valid points)

Saves:
  outputs/<method>/eval.json
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from romatch import roma_indoor


def load_method(out_dir: Path):
    pts = np.load(out_dir / "points.npy")          # (N, H, W, 3)
    return {
        "points": pts,
        "depth": np.load(out_dir / "depth.npy"),
        "extr": np.load(out_dir / "extrinsic.npy"),
        "intr": np.load(out_dir / "intrinsic.npy"),
        "names": json.loads((out_dir / "frame_names.json").read_text()),
    }


def sample_matches_to_world(method, i, j, kpts_i, kpts_j):
    """Map sampled pixel correspondences to world-coords using each method's
    saved per-frame world-coord point map. Returns (M, 3), (M, 3), valid mask."""
    Hi, Wi, _ = method["points"][i].shape
    Hj, Wj, _ = method["points"][j].shape
    xi = np.clip(np.rint(kpts_i[:, 0]).astype(int), 0, Wi - 1)
    yi = np.clip(np.rint(kpts_i[:, 1]).astype(int), 0, Hi - 1)
    xj = np.clip(np.rint(kpts_j[:, 0]).astype(int), 0, Wj - 1)
    yj = np.clip(np.rint(kpts_j[:, 1]).astype(int), 0, Hj - 1)
    pi = method["points"][i][yi, xi]   # (M, 3)
    pj = method["points"][j][yj, xj]
    finite = np.isfinite(pi).all(-1) & np.isfinite(pj).all(-1)
    return pi, pj, finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-dir", required=True)
    ap.add_argument("--frames-dir", default="/home/haoming/mosaic_thinker/frames")
    ap.add_argument("--n-pairs", type=int, default=15,
                    help="number of random adjacent-ish pairs to evaluate")
    ap.add_argument("--samples-per-pair", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    method = load_method(Path(args.method_dir))
    names = method["names"]
    N = len(names)
    H_pred, W_pred = method["points"].shape[1:3]
    print(f"Loaded {N} frames, point-map shape {method['points'].shape}")

    # Per-frame extents (catches scale collapse)
    extents = []
    valid_pts = []
    for i in range(N):
        p = method["points"][i].reshape(-1, 3)
        ok = np.isfinite(p).all(-1)
        p = p[ok]
        valid_pts.append(int(ok.sum()))
        if len(p) == 0:
            extents.append(0.0); continue
        ext = np.linalg.norm(p.max(0) - p.min(0))
        extents.append(float(ext))
    print(f"Per-frame extent (diagonal): median={np.median(extents):.2f}m, "
          f"max={np.max(extents):.2f}m, min={np.min(extents):.2f}m")

    device = torch.device(args.device)
    roma = roma_indoor(device="cuda" if device.type == "cuda" else "cpu")

    # Pick pairs: first the natural adjacencies in time + a random sample
    rng = np.random.default_rng(0)
    pairs = set()
    # adjacent in name order
    for k in range(N - 1):
        pairs.add((k, k + 1))
    # add a few random pairs
    while len(pairs) < args.n_pairs:
        a, b = rng.integers(0, N, size=2)
        if a == b: continue
        pairs.add((min(a, b), max(a, b)))
    pairs = sorted(pairs)[: args.n_pairs]

    frames_dir = Path(args.frames_dir)
    pair_results = []
    for i, j in pairs:
        path_i = frames_dir / names[i]
        path_j = frames_dir / names[j]
        with torch.no_grad():
            warp, certainty = roma.match(str(path_i), str(path_j), device=device)
            matches, certainty = roma.sample(warp, certainty, num=args.samples_per_pair)
            img_i = Image.open(path_i); img_j = Image.open(path_j)
            kpi, kpj = roma.to_pixel_coordinates(
                matches, img_i.height, img_i.width, img_j.height, img_j.width
            )
            kpi = kpi.cpu().numpy(); kpj = kpj.cpu().numpy()
            cert = certainty.cpu().numpy()
        # Filter to high-confidence matches
        keep = cert > 0.5
        kpi = kpi[keep]; kpj = kpj[keep]
        if len(kpi) < 5:
            continue
        # Rescale to predicted (H_pred, W_pred) since point maps live there
        sx_i = W_pred / img_i.width; sy_i = H_pred / img_i.height
        sx_j = W_pred / img_j.width; sy_j = H_pred / img_j.height
        kpi_p = kpi * np.array([sx_i, sy_i])
        kpj_p = kpj * np.array([sx_j, sy_j])
        pi, pj, ok = sample_matches_to_world(method, i, j, kpi_p, kpj_p)
        pi = pi[ok]; pj = pj[ok]
        if len(pi) < 5:
            continue
        d = np.linalg.norm(pi - pj, axis=1)
        pair_results.append({
            "pair": [names[i], names[j]],
            "n": int(len(d)),
            "median_m": float(np.median(d)),
            "mean_m": float(np.mean(d)),
            "p90_m": float(np.percentile(d, 90)),
            "p99_m": float(np.percentile(d, 99)),
        })
        print(f"  {names[i]} <-> {names[j]}: n={len(d):3d}, "
              f"median={np.median(d):.3f}m, p90={np.percentile(d, 90):.3f}m")

    summary = {
        "n_frames": N,
        "valid_points_per_frame_median": int(np.median(valid_pts)),
        "extent_median_m": float(np.median(extents)),
        "extent_max_m": float(np.max(extents)),
        "extent_min_m": float(np.min(extents)),
        "pairs": pair_results,
        "agg_median_m": float(np.median([r["median_m"] for r in pair_results])) if pair_results else None,
        "agg_mean_m": float(np.mean([r["mean_m"] for r in pair_results])) if pair_results else None,
        "agg_p90_m": float(np.median([r["p90_m"] for r in pair_results])) if pair_results else None,
    }
    out_file = Path(args.method_dir) / "eval.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(f"Frames: {N}, valid points/frame median: {summary['valid_points_per_frame_median']}")
    print(f"Per-frame extent median: {summary['extent_median_m']:.2f} m "
          f"(min {summary['extent_min_m']:.2f}, max {summary['extent_max_m']:.2f})")
    print(f"Multi-view consistency (median over pairs): "
          f"{summary['agg_median_m']:.3f} m, p90 (median over pairs): {summary['agg_p90_m']:.3f} m")
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
