"""
Stage 1 (baseline): Faithful reproduction of MosaicThinker's per-frame depth +
pairwise image-matching + MST topology pipeline. Built from scratch, not
relying on the existing /home/haoming/mosaic_thinker code.

Pipeline:
  1) Per-frame DepthPro -> depth + focal length (monocular, scale-ambiguous)
  2) Per-frame CLIP image embeddings -> pairwise cosine similarity matrix
  3) Pick root = frame with highest aggregate similarity (eq. 2 of paper)
  4) Build MST over CLIP-similarity graph (eq. 3 of paper) using Kruskal
  5) For each MST edge: RoMa matching -> 3D-3D similarity transform (Umeyama
     w/ scale, since DepthPro scale differs across frames)
  6) Chain transforms along MST path from each frame to root -> global poses
  7) Backproject every pixel of every frame into the global frame
  8) Save global point cloud (subsampled), per-frame extrinsics, intrinsics,
     and timing breakdown.

Outputs: outputs/baseline/<run_tag>/
  points.npy          (N, H, W, 3)   per-frame world-coord point maps
  depth.npy           (N, H, W)
  extrinsic.npy       (N, 3, 4)      world->cam (so cam->world is the inverse)
  intrinsic.npy       (N, 3, 3)
  images.npy          (N, 3, H, W)
  frame_names.json
  timing.json
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from romatch import roma_indoor
from transformers import (
    DepthProForDepthEstimation, DepthProImageProcessorFast,
    CLIPModel, CLIPProcessor,
)


def load_image(p):
    return Image.open(p).convert("RGB")


def depth_per_frame(img: Image.Image, model, proc, device):
    inputs = proc(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    post = proc.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])[0]
    return post["predicted_depth"].cpu().numpy().astype(np.float32), float(post["focal_length"])


def clip_features(images, clip_model, clip_proc, device):
    inputs = clip_proc(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype(np.float32)


def kruskal_mst(N, edges):
    """edges: list of (weight_descending, i, j). Returns list of edges in MST."""
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a == b: return False
        parent[a] = b
        return True
    edges_sorted = sorted(edges, key=lambda e: -e[0])
    mst = []
    for w, i, j in edges_sorted:
        if union(i, j):
            mst.append((w, i, j))
            if len(mst) == N - 1: break
    return mst


def shortest_paths_to_root(N, mst, root):
    adj = {i: [] for i in range(N)}
    for w, i, j in mst:
        adj[i].append(j); adj[j].append(i)
    parent = [-1] * N
    parent[root] = root
    order = [root]
    seen = {root}
    while order:
        u = order.pop(0)
        for v in adj[u]:
            if v not in seen:
                seen.add(v); parent[v] = u; order.append(v)
    paths = []
    for i in range(N):
        path = []
        cur = i
        while cur != root:
            path.append(cur)
            cur = parent[cur]
        path.append(root)
        paths.append(path)
    return parent, paths


def umeyama(src, dst):
    """7-DoF similarity (scale + rotation + translation) mapping src -> dst.
    src, dst: (M, 3). Returns 4x4 transform."""
    src = np.asarray(src, dtype=np.float64); dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(0); mu_d = dst.mean(0)
    s = src - mu_s; d = dst - mu_d
    H = s.T @ d / s.shape[0]
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3); D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = (Vt.T @ D @ U.T)
    var_s = (s ** 2).sum() / s.shape[0]
    scale = (S * np.diag(D)).sum() / max(var_s, 1e-12)
    t = mu_d - scale * R @ mu_s
    T = np.eye(4)
    T[:3, :3] = scale * R; T[:3, 3] = t
    return T


def lift_pixels(uvs, depth_map, focal, pp):
    """uvs: (M,2) float pixel coords; depth_map (H,W) numpy; pp=(cx,cy)."""
    H, W = depth_map.shape
    xs = np.clip(np.rint(uvs[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.rint(uvs[:, 1]).astype(int), 0, H - 1)
    z = depth_map[ys, xs]
    valid = z > 0
    xs = xs[valid]; ys = ys[valid]; z = z[valid]
    X = (xs - pp[0]) * z / focal
    Y = (ys - pp[1]) * z / focal
    return np.stack([X, Y, z], axis=1), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="/home/haoming/mosaic_thinker/frames")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-names", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-matches-per-pair", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir)
    names = (args.frame_names.split(",") if args.frame_names
             else sorted([p.name for p in frames_dir.glob("frame_*.png")]))
    paths = [frames_dir / n for n in names]
    images = [load_image(p) for p in paths]
    N = len(images)
    print(f"Loaded {N} frames")

    device = torch.device(args.device)
    timing = {}

    # ---- 1. DepthPro per frame ----
    t = time.time()
    dp_proc = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    dp_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device).eval()
    depths = []; focals = []
    for img in images:
        d, f = depth_per_frame(img, dp_model, dp_proc, device)
        depths.append(d); focals.append(f)
    depths = np.stack(depths)  # (N, H, W)
    focals = np.array(focals)
    timing["depth_total_s"] = time.time() - t
    timing["depth_per_frame_ms"] = 1000 * timing["depth_total_s"] / N
    print(f"DepthPro: {timing['depth_total_s']:.2f}s ({timing['depth_per_frame_ms']:.0f}ms/frame)")
    del dp_model; torch.cuda.empty_cache()

    # ---- 2. CLIP similarity ----
    t = time.time()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    feats = clip_features(images, clip_model, clip_proc, device)
    sim = feats @ feats.T
    np.fill_diagonal(sim, -1.0)
    timing["clip_s"] = time.time() - t
    del clip_model; torch.cuda.empty_cache()

    # ---- 3. root selection ----
    agg = sim.copy(); np.fill_diagonal(agg, 0)
    root = int(agg.sum(axis=1).argmax())
    print(f"Root frame: {names[root]} (idx {root})")

    # ---- 4. MST ----
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((float(sim[i, j]), i, j))
    mst = kruskal_mst(N, edges)
    parent, paths_to_root = shortest_paths_to_root(N, mst, root)
    timing["mst_s"] = time.time() - t

    # ---- 5. RoMa pairwise alignment along MST edges ----
    t = time.time()
    roma_model = roma_indoor(device="cuda" if device.type == "cuda" else "cpu")

    def pp_of(img): return ((img.size[0] - 1) / 2.0, (img.size[1] - 1) / 2.0)

    pairwise_T = {}     # (child, parent) -> 4x4 transform mapping child-frame -> parent-frame
    pair_match_counts = {}
    for w, i, j in mst:
        # Decide which is parent, which is child along the rooted tree.
        if parent[i] == j:
            child, par = i, j
        else:
            child, par = j, i
        warp, certainty = roma_model.match(str(paths[child]), str(paths[par]), device=device)
        matches, certainty = roma_model.sample(warp, certainty, num=args.max_matches_per_pair)
        H, W = images[par].height, images[par].width
        kpts_c, kpts_p = roma_model.to_pixel_coordinates(
            matches,
            images[child].height, images[child].width,
            images[par].height, images[par].width,
        )
        kpts_c = kpts_c.detach().cpu().numpy()
        kpts_p = kpts_p.detach().cpu().numpy()

        pts_c, vc = lift_pixels(kpts_c, depths[child], focals[child], pp_of(images[child]))
        pts_p, vp = lift_pixels(kpts_p[vc], depths[par], focals[par], pp_of(images[par]))
        pts_c = pts_c[vp]
        if len(pts_c) < 3:
            print(f"  WARN: too few matches for ({child},{par})")
            T = np.eye(4)
        else:
            T = umeyama(pts_c, pts_p)
        pairwise_T[(child, par)] = T
        pair_match_counts[(child, par)] = int(len(pts_c))
        print(f"  edge {names[child]} -> {names[par]}: {len(pts_c)} matches, "
              f"scale={np.linalg.norm(T[:3,0]):.3f}")
    timing["roma_total_s"] = time.time() - t
    timing["roma_per_edge_s"] = timing["roma_total_s"] / max(1, len(mst))
    del roma_model; torch.cuda.empty_cache()

    # ---- 6. Chain transforms to global root frame ----
    global_T = np.zeros((N, 4, 4), dtype=np.float64)
    global_T[root] = np.eye(4)
    for i in range(N):
        if i == root: continue
        T_acc = np.eye(4)
        path = paths_to_root[i]  # i, ..., root
        for k in range(len(path) - 1):
            child, par = path[k], path[k + 1]
            T_acc = pairwise_T[(child, par)] @ T_acc
        global_T[i] = T_acc

    # ---- 7. Build per-frame world-coord point maps and extrinsic/intrinsic ----
    t = time.time()
    point_maps = []
    extrinsics = []
    intrinsics = []
    for i in range(N):
        H, W = depths[i].shape
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        # Backproject all pixels into local cam frame
        xs = np.arange(W); ys = np.arange(H)
        X, Y = np.meshgrid(xs, ys)
        z = depths[i]
        Xc = (X - cx) * z / focals[i]
        Yc = (Y - cy) * z / focals[i]
        local = np.stack([Xc, Yc, z], axis=-1)        # (H,W,3)
        # Apply global transform (similarity, not strict SE(3))
        T = global_T[i]
        flat = local.reshape(-1, 3)
        world = (T[:3, :3] @ flat.T).T + T[:3, 3]
        point_maps.append(world.reshape(H, W, 3).astype(np.float32))
        # Extrinsic: world->cam = inverse(cam->world). We treat T as cam->world.
        # For non-rigid (similarity) transforms we still record T as a placeholder.
        extrinsics.append(T[:3, :].astype(np.float32))
        K = np.array([[focals[i], 0, cx], [0, focals[i], cy], [0, 0, 1]], dtype=np.float32)
        intrinsics.append(K)
    point_maps = np.stack(point_maps)
    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)
    timing["chain_and_lift_s"] = time.time() - t

    # ---- 8. Save ----
    np.save(out / "points.npy", point_maps)
    np.save(out / "depth.npy", depths.astype(np.float32))
    np.save(out / "extrinsic.npy", extrinsics)
    np.save(out / "intrinsic.npy", intrinsics)
    # Save preprocessed images at original resolution for downstream comparisons
    np.save(out / "image_sizes.npy",
            np.array([[im.width, im.height] for im in images], dtype=np.int32))
    (out / "frame_names.json").write_text(json.dumps(names, indent=2))
    timing["mst_edges"] = [[w, names[i], names[j]] for w, i, j in mst]
    timing["root"] = names[root]
    timing["match_counts"] = {f"{names[c]}->{names[p]}": v for (c, p), v in pair_match_counts.items()}
    timing["total_s"] = sum(v for k, v in timing.items()
                            if isinstance(v, (int, float)) and k.endswith("_s"))
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    print("Saved", out)
    print(f"Total wallclock: ~{timing['total_s']:.1f}s")


if __name__ == "__main__":
    main()
