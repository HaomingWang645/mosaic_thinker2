"""
Stage 1: Feed-forward 3D reconstruction with VGGT on the 22-frame ScanNet/VSI scene.

Produces:
  outputs/vggt/<run_tag>/
    points.npy          (N_views, H, W, 3)   world-coordinate points
    points_conf.npy     (N_views, H, W)      confidence
    depth.npy           (N_views, H, W)      depth maps
    extrinsic.npy       (N_views, 3, 4)      camera extrinsics (world->cam)
    intrinsic.npy       (N_views, 3, 3)      camera intrinsics
    images.npy          (N_views, 3, H, W)   preprocessed images used
    frame_names.json    list of source filenames
    timing.json         latency breakdown
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="/home/haoming/mosaic_thinker/frames")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-names", default=None,
                    help="comma-separated subset of filenames; default = all PNGs sorted")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir)
    if args.frame_names:
        names = args.frame_names.split(",")
    else:
        names = sorted([p.name for p in frames_dir.glob("frame_*.png")])
    image_paths = [str(frames_dir / n) for n in names]
    print(f"Loading {len(image_paths)} images")

    device = args.device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    timing = {}
    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    timing["load_model_s"] = time.time() - t0

    t0 = time.time()
    images = load_and_preprocess_images(image_paths).to(device)  # (N, 3, H, W)
    timing["load_images_s"] = time.time() - t0

    print(f"images shape: {tuple(images.shape)}")

    t0 = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(images)
    torch.cuda.synchronize()
    timing["forward_s"] = time.time() - t0
    timing["forward_per_frame_ms"] = 1000 * timing["forward_s"] / len(image_paths)

    # Decode poses
    pose_enc = preds["pose_enc"]              # (1, N, 9)
    extr, intr = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    extr = extr[0].cpu().numpy()              # (N, 3, 4)
    intr = intr[0].cpu().numpy()              # (N, 3, 3)

    # Use VGGT's `world_points` head (preferred). Fall back to depth-based unprojection.
    if "world_points" in preds:
        world_points = preds["world_points"][0].cpu().numpy()       # (N, H, W, 3)
        world_points_conf = preds["world_points_conf"][0].cpu().numpy()  # (N, H, W)
    else:
        world_points = unproject_depth_map_to_point_map(
            preds["depth"][0].cpu().numpy(),
            extr, intr,
        )
        world_points_conf = preds.get("depth_conf", torch.ones_like(preds["depth"]))[0].cpu().numpy()

    depth = preds["depth"][0].cpu().numpy()                          # (N, H, W, 1) or (N, H, W)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    print(f"world_points: {world_points.shape}, depth: {depth.shape}")
    print(f"forward total: {timing['forward_s']:.2f}s, per-frame: {timing['forward_per_frame_ms']:.1f}ms")

    np.save(out / "points.npy", world_points.astype(np.float32))
    np.save(out / "points_conf.npy", world_points_conf.astype(np.float32))
    np.save(out / "depth.npy", depth.astype(np.float32))
    np.save(out / "extrinsic.npy", extr.astype(np.float32))
    np.save(out / "intrinsic.npy", intr.astype(np.float32))
    np.save(out / "images.npy", images.cpu().numpy().astype(np.float32))
    (out / "frame_names.json").write_text(json.dumps(names, indent=2))
    (out / "timing.json").write_text(json.dumps(timing, indent=2))

    print("Saved to", out)


if __name__ == "__main__":
    main()
