"""
Stage 2: Lift 2D segmentation masks (GroundingDINO + SAM) into the 3D world
using each method's per-frame world-coord point map.

Produces, for each method:
  outputs/<method>/semantic_points.npz
    points:  (M, 3) world-coord
    labels:  (M,)   label-string indices
    label_names: list[str]
    confidences: (M,)
    frame_idx:   (M,)
  outputs/<method>/semantic_map.png   (BEV plot)
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    SamModel, SamProcessor,
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches


DEFAULT_QUERIES = [
    "couch", "sofa", "chair", "table", "bed", "tv", "lamp", "pillow", "blanket",
    "cabinet", "bookshelf", "rug", "window", "door", "plant",
    "stove", "refrigerator", "oven", "microwave", "sink", "toilet", "desk",
]


def detect(image: Image.Image, queries, det_model, det_proc, device, threshold=0.3):
    text = ". ".join(queries) + "."
    inputs = det_proc(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = det_model(**inputs)
    results = det_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold,
        target_sizes=[(image.height, image.width)]
    )[0]
    return results  # dict with boxes (N,4) xyxy, scores, labels (text)


def segment(image: Image.Image, boxes, sam_model, sam_proc, device):
    if len(boxes) == 0:
        return np.zeros((0, image.height, image.width), dtype=bool)
    inputs = sam_proc(image, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = sam_proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]   # (N, 3, H, W) — pick best of 3
    # outputs.iou_scores (1, N, 3); pick best per box
    iou = outputs.iou_scores[0].detach().cpu().numpy()  # (N, 3)
    best = iou.argmax(-1)
    out = np.stack([masks[i, best[i]].numpy() for i in range(len(boxes))])
    return out  # (N, H, W) bool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-dir", required=True)
    ap.add_argument("--frames-dir", default="/home/haoming/mosaic_thinker/frames")
    ap.add_argument("--queries", default=",".join(DEFAULT_QUERIES))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-points-per-mask", type=int, default=2000)
    ap.add_argument("--det-threshold", type=float, default=0.3)
    args = ap.parse_args()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    method_dir = Path(args.method_dir)
    pts_world = np.load(method_dir / "points.npy")    # (N, Hp, Wp, 3)
    names = json.loads((method_dir / "frame_names.json").read_text())
    N, Hp, Wp, _ = pts_world.shape
    print(f"Method: {method_dir.name}, point map shape ({N},{Hp},{Wp},3)")

    device = torch.device(args.device)
    det_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny").to(device).eval()
    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    rng = np.random.default_rng(0)
    all_pts, all_lbls, all_conf, all_frame = [], [], [], []
    label_to_idx = {}
    timing = {"detect_s": 0, "segment_s": 0, "lift_s": 0}

    for fi, name in enumerate(names):
        img = Image.open(Path(args.frames_dir) / name).convert("RGB")
        Wi, Hi = img.size
        sx, sy = Wp / Wi, Hp / Hi   # scale to predicted point-map resolution

        t = time.time()
        det = detect(img, queries, det_model, det_proc, device, args.det_threshold)
        timing["detect_s"] += time.time() - t
        boxes = det["boxes"].cpu().numpy()   # xyxy, original size
        scores = det["scores"].cpu().numpy()
        # Some HF versions return 'labels' as list of str, others as 'text_labels'
        labels = det.get("labels", None) or det.get("text_labels", None) or det.get("text", None)
        if labels is None:
            labels = [""] * len(boxes)
        else:
            labels = list(labels)

        if len(boxes) == 0:
            continue

        t = time.time()
        masks = segment(img, boxes, sam_model, sam_proc, device)  # (M, Hi, Wi)
        timing["segment_s"] += time.time() - t

        # Lift each mask -> 3D points using point map
        t = time.time()
        for mi in range(len(boxes)):
            mask_full = masks[mi]   # (Hi, Wi)
            ys, xs = np.where(mask_full)
            if len(xs) == 0:
                continue
            # subsample
            if len(xs) > args.max_points_per_mask:
                idx = rng.choice(len(xs), args.max_points_per_mask, replace=False)
                xs = xs[idx]; ys = ys[idx]
            # rescale to point-map resolution
            xs_p = np.clip(np.rint(xs * sx).astype(int), 0, Wp - 1)
            ys_p = np.clip(np.rint(ys * sy).astype(int), 0, Hp - 1)
            p3d = pts_world[fi][ys_p, xs_p]  # (M, 3)
            ok = np.isfinite(p3d).all(-1)
            p3d = p3d[ok]
            if len(p3d) == 0:
                continue
            lab = labels[mi].strip() or "obj"
            if lab not in label_to_idx:
                label_to_idx[lab] = len(label_to_idx)
            all_pts.append(p3d.astype(np.float32))
            all_lbls.append(np.full(len(p3d), label_to_idx[lab], dtype=np.int32))
            all_conf.append(np.full(len(p3d), float(scores[mi]), dtype=np.float32))
            all_frame.append(np.full(len(p3d), fi, dtype=np.int32))
        timing["lift_s"] += time.time() - t

    if not all_pts:
        print("No semantic points lifted.")
        return

    pts = np.concatenate(all_pts)
    lbls = np.concatenate(all_lbls)
    conf = np.concatenate(all_conf)
    fidx = np.concatenate(all_frame)
    label_names = [None] * len(label_to_idx)
    for k, v in label_to_idx.items():
        label_names[v] = k
    print(f"Lifted {len(pts)} points across {len(label_names)} labels: {label_names}")
    print(f"timing: {timing}")
    np.savez(method_dir / "semantic_points.npz",
             points=pts, labels=lbls, confidences=conf, frame_idx=fidx,
             label_names=np.array(label_names))
    (method_dir / "semantic_timing.json").write_text(json.dumps(timing, indent=2))

    # ---- BEV plot ----
    # Use floor-projection: drop the height axis. We don't know which axis is up
    # in either coord system, so pick the axis with smallest variance as height.
    var = pts.var(0)
    height_axis = int(var.argmin())
    plane = [a for a in [0, 1, 2] if a != height_axis]
    P = pts[:, plane]

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.tab20
    for li, ln in enumerate(label_names):
        m = lbls == li
        ax.scatter(P[m, 0], P[m, 1], s=2, alpha=0.5, color=cmap(li % 20), label=ln)
    ax.set_aspect("equal")
    ax.set_title(f"BEV semantic map — {method_dir.name}\n"
                 f"{len(pts)} points, height axis dropped = {['x','y','z'][height_axis]}")
    ax.legend(loc="upper right", markerscale=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(method_dir / "semantic_map.png", dpi=120)
    print(f"Saved {method_dir / 'semantic_map.png'}")


if __name__ == "__main__":
    main()
