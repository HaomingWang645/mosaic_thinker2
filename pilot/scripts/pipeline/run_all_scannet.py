"""Run VGGT + GroundingDINO+SAM lift + gravity-aligned aggregation on the
three ScanNet scenes."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamModel, SamProcessor)

PILOT = Path("/home/haoming/mosaicthinker2/pilot")
FRAMES_ROOT = PILOT / "outputs/scannet_frames"
OUT_ROOT = PILOT / "outputs/scannet_pred"

DEFAULT_QUERIES = [
    "couch", "sofa", "chair", "table", "bed", "tv", "television", "lamp",
    "pillow", "blanket", "cabinet", "bookshelf", "rug", "window", "door",
    "plant", "stove", "refrigerator", "oven", "microwave", "sink", "toilet",
    "desk", "drawer", "countertop", "trash can", "shelf",
]


def run_vggt(model, frame_dir, out_dir, device):
    paths = sorted(str(p) for p in Path(frame_dir).glob("*.png"))
    if not paths:
        raise RuntimeError(f"no frames in {frame_dir}")
    images = load_and_preprocess_images(paths).to(device)
    t0 = time.time()
    dtype = torch.bfloat16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        preds = model(images)
    torch.cuda.synchronize()
    fwd_s = time.time() - t0
    pose_enc = preds["pose_enc"]
    extr, intr = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    extr = extr[0].cpu().numpy(); intr = intr[0].cpu().numpy()
    pts = preds["world_points"][0].cpu().numpy()
    conf = preds["world_points_conf"][0].cpu().numpy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(out_dir) / "points.npy", pts.astype(np.float32))
    np.save(Path(out_dir) / "points_conf.npy", conf.astype(np.float32))
    np.save(Path(out_dir) / "extrinsic.npy", extr.astype(np.float32))
    np.save(Path(out_dir) / "intrinsic.npy", intr.astype(np.float32))
    np.save(Path(out_dir) / "images.npy", images.cpu().numpy().astype(np.float32))
    names = [Path(p).name for p in paths]
    (Path(out_dir) / "frame_names.json").write_text(json.dumps(names, indent=2))
    print(f"  VGGT forward: {fwd_s:.2f}s ({len(paths)} frames)")
    return fwd_s


def run_semantic_lift(out_dir, frames_dir, det_model, det_proc, sam_model, sam_proc,
                      queries, device, det_thresh=0.3, max_pts_per_mask=2000):
    out_dir = Path(out_dir)
    pts = np.load(out_dir / "points.npy")
    names = json.loads((out_dir / "frame_names.json").read_text())
    N, Hp, Wp, _ = pts.shape
    rng = np.random.default_rng(0)
    all_pts, all_lbls, all_conf, all_frame = [], [], [], []
    label_to_idx = {}
    text = ". ".join(queries) + "."
    for fi, name in enumerate(names):
        img = Image.open(Path(frames_dir) / name).convert("RGB")
        Wi, Hi = img.size
        sx, sy = Wp / Wi, Hp / Hi
        inputs = det_proc(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = det_model(**inputs)
        det = det_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=det_thresh,
            target_sizes=[(img.height, img.width)])[0]
        boxes = det["boxes"].cpu().numpy()
        scores = det["scores"].cpu().numpy()
        labels = det.get("text_labels") or det.get("labels") or [""] * len(boxes)
        labels = list(labels)
        if len(boxes) == 0:
            continue
        s_inputs = sam_proc(img, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
        with torch.no_grad():
            s_outputs = sam_model(**s_inputs)
        masks = sam_proc.image_processor.post_process_masks(
            s_outputs.pred_masks.cpu(),
            s_inputs["original_sizes"].cpu(),
            s_inputs["reshaped_input_sizes"].cpu())[0]
        iou = s_outputs.iou_scores[0].detach().cpu().numpy()
        best = iou.argmax(-1)
        for mi in range(len(boxes)):
            mask_full = masks[mi, best[mi]].numpy()
            ys, xs = np.where(mask_full)
            if len(xs) == 0: continue
            if len(xs) > max_pts_per_mask:
                idx = rng.choice(len(xs), max_pts_per_mask, replace=False)
                xs = xs[idx]; ys = ys[idx]
            xs_p = np.clip(np.rint(xs * sx).astype(int), 0, Wp - 1)
            ys_p = np.clip(np.rint(ys * sy).astype(int), 0, Hp - 1)
            p3d = pts[fi][ys_p, xs_p]
            ok = np.isfinite(p3d).all(-1)
            p3d = p3d[ok]
            if len(p3d) == 0: continue
            lab = labels[mi].strip() or "obj"
            if lab not in label_to_idx:
                label_to_idx[lab] = len(label_to_idx)
            all_pts.append(p3d.astype(np.float32))
            all_lbls.append(np.full(len(p3d), label_to_idx[lab], dtype=np.int32))
            all_conf.append(np.full(len(p3d), float(scores[mi]), dtype=np.float32))
            all_frame.append(np.full(len(p3d), fi, dtype=np.int32))
    pts_arr = np.concatenate(all_pts) if all_pts else np.zeros((0, 3), dtype=np.float32)
    lbls = np.concatenate(all_lbls) if all_lbls else np.zeros((0,), dtype=np.int32)
    conf = np.concatenate(all_conf) if all_conf else np.zeros((0,), dtype=np.float32)
    fidx = np.concatenate(all_frame) if all_frame else np.zeros((0,), dtype=np.int32)
    label_names = [None] * len(label_to_idx)
    for k, v in label_to_idx.items():
        label_names[v] = k
    np.savez(out_dir / "semantic_points.npz",
             points=pts_arr, labels=lbls, confidences=conf, frame_idx=fidx,
             label_names=np.array(label_names))
    print(f"  semantic: {len(pts_arr)} points across {len(label_names)} labels")


def main():
    SCENES = ["scene0011_00", "scene0050_00", "scene0046_00"]
    device = torch.device("cuda:0")
    print("Loading VGGT...")
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    print("Loading GroundingDINO + SAM...")
    det_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny").to(device).eval()
    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    for s in SCENES:
        print(f"\n=== {s} ===")
        out = OUT_ROOT / s
        run_vggt(vggt, FRAMES_ROOT / s, out, device)
        run_semantic_lift(out, FRAMES_ROOT / s, det_model, det_proc, sam_model, sam_proc,
                          DEFAULT_QUERIES, device)


if __name__ == "__main__":
    main()
