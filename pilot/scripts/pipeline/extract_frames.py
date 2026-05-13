"""Extract evenly-spaced RGB frames from a VSI-Bench mp4."""
import sys
from pathlib import Path
import cv2


def extract(mp4_path, out_dir, n_frames=24):
    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{mp4_path.name}: {total} frames total, extracting {n_frames}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    idxs = [int(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            print(f"  failed frame {fi}"); continue
        cv2.imwrite(str(Path(out_dir) / f"frame_{fi:05d}.png"), fr)
    cap.release()
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    SCENES = ["scene0011_00", "scene0050_00", "scene0046_00"]
    VSI = Path("/home/haoming/x-spatial-manual/data/vsi_bench_full/scannet")
    OUT = Path("/home/haoming/mosaicthinker2/pilot/outputs/scannet_frames")
    for s in SCENES:
        extract(VSI / f"{s}.mp4", OUT / s, n_frames=24)
