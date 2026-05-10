"""
Try to identify which VSI-Bench / ARKitScenes video our 22 mosaic_thinker
frames came from.

For each candidate ARKitScenes mp4 (150 of them), extract every Nth frame and
compute a fast perceptual hash. Compare against frame_0067.png (a distinctive
'corridor with TV' frame) and frame_0122.png (sofa with pillow). Score by
minimum-distance match. Report the top 5.
"""
import sys, time
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


VSI = Path("/home/haoming/x-spatial-manual/data/vsi_bench_full/arkitscenes")
QUERY_FRAMES = ["frame_0067.png", "frame_0122.png", "frame_0080.png"]
QUERY_DIR = Path("/home/haoming/mosaic_thinker/frames")


def phash(img_arr_rgb, size=16):
    g = cv2.cvtColor(img_arr_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size, size))
    avg = g.mean()
    return (g > avg).flatten()


def hamming(a, b):
    return int((a != b).sum())


def main():
    queries = []
    for qn in QUERY_FRAMES:
        img = np.asarray(Image.open(QUERY_DIR / qn).convert("RGB"))
        queries.append((qn, phash(img)))

    candidates = sorted(VSI.glob("*.mp4"))
    print(f"Scanning {len(candidates)} ARKitScenes videos (this may take a couple of minutes)…")
    best_per_scene = []
    t0 = time.time()
    for i, mp4 in enumerate(candidates):
        scene_id = mp4.stem
        cap = cv2.VideoCapture(str(mp4))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframes <= 0:
            cap.release(); continue
        # Sample every ~max(1, nframes/200) frames
        step = max(1, nframes // 100)
        best_for_query = {q: (None, 1e9) for q, _ in queries}
        for fi in range(0, nframes, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok: continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h = phash(frame_rgb)
            for q_name, q_hash in queries:
                d = hamming(q_hash, h)
                if d < best_for_query[q_name][1]:
                    best_for_query[q_name] = (fi, d)
        cap.release()
        agg = sum(d for _, d in best_for_query.values())
        best_per_scene.append((agg, scene_id, nframes, best_for_query))
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  scanned {i+1}/{len(candidates)} in {elapsed:.0f}s")

    best_per_scene.sort()
    print("\nTop 8 candidate scenes by aggregate Hamming distance over 3 query frames:\n")
    print(f"{'rank':<5} {'scene_id':<12} {'nframes':<8} {'sum_dist':<10}  per-query (frame_idx, dist)")
    for r, (agg, sid, nf, bq) in enumerate(best_per_scene[:8]):
        per_q = "  ".join(f"{q.split('.')[0]}: ({bq[q][0]}, {bq[q][1]})" for q in [q for q, _ in queries])
        print(f"{r:<5} {sid:<12} {nf:<8} {agg:<10}  {per_q}")


if __name__ == "__main__":
    main()
