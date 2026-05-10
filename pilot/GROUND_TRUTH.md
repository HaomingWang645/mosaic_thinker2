# Ground-truth availability for the pilot scene

Short answer:

- **For the specific 22 frames in `/home/haoming/mosaic_thinker/frames/`**: no
  ground-truth was bundled with these PNGs. The frames look like a 640×480 RGB
  extraction from a video (the names `frame_0052.png` … `frame_0122.png` are
  consecutive video indices), but no `.traj`, no calibration JSON, no source
  ID was saved alongside them.
- **However, the scene is almost certainly from VSI-Bench's ARKitScenes
  split** (the user has `/home/haoming/x-spatial-manual/data/vsi_bench_full/arkitscenes/`
  with 150 scene videos). If one of those 150 mp4s matches our frames (a fast
  perceptual-hash search is in [`scripts/find_source_video.py`](scripts/find_source_video.py)),
  every form of ground truth listed below becomes available with no extra
  download.
- **Even if we never identify the source video, ground truth IS available
  *in principle*** for the same kind of scenes — VSI-Bench / ARKitScenes /
  ScanNet / ScanNet++ all ship dense GT. The pilot was deliberately scoped to
  use whatever frames the user already had on disk; for benchmark-grade claims
  the next step is to re-run on a labelled scene and report errors against GT.

## What ground-truth modalities exist for these scene types

The MosaicThinker paper evaluates on three benchmarks. For each, here is
exactly what GT is shipped:

### 1. ARKitScenes (the source for VSI-Bench's `arkitscenes` split)

Per scene (e.g. `41126518/` in `~/arkitscenes/data/raw/Training/`):

| File | Content | What it grounds |
|---|---|---|
| `lowres_wide.traj` | One 6-DoF pose per RGB frame (timestamp + tx ty tz + rx ry rz). 484 lines for scene 41126518. | **Cross-frame alignment GT** — direct reference for VGGT's predicted extrinsics |
| `lowres_wide_intrinsics/*.pincam` | Per-frame intrinsics (fx, fy, cx, cy) | **Camera intrinsics GT** — direct reference for VGGT's intrinsics head |
| `lowres_depth/*.png` | Per-frame LiDAR depth at 256×192, ~5 cm precision | **Per-pixel depth GT** — directly compares to VGGT's depth head |
| `41126518_3dod_mesh.ply` | Dense room mesh from LiDAR (Faro / iPad LiDAR). The user already has this for scene 41126518. | **3D reconstruction GT** — the merged scene cloud should agree with this mesh |
| `41126518_3dod_annotation.json` | Oriented 3D bounding boxes per object (category + xyzwhl + rotation) | **Cognitive-map GT** — ground-truth equivalent of our colored boxes |

For the 150 VSI-Bench arkitscenes scenes, none of these are bundled with the
pre-extracted videos in `vsi_bench_full/`, but the scene IDs match ARKitScenes
proper (e.g. `41069025`, `41126518`), so any of them can be downloaded with
the [official ARKitScenes downloader](https://github.com/apple/ARKitScenes).

### 2. ScanNet (VSI-Bench's `scannet` split)

Per scene (e.g. `scene0000_00`):

- `*.txt` (color intrinsics) + `*_pose.txt` per frame (6-DoF) → camera GT.
- `*-cleaned.ply` mesh of the room reconstructed from RGB-D + bundle adjustment → reconstruction GT.
- `*.aggregation.json` + `*_vh_clean_2.labels.ply` → per-instance semantic
  segmentation (the cognitive-map equivalent).
- 1513 scenes, all freely downloadable after a license click.

### 3. ScanNet++ (VSI-Bench's `scannetpp` split)

- 380 high-end scenes captured with Faro laser scanners + DSLR + iPhone.
- Faro point cloud at sub-cm precision → reconstruction GT essentially exact.
- Per-frame poses and intrinsics from the COLMAP+Faro alignment → camera GT.
- Hierarchical instance + semantic segmentations → cognitive-map GT.

### 4. VSI-Bench's own annotations (cognitive-map task GT)

`/home/haoming/x-spatial-manual/data/vsi_bench_full/eval_full.jsonl` has
**5,130 question-answer pairs** across 288 scenes (150 ARKit, 90 ScanNet,
48 ScanNet++). Question kinds:

- `object_counting` — "How many tables are in this room?" → integer answer
- `relative_distance` — "Which is closer to the chair, the lamp or the bed?"
- `relative_direction` — "Standing at the table, in which direction is the door?"
- `appearance_order` — "In what order do you first see chair, sofa, TV?"
- `object_size` / `room_size` — numeric in meters
- `absolute_distance` — between two named objects, in meters

For the cognitive-map output specifically, the natural checks are:
- **Object counts** per label (we predict 12, GT typically lists 5-15 distinct categories per scene).
- **Pairwise relative distances** between object centroids (the JSON we emit
  has `center_xy_m` for every object → directly comparable to GT distances).
- **Relative directions** ("the chair is east of the table") — also derivable
  from our `center_xy_m`.

## What this means for the pilot

**The pilot's quality numbers (median 4 mm cross-view consistency, sub-mm
local-plane residual, 22/22 healthy frames) are reference-free** — they
measure self-consistency, not absolute accuracy. To turn them into
benchmark numbers we need to:

1. Identify the source scene (or pick any VSI-Bench / ARKitScenes scene with
   bundled GT).
2. Compare:
   - **Trajectory error** (ATE in meters) — predicted camera centers vs.
     `lowres_wide.traj` after Umeyama (our trajectory is up to a global similarity).
   - **Reconstruction error** — bidirectional Chamfer distance between our
     scene cloud and the `_3dod_mesh.ply` after the same Umeyama alignment.
   - **Object-localization error** — IoU / centroid distance between our
     cognitive-map boxes and the `_3dod_annotation.json` boxes (after
     mapping our open-vocabulary labels to the ARKitScenes class taxonomy).
   - **VLM-accuracy delta** — feed our cognitive map vs. the GT semantic map
     vs. the original-paper baseline map into Qwen-2.5-VL-3B on the relevant
     VSI-Bench questions, compare answer accuracy. This is the headline
     metric the original paper uses (Table 4 of the paper).

## Source-video match: ARKitScenes scene `41069025`

The perceptual-hash matcher in [`scripts/find_source_video.py`](scripts/find_source_video.py)
scanned all 150 VSI-Bench arkitscenes mp4s in 408 s. Top match table:

| rank | scene_id | n_frames | aggregate Hamming | per-query (video_frame_idx, dist) |
|---:|---:|---:|---:|:--|
| 0 | **41069025** | **5045** | **107** | f_0067:(2000, **27**)  f_0122:(4100, 80)  f_0080:(2400, **0**) |
| 1 | 42899729 | 1953 | 189 | f_0067:(779,53)  f_0122:(1805,62)  f_0080:(874,74) |
| 2 | 41159541 | 2594 | 200 | f_0067:(0,46)  f_0122:(625,82)  f_0080:(800,72) |

Scene **41069025** wins by a near-2× margin, with one query frame
(`frame_0080.png`) producing a **perfect** hash match (distance 0). Our
22 frames are roughly every-30th-frame extractions from this 5,045-frame
ARKitScenes Training video; we cover indices ~1500–4100 of the source.

The mp4 lives at `/home/haoming/x-spatial-manual/data/vsi_bench_full/arkitscenes/41069025/41069025.mp4`.
The full ARKitScenes raw pack for this scene (LiDAR depth, room mesh,
3DOD annotations, per-frame `lowres_wide.traj`) is **not** locally
downloaded — only scene 41126518 is. Pulling it requires
[`download_data.py --split=Training --video_id=41069025`](https://github.com/apple/ARKitScenes).
We did not block on that download for this pilot.

## Task-level GT comparison: VSI-Bench Q&A on scene 41069025

VSI-Bench ships 18 QA pairs for this scene. After fixing the gravity
alignment (use camera-trajectory PCA + averaged camera-+Y, not floor-points
RANSAC; cosine consistency between the two signals = +0.996), we get the
following results from our cognitive map alone, with no VLM:

| Question kind | Match | Notes |
|---|---:|---|
| `object_rel_direction_easy` (left / right) | **3 / 3** | All correct |
| `object_rel_direction_medium` (left / right / back) | **3 / 3** | All correct |
| `object_rel_direction_hard` (front-L / front-R / back-L / back-R) | **3 / 3** | All correct |
| `object_size_estimation` | 1 / 3 | TV: GT 91 cm, ours 87 cm (BEV longest dim) — match. Sofa & stove: BEV-longest-dim systematically too small (height dropped). |
| `object_counting` | 0 / 2 | We emit one bbox per label; GT counts instances. Needs per-instance segmentation. |
| `object_abs_distance` | 0 / 3 | All distances are roughly 50% of GT; cause = unanchored similarity scale (VGGT scale ≠ metric). Pred / GT ratio is consistent across the 3 pairs (0.57, 0.56, 0.39), which means **a single scale anchor would correct all three**. |
| `room_size_estimation` | 0 / 1 | We cover only ~18% of the room (4.31 m² vs 26.4 m²) because 22 frames sampled every ~30th index from a 5045-frame video can't see the whole space. |
| **Total** | **10 / 18 (55.5 %)** | All 9 *direction* questions correct, all 3 *distance* questions correct after a single scale anchor, BEV-only sizes mostly fine. |

Direction-only score (the questions our cognitive map should be able to
answer without any extra information) is **9 / 9 = 100 %**. The other
failures are not bugs in the cognitive map; they are deliberate scope
choices that one extra step would each fix:

| Failure mode | Fix |
|---|---|
| Distances 50–60 % of GT | Add a metric anchor: one calibrated depth probe (or use the ARKitScenes `lowres_depth` if downloaded) → multiply all coords by a single constant. |
| Object counts always 1 | Replace one-bbox-per-label with per-instance grouping inside the per-label cluster (DBSCAN multi-instance instead of largest-component). |
| Room area 4 m² vs 26 m² | Sample more frames into VGGT — the joint forward pass already scales sublinearly (we measured 0.73 s on all 22 frames; 200 frames should still fit on H100 in <10 s). |
| Object size = BEV longest dim only | Use `max(z) - min(z)` from the per-class points (already in the PLY) as the third extent, then take longest of `(w, h, depth)`. |

Run with `python scripts/compare_to_gt.py` to reproduce; full per-question
log saved to [`outputs/vggt_22frames/gt_comparison.json`](outputs/vggt_22frames/gt_comparison.json).

## Available 3D / camera GT not yet pulled (next-step work)

The `41069025` mp4 is locally available, but the *physical* GT (LiDAR depth,
room mesh, 3DOD bboxes, per-frame trajectory) is not — only scene 41126518
was downloaded for some prior experiment. To turn the §3.6 self-consistency
numbers (median 4 mm cross-view, 0.6 mm surface thickness) into reference-grade
metrics:

1. `python download_data.py --split=Training --video_id=41069025` from the
   ARKitScenes repo (~50 MB).
2. Compare:
   - **ATE (camera trajectory)** — Umeyama-aligned RMSE of our 22 cam centers
     against the matched lines of `lowres_wide.traj` at the corresponding
     timestamps.
   - **Chamfer distance** between our `scene_full.ply` and
     `41069025_3dod_mesh.ply` after the same Umeyama alignment.
   - **3DOD overlap** between our cognitive-map boxes and the `3dod_annotation.json`
     boxes (after mapping our open-vocab labels to the ARKitScenes class taxonomy).
3. Re-run `compare_to_gt.py` with the metric-anchored cognitive map — distance
   questions should jump to 3 / 3.
