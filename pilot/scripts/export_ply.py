"""
Export VGGT 3D reconstruction as PLY files (binary, little-endian).

Produces three files in pilot/outputs/vggt_22frames/ply/:

  scene_full.ply        — full gravity-aligned scene, top-50% confidence,
                          per-vertex (x, y, z, red, green, blue).
                          Open with MeshLab / CloudCompare / Open3D / Blender.

  scene_semantic.ply    — only lifted semantic points, colored by label class
                          via the same palette used in the cognitive map,
                          with extra per-vertex fields:
                            label_id (uint16)         — class index
                            confidence (float32)      — detection confidence
                            frame_idx (uint16)        — frame the point came from

  scene_combined.ply    — every full-scene point + every semantic point, with
                          a `is_semantic` flag (0/1) and label_id (0xFFFF when
                          unlabeled), so a single file can be loaded for both
                          uses. Larger but most flexible.

  labels.json           — mapping {label_id: label_name, color_rgb}.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METHOD = Path("outputs/vggt_22frames")
OUT = METHOD / "ply"; OUT.mkdir(parents=True, exist_ok=True)


def write_ply_binary(path, *, vertices, colors=None, fields=None):
    """Write an ASCII-header / binary-little-endian PLY.

    Args:
      vertices: (N, 3) float32
      colors:   (N, 3) uint8 or None
      fields:   list of (name, dtype-str-numpy, ndarray) for extra per-vertex
                attributes; numpy dtypes accepted: 'uint8','uint16','int32','float32'
    """
    N = len(vertices)
    fields = fields or []
    dtype_map = {"uint8": "uchar", "uint16": "ushort", "int32": "int", "float32": "float"}

    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {N}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    for name, dt, _ in fields:
        header.append(f"property {dtype_map[dt]} {name}")
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    # Compose record dtype
    record_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        record_fields += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    np_to_dt = {"uint8": "u1", "uint16": "<u2", "int32": "<i4", "float32": "<f4"}
    for name, dt, _ in fields:
        record_fields.append((name, np_to_dt[dt]))
    rec = np.zeros(N, dtype=record_fields)
    rec["x"] = vertices[:, 0].astype(np.float32)
    rec["y"] = vertices[:, 1].astype(np.float32)
    rec["z"] = vertices[:, 2].astype(np.float32)
    if colors is not None:
        rec["r"] = colors[:, 0].astype(np.uint8)
        rec["g"] = colors[:, 1].astype(np.uint8)
        rec["b"] = colors[:, 2].astype(np.uint8)
    for name, dt, arr in fields:
        rec[name] = arr.astype({"uint8": np.uint8, "uint16": np.uint16,
                                 "int32": np.int32, "float32": np.float32}[dt])

    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(rec.tobytes())
    print(f"  wrote {path}  ({N:,} vertices, {path.stat().st_size/1e6:.1f} MB)")


def main():
    # Already gravity-aligned; load from the cognitive-map cache
    aligned = np.load(METHOD / "scene_aligned_pts.npz", allow_pickle=True)
    scene_pts = aligned["scene"]              # (M, 3) gravity-aligned
    scene_rgb = aligned["scene_rgb"]          # (M, 3) in [0,1]
    sem_pts = aligned["semantic"]             # (K, 3)
    sem_lbl = aligned["sem_labels"]           # (K,)
    sem_names = list(aligned["sem_label_names"])
    cams = aligned["cams"]
    # Load semantic confidences & frame_idx from the original npz
    sem_full = np.load(METHOD / "semantic_points.npz", allow_pickle=True)
    sem_conf = sem_full["confidences"]
    sem_frame = sem_full["frame_idx"]

    # Canonicalize labels to a fixed vocab so the PLY label_id is meaningful.
    base_vocab = ["chair", "couch", "table", "bed", "tv", "lamp", "pillow",
                  "blanket", "cabinet", "bookshelf", "rug", "window", "door",
                  "plant", "stove", "refrigerator", "oven", "microwave",
                  "sink", "toilet", "desk"]
    canon_to_id = {n: i for i, n in enumerate(base_vocab)}
    UNLABELED = 0xFFFF

    import re
    def canonicalize(name):
        words = re.findall(r"[a-z]+", name.lower())
        if "sofa" in words: return "couch"
        for w in words:
            if w in canon_to_id: return w
        return None

    # Per-point canonical label id
    canon_ids = np.full(len(sem_lbl), UNLABELED, dtype=np.uint16)
    for li, ln in enumerate(sem_names):
        c = canonicalize(ln)
        if c is None: continue
        m = sem_lbl == li
        canon_ids[m] = canon_to_id[c]
    valid_sem = canon_ids != UNLABELED
    print(f"Semantic points after canonicalization: {valid_sem.sum():,}/{len(canon_ids):,}")

    # Color palette tab10
    cmap = plt.cm.tab10
    palette = np.array([np.array(cmap(i % 10)[:3]) * 255 for i in range(len(base_vocab))],
                       dtype=np.uint8)

    # ---- 1. scene_full.ply ----
    print("\nscene_full.ply:")
    rgb_u8 = (np.clip(scene_rgb, 0, 1) * 255).astype(np.uint8)
    write_ply_binary(OUT / "scene_full.ply",
                     vertices=scene_pts.astype(np.float32),
                     colors=rgb_u8)

    # ---- 2. scene_semantic.ply (canonical-labelled subset) ----
    print("\nscene_semantic.ply:")
    sem_pts_c = sem_pts[valid_sem]
    sem_lbl_c = canon_ids[valid_sem]
    sem_conf_c = sem_conf[valid_sem]
    sem_frame_c = sem_frame[valid_sem]
    sem_color = palette[sem_lbl_c]
    write_ply_binary(OUT / "scene_semantic.ply",
                     vertices=sem_pts_c.astype(np.float32),
                     colors=sem_color,
                     fields=[("label_id", "uint16", sem_lbl_c),
                             ("confidence", "float32", sem_conf_c),
                             ("frame_idx", "uint16", sem_frame_c.astype(np.uint16))])

    # ---- 3. scene_combined.ply (scene + sem, with is_semantic + label_id) ----
    print("\nscene_combined.ply:")
    all_pts = np.concatenate([scene_pts, sem_pts_c]).astype(np.float32)
    # Color: scene uses RGB, semantic gets blended (RGB ⊕ palette to make it
    # visually obvious). Keep separate fields so loaders can re-color by label.
    sem_blend = (0.4 * sem_color + 0.6 * 255).astype(np.uint8)  # not used as color; the file
                                                                # uses scene RGB for scene,
                                                                # palette for semantic
    all_rgb = np.concatenate([rgb_u8, sem_color]).astype(np.uint8)
    is_sem = np.concatenate([np.zeros(len(scene_pts), dtype=np.uint8),
                              np.ones(len(sem_pts_c), dtype=np.uint8)])
    label_id_all = np.concatenate([np.full(len(scene_pts), UNLABELED, dtype=np.uint16),
                                    sem_lbl_c.astype(np.uint16)])
    confidence_all = np.concatenate([np.zeros(len(scene_pts), dtype=np.float32),
                                      sem_conf_c.astype(np.float32)])
    write_ply_binary(OUT / "scene_combined.ply",
                     vertices=all_pts,
                     colors=all_rgb,
                     fields=[("is_semantic", "uint8", is_sem),
                             ("label_id", "uint16", label_id_all),
                             ("confidence", "float32", confidence_all)])

    # ---- 4. cameras.ply (camera trajectory as a polyline-style point set) ----
    print("\ncameras.ply:")
    cam_color = np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(cams), 1))
    write_ply_binary(OUT / "cameras.ply",
                     vertices=cams.astype(np.float32),
                     colors=cam_color,
                     fields=[("frame_idx", "uint16",
                              np.arange(len(cams), dtype=np.uint16))])

    # ---- 5. labels.json (legend) ----
    legend = {
        "label_to_id": canon_to_id,
        "id_to_label": {i: n for n, i in canon_to_id.items()},
        "id_to_color_rgb": {i: palette[i].tolist() for i in range(len(base_vocab))},
        "unlabeled_id": UNLABELED,
        "coordinate_frame": ("gravity-aligned: floor at z=0, +Z = up. "
                             "X-Y is the BEV plane. Units = meters."),
        "scale_caveat": ("VGGT outputs are up to a global similarity (no metric "
                         "anchor). Distances are *relative*; absolute scale may "
                         "differ from the real world by an unknown constant."),
    }
    (OUT / "labels.json").write_text(json.dumps(legend, indent=2))
    print(f"  wrote {OUT / 'labels.json'}")

    print("\nAll PLY files saved under", OUT.resolve())


if __name__ == "__main__":
    main()
