"""
Compare our cognitive-map output (outputs/vggt_22frames/cognitive_map.json)
against VSI-Bench's task-level ground truth for scene 41069025.

Reports per-question:
  - kind, question, GT answer
  - our prediction (when computable from the cognitive map)
  - error / verdict

Saves: pilot/outputs/vggt_22frames/gt_comparison.json
"""
import json
from pathlib import Path

import numpy as np


METHOD = Path("outputs/vggt_22frames")
SCENE = "41069025"
VSI_JSONL = "/home/haoming/x-spatial-manual/data/vsi_bench_full/eval_full.jsonl"


def load_qas():
    return [json.loads(l) for l in open(VSI_JSONL)
            if json.loads(l).get("scene_id") == SCENE]


VSI_TO_OURS = {"sofa": "couch"}    # VSI-Bench uses 'sofa', our vocab uses 'couch'

def normalize(name):
    n = name.lower().strip()
    return VSI_TO_OURS.get(n, n)


def find_object_in_q(q, objects):
    """Return the canonical label of any object whose name appears in the
    question (after sofa↔couch normalization), or None."""
    ql = q.lower()
    for vsi_word, our_word in VSI_TO_OURS.items():
        if vsi_word in ql and our_word in objects:
            return our_word
    for label in objects:
        if label in ql:
            return label
    return None


def find_two_objects_in_q(q, objects):
    """Return ordered list of up to N labels found in q."""
    ql = q.lower()
    found = []
    for w in ql.replace(",", " ").replace(".", " ").replace("?", " ").split():
        w = w.strip()
        nw = VSI_TO_OURS.get(w, w)
        if nw in objects and nw not in found:
            found.append(nw)
    return found


def main():
    qas = load_qas()
    cog = json.loads((METHOD / "cognitive_map.json").read_text())
    objects = {o["label"]: o for o in cog["objects"]}

    # Quick stats from our prediction
    print(f"\n=== Cognitive map summary ===")
    extent_x = cog["scene_extent_m"]["x"]
    extent_y = cog["scene_extent_m"]["y"]
    pred_room_m2 = (extent_x[1] - extent_x[0]) * (extent_y[1] - extent_y[0])
    print(f"Predicted scene extent: {extent_x[1]-extent_x[0]:.2f} m × "
          f"{extent_y[1]-extent_y[0]:.2f} m  ≈ {pred_room_m2:.2f} m² (BEV bbox)")
    print(f"Predicted objects: {sorted(objects.keys())}")

    results = []
    matches = mismatches = unanswerable = 0
    for qa in qas:
        kind = qa["kind"]; q = qa["question"]; a_gt = qa["answer"]
        pred = None; verdict = "unanswerable"; note = ""

        if kind == "object_counting":
            target = q.lower().split("how many")[1].split("(s)")[0].strip()
            target = normalize(target.replace("are in this room?", "").strip())
            if target in objects:
                pred = 1
                verdict = ("undercount" if int(a_gt) > 1 else "match" if int(a_gt) == 1 else "mismatch")
                note = ("our pipeline emits one bbox per label; "
                        "instance-level count needs per-instance segmentation")
            else:
                note = f"label '{target}' not in our 14-word vocab"

        elif kind == "object_size_estimation":
            target = find_object_in_q(q, objects)
            if target:
                size = objects[target]["size_m"]
                pred_cm = max(size) * 100   # BEV longest dim only
                pred = round(pred_cm, 0)
                note = "BEV-longest-dim only; full size needs z extent (avail in PLY)"
                err = abs(pred - float(a_gt))
                verdict = ("rough_match" if err < 30 else "mismatch")
            else:
                note = f"target object not in our vocab"

        elif kind == "object_abs_distance":
            named = find_two_objects_in_q(q, objects)
            if len(named) >= 2:
                a_, b_ = named[0], named[1]
                ca = np.array(objects[a_]["center_xy_m"])
                cb = np.array(objects[b_]["center_xy_m"])
                pred = round(float(np.linalg.norm(ca - cb)), 2)
                err = abs(pred - float(a_gt))
                verdict = ("rough_match" if err < 1.0 else "mismatch")
                note = f"centroid-to-centroid distance ({a_} ↔ {b_}); GT uses closest-point"
            else:
                note = f"could not find both objects (found: {named})"

        elif kind in ("object_rel_direction_easy", "object_rel_direction_medium",
                       "object_rel_direction_hard"):
            # Format: "If I am standing by A and facing B, is C to my <option>?"
            named = find_two_objects_in_q(q, objects)
            if len(named) >= 3:
                A_, B_, C_ = named[0], named[1], named[2]
                pA = np.array(objects[A_]["center_xy_m"])
                pB = np.array(objects[B_]["center_xy_m"])
                pC = np.array(objects[C_]["center_xy_m"])
                forward = pB - pA; forward = forward / (np.linalg.norm(forward) + 1e-9)
                # right vector = rotate forward by -90° (clockwise)
                right = np.array([forward[1], -forward[0]])
                vAC = pC - pA
                fwd_proj = float(vAC @ forward)
                rgt_proj = float(vAC @ right)
                if kind == "object_rel_direction_easy":
                    pred = "left" if rgt_proj < 0 else "right"
                elif kind == "object_rel_direction_medium":
                    if fwd_proj < -0.2 * np.linalg.norm(vAC):
                        pred = "back"
                    else:
                        pred = "left" if rgt_proj < 0 else "right"
                else:  # hard
                    fl = fwd_proj > 0
                    rg = rgt_proj > 0
                    pred = ("front-right" if (fl and rg) else
                            "front-left"  if (fl and not rg) else
                            "back-right"  if (not fl and rg) else "back-left")
                # Decode GT letter -> option string from VSI-Bench's options list
                opts = qa.get("options") or []
                gt_str = None
                for o in opts:
                    if o.startswith(f"{a_gt}. "): gt_str = o.split(". ", 1)[1].strip()
                if gt_str:
                    verdict = "match" if gt_str == pred else "mismatch"
                    note = (f"prompt: at {A_} facing {B_}, asking about {C_}. "
                            f"pred={pred}, GT={gt_str}")
                else:
                    verdict = "computed-but-no-options"
                    note = f"pred={pred}, GT-letter={a_gt}, options not parsable"
            else:
                note = f"need 3 objects, only found: {named}"

        elif kind == "room_size_estimation":
            pred = round(pred_room_m2, 2)
            err = abs(pred - float(a_gt))
            verdict = ("rough_match" if err < 5 else "mismatch")
            note = ("our 22-frame BEV bbox covers only a fragment of the room; "
                    "fuller coverage requires more video frames as VGGT input")

        else:
            note = f"kind '{kind}' not handled by simple cognitive-map comparison"

        if verdict in ("match", "rough_match"):
            matches += 1
        elif verdict == "unanswerable":
            unanswerable += 1
        else:
            mismatches += 1

        results.append({"kind": kind, "question": q, "gt": a_gt,
                        "pred": pred, "verdict": verdict, "note": note})

    print(f"\n=== Q-by-Q comparison ({len(qas)} VSI-Bench QAs) ===\n")
    for r in results:
        print(f"[{r['kind']:25s}]")
        print(f"  Q: {r['question'][:120]}")
        print(f"  GT  : {r['gt']}")
        print(f"  PRED: {r['pred']}   ({r['verdict']})")
        print(f"  Note: {r['note']}")
        print()

    summary = {
        "n_total": len(qas),
        "n_rough_match_or_match": matches,
        "n_mismatch": mismatches,
        "n_unanswerable_with_current_map": unanswerable,
        "predicted_room_extent_m2": round(pred_room_m2, 2),
        "results": results,
    }
    out = METHOD / "gt_comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {matches} match-ish, {mismatches} mismatch, "
          f"{unanswerable} unanswerable (out of {len(qas)})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
