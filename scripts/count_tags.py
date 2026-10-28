"""Detect the AprilTags of a single-shot dataset (first image per camera), count them per camera and per tag-set (set label
from the board pose in the dataset's config.yaml, H-scene naming), write a 3x3 contact sheet with the detections drawn. Runs in
the calibration container: docker run --rm -v $PWD:/root/calib -v <dataset>:/root/ds:ro -v $PWD/output:/root/output -w /root
  or_ipb_calibration-systemcalib:latest python3 calib/scripts/count_tags.py ds output/<name>_detections.png"""
import sys, glob, os, yaml, numpy as np, cv2
from ipb_calibration.apriltag import Apriltags

ds, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(os.path.join(ds, "config.yaml")))
calib = yaml.safe_load(open(os.path.join(ds, "input", "calibration.yaml")))


def normal(q):
    w, x, y, z = q
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def label(b):
    p = np.array(b["position"]); n = normal(b["quaternion"])
    if abs(n[2]) > 0.9:
        return f"floor {'N' if p[1] > 3 else 'S' if p[1] < -3 else 'E' if p[0] > 3 else 'W'}{'+' if (p[0] if abs(p[1]) > 3 else p[1]) > 0 else '-'}"
    if abs(n[0]) > 0.6 and abs(n[1]) > 0.6:
        return f"corner {'N' if p[1] > 0 else 'S'}{'E' if p[0] > 0 else 'W'} {'up' if p[2] > 1 else 'low'}"
    w = "N" if p[1] > 4.5 else "S" if p[1] < -4.5 else "E" if p[0] > 4.5 else "W"
    s = p[0] if w in "NS" else p[1]
    return f"wall {w} {'tilt' if abs(s) > 2 else 'flat'}{'+' if s > 0 else '-'}"


tag_label = {int(b["filename"].split("_")[-1].split(".")[0]): label(b) for b in cfg["targets"]["boards"]}
april = Apriltags(os.path.join(ds, "reference", "apriltag_coords.txt"), "tag36h11")
tiles = []
print(f"{'camera':17s} tags  planes>=6  sets(tags)")
for topic in calib["image_topics"]:
    folder = os.path.join(ds, "input", topic.replace("/", ""))
    img = cv2.imread(sorted(glob.glob(f"{folder}/*.png"))[0])
    dets = april.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    sets = {}
    for d in dets:
        sets.setdefault(tag_label.get(d["tag_id"], "?"), []).append(d)
        c = np.asarray(d["corners"]).reshape(-1, 2).astype(int) if "corners" in d else None
        if c is not None:
            cv2.polylines(img, [c], True, (0, 255, 0), 4)
    # coplanar sets: the two corner sets share a plane, the two flat wall sets share a plane, the floor sets share a plane
    planes = {}
    for k, v in sets.items():
        pk = k.replace(" up", "").replace(" low", "") if k.startswith("corner") else k[:-1] if "flat" in k else "floor" if k.startswith("floor") else k
        planes[pk] = planes.get(pk, 0) + len(v)
    cam = topic.split("/")[1]
    print(f"{cam:17s} {len(dets):4d}  {sum(n >= 6 for n in planes.values()):5d}      " + ", ".join(f"{k} {len(v)}" for k, v in sorted(sets.items())))
    small = cv2.resize(img, (960, 620))
    cv2.putText(small, f"{cam}: {len(dets)} tags", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    tiles.append(small)
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)]
cv2.imwrite(out, np.vstack(rows))
print("wrote", out)
