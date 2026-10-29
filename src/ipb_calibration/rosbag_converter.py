"""Convert a ROS2 bag into the input folder of scripts/lcba.py (see SPEC.md).

Poses are picked by time-sync to the reference camera (first camera in the yaml):
every `every_n`-th image of that camera, and the nearest message of every other
sensor, dropped if further than `max_sync_dt`. Sensor names are the frame_id of
the message headers; init extrinsics come from the tf_static tree.
"""
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# PointCloud2 layouts: field -> (byte offset, numpy dtype), from the C++ structs (alignas(16)).
POINT_FORMATS = {
    "PointXYZIR": {"x": (0, "f4"), "y": (4, "f4"), "z": (8, "f4"),
                   "intensity": (16, "f4"), "ring": (20, "u2")},
    "PointXYZIRADT": {"x": (0, "f4"), "y": (4, "f4"), "z": (8, "f4"),
                      "intensity": (16, "f4"), "ring": (20, "u2"),
                      "azimuth": (24, "f4"), "distance": (28, "f4"),
                      "return_type": (32, "u1"), "time_stamp": (40, "f8")},
}
POINT_SIZES = {"PointXYZIR": 32, "PointXYZIRADT": 48}

# CameraInfo.distortion_model -> calibration.yaml camera_models
CAMERA_MODELS = {"plumb_bob": "pinhole", "rational_polynomial": "pinhole",
                 "equidistant": "fisheye", "fisheye": "fisheye"}

# sensor-name filter of io.py: camera needs one of these in its name, LiDAR one of the others
CAM_KEYS, LIDAR_KEYS = ("image", "cam"), ("points", "laser", "lidar")


def decode_cloud(msg, point_format):
    """PointCloud2 -> float32 [N,3] without NaN/inf points. Raises if msg does not match point_format."""
    layout, size = POINT_FORMATS[point_format], POINT_SIZES[point_format]
    if msg.point_step != size:
        raise ValueError(f"point_step {msg.point_step} != {size} of {point_format}")
    offsets = {f.name: f.offset for f in msg.fields}
    for name, (off, _) in layout.items():
        if offsets.get(name) != off:
            raise ValueError(f"{point_format}: field '{name}' at {offsets.get(name)}, expected {off}")
    end = ">" if msg.is_bigendian else "<"
    dt = np.dtype({"names": ["x", "y", "z"], "formats": [end + "f4"] * 3,
                   "offsets": [layout[n][0] for n in "xyz"], "itemsize": msg.point_step})
    raw = np.asarray(msg.data, dtype=np.uint8).reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step]
    p = np.ascontiguousarray(raw).reshape(-1).view(dt)
    xyz = np.stack([p["x"], p["y"], p["z"]], axis=-1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def select_poses(ref_stamps, other_stamps, every_n, max_sync_dt, max_poses=None):
    """Pose rows [ref_idx, idx_sensor1, ...]. other_stamps: one sorted stamp array per other sensor."""
    rows = []
    for r in range(0, len(ref_stamps), every_n):
        row = [r]
        for s in other_stamps:
            i = int(np.clip(np.searchsorted(s, ref_stamps[r]), 1, len(s) - 1)) if len(s) > 1 else 0
            if len(s) > 1 and abs(s[i - 1] - ref_stamps[r]) < abs(s[i] - ref_stamps[r]):
                i -= 1
            if abs(s[i] - ref_stamps[r]) > max_sync_dt:
                break
            row.append(i)
        else:
            rows.append(row)
    # a sensor message may not serve two poses (a stationary pose twice would be a duplicate view)
    kept = []
    for row in rows:
        if not kept or all(a != b for a, b in zip(row[1:], kept[-1][1:])):
            kept.append(row)
    rows = kept
    if max_poses and len(rows) > max_poses:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, max_poses).round().astype(int)]
    return np.array(rows, dtype=int).reshape(len(rows), len(other_stamps) + 1)


def _T(tf):
    T = np.eye(4)
    q, t = tf.transform.rotation, tf.transform.translation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def pose_in_root(tree, frame):
    """T_root_frame (frame -> root) walking child->parent links of the tf tree."""
    T = np.eye(4)
    while frame in tree:
        parent, T_pc = tree[frame]
        T = T_pc @ T
        frame = parent
    return T, frame


def init_pose(tree, cam0_frame, frame):
    """[rx, ry, rz, tx, ty, tz] (xyz Euler deg, m) of `frame` expressed in cam0, as calibration.yaml wants."""
    T0, root0 = pose_in_root(tree, cam0_frame)
    T, root = pose_in_root(tree, frame)
    if root != root0:
        raise ValueError(f"tf_static: '{frame}' (root '{root}') and '{cam0_frame}' (root '{root0}') are not connected")
    T = np.linalg.inv(T0) @ T
    return [*Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True).tolist(), *T[:3, 3].tolist()]


def convert(cfg):
    from rosbags.highlevel import AnyReader  # needs `pip install rosbags`

    cams, lidars = cfg["cameras"], cfg["lidars"]
    img_topics = [c["image_topic"] for c in cams]
    info_topics = [c["info_topic"] for c in cams]
    pc_topics = [l["topic"] for l in lidars]
    tf_topic = cfg.get("tf_static_topic", "/tf_static")
    out = Path(cfg["output"])
    use_header = cfg.get("use_header_stamp", True)

    stamps = defaultdict(list)   # topic -> [stamp] per message
    frame_ids, infos, tree = {}, {}, {}
    with AnyReader([Path(cfg["bag"])]) as reader:
        conns = [c for c in reader.connections if c.topic in {*img_topics, *pc_topics, *info_topics, tf_topic}]
        missing = {*img_topics, *pc_topics, *info_topics, tf_topic} - {c.topic for c in conns}
        if missing:
            raise ValueError(f"topics not in bag: {sorted(missing)}")
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == tf_topic:
                for tf in msg.transforms:
                    tree[tf.child_frame_id.lstrip("/")] = (tf.header.frame_id.lstrip("/"), _T(tf))
            elif conn.topic in info_topics:
                infos.setdefault(conn.topic, msg)
            else:
                stamps[conn.topic].append(stamp(msg) if use_header else ts * 1e-9)
                frame_ids.setdefault(conn.topic, msg.header.frame_id.lstrip("/"))

        names = [frame_ids[t] for t in img_topics + pc_topics]
        if len(set(names)) != len(names):
            raise ValueError(f"frame_ids must be unique per topic, got {names}")
        for n, keys in [(frame_ids[t], CAM_KEYS) for t in img_topics] + [(frame_ids[t], LIDAR_KEYS) for t in pc_topics]:
            if not any(k in n for k in keys):
                print(f"WARNING: sensor name '{n}' has none of {keys}: io.py will not classify it")

        arr = {t: np.array(stamps[t]) for t in img_topics + pc_topics}
        empty = [t for t, a in arr.items() if not len(a)]
        if empty:
            raise ValueError(f"no messages on {empty}")
        for t, a in arr.items():
            if np.any(np.diff(a) < 0):
                raise ValueError(f"{t}: stamps not increasing")
        poses = select_poses(arr[img_topics[0]], [arr[t] for t in img_topics[1:] + pc_topics],
                             cfg.get("every_n", 1), cfg.get("max_sync_dt", 0.05), cfg.get("max_poses"))
        if not len(poses):
            raise ValueError("no pose where every sensor has a message within max_sync_dt")
        print(f"{len(poses)} poses selected")

        # pass 2: write the selected messages
        topics = img_topics + pc_topics
        wanted = {t: {int(i): k for k, i in enumerate(poses[:, c])} for c, t in enumerate(topics)}
        for n in names:
            (out / n.replace("/", "")).mkdir(parents=True, exist_ok=True)
        count = defaultdict(int)
        fmt = {l["topic"]: l.get("point_format", cfg.get("point_format")) for l in lidars}
        for conn, ts, raw in reader.messages(connections=[c for c in conns if c.topic in topics]):
            i = count[conn.topic]
            count[conn.topic] += 1
            if i not in wanted[conn.topic]:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            folder = out / frame_ids[conn.topic].replace("/", "")
            k = wanted[conn.topic][i]
            if conn.topic in img_topics:
                img = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"{conn.topic}: cannot decode image {i}")
                cv2.imwrite(str(folder / f"{k:04d}.png"), img)
            else:
                np.save(folder / f"{k:04d}.npy", decode_cloud(msg, fmt[conn.topic]))

    models = []
    for c in cams:
        m = infos[c["info_topic"]].distortion_model
        if m not in CAMERA_MODELS:
            raise ValueError(f"{c['info_topic']}: unknown distortion_model '{m}', known {sorted(CAMERA_MODELS)}")
        models.append(CAMERA_MODELS[m])

    cam0 = names[0]
    calib = {
        "image_topics": names[:len(cams)],
        "point_cloud_topics": names[len(cams):],
        "camera_models": models,
        "init_cami_to_cam0": {n.replace("/", ""): init_pose(tree, cam0, n) for n in names[:len(cams)]},
        "init_lidari_to_cam0": {n.replace("/", ""): init_pose(tree, cam0, n) for n in names[len(cams):]},
    }
    with open(out / "calibration.yaml", "w") as f:
        yaml.safe_dump(calib, f, default_flow_style=False)
    print(f"written to {out}")
    return calib
