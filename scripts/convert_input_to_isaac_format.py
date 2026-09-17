"""Converts this repo's legacy input/ layout (calibration.yaml + per-topic
image/.npy folders) into the Isaac Sim capture layout lcba.py also accepts
(config.yaml + camera/ + lidar/ + gt/), so the real recorded rig data can be
run through the same path as a synthetic capture.

Camera/LiDAR extrinsics come straight from calibration.yaml's
init_cami_to_cam0/init_lidari_to_cam0 (cam0 stands in for base_link, since
the legacy format has no separate base_link concept and cam0's own offset is
already identity). Camera intrinsics/distortion are *estimated* from the
real detections (cv2.calibrateCamera via estimate_initial_k) since, unlike a
synthetic Isaac capture, no exact ground truth intrinsics exist for a real
camera -- this is a best-available initial guess, not ground truth.
`gt/apriltag_<frame_id>.csv` ground truth is built by running the same
AprilTag detector the legacy path itself uses against every real image, so
"visible" means "the detector found it", not a geometric projection check
(there's no independent ground-truth pose to project from).
"""
import csv
import shutil
from os.path import join
from pathlib import Path

import click
import numpy as np
import yaml
from matplotlib import pyplot as plt

from ipb_calibration import ipb_preprocessing as pp
from ipb_calibration.apriltag import Apriltags
from npy2pcd import write_pcd

_CORNER_COLS = ("top_left", "top_right", "bottom_right", "bottom_left")


def sensor_name_for(topic, prefix):
    # "/camera/front/image_raw" -> "camera_front", "/lidar/horizontal/points" -> "lidar_horizontal"
    return f"{prefix}_{topic.strip('/').split('/')[1]}"


@click.command()
@click.option("--path_data", "-p", default="input", help="Legacy input/ directory (calibration.yaml + topic folders).")
@click.option("--apriltag_file", "-a", default="reference/apriltag_coords.txt", help=".txt or .csv reference AprilTag corner coordinates.")
@click.option("--path_out", "-o", required=True, help="Output Isaac-format capture directory (created fresh).")
def main(path_data, apriltag_file, path_out):
    cfg = yaml.safe_load(open(join(path_data, "calibration.yaml")))
    april = Apriltags(apriltag_file)

    camera_topics = cfg["image_topics"]
    lidar_topics = cfg["point_cloud_topics"]
    camera_names = [sensor_name_for(t, "camera") for t in camera_topics]
    lidar_names = [sensor_name_for(t, "lidar") for t in lidar_topics]
    models = cfg.get("camera_models", len(camera_topics) * ["pinhole"])

    out = Path(path_out)
    (out / "camera").mkdir(parents=True, exist_ok=True)
    (out / "lidar").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)

    # --- cameras: copy images, detect tags, collect per-frame observations ---
    img_sizes = np.zeros([len(camera_topics), 2], dtype=np.int64)
    observations = [[] for _ in camera_topics]
    frame_ids = None
    for c, (topic, name) in enumerate(zip(camera_topics, camera_names)):
        folder = join(path_data, topic.replace("/", ""))
        images = sorted(Path(folder).glob("*.png"))
        if frame_ids is None:
            frame_ids = [int(p.stem) for p in images]

        for frame_id, image in zip(frame_ids, images):
            img = plt.imread(image)
            img_sizes[c] = img.shape[1], img.shape[0]
            gray = (np.mean(img, axis=-1) * 255).astype("uint8")
            dets = april.detect(gray)
            observations[c].append(dets)
            shutil.copy2(image, out / "camera" / f"{name}_{frame_id:04d}.png")
        print(f"{name}: {len(images)} frames copied, "
              f"{sum(len(d) for d in observations[c])} tag detections")

    cam_is_pinhole = [m == "pinhole" for m in models]
    init_k, init_coeff = pp.estimate_initial_k(
        observations, cam_is_pinhole, max_num_images=10, image_size=img_sizes)

    # --- lidars: npy -> pcd, same frame_ids as the cameras ---
    for topic, name in zip(lidar_topics, lidar_names):
        folder = join(path_data, topic.replace("/", ""))
        scans = sorted(Path(folder).glob("*.npy"))
        for frame_id, scan_f in zip(frame_ids, scans):
            points = np.load(scan_f)
            write_pcd(points, str(out / "lidar" / f"{name}_{frame_id:04d}_local.pcd"))
        print(f"{name}: {len(scans)} scans converted")

    # --- config.yaml: cam0 stands in for base_link (its own offset is already identity) ---
    def offset_dict(values):
        roll, pitch, yaw, x, y, z = values
        return {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw}

    cameras_cfg = []
    for c, (topic, name) in enumerate(zip(camera_topics, camera_names)):
        cx, cy, fx, fy = init_k[c]
        k1, k2, p1, p2, k3 = np.asarray(init_coeff[c]).flatten()[:5]
        cameras_cfg.append({
            "sensor_name": name,
            "model": models[c],
            "resolution": {"width": int(img_sizes[c, 0]), "height": int(img_sizes[c, 1])},
            "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
            "distortion": {"k1": float(k1), "k2": float(k2), "p1": float(p1),
                          "p2": float(p2), "k3": float(k3)},
        })
    lidars_cfg = [{"sensor_name": name} for name in lidar_names]

    base_link_sensors = {}
    for topic, name in zip(camera_topics, camera_names):
        base_link_sensors[name] = offset_dict(cfg["init_cami_to_cam0"][topic.replace("/", "")])
    for topic, name in zip(lidar_topics, lidar_names):
        base_link_sensors[name] = offset_dict(cfg["init_lidari_to_cam0"][topic.replace("/", "")])

    isaac_cfg = {
        "cameras": cameras_cfg,
        "lidars": lidars_cfg,
        "base_link": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                      "sensors": base_link_sensors},
    }
    with open(out / "config.yaml", "w") as f:
        f.write("# Converted from legacy calibration.yaml by convert_input_to_isaac_format.py.\n"
               "# Camera intrinsics/distortion are ESTIMATED from real detections, not exact\n"
               "# ground truth (unlike a synthetic Isaac capture's config.yaml).\n")
        yaml.safe_dump(isaac_cfg, f, default_flow_style=False)

    # --- gt/apriltag_<frame_id>.csv, one file per frame, rows per (camera, visible tag) ---
    header = ["frame_id", "tag_family", "tag_id", "sensor_name", "image_filename",
              "top_left_x_2d", "top_left_y_2d", "top_right_x_2d", "top_right_y_2d",
              "bottom_right_x_2d", "bottom_right_y_2d", "bottom_left_x_2d", "bottom_left_y_2d",
              "coordinate_frame", "pointcloud_filenames",
              "center_x_3d", "center_y_3d", "center_z_3d"]
    for col in _CORNER_COLS:
        header += [f"{col}_x_3d", f"{col}_y_3d", f"{col}_z_3d"]

    pointcloud_filenames = ";".join(
        f"lidar/{name}_{{frame_id:04d}}_local.pcd" for name in lidar_names)

    for t, frame_id in enumerate(frame_ids):
        with open(out / "gt" / f"apriltag_{frame_id:04d}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for c, name in enumerate(camera_names):
                for det in observations[c][t]:
                    tl, tr, bl, br = det["corners"]  # internal order
                    ctl, ctr, cbl, cbr = det["coords"]
                    center = det["coords"].mean(axis=0)
                    row = [frame_id, "tag36h11", det["tag_id"], name,
                          f"camera/{name}_{frame_id:04d}.png",
                          *tl, *tr, *br, *bl,
                          "global", pointcloud_filenames.format(frame_id=frame_id),
                          *center, *ctl, *ctr, *cbr, *cbl]
                    w.writerow(row)

    print(f"Wrote Isaac-format capture to {out}")


if __name__ == "__main__":
    main()
