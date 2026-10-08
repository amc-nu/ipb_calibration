"""Headless export to verify a calibration: aligned scans as .ply and LiDAR
points projected into every camera image."""
import glob
from os.path import basename, join
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from numpy.linalg import inv

from ipb_calibration.ipb_preprocessing import load_scan
from ipb_calibration.utils import homogenous, np2o3d


def _overlay(img, cam, pts_world, T_cam_map, radius=1, max_depth=None):
    """Draw points (world frame) into a BGR image, colored by depth."""
    x_c = (inv(cam.T_cam_ref) @ inv(T_cam_map) @ homogenous(pts_world).T)[:3].T
    depth = np.linalg.norm(x_c, axis=-1)
    ok = depth > 0
    if cam.is_pinhole:
        ok &= x_c[:, 2] > 0
    if not ok.any():
        return img
    uv = np.round(cam.project(pts_world[ok], T_cam_map)).astype(int)
    depth = depth[ok]
    h, w = img.shape[:2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, depth = uv[ok], depth[ok]
    if len(uv) == 0:
        return img
    max_depth = max_depth or np.percentile(depth, 95)
    colors = (plt.get_cmap("turbo")(np.clip(depth / max_depth, 0, 1))[:, 2::-1]
              * 255).astype(np.uint8)  # RGB -> BGR
    order = np.argsort(-depth)  # far first, near points end up on top
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            v = np.clip(uv[order, 1] + dy, 0, h - 1)
            u = np.clip(uv[order, 0] + dx, 0, w - 1)
            img[v, u] = colors[order]
    return img


def export_verification(lcba, path_data, path_out):
    """Write to path_out:
      - lidar_<name>_scans.ply: all scans in the world (reference map) frame,
        colored by pose index
      - <camera>/<image>_proj.png: every image with the LiDAR scan of the same
        pose projected into it (colored by depth)
    Uses the full (non-downsampled) scans from path_data.
    """
    if lcba.num_lidar == 0:
        print("No lidar data, nothing to export.")
        return
    path_out = Path(path_out)
    path_out.mkdir(parents=True, exist_ok=True)
    cfg = lcba.cfg

    # world-frame scans per lidar: list over poses
    world = []
    for topic, lidar in zip(cfg["point_cloud_topics"], lcba.lidars):
        name = topic.replace("/", "")
        files = sorted(glob.glob(join(path_data, name, "*.npy")))
        scans = [lidar.scan2world(np.asarray(load_scan(f).points), lcba.T_cam_map[t])
                 for t, f in enumerate(files[:lcba.num_poses])]
        world.append(scans)

        cmap = plt.get_cmap("viridis")
        pts = np.concatenate(scans)
        cols = np.concatenate([np.tile(cmap(t / len(scans))[:3], (len(s), 1))
                               for t, s in enumerate(scans)])
        o3d.io.write_point_cloud(str(path_out / f"lidar_{name}_scans.ply"),
                                 np2o3d(pts, cols))

    for topic, cam in zip(cfg["image_topics"], lcba.cameras):
        name = topic.replace("/", "")
        (path_out / name).mkdir(exist_ok=True)
        images = sorted(glob.glob(join(path_data, name, "*.png")))
        for t, f in enumerate(images[:lcba.num_poses]):
            img = cv2.imread(f)
            for scans in world:
                if t < len(scans):
                    img = _overlay(img, cam, scans[t], lcba.T_cam_map[t])
            cv2.imwrite(str(path_out / name / (basename(f)[:-4] + "_proj.png")), img)
    print(f"Verification export written to {path_out}")
