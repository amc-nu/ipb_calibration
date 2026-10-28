"""Post-calibration check of the input data: flags properties of the dataset that made the
calibration fail, converge slowly or be poorly constrained, with a hint on how to record better data."""
import numpy as np
from scipy.spatial.transform import Rotation


def check_data(lcba, img_sizes, out_file=None):
    issues = []  # (problem, suggestion)

    def add(problem, suggestion):
        issues.append((problem, suggestion))

    if not (lcba.converged and lcba.final_step):
        add(f"Optimization did not converge within {lcba.num_iter_used} iterations.",
            "Results may be unreliable; see the issues below, or raise num_iter in scripts/lcba.py.")

    cam_names = [t.replace("/", "") for t in lcba.cfg["image_topics"]]
    for c in range(lcba.num_cameras):
        name, cam = cam_names[c], lcba.cameras[c]
        f_init, f_final = lcba.k[c][2], cam.K[0, 0]
        if abs(f_init - f_final) > 0.2 * f_final:
            add(f"{name}: initial focal length {f_init:.0f}px is far from the calibrated {f_final:.0f}px "
                "(initial intrinsics come from one plane of tags per image).",
                "Record some images close to a wall of tags, tilted (30-60deg) and from several angles, so that "
                ">= 8 tags on one plane fill most of the image.")

        obs = [p for p in lcba.p_img[c] if len(p) > 0]
        if len(obs) < 10:
            add(f"{name}: tags detected in only {len(obs)} images.",
                "Record more frames with tags in view (>= 10, better 20+), at different distances.")
        if obs:
            w, h = img_sizes[c]
            pts = np.concatenate(obs)
            cells = np.unique((np.clip(pts[:, 0] / w, 0, 0.999) * 8).astype(int) * 8 +
                              (np.clip(pts[:, 1] / h, 0, 0.999) * 8).astype(int))
            if len(cells) < 32:
                add(f"{name}: tag corners cover only {len(cells)/64:.0%} of the image area (8x8 grid).",
                    "Distortion is unconstrained where no tags are seen: record frames with tags near the image "
                    "borders and corners.")

        err = np.concatenate([e for e in lcba.errors[f"cam_{c}"] if len(e) > 0])
        rms = np.sqrt(np.mean(err**2))
        if rms > 3 * lcba.std_pix:
            add(f"{name}: reprojection RMS {rms:.2f}px is {rms/lcba.std_pix:.1f}x the assumed --std_pix "
                f"{lcba.std_pix}.",
                f"Set --std_pix ~{rms:.1f}, or check for blur / wrong camera model / wrong tag coordinates.")

    if lcba.num_cameras > 1:
        seen = np.array([[len(p) > 0 for p in lcba.p_img[c]] for c in range(lcba.num_cameras)])
        for c in range(1, lcba.num_cameras):
            n = np.sum(seen[0] & seen[c]) if np.any(seen[c]) else 0
            if n < 3:
                add(f"{cam_names[c]}: sees tags together with {cam_names[0]} in only {n} frames.",
                    "Its extrinsics rely on the LiDAR/poses only; record frames where both cameras see tags.")

    # Trajectory diversity: pure translation along a line leaves the extrinsics weakly constrained
    if lcba.num_poses > 1:  # single shot: no trajectory, the targets alone must constrain everything
        R0 = lcba.T_cam_map[0][:3, :3]
        max_rot = max(np.rad2deg(np.linalg.norm(Rotation.from_matrix(R0.T @ T[:3, :3]).as_rotvec()))
                      for T in lcba.T_cam_map)
        pos = np.stack([T[:3, 3] for T in lcba.T_cam_map])
        sv = np.linalg.svd(pos - pos.mean(0), compute_uv=False) / np.sqrt(len(pos))
        if max_rot < 10:
            add(f"Trajectory: orientation changes by at most {max_rot:.1f}deg over all frames (pure translation).",
                "Rotate the platform during recording (turns of >= 30-90deg in yaw, and some pitch/roll if possible).")
        if sv[1] < 0.2:
            add(f"Trajectory: frames lie on a straight line (spread across the motion {sv[1]:.2f}m).",
                "Drive curves / lateral offsets and, if possible, vary the height.")

    if lcba.num_lidar > 0:
        lo, hi = lcba.map_points.min(0) - 0.5, lcba.map_points.max(0) + 0.5
        lidar_names = [t.replace("/", "") for t in lcba.cfg["point_cloud_topics"]]
        for l, (lidar, scans) in enumerate(zip(lcba.lidars, lcba.points)):
            name = lidar_names[l]
            if lcba.lidar_inlier_ratio_init is None:  # the LiDAR stage never ran (camera stage diverged)
                add(f"{name}: the LiDAR stage was not reached (the camera stage failed).", "Fix the camera stage first (see the log).")
                continue
            init, final = lcba.lidar_inlier_ratio_init[l].mean(), lcba.lidar_inlier_ratio[l].mean()
            if init < 0.3:
                add(f"{name}: only {init:.0%} of points matched the reference map with the initial guess.",
                    f"Improve init_lidari_to_cam0/{name} in calibration.yaml (roughly within 0.1m / 5deg).")
            if final < 0.5:
                add(f"{name}: only {final:.0%} of points match the reference map after calibration.",
                    "Check that the reference map is from the same environment and the LiDAR/camera "
                    "frames are time-synchronized.")
            w = np.concatenate([lidar.scan2world(s, T) for s, T in zip(scans, lcba.T_cam_map)])
            outside = np.mean(np.any((w < lo) | (w > hi), axis=-1))
            if outside > 0.2:
                add(f"{name}: {outside:.0%} of points lie outside the reference map.",
                    "Extend the TLS scan to everything the LiDAR sees, or record where the map covers the "
                    "LiDAR's view.")

    lines = ["Input data check: " + ("no issues found." if not issues else f"{len(issues)} issue(s)")]
    for i, (p, s) in enumerate(issues, 1):
        lines += [f"{i}. {p}", f"   -> {s}"]
    report = "\n".join(lines)
    print(report)
    if out_file:
        with open(out_file, "w") as f:
            f.write(report + "\n")
    return issues
