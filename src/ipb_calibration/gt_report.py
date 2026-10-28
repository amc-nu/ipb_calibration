"""Compare a calibration result with the ground truth of an Isaac Sim rig config (`base_link.sensors`, `cameras`)."""
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ipb_calibration import camera as cd
from ipb_calibration.metrics import reproj_residuals, stats, table
from ipb_calibration.utils import update_pose

USD2CV = np.diag([1., -1., -1.])  # USD camera looks along -Z with +Y up, the model uses OpenCV (Z forward, Y down)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float))**2)))


def _match(key, names):
    """GT sensor name for a folder key (topic without '/'), longest prefix wins: camera_frontimage_raw -> camera_front."""
    hits = [n for n in names if key.startswith(n.replace("/", ""))]
    return max(hits, key=len) if hits else None


def _pose(s, deg, is_cam):
    # Isaac applies roll/pitch/yaw as intrinsic XYZ (checked against a converged solve: 0.02deg/7mm)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("XYZ", [s["roll"], s["pitch"], s["yaw"]], degrees=deg).as_matrix()
    if is_cam:
        T[:3, :3] = T[:3, :3] @ USD2CV
    T[:3, 3] = [s["x"], s["y"], s["z"]]
    return T


def _gt_camera(c, T):
    d = c["distortion_coefficients"]["data"]
    if c["distortion_model"] == "equidistant":
        # ponytail: equidistant k4 dropped, the model's radial polynomial stops at k3
        d = [d[0], d[1], 0, 0, d[2]]
    dist = cd.CVDistortionModel(degree=3, division_model=False, cv2_coeff=list(d[:5]))
    return cd.Camera(np.array(c["camera_matrix"]["data"], float).reshape(3, 3), T, dist,
                     is_pinhole=c["distortion_model"] != "equidistant")


def _refit_poses(lcba, gt_cams):
    """cam0 pose per frame re-fitted to the GT rig and the reference tag coords (start: the estimated pose)."""
    poses = []
    for t, T0 in enumerate(lcba.T_cam_map):
        obs = [(cam, lcba.coords_prior[lcba.ray2apriltag_idx[c][t].astype(int)], lcba.p_img[c][t])
               for c, cam in enumerate(gt_cams) if cam is not None and len(lcba.p_img[c][t])]
        if not obs:
            poses.append(T0)
            continue
        f = lambda x: np.concatenate([(cam.project(X, update_pose(T0, x[:3], x[3:])) - p).ravel() for cam, X, p in obs])
        x = least_squares(f, np.zeros(6)).x
        poses.append(update_pose(T0, x[:3], x[3:]))
    return poses


def _fmt(v, p=4):
    return " ".join(f"{x:.{p}f}" for x in np.atleast_1d(v))


def gt_report(lcba, metrics, gt_file, apriltag_file=None, csv_file=None):
    """Returns a text table (sensor, GT, result, RMSE, err/sigma; sigmas from metrics.compute_metrics) and
    optionally writes per-corner reprojection residuals of GT and result to csv_file."""
    gt = yaml.safe_load(open(gt_file))
    deg = gt.get("rotational_units", "deg") == "deg"
    sensors = gt["base_link"]["sensors"]
    gt_specs = {c["sensor_name"]: c for c in gt.get("cameras", [])}
    cam_keys = [t.replace("/", "") for t in lcba.cfg["image_topics"]]
    lidar_keys = [t.replace("/", "") for t in lcba.cfg.get("point_cloud_topics", [])][:lcba.num_lidar]

    name0 = _match(cam_keys[0], sensors)
    if name0 is None:
        return f"GT report skipped: no sensor in {gt_file} matches reference camera {cam_keys[0]}\n"
    T_w_c0 = _pose(sensors[name0], deg, True)
    T_gt = {k: np.linalg.inv(T_w_c0) @ _pose(sensors[n], deg, k in cam_keys)
            for k in cam_keys + lidar_keys for n in [_match(k, sensors)] if n is not None}

    def z(err, sigma):  # largest |error| / sigma, blank where sigma is ~0 (camera 0 extrinsics)
        sigma = np.asarray(sigma)
        return f"{np.max(np.abs(err) / sigma):.2f}" if np.all(sigma > 1e-6) else ""

    rows = []

    def extr(key, T_res):
        G, sg = T_gt[key], metrics[key]["sigma"]
        e_r = np.degrees(Rotation.from_matrix(G[:3, :3].T @ T_res[:3, :3]).as_rotvec())  # local rotvec, as the solver's update
        eul = [Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True) for T in (G, T_res)]  # calibration.yaml format
        rows.append((key, "t [m]", _fmt(G[:3, 3]), _fmt(T_res[:3, 3]), f"{rmse(G[:3, 3], T_res[:3, 3]):.4f}",
                     z((T_res[:3, 3] - G[:3, 3]) * 1e3, sg["t_mm"])))
        rows.append(("", "rpy xyz [deg]", _fmt(eul[0], 3), _fmt(eul[1], 3), f"{rmse(e_r, 0):.4f}", z(e_r, sg["rot_deg"])))

    gt_cams = [_gt_camera(gt_specs[n], T_gt[k]) if (n := _match(k, sensors)) in gt_specs else None for k in cam_keys]
    poses_gt = _refit_poses(lcba, gt_cams)
    tag_ids = None
    if apriltag_file is not None:  # label corners id*100+corner like apriltag_coords.txt
        ref = np.loadtxt(apriltag_file, skiprows=1)
        tag_ids = ref[np.argmin(np.linalg.norm(lcba.coords_prior[:, None] - ref[None, :, 1:], axis=-1), axis=1), 0]
    corners = []
    for c, (key, gt_cam) in enumerate(zip(cam_keys, gt_cams)):
        if gt_cam is None:
            rows.append((key, "-", "no GT", "", "", ""))
            continue
        cam = lcba.cameras[c]
        extr(key, cam.T_cam_ref)
        k_gt, k_res = [[C.K[0, 2], C.K[1, 2], C.K[0, 0], C.K[1, 1]] for C in (gt_cam, cam)]  # solver order
        rows.append(("", "cx cy fx fy [px]", _fmt(k_gt, 2), _fmt(k_res, 2), f"{rmse(k_gt, k_res):.4f}",
                     z(np.subtract(k_res, k_gt), metrics[key]["sigma"]["cx_cy_fx_fy_px"])))
        d_gt, d_res = gt_cam.distortion.to_cv2()[:5], cam.distortion.to_cv2()[:5]
        # ponytail: no err/sigma for distortion, its cov order depends on --dist_degree/--division_model
        rows.append(("", "k1 k2 p1 p2 k3", _fmt(d_gt), _fmt(d_res), f"{rmse(d_gt, d_res):.4f}", ""))

        args = (lcba.p_img[c], lcba.ray2apriltag_idx[c])
        r_gt = reproj_residuals(gt_cam, lcba.coords_prior, poses_gt, *args)
        r_res = reproj_residuals(cam, lcba.apriltag_coords, lcba.T_cam_map, *args)
        if not r_gt:
            continue
        s_gt, s_res = [stats(np.concatenate([np.linalg.norm(e, axis=1) for _, _, e in r])) for r in (r_gt, r_res)]
        rows.append(("", f"reproj rmse [px] ({s_gt['n']} corners)", f"{s_gt['rmse']:.4f}", f"{s_res['rmse']:.4f}", "", ""))
        rows.append(("", "reproj max [px]", f"{s_gt['max']:.4f}", f"{s_res['max']:.4f}", "", ""))
        for (t, idx, eg), (_, _, er) in zip(r_gt, r_res):
            for i, pd, e1, e2 in zip(idx, lcba.p_img[c][t], eg, er):
                corners.append((key, t, tag_ids[i] if tag_ids is not None else i, *pd, *e1, *e2))

    for l, key in enumerate(lidar_keys):
        if key not in T_gt:
            rows.append((key, "-", "no GT", "", "", ""))
            continue
        extr(key, lcba.lidars[l].T_lidar_ref)

    out = [f"Ground truth comparison ({gt_file}), extrinsics are sensor -> {cam_keys[0]}",
           table(("sensor", "parameter", "GT", "result", "RMSE", "err/sigma"), rows)]

    if csv_file is not None and corners:
        np.savetxt(csv_file, np.array(corners, dtype=object), delimiter=",", comments="",
                   fmt=["%s", "%d", "%d", "%.3f", "%.3f", "%.4f", "%.4f", "%.4f", "%.4f"],
                   header="camera,frame,corner_id,u_det,v_det,du_gt,dv_gt,du_result,dv_result")
        out.append(f"Per-corner reprojection residuals (projected - detected) written to {csv_file}")
    return "\n".join(out) + "\n"
