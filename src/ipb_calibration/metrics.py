"""Quality metrics of a solved calibration: stored under `metrics` in results.yaml and printed at the end of the run."""
import numpy as np
from scipy.spatial.transform import Rotation

from ipb_calibration.utils import update_pose

MEANINGS = """Metric meanings
  reproj [px]          Distance between a detected tag corner and the same corner (adjusted 3D coords) projected
                       through the estimated camera model and frame pose. rmse = sqrt(mean(d^2)) over all corners;
                       mean/median/p95/max of d. Low values (~std_pix) mean the camera model explains the images.
                       A large max with a small median points to single bad detections.
  worst frames         Frames with the highest per-frame reprojection RMSE ("frame:rmse"); frame = 0-based index in
                       the sorted image files (not the file name). Check those images
                       (verification/<camera>/*_reproj.png) for blur, partial tags or wrong detections.
  point-to-plane [m]   Distance of LiDAR points (downsampled, final association within 3*std_lidar) to the local
                       plane of the reference map. rmse/median/p95 of |d|. Should be close to the sensor noise
                       (std_lidar); higher means a residual misalignment or map/scan inconsistency.
  plane-to-plane [m]   Multi-LiDAR consistency: for every map plane seen by two LiDARs (>= 50 inlier points each),
                       the difference of the LiDARs' mean signed point-to-plane offsets = distance between the two
                       LiDARs' versions of that plane. rmse/max over all LiDAR pairs and shared planes, per pair in
                       results.yaml. ~0 means the LiDARs agree with each other (not only with the map).
  inlier ratio         Share of LiDAR points that found a map plane within 3*std_lidar (mean over frames, worst
                       frame). Low values mean the scans are outside the map or still misaligned.
  sigma t/rot/K        1-sigma precision of the estimate from the inverse normal matrix, valid for the assumed
                       std_pix/std_lidar/std_apriltags. t/rot are relative to camera 0 (0 for camera 0 itself).
                       Says how well the data constrains a parameter (geometry, coverage), not how close it is to
                       the truth. Multiply by sigma0 if sigma0 is clearly >1.
  sigma0               A-posteriori variance factor sqrt(sum((residual/std)^2) / redundancy), per sensor type
                       (total redundancy incl. tag/camera-0 priors, split by observation count).
                       ~1: the assumed --std_pix / --std_lidar match the data. >1: residuals larger than assumed
                       (noise underestimated or a model error), sigmas above are too optimistic by this factor.
                       <1: std set too pessimistic.
  GT table             Sensor -> camera-0 extrinsics, intrinsics (cx cy fx fy) and distortion (k1 k2 p1 p2 k3),
                       GT vs result. RMSE = sqrt(mean((result-GT)^2)) over the listed components; rotation uses the
                       rotation vector of R_gt^T R_result [deg].
  err/sigma            Largest |result-GT| / sigma over the components. <~3: the error is within the predicted
                       precision. >>3: a bias the solver is not aware of (model error, wrong std, bad data).
  GT reproj            Reprojection with GT intrinsics/extrinsics and the reference tag coords, frame poses
                       re-fitted per frame to the GT rig (the GT config has no trajectory). This is the floor the
                       detections allow; a result far below it is overfitting, far above is a calibration error.
"""


def stats(d):
    d = np.abs(np.asarray(d, float))
    if len(d) == 0:
        return {"n": 0}
    return {"rmse": float(np.sqrt(np.mean(d**2))), "mean": float(d.mean()), "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)), "max": float(d.max()), "n": int(len(d))}


def reproj_residuals(cam, coords, poses, p_img, idx):
    """[(frame, coord idx, projected - detected [n,2] px)] for every frame with detections."""
    return [(t, i.astype(int), cam.project(coords[i.astype(int)], T) - p)
            for t, (T, p, i) in enumerate(zip(poses, p_img, idx)) if len(p)]


def lidar_residuals(lcba, l):
    """Inlier point-to-plane distances [m] per frame, the inlier ratio and a map-plane key per inlier point,
    associated like the solver's final step."""
    thr = lcba.std_lidar * lcba.outlier_mult
    res, ratio, keys = [], [], []
    for points, T in zip(lcba.points[l], lcba.T_cam_map):
        p = lcba.lidars[l].scan2world(points, T)
        d, i = lcba.map_kdtree.query(p, distance_upper_bound=thr, workers=5)
        ok = d < thr
        n_, q = lcba.map_normals[i[ok]], lcba.map_points[i[ok]]
        # ponytail: plane id = normal rounded to 0.2 (sign fixed by the largest component) + offset rounded to 0.1 m;
        # fine for walls/floor, a plane straddling a rounding edge splits in two (then it is just skipped as too small)
        n_ = n_ * np.sign(n_[np.arange(len(n_)), np.abs(n_).argmax(1)])[:, None]
        res.append(np.einsum("nd,nd->n", n_, p[ok] - q))
        keys.append(np.column_stack([np.round(n_ / 0.2), np.round(np.einsum("nd,nd->n", n_, q) / 0.1)]))
        ratio.append(ok.mean())
    return res, np.array(ratio), keys


def plane_to_plane(per_lidar, min_pts=50):
    """per_lidar: {name: (signed distances, plane keys)}; -> {pair: [mean offset diff on each shared plane, m]}"""
    off = {}
    for name, (d, k) in per_lidar.items():
        u, inv = np.unique(k, axis=0, return_inverse=True)
        inv = inv.ravel()
        cnt = np.bincount(inv)
        mean = np.bincount(inv, d) / np.maximum(cnt, 1)
        off[name] = {tuple(u[j]): mean[j] for j in range(len(u)) if cnt[j] >= min_pts}
    names = list(off)
    return {f"{a}|{b}": [off[a][p] - off[b][p] for p in off[a].keys() & off[b].keys()]
            for i, a in enumerate(names) for b in names[i + 1:]}


def _sigma(lcba, key, T):
    """1-sigma of t [mm], rotation [deg] relative to camera 0 and of cx cy fx fy [px].

    Every sensor pose is relative to a reference frame held to camera 0 only by a 1e-3 prior; propagating the
    joint covariance through inv(T_cam0) @ T removes that gauge from the sensor sigmas."""
    a, b = lcba.param_idx("cam_0")[0], lcba.param_idx(key)[0]
    T0 = lcba.cameras[0].T_cam_ref
    R = np.linalg.inv(T0) @ T

    def f(x):  # [t, rotvec] change of the relative pose for a solver update x of cam0 (x[:6]) and the sensor (x[6:])
        Q = np.linalg.inv(update_pose(T0, x[:3], x[3:6])) @ update_pose(T, x[6:9], x[9:])
        return np.concatenate([Q[:3, 3] - R[:3, 3], Rotation.from_matrix(R[:3, :3].T @ Q[:3, :3]).as_rotvec()])

    J = np.stack([f(e) for e in np.eye(12) * 1e-7], axis=1) / 1e-7
    idx = np.r_[a:a + 6, b:b + 6]
    sd = np.sqrt(np.abs(np.diag(J @ lcba.cov[np.ix_(idx, idx)] @ J.T)))
    out = {"t_mm": (sd[:3] * 1e3).tolist(), "rot_deg": np.degrees(sd[3:]).tolist()}
    if key.startswith("cam"):
        out["cx_cy_fx_fy_px"] = np.sqrt(np.abs(np.diag(lcba.cov)[b + 6:b + 10])).tolist()
    return out


def compute_metrics(lcba):
    m = {"sigma0": {}}
    groups = {}  # sigma0 group -> [sum((residual/std)^2), number of observations]
    w_sq, n = 0.0, 0
    for c, topic in enumerate(lcba.cfg["image_topics"][:lcba.num_cameras]):
        key = topic.replace("/", "")
        r = reproj_residuals(lcba.cameras[c], lcba.apriltag_coords, lcba.T_cam_map, lcba.p_img[c], lcba.ray2apriltag_idx[c])
        d = np.concatenate([np.linalg.norm(e, axis=1) for _, _, e in r]) if r else np.zeros(0)
        per_frame = {int(t): float(np.sqrt(np.mean(np.sum(e**2, axis=1)))) for t, _, e in r}
        w_sq += np.sum(d**2) / lcba.std_pix**2
        n += 2 * len(d)
        m[key] = {"reprojection_px": stats(d), "reprojection_rmse_per_frame_px": per_frame,
                  "worst_frames": [[t, v] for t, v in sorted(per_frame.items(), key=lambda x: -x[1])[:5]],
                  "sigma": _sigma(lcba, f"cam_{c}", lcba.cameras[c].T_cam_ref)}
    if n:
        groups["cameras"] = (w_sq, n)

    w_sq, n = 0.0, 0
    per_lidar = {}
    for l, topic in enumerate(lcba.cfg.get("point_cloud_topics", [])[:lcba.num_lidar]):
        key = topic.replace("/", "")
        res, ratio, keys = lidar_residuals(lcba, l)
        d = np.concatenate(res)
        per_lidar[key] = (d, np.concatenate(keys))
        w_sq += np.sum(d**2) / lcba.std_lidar**2
        n += len(d)
        m[key] = {"point_to_plane_m": stats(d), "inlier_ratio": float(ratio.mean()),
                  "inlier_ratio_worst_frame": [int(ratio.argmin()), float(ratio.min())],
                  "sigma": _sigma(lcba, f"lidar_{l}", lcba.lidars[l].T_lidar_ref)}
    if n:
        groups["lidars"] = (w_sq, n)
    pp = plane_to_plane(per_lidar)
    allv = np.concatenate([np.abs(v) for v in pp.values()]) if pp else np.zeros(0)
    if len(allv):
        m["plane_to_plane_m"] = {"rmse": float(np.sqrt(np.mean(allv**2))), "max": float(allv.max()), "n": int(len(allv)),
                                 "pairs": {k: {"rmse": float(np.sqrt(np.mean(np.square(v)))), "n": len(v)} for k, v in pp.items() if v}}
    # Redundancy = all observations - parameters. The tag-coordinate priors (3 per corner) and the camera-0 prior
    # are observations too; without them a single shot (each corner seen once, 3 coords each) gets a negative
    # redundancy. ponytail: total redundancy split by observation count, exact would be trace(I - H) per group
    n_all = sum(n for _, n in groups.values()) + 3 * lcba.num_tags + 6
    r_all = n_all - lcba.num_params
    for k, (w_sq, n) in groups.items():
        m["sigma0"][k] = float(np.sqrt(w_sq / max(r_all * n / n_all, 1)))
    return m


def table(head, rows):
    w = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    line = lambda r: " | ".join(s.ljust(w[i]) for i, s in enumerate(r))
    sep = "-+-".join("-" * x for x in w)
    return "\n".join([sep, line(head), sep] + [line(r) for r in rows] + [sep])


def metrics_report(m):
    f = lambda v, p=3: " ".join(f"{x:.{p}f}" for x in np.atleast_1d(v))
    rows = []
    for key, v in m.items():
        if key in ("sigma0", "plane_to_plane_m"):
            continue
        if "reprojection_px" in v:
            s = v["reprojection_px"]
            if s["n"]:
                rows.append((key, f"reproj [px] ({s['n']} corners)",
                             f"rmse {s['rmse']:.3f}  mean {s['mean']:.3f}  median {s['median']:.3f}  p95 {s['p95']:.3f}  max {s['max']:.3f}"))
            rows.append(("", "worst frames (frame:rmse px)", "  ".join(f"{t}:{e:.3f}" for t, e in v["worst_frames"])))
        else:
            s, wf = v["point_to_plane_m"], v["inlier_ratio_worst_frame"]
            rows.append((key, f"point-to-plane [m] ({s['n']} pts)",
                         f"rmse {s['rmse']:.4f}  median {s['median']:.4f}  p95 {s['p95']:.4f}  max {s['max']:.4f}"))
            rows.append(("", "inlier ratio", f"mean {v['inlier_ratio']:.3f}  worst frame {wf[0]}: {wf[1]:.3f}"))
        sg = v["sigma"]
        rows.append(("", "sigma t [mm] / rot [deg]", f"{f(sg['t_mm'], 2)} / {f(sg['rot_deg'], 4)}"))
        if "reprojection_px" in v:
            rows.append(("", "sigma cx cy fx fy [px]", f(sg["cx_cy_fx_fy_px"])))
    if "plane_to_plane_m" in m:
        p = m["plane_to_plane_m"]
        rows.append(("lidar pairs", f"plane-to-plane [m] ({p['n']} planes)", f"rmse {p['rmse']:.4f}  max {p['max']:.4f}"))
    rows += [("sigma0", k, f"{v:.3f}") for k, v in m["sigma0"].items()]
    return "Calibration quality\n" + table(("sensor", "metric", "value"), rows) + "\n"
