"""Single-run HTML report (no ground truth): the look of the experiment deck (templates/report/slides_template.html)
for one calibration. Written by scripts/lcba.py to <out>/report/report.html (+ images/, lib/), self-contained:
copy the report/ folder anywhere and open report.html (works from file://)."""
import base64
import io
import json
import shutil
from datetime import datetime
from html import escape
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ipb_calibration.metrics import MEANINGS  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[2] / "templates" / "report"
BLUE, ORANGE, GREY, INK = "#2f6fdb", "#e07b24", "#8a8f98", "#1b2430"
MAX_PTS = 12000  # per LiDAR in the 3D view


def _rig_views(lcba, results):
    """Top / front / side view of the sensors at pose 0 in the reference frame, x forward = heading of camera 0.
    Returns (png data URI, {sensor: xyz})."""
    pos, fwd = {}, {}
    for k, v in results.items():
        if not k.startswith(("camera", "lidar")):
            continue
        T = np.asarray(lcba.T_cam_map[0]) @ np.asarray(v["extrinsics"], float)
        pos[k], fwd[k] = T[:3, 3], T[:3, 2]
    f0 = fwd[next(k for k in pos if k.startswith("camera"))]
    yaw = np.arctan2(f0[1], f0[0])
    R = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])  # world -> rig heading
    c = np.mean(list(pos.values()), axis=0)
    pos = {k: R @ (p - c) + [0, 0, c[2]] for k, p in pos.items()}
    fwd = {k: R @ f for k, f in fwd.items()}

    fig = plt.figure(figsize=(9.6, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    views = [(fig.add_subplot(gs[:, 0]), 0, 1, "top view", "x forward [m]", "y left [m]"),
             (fig.add_subplot(gs[0, 1]), 1, 2, "front view", "y left [m]", "z [m]"),
             (fig.add_subplot(gs[1, 1]), 0, 2, "side view", "x forward [m]", "z [m]")]
    for ax, i, j, title, xl, yl in views:
        for k, p in pos.items():
            lid = k.startswith("lidar")
            ax.scatter(p[i], p[j], c=ORANGE if lid else BLUE, marker="s" if lid else "o", s=34, zorder=3)
            ax.annotate(k.replace("image_raw", "").replace("camera_", "").replace("lidar_", "").replace("points", ""),
                        (p[i], p[j]), fontsize=6, color=INK, xytext=(3, 3), textcoords="offset points")
            if not lid:
                d = fwd[k][[i, j]]
                if np.linalg.norm(d) > .05:
                    ax.arrow(p[i], p[j], d[0] * .4, d[1] * .4, head_width=.05, color=BLUE, alpha=.7, length_includes_head=True)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=7); ax.grid(alpha=.25); ax.margins(.15)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 1:
            ax.invert_xaxis()  # looking along +x: left is on the left
    fig.tight_layout(pad=.4)
    b = io.BytesIO()
    fig.savefig(b, dpi=130, bbox_inches="tight", transparent=True, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _table(head, rows):
    h = "".join(f"<th>{escape(c)}</th>" for c in head)
    r = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{r}</tbody></table>"


def _rpy(R):
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-R[2, 0]), np.arctan2(R[1, 0], R[0, 0])])


def _tables(results):
    m = results["metrics"]
    short = lambda k: k.replace("image_raw", "")
    cams = [k for k in results if k.startswith("camera")]
    lids = [k for k in results if k.startswith("lidar")]
    rows = []
    for k in cams:
        s, g = m[k]["reprojection_px"], m[k]["sigma"]
        rows.append([short(k), s["n"], f"{s['rmse']:.3f}", f"{s['mean']:.3f}", f"{s['median']:.3f}", f"{s['p95']:.3f}", f"{s['max']:.3f}",
                     "  ".join(f"{t}:{e:.2f}" for t, e in m[k]["worst_frames"][:3])])
    cam_metrics = _table(["camera", "corners", "RMSE [px]", "mean", "median", "p95", "max", "worst frames (frame:RMSE)"], rows)
    rows = []
    for k in lids:
        s, wf = m[k]["point_to_plane_m"], m[k]["inlier_ratio_worst_frame"]
        rows.append([k.replace("lidar_", ""), s["n"], f"{s['rmse'] * 1e3:.2f}", f"{s['median'] * 1e3:.2f}", f"{s['p95'] * 1e3:.2f}",
                     f"{m[k]['inlier_ratio']:.3f}", f"{wf[0]}: {wf[1]:.3f}"])
    lid_metrics = _table(["LiDAR", "points", "RMSE [mm]", "median", "p95", "inlier ratio", "worst frame"], rows)

    rows = []
    for k in cams:
        K, d, g = np.asarray(results[k]["K"]), results[k]["distortion_coeff"], m[k]["sigma"]["cx_cy_fx_fy_px"]
        rows.append([short(k), "pinhole" if results[k]["is_pinhole"] else "fisheye", f"{K[0, 0]:.2f}", f"{K[1, 1]:.2f}", f"{K[0, 2]:.2f}", f"{K[1, 2]:.2f}",
                     f"{g[2]:.2f} / {g[3]:.2f}", f"{g[0]:.2f} / {g[1]:.2f}", " ".join(f"{x:.4g}" for x in d)])
    intr = _table(["camera", "model", "fx [px]", "fy [px]", "cx [px]", "cy [px]", "σ fx / fy [px]", "σ cx / cy [px]", "distortion (k1 k2 p1 p2 k3 …)"], rows)

    rows = []
    for k in cams + lids:
        T, g = np.asarray(results[k]["extrinsics"], float), m[k]["sigma"]
        rows.append([short(k).replace("lidar_", ""), "camera" if k in cams else "LiDAR", " ".join(f"{x:.4f}" for x in T[:3, 3]),
                     " ".join(f"{x:.3f}" for x in _rpy(T[:3, :3])), " ".join(f"{x:.2f}" for x in g["t_mm"]), " ".join(f"{x:.4f}" for x in g["rot_deg"])])
    extr = _table(["sensor", "type", "t x y z [m]", "roll pitch yaw [°]", "σ t [mm]", "σ rot [°]"], rows)
    return dict(cam_metrics=cam_metrics, lid_metrics=lid_metrics, intr=intr, extr=extr)


def _clouds(lcba, world):
    """{lo, hi, lidars: {name: {n, xyz (uint16 b64), rgb (uint8 b64, colour = frame)}}} like make_report.cloud_js."""
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("viridis")
    sel = {}
    for name, scans in world.items():
        p = np.concatenate(scans)
        t = np.concatenate([np.full(len(s), i / max(len(scans) - 1, 1)) for i, s in enumerate(scans)])
        i = rng.choice(len(p), min(len(p), MAX_PTS), replace=False)
        sel[name] = (p[i], (cmap(t[i])[:, :3] * 255).astype(np.uint8))
    if lcba.ref_map is not None and len(lcba.ref_map.points):
        r = np.asarray(lcba.ref_map.points)
        sel["map"] = (r[rng.choice(len(r), min(len(r), MAX_PTS), replace=False)], None)
    allp = np.concatenate([p for p, _ in sel.values()])
    lo, hi = np.percentile(allp, 0.5, 0) - 1, np.percentile(allp, 99.5, 0) + 1
    out = {}
    for name, (p, c) in sel.items():
        q = (np.clip((p - lo) / (hi - lo), 0, 1) * 65535).astype("<u2")
        out[name] = dict(n=len(p), xyz=base64.b64encode(q.tobytes()).decode(),
                         rgb=base64.b64encode(c.tobytes()).decode() if c is not None else "")
    return dict(lo=lo.tolist(), hi=hi.tolist(), lidars=out)


def _images(verif, rep):
    """verification PNGs -> report/images/<camera>/<frame>_<proj|reproj>.jpg (900 px wide). Returns (cameras, frames)."""
    cams, frames = [], set()
    for d in sorted(Path(verif).glob("camera_*")):
        cams.append(d.name)
        (rep / "images" / d.name).mkdir(parents=True, exist_ok=True)
        for p in d.glob("*_*proj.png"):
            im = cv2.imread(str(p))
            im = cv2.resize(im, (900, round(im.shape[0] * 900 / im.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(rep / "images" / d.name / (p.stem + ".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 78])
            frames.add(p.name.split("_")[0])
    return cams, sorted(frames)


def make_run_report(lcba, results, world, verif_dir, path_out, experiment_name, config, report_text):
    """world = export_verification()'s {lidar name: [scan per pose in the reference frame]}."""
    rep = Path(path_out) / "report"
    rep.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATES / "lib", rep / "lib", dirs_exist_ok=True)
    cams, frames = _images(verif_dir, rep)
    m = results["metrics"]
    corners = sum(m[k]["reprojection_px"]["n"] for k in results if k.startswith("camera"))
    lid_pts = sum(m[k]["point_to_plane_m"]["n"] for k in results if k.startswith("lidar"))
    p2p = m.get("plane_to_plane_m")
    data = dict(cams=cams, frames=frames, clouds=_clouds(lcba, world) if world else None)
    fill = dict(
        name=escape(experiment_name), date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_cam=lcba.num_cameras, n_lid=lcba.num_lidar, n_pose=lcba.num_poses, n_tags=len(lcba.apriltag_coords) // 4, n_corners=corners, n_lid_pts=lid_pts,
        reproj=f"{np.sqrt(sum(m[k]['reprojection_px']['rmse'] ** 2 * m[k]['reprojection_px']['n'] for k in results if k.startswith('camera')) / corners):.3f}",
        p2p=f"{p2p['rmse'] * 1e3:.2f} mm" if p2p else "–", s0c=f"{m['sigma0']['cameras']:.2f}", s0l=f"{m['sigma0']['lidars']:.2f}",
        params=escape(f"tag family {config['tag_family']} · std_pix {config['std_pix']} px · std_apriltags {config['std_apriltags']} m · std_lidar {config['std_lidar']} m · "
                      f"dist_degree {config['dist_degree']} · max_scanpoints {config['max_scanpoints']}"),
        rig=_rig_views(lcba, results), meanings=escape(MEANINGS), report=escape(report_text), **_tables(results))
    html = (TEMPLATES / "run_report_template.html").read_text()
    for k, v in fill.items():
        html = html.replace("{{" + k + "}}", str(v))
    html = html.replace("/*DATA*/{}", json.dumps(data))
    (rep / "report.html").write_text(html)
    print(f"Run report written to {rep / 'report.html'}")
