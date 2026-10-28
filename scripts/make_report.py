"""Static HTML report of the Isaac box-rig calibration experiments (host python: numpy + pyyaml only).

usage: python scripts/make_report.py [output_dir=output]
Reads output/exp_Box*/ (results.yaml, reprojection_corners.csv, report.txt, verification/) and the matching
output/dataset_Box*_2*/config.yaml (= ground truth + scene), writes output/report/index.html, data/<exp>.js
(3D point clouds, loaded lazily), lib/ (three.js, from report/lib) and images/<exp>/<camera>/<frame>_<proj|reproj>.jpg
(verification images downsized to 1100 px wide, only (re)written when missing or older than the source), so output/report/
is self-contained: copy that folder anywhere and open index.html in a browser (works from file://).
"""
import base64, csv, glob, json, re, shutil, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "output").resolve()
REP = OUT / "report"
ISAAC = Path.home() / "docker/isaac"
MAX_PTS = 12000  # per lidar in the 3D pane


# ------------------------------------------------------------------ experiment descriptions
def describe(name):
    """(group, title, description) from the experiment name."""
    n = name.removeprefix("Box")
    if n == "C":
        return "C", "C  corridor trajectory", ("26-frame straight trajectory through the 20x20 m corridor room (openings in the N/S walls), 0.75 m tags on the "
                "walls and on four 45deg-yaw planes cutting the room corners. Failed: east/south/west cameras only ever see parallel wall planes, "
                "Zhang's initial focal length is degenerate (f ~ 1e5) and the camera stage diverges.")
    mp = re.match(r"P_(a3|09)_f(\d+)_k(\d+)$", n)
    if mp:
        sz, f_, k = mp.group(1), int(mp.group(2)), int(mp.group(3))
        size = "A3 (0.30 m plate, 0.24 m tag)" if sz == "a3" else "0.9 m plate (0.72 m tag)"
        return f"P-{sz}", f"P spin, {'A3' if sz == 'a3' else '0.9 m'} boards, {f_}/face ({5 * f_}), K={k} positions", (
            f"Rig spinning in place in a 10x10x3 m room (rig x/y offsets halved), {size} tag36h11 boards, exactly {f_} on each of the 4 walls "
            f"and {f_} on the floor ({5 * f_} total, farthest-point sampling per face); 12 yaw positions captured (every 30 deg), {k} of them used.")
    mw = re.match(r"W_(a3|09)_(all|n(\d+))_k(\d+)$", n)
    if mw:
        sz, sel, k = mw.group(1), mw.group(3), int(mw.group(4))
        size = "A3 (0.30 m plate, 0.24 m tag)" if sz == "a3" else "0.9 m plate (0.72 m tag)"
        return f"W-{sz}", f"W spin, {'A3' if sz == 'a3' else '0.9 m'} boards, {'all' if sel is None else 'N=' + sel} boards, K={k} positions", (
            f"Rig spinning in place in a 10x10x3 m room (rig x/y offsets halved), {size} tag36h11 boards only on the walls and the floor "
            f"({'all non-overlapping candidates, capped' if sel is None else 'farthest-point subset, N=' + sel}); thinned to {k} evenly spaced yaw positions "
            f"(captured every {10 if sel is None else 30} deg).")
    if n == "S":
        return "S", "S  spin, C2 scene", ("Same rig and scene as C2 (corridor room, wall boards, 45deg corner planes, chevrons; 0.75 m tags) but the rig "
                "spins in place at the origin, 36 frames at 10deg yaw steps (trajectory_spin.csv): every camera sees many plane orientations.")
    if n == "C2":
        return "C", "C2  corridor + chevrons", ("C plus free-standing boards turned alternately +-45deg in yaw 3 m off the walls so every camera sees "
                "non-parallel planes (fixes the degenerate initialisation).")
    if n == "C3":
        return "C", "C3  corridor + stacked chevrons", ("C2 with the N/S chevrons stacked three high on one plane: C2 fixed 8 of 9 cameras, but "
                "camera_south only saw single (non-coplanar) chevron tags, which the initialisation drops (it needs >= 8 points per plane).")
    if n == "I2_v2":
        return "I", "I2_v2  I2 with real 3x3-tag boards", (
            "Same layout and poses as I2 (28 tag-sets, 5 m rig, 10x10x3 m room, single shot) but each tag-set is one 0.9x0.9 m board with 3x3 tag36h11 tags "
            "(0.24 m tag, 0.30 m tile; generate_boards.py --configuration-matrix 3x3) instead of 9 separate A3 plates: 28 boards, 252 tags.")
    if n == "I2_big":
        return "I", "I2_big  I2 scaled to the 20x20x3 m room", (
            "I2 (28 tag-sets = 252 A3 plates, 5 m rig at the room centre, single shot) in a 20x20x3 m room: the sets stand on the walls, spread over the 20 m sides "
            "(wall sets at 4/8/12/16 m along each side, z 1.5 m; floor sets 0.5 m off the walls; 1 per corner chamfer). Failed: the camera stage runs out of its 150 "
            "iterations (the cameras see only 3 tags per plane at 10 m, f vs depth ambiguity) and the LiDARs are never added.")
    if n == "I3_big":
        return "I", "I3_big  I2_big with the floor sets 5 m off the walls", (
            "I2_big (28 tag-sets = 252 A3 plates, 5 m rig at the centre of a 20x20x3 m room, single shot) with the 8 floor sets moved from 0.5 m to 5 m off the walls "
            "(same positions along the walls, +-4 m on the N/S sides, +-5 m on the E/W sides); wall sets at 4/8/12/16 m along each side, 1 set per corner chamfer unchanged.")
    if n == "K":
        return "K", "K  40 boards of 3x3 tags, 20x20x3 m room", (
            "Single shot, 5 m rig at the centre of a 20x20x3 m room; 40 real 0.9x0.9 m boards with 3x3 tag36h11 tags (0.24 m tag, 0.30 m tile; 360 tags): per wall 5 boards "
            "(z 1.5 m, 4 m apart), 4 flat floor boards 3 m off the wall (+-1.5 / +-5.5 m along it), one board per corner on the 45deg chamfer facing the centre (z 0.47 m).")
    mj = re.match(r"J_(small|big)$", n)
    if mj:
        big = mj.group(1) == "big"
        return "J", f"J_{mj.group(1)}  7 boards of 2x2 tags, {'20x20x3' if big else '10x10x3'} m room", (
            f"Single shot, 5 m rig at the centre of a {'20x20x3' if big else '10x10x3'} m room; 7 real 0.9x0.9 m boards with 2x2 tag36h11 tags (0.36 m tag, 0.45 m tile; "
            f"28 tags): one per corner on the 45deg chamfer facing the centre (0.6 m off the corner, z 1.5 m) and 3 flat floor boards (N, S, E sides, centred, "
            f"{'5' if big else '1'} m off the wall). "
            + ("Failed: every camera sees one board (4 tags, 16 corners); the camera stage ran out of its 150 iterations, the cameras end up tens of metres off." if big else
               "Failed: camera_east sees no board and the initialisation stops (no 2-tag plane); the others see a single 4-tag board."))
    if n == "I2_flat":
        return "I", "I2  tag-sets, no wall tilt, 1 per corner", (
            "I without the 30deg tilt on the 2 m / 8 m wall sets and with a single tag-set per corner (the floor-level one): 28 tag-sets (252 plates) on the 5 m rig, "
            "10x10x3 m room, single shot: 1 per corner on the 45deg chamfer, 4 flat per wall at 2/4/6/8 m (z 1.5 m), 2 flat floor sets per wall 0.5 m off the wall.")
    if n == "I_sets":
        return "I", "I  tag-sets, 5 m rig, single shot", (
            "Single shot, 5 m rig (end sensors at x = +-2.5 m, cameras 2.0 m, corner LiDARs 2.2 m) in a 10x10x3 m room; 32 tag-sets (3x3 A3 plates at 0.30 m pitch, no gaps: "
            "0.9x0.9 m, 9 tags): 2 per corner on the 45deg chamfer facing the centre (floor level and 1.5 m), 4 per wall at 2/4/6/8 m (z 1.5 m, the 2 m and 8 m ones yawed 30deg "
            "away from the adjacent wall), 2 flat floor sets per wall 0.5 m off the wall.")
    mg = re.match(r"G_([sb])_(g1|g2|g2line|g2arc)$", n)
    if mg:
        room, sc = mg.groups()
        rm = "10x10x3 m room (rig x/y offsets halved)" if room == "s" else "20x20x8 m room"
        traj = {"g1": ("single shot", "g1, single shot"), "g2": ("single shot", "g2, single shot"),
                "g2line": ("7 frames, straight line x = -1.5..+1.5 m, no rotation", "g2 + straight line (7 frames)"),
                "g2arc": ("7 frames, 2 m radius arc turning 45deg to the right (yaw 0..-45deg)", "g2 + 45deg arc (7 frames)")}[sc]
        lay = ("per camera direction (8; camera_north_2 shares camera_north's) one rigid 3x3 init panel of A3 plates (0.30 m plate, 0.24 m tag36h11 tag, "
               "0.35 m pitch, 1x1 m) 2.5 m ahead facing the camera, centre 2 m high, plus a flat 2x2 floor group 3.5 m ahead where the floor is visible"
               + (" (not for the E/W cameras)" if room == "s" else ""))
        if sc != "g1":
            lay += (", plus two near 2x2 groups per camera: 1.5 m ahead at the left image side yawed 35deg and 4 m ahead at the right side pitched 25deg "
                    "(clamped to the room, plates occluding another group in some camera dropped)")
        return f"G-{room}", f"G  panels, {'small' if room == 's' else 'large'} room, {traj[1]}", f"Multi-tag panel layout, {lay}; {rm}; {traj[0]}."
    m = re.match(r"([DEF])_?(a3|1mx|1m)?_?(all|([npq])(\d+))$", n)
    kind, sz, sel, mode, num = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    if kind == "D":
        sz = "a3"
    dense = " on a denser candidate grid" if sz == "1mx" else ""
    sz = "1m" if sz == "1mx" else sz
    how = {"n": "greedy image-cell coverage", "p": "plane-aware selection: >= 2 differently oriented planes with >= 3 tags per camera first, then coverage",
           "q": "plane-aware selection: >= 2 differently oriented planes with >= 6 tags per camera first, then coverage"}
    sub = "all boards (greedy cap 300, no image overlap)" if sel == "all" else f"reduced to {num} boards ({how[mode]})"
    room = {"D": "20x20x8 m room", "E": "20x20x8 m room", "F": "10x10x3 m room (rig x/y offsets halved)"}[kind]
    size = "A3 (0.30 m plate, 0.24 m tag)" if sz == "a3" else "1 m plate (0.8 m tag)"
    exp = {"D": "D  single shot, A3 targets", "E": "E  single shot, 1 m targets", "F": "F  small room, "}[kind]
    if kind == "F":
        exp += "A3 targets" if sz == "a3" else "1 m targets"
    label = "all" if sel == "all" else {"n": "N", "p": "plane3 N", "q": "plane6 N"}[mode] + f"={num}"
    return kind + ("-" + sz if kind == "F" else ""), f"{exp}, {label}", (
        f"One frame, {size} tag36h11 boards on the walls and on 45deg-yaw planes cutting the room corners facing the centre{dense}, {sub}; {room}.")


# ------------------------------------------------------------------ ground truth comparison (numpy only)
def rot_xyz(r, p, y):  # Isaac applies roll/pitch/yaw as intrinsic XYZ: R = Rx Ry Rz (see gt_report._pose)
    r, p, y = np.radians([r, p, y])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def pose(s, is_cam):
    T = np.eye(4)
    T[:3, :3] = rot_xyz(s["roll"], s["pitch"], s["yaw"]) @ (np.diag([1., -1, -1]) if is_cam else np.eye(3))
    T[:3, 3] = [s["x"], s["y"], s["z"]]
    return T


def match(key, names):
    hits = [n for n in names if key.startswith(n.replace("/", ""))]
    return max(hits, key=len) if hits else None


def rot_err_deg(A, B):
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def gt_compare(res, gt):
    """per-sensor rows: extrinsic errors (mm, deg), intrinsics errors (px), err/sigma"""
    sensors = gt["base_link"]["sensors"]
    cams = {c["sensor_name"]: c for c in gt["cameras"]}
    keys = [k for k in res if k.startswith(("camera", "lidar"))]
    cam_keys = [k for k in keys if k.startswith("camera")]
    # reference camera = first camera of the config (corner_ne); its pose defines the frame
    ref = next(k for k in cam_keys if match(k, sensors) == gt["cameras"][0]["sensor_name"])
    Tc0 = pose(sensors[match(ref, sensors)], True)
    rows = []
    for k in keys:
        n = match(k, sensors)
        is_cam = k.startswith("camera")
        G = np.linalg.inv(Tc0) @ pose(sensors[n], is_cam)
        T = np.array(res[k]["extrinsics"], float)
        sg = res.get("metrics", {}).get(k, {}).get("sigma", {})
        row = dict(sensor=n, kind="camera" if is_cam else "lidar", t_err_mm=(T[:3, 3] - G[:3, 3]) * 1e3,
                   t_err_norm_mm=float(np.linalg.norm(T[:3, 3] - G[:3, 3]) * 1e3), rot_err_deg=rot_err_deg(G, T))
        if sg.get("t_mm"):
            row["t_sigma_mm"] = sg["t_mm"]
        if is_cam:
            K = np.array(res[k]["K"], float)
            g = cams[n]["camera_matrix"]["data"]
            d = [K[0, 2] - g[2], K[1, 2] - g[5], K[0, 0] - g[0], K[1, 1] - g[4]]
            row.update(dc_px=d, f_err_px=float(max(abs(d[2]), abs(d[3]))), pp_err_px=float(np.hypot(d[0], d[1])))
            if sg.get("cx_cy_fx_fy_px"):
                row["k_sigma_px"] = sg["cx_cy_fx_fy_px"]
            row["reproj_rmse_px"] = res.get("metrics", {}).get(k, {}).get("reprojection_px", {}).get("rmse")
            row["n_corners"] = res.get("metrics", {}).get(k, {}).get("reprojection_px", {}).get("n")
        else:
            m = res.get("metrics", {}).get(k, {})
            row["p2plane_rmse_mm"] = m.get("point_to_plane_m", {}).get("rmse", float("nan")) * 1e3
            row["inlier"] = m.get("inlier_ratio")
        rows.append(row)
    return rows


def err_sigma(r):
    """largest |t err| / sigma over axes (camera 0 sigma ~0 -> None)"""
    s = r.get("t_sigma_mm")
    return float(np.max(np.abs(r["t_err_mm"]) / np.array(s))) if s and min(s) > 1e-3 else None


# ------------------------------------------------------------------ point clouds for the 3D pane
def read_ply(path, n):
    with open(path, "rb") as f:
        head = b""
        while b"end_header\n" not in head:
            head += f.readline()
        nv = int(re.search(rb"element vertex (\d+)", head).group(1))
        a = np.frombuffer(f.read(nv * 27), dtype=np.dtype([("p", "<f8", 3), ("c", "u1", 3)]), count=nv)
    idx = np.random.default_rng(0).choice(nv, min(nv, n), replace=False)
    return a["p"][idx], a["c"][idx]


def cloud_js(exp):
    """data/<exp>.js: window.CLOUDS[exp] = {bbox, lidars: {name: {n, frame(uint8 pose idx colour), b64 int16 xyz}}}"""
    clouds = {}
    for f in sorted(glob.glob(str(OUT / f"exp_{exp}/verification/lidar_*_scans.ply"))):
        name = re.sub(r"^lidar_|_scans\.ply$", "", Path(f).name)
        p, c = read_ply(f, MAX_PTS)
        clouds[name] = (p, c)
    if not clouds:
        return None
    allp = np.concatenate([p for p, _ in clouds.values()])
    lo, hi = np.percentile(allp, 0.5, 0), np.percentile(allp, 99.5, 0)
    lo, hi = lo - 1, hi + 1
    out = {}
    for name, (p, c) in clouds.items():
        q = np.clip((p - lo) / (hi - lo), 0, 1) * 65535
        out[name] = dict(n=len(p), xyz=base64.b64encode(q.astype("<u2").tobytes()).decode(),
                         rgb=base64.b64encode(c.tobytes()).decode())
    return dict(lo=lo.tolist(), hi=hi.tolist(), lidars=out)


# ------------------------------------------------------------------ collect
def board_info(cfg):
    assets = Path(cfg["assets_dir"]).name
    mf = next((m for m in (ISAAC / "boards" / assets / "manifest.json", ISAAC / assets / "manifest.json") if m.exists()), None)
    man = json.load(open(mf)) if mf else {}
    w = cfg["room"]["walls"]
    return dict(n_boards=len(cfg["targets"]["boards"]), family=man.get("family", "tag36h11"),
                plate=man.get("dimensions_m", [None])[0], tag=man.get("black_border_outer_size_m"),
                room=dict(x=2 * w["west"]["distance"], y=2 * w["north"]["distance"], h=w["north"]["height"],
                          corridor="corridor_width" in w["north"]))


def init_info(text):
    """Zhang init diagnostics from the log: tags in the largest coplanar view per camera, range of initial focal lengths"""
    text = re.sub(r".*%\|.*\n", "", text)
    tags = [int(float(n.split()[0]) / 4) if n.split() else 0 for n in re.findall(r"cam\d, num pts \[([^\]]*)\]", text)]
    tail = text[text.rfind("Intrinsics reprojection error"):]
    m = re.search(r"\n\[\[(.*?)\]\]", tail, re.S)
    fs = []
    if m:
        fs = [float(r.split()[2]) for r in re.findall(r"\[?([\d\.\s\-e\+]+)\]", m.group(1)) if len(r.split()) == 4]
    reason = ""
    if "nimages > 0" in text:
        reason = "no coplanar view with >= 2 tags for some camera (Zhang init impossible)"
    elif "the LiDAR stage was not reached" in text:
        reason = "camera stage did not converge within the iteration budget (initial focal lengths " + (f"{min(fs):.0f}..{max(fs):.0f} px" if fs else "n/a") + ")"
    elif "LinAlgError" in text or "TypeError" in text or "cv2.error" in text:
        reason = "solver diverged (initial focal lengths " + (f"{min(fs):.0f}..{max(fs):.0f} px" if fs else "n/a") + ")"
    return dict(tags=tags, cams=len(tags), fmin=min(fs) if fs else None, fmax=max(fs) if fs else None, reason=reason)


def collect():
    exps = []
    names = sorted({Path(p).name.removeprefix("exp_").removesuffix(".log") for p in glob.glob(str(OUT / "exp_Box*.log"))}
                   | {Path(p).name.removeprefix("exp_") for p in glob.glob(str(OUT / "exp_Box*/"))})
    for name in names:
        if name.startswith("BoxH_"):
            continue  # scene H is not part of the report (user request 2026-10-22)
        d = str(OUT / f"exp_{name}")
        ds = sorted(glob.glob(str(OUT / f"dataset_{name}_2*/config.yaml")))
        if not ds:
            continue
        cfg = yaml.safe_load(open(ds[-1]))
        group, title, desc = describe(name)
        e = dict(name=name, group=group, title=title, desc=desc, dataset=Path(ds[-1]).parent.name, **board_info(cfg))
        sensors = cfg["base_link"]["sensors"]
        e["rig"] = dict(cameras=len(cfg["cameras"]), lidars=len(cfg["lidars"]),
                        cam_model=f'{cfg["cameras"][0]["image_width"]}x{cfg["cameras"][0]["image_height"]}, f={cfg["cameras"][0]["camera_matrix"]["data"][0]:.1f} px, pinhole',
                        lidar_models=sorted({l["config"] for l in cfg["lidars"]}),
                        span=[round(max(s["x"] for s in sensors.values()) - min(s["x"] for s in sensors.values()), 2),
                              round(max(s["y"] for s in sensors.values()) - min(s["y"] for s in sensors.values()), 2)])
        mw = re.match(r"Box[WP]_(a3|09)_(all|[nf]\d+)_k(\d+)$", name)
        if mw:
            e["spin"] = dict(size=mw.group(1), sel=mw.group(2), K=int(mw.group(3)))
        e["frames"] = len(glob.glob(str(OUT / f"dataset_{name}_2*/input/camera_northimage_raw/*.png")))
        rpath = Path(d) / "results.yaml"
        e["converged"] = False
        e["log_tail"] = ""
        log = OUT / f"exp_{name}.log"
        if log.exists():
            lines = [l for l in log.read_text(errors="replace").splitlines() if "%|" not in l]
            e["log_tail"] = "\n".join(lines[-12:])[-1500:]
        logtext = log.read_text(errors="replace") if log.exists() else ""
        e["init"] = init_info(logtext)
        if rpath.exists() and (Path(d) / "report.txt").exists():
            res = yaml.safe_load(open(rpath))
            # a report is also written when the camera stage ran out of iterations (LiDARs never added): not converged
            if "metrics" in res and any(k.startswith("lidar") for k in res) and "adding LiDARs" in logtext:
                e["converged"] = True
                e["report_txt"] = (Path(d) / "report.txt").read_text()
                e["data_report"] = (Path(d) / "data_report.txt").read_text() if (Path(d) / "data_report.txt").exists() else ""
                rows = gt_compare(res, cfg)
                for r in rows:
                    r["err_sigma"] = err_sigma(r)
                e["rows"] = rows
                e["sigma0"] = res["metrics"].get("sigma0", {})
                cams = [r for r in rows if r["kind"] == "camera"]
                lids = [r for r in rows if r["kind"] == "lidar"]
                es = [r["err_sigma"] for r in rows if r["err_sigma"] is not None]
                csvf = Path(d) / "reprojection_corners.csv"
                tags, obs = set(), 0
                if csvf.exists():
                    for rr in csv.DictReader(open(csvf)):
                        tags.add(int(rr["corner_id"]) // 100); obs += 1
                e["tags_detected"], e["corner_obs"] = len(tags), obs
                nc = [r["n_corners"] for r in cams if r["n_corners"]]
                e["summary"] = dict(
                    reproj=float(np.average([r["reproj_rmse_px"] for r in cams], weights=nc)),
                    f_mean=float(np.mean([r["f_err_px"] for r in cams])), f_max=float(max(r["f_err_px"] for r in cams)),
                    pp_mean=float(np.mean([r["pp_err_px"] for r in cams])),
                    cam_t_mean=float(np.mean([r["t_err_norm_mm"] for r in cams[1:]] or [0])),
                    cam_t_max=float(max([r["t_err_norm_mm"] for r in cams[1:]] or [0])),
                    cam_rot_max=float(max(r["rot_err_deg"] for r in cams)),
                    lid_t_mean=float(np.mean([r["t_err_norm_mm"] for r in lids])), lid_t_max=float(max(r["t_err_norm_mm"] for r in lids)),
                    lid_rot_max=float(max(r["rot_err_deg"] for r in lids)),
                    err_sigma_max=float(max(es)) if es else None,
                    p2p_rmse_mm=(res["metrics"].get("plane_to_plane_m") or {}).get("rmse", float("nan")) * 1e3,
                    p2p_max_mm=(res["metrics"].get("plane_to_plane_m") or {}).get("max", float("nan")) * 1e3,
                    sigma0_cam=res["metrics"]["sigma0"].get("cameras"), min_corners=int(min(nc)))
                frames = sorted({Path(p).name[:-9] for p in glob.glob(str(OUT / f"exp_{name}/verification/camera_*/*_proj.png"))})
                e["verif_frames"] = frames
                e["cams"] = sorted(Path(p).name for p in glob.glob(str(OUT / f"exp_{name}/verification/camera_*")))
                cl = cloud_js(name)
                if cl:
                    (REP / "data").mkdir(parents=True, exist_ok=True)
                    (REP / "data" / f"{name}.js").write_text(f"window.CLOUDS=window.CLOUDS||{{}};window.CLOUDS[{json.dumps(name)}]={json.dumps(cl)};")
        exps.append(e)
    return exps


def _resize(job):
    src, dst = job
    from PIL import Image
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB"); im.thumbnail((1100, 1100)); im.save(dst, "JPEG", quality=78)


def copy_assets(exps):
    """lib/ and the verification images of every experiment into the report directory"""
    shutil.copytree(Path(__file__).parent.parent / "templates" / "report" / "lib", REP / "lib", dirs_exist_ok=True)
    jobs = []
    for e in exps:
        for src in glob.glob(str(OUT / f"exp_{e['name']}/verification/*/*_*proj.png")):
            src = Path(src); dst = REP / "images" / e["name"] / src.parent.name / (src.stem + ".jpg")
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                jobs.append((src, dst))
    with ProcessPoolExecutor() as ex:
        list(ex.map(_resize, jobs, chunksize=16))
    print(f"images: {len(jobs)} written")


def main():
    REP.mkdir(exist_ok=True)
    exps = collect()
    html = (Path(__file__).parent.parent / "templates" / "report" / "report_template.html").read_text()
    prev = (Path(__file__).parent.parent / "templates" / "report" / "report_previous.json").read_text()
    html = html.replace("/*EXPS*/[]", json.dumps(exps, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    html = html.replace("/*PREV*/[]", prev)
    (REP / "index.html").write_text(html)
    copy_assets(exps)
    import make_slides  # the slide deck goes to the same output/report/
    make_slides.main(exps)
    ok = sum(e["converged"] for e in exps)
    print(f"{len(exps)} experiments ({ok} converged) -> {REP / 'index.html'}")


if __name__ == "__main__":
    main()
