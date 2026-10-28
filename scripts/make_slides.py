"""Builds output/report/report.html: the 7-slide explanation deck (process, scenarios, rigs, results, findings).
Images are rendered here (matplotlib/PIL/pdftoppm) and embedded as data URIs, the numbers come from make_report.collect(),
so the deck is one self-contained file. Run: python scripts/make_slides.py   (after make_report.py's experiments exist in output/)
"""
import base64, csv, glob, io, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(Path.home() / "docker/isaac/configs")]
import make_report
OUT = make_report.OUT  # same output dir as make_report.py (argv[1], default ./output)
ISAAC = Path.home() / "docker/isaac"
sys.path[:0] = [str(HERE), str(ISAAC / "configs")]
from make_box_scenarios import cam_poses, board_corners

BLUE, ORANGE, GREEN, GREY, INK = "#2f6fdb", "#e07b24", "#1f9d6a", "#8a8f98", "#1b2430"


def uri(img, fmt="JPEG", q=82):
    b = io.BytesIO(); img.convert("RGB").save(b, fmt, quality=q)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode()


def fig_uri(fig, w=None):
    b = io.BytesIO(); fig.savefig(b, dpi=130, bbox_inches="tight", transparent=True, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def last(pattern):
    return sorted(glob.glob(str(OUT / pattern)))[-1]


# ------------------------------------------------------------------ images
def board_images():
    d = ISAAC / "boards" / "boards_tag36h11_a3" / "pdf"
    tmp = Path(tempfile.mkdtemp())
    ims = []
    for i in range(9):
        subprocess.run(["pdftoppm", "-r", "60", "-png", "-singlefile", str(d / f"board_tag36h11_{i:02d}.pdf"), str(tmp / f"b{i}")], check=True)
        ims.append(Image.open(tmp / f"b{i}.png").convert("RGB"))
    w = ims[0].size[0]
    sheet = Image.new("RGB", (3 * w, 3 * w), "white")
    for i, im in enumerate(ims):
        sheet.paste(im.resize((w, w)), ((i % 3) * w, (i // 3) * w))
    return uri(ims[0].resize((w * 2, w * 2)), "PNG"), uri(sheet, "PNG")


def rig_plot(cfg_name, title):
    cfg = yaml.safe_load(open(ISAAC / "configs" / cfg_name))
    sens = cfg["base_link"]["sensors"]
    poses = cam_poses(sens, dict(x=0, y=0, z=0, roll=0, pitch=0, yaw=0))
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 2.75), gridspec_kw=dict(height_ratios=[1.7, 1]))
    for n, s in sens.items():
        lid = n.startswith("lidar")
        a.scatter(s["x"], s["y"], c=ORANGE if lid else BLUE, marker="s" if lid else "o", s=34, zorder=3)
        b.scatter(s["x"], s["z"], c=ORANGE if lid else BLUE, marker="s" if lid else "o", s=34, zorder=3)
        if not lid:
            f = poses[n][0][:, 2]; a.arrow(s["x"], s["y"], f[0] * .55, f[1] * .55, head_width=.1, color=BLUE, alpha=.7, length_includes_head=True)
    a.set_aspect("equal"); a.set_ylabel("y [m]", fontsize=8); b.set_xlabel("x [m]", fontsize=8); b.set_ylabel("z [m]", fontsize=8)
    b.set_ylim(0, 3.4); b.axhline(0, color=GREY, lw=.8)
    a.set_title(title, fontsize=9.5, color=INK)
    for ax in (a, b):
        ax.tick_params(labelsize=7); ax.grid(alpha=.25)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    a.margins(.1)
    fig.tight_layout(pad=.4)
    return fig_uri(fig)


def layout_plot(cfg_path, title, traj=None, half=None, ax_lim=None):
    cfg = yaml.safe_load(open(cfg_path))
    walls = cfg["room"]["walls"]; hx, hy = walls["east"]["distance"], walls["north"]["distance"]
    fig, a = plt.subplots(figsize=(3.9, 3.9))
    a.plot([-hx, hx, hx, -hx, -hx], [-hy, -hy, hy, hy, -hy], color=INK, lw=1.6)
    if "corridor_width" in walls["north"]:  # openings in the N/S walls (x axis = north)
        w = walls["north"]["corridor_width"] / 2
        for yy in (-hy, hy): a.plot([-w, w], [yy, yy], color="white", lw=3.5, zorder=2)
    for b in cfg["targets"]["boards"]:
        p = np.array(b["position"]); c, n = board_corners(b, 0.30)
        if abs(n[2]) > .7:
            a.scatter(p[0], p[1], marker="s", s=7, c=GREEN, zorder=3)
        else:
            t = np.array([-n[1], n[0]]) * .15; a.plot([p[0] - t[0], p[0] + t[0]], [p[1] - t[1], p[1] + t[1]], color=BLUE, lw=1.8, zorder=3)
    for n, s in cfg["base_link"]["sensors"].items():
        a.scatter(s["x"], s["y"], c=ORANGE if n.startswith("lidar") else "#c0392b", s=14, zorder=4, marker="s" if n.startswith("lidar") else "o")
    if traj:
        r = list(csv.DictReader(open(traj))); a.plot([float(x["x"]) for x in r], [float(x["y"]) for x in r], "-", color="#c0392b", lw=1.2, zorder=1)
    a.set_aspect("equal"); a.set_title(title, fontsize=9.5, color=INK); a.tick_params(labelsize=7); a.grid(alpha=.2)
    if ax_lim: a.set_xlim(-ax_lim[0], ax_lim[0]); a.set_ylim(-ax_lim[1], ax_lim[1])
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig_uri(fig)


def read_ply(path):
    raw = open(path, "rb").read(); i = raw.index(b"end_header\n") + 11
    head = raw[:i].decode(); n = int(head.split("element vertex ")[1].split()[0])
    props = [l.split()[1:] for l in head.split("\n") if l.startswith("property")]
    dt = np.dtype([(p[1], {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1"}[p[0]]) for p in props])
    return np.frombuffer(raw[i:i + n * dt.itemsize], dt)


def cloud_plots(ds):
    ref = read_ply(Path(ds) / "reference/reference_pointcloud.ply")
    xyz = np.stack([ref["x"], ref["y"], ref["z"]], 1)[::6]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    a.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], s=.3, cmap="viridis"); a.set_aspect("equal"); a.set_title("reference cloud, top view (colour = height)", fontsize=8.5)
    b.scatter(xyz[:, 0], xyz[:, 2], c=xyz[:, 2], s=.3, cmap="viridis"); b.set_aspect("equal"); b.set_title("side view", fontsize=8.5)
    for ax in (a, b): ax.tick_params(labelsize=7)
    fig.tight_layout(); ref_uri = fig_uri(fig)
    fig, a = plt.subplots(figsize=(3.6, 3.2))
    exp = OUT / "exp_BoxI2_flat" / "verification"; cols = plt.cm.tab10(np.arange(8))
    for k, p in enumerate(sorted(exp.glob("lidar_*_scans.ply"))):
        c = read_ply(p); c = c[::12]; a.scatter(c["x"], c["y"], s=.25, color=cols[k])
    a.set_aspect("equal"); a.set_title("8 calibrated LiDAR scans (one colour each)", fontsize=8.5); a.tick_params(labelsize=7)
    fig.tight_layout()
    return ref_uri, fig_uri(fig)


def sample_images():
    d = Path(last("dataset_BoxI2_flat_2*")) / "input"
    v = OUT / "exp_BoxI2_flat" / "verification"
    sm = lambda p: Image.open(p).convert("RGB").resize((720, 465))
    return [uri(sm(d / "camera_northimage_raw/0000.png")), uri(sm(v / "camera_northimage_raw/0000_proj.png")), uri(sm(v / "camera_corner_neimage_raw/0000_reproj.png"))]


# ------------------------------------------------------------------ data
def modality(e):
    n = e["name"].removeprefix("Box")
    if e["group"] in ("S", "W-a3", "W-09", "P-a3", "P-09"):
        return "spin"
    if n.startswith("C") or e["frames"] > 1:
        return "trajectory"
    return "single"


def data(exps=None):
    exps = exps or make_report.collect()
    es = [e for e in exps if e["converged"] and e.get("summary")]
    keys = ["reproj", "f_mean", "pp_mean", "cam_t_mean", "cam_t_max", "cam_rot_max", "lid_t_mean", "lid_t_max", "lid_rot_max", "p2p_rmse_mm", "p2p_max_mm", "err_sigma_max"]
    rk = {e["name"]: [] for e in es}
    for k in keys:
        r = sorted([e for e in es if e["summary"].get(k) is not None and e["summary"][k] == e["summary"][k]], key=lambda e: e["summary"][k])
        for i, e in enumerate(r): rk[e["name"]].append(i / (len(r) - 1))
    rows = []
    for e in exps:
        n = e["name"].removeprefix("Box"); t = n.split("_"); en = "I1" if t[0] == "I" else " ".join(t) if t[0] == "G" else n if n in ("I2_v2", "I2_big", "I3_big", "J_small", "J_big") else t[0]
        s = e.get("summary") or {}
        rows.append(dict(name=n, exp=en, mod=modality(e), ok=bool(e["converged"] and s), boards=e["n_boards"], frames=e["frames"], title=e["title"],
                         score=float(np.mean(rk[e["name"]])) if e["name"] in rk else None, **{k: s.get(k) for k in keys}))
    return rows


def main(exps=None):
    board, tagset = board_images()
    ds_i = last("dataset_BoxI2_flat_2*")
    ref_uri, lid_uri = cloud_plots(ds_i)
    img = dict(
        board=board, tagset=tagset,
        rig10=rig_plot("calibration_box_tag36h11.config.yaml", "10 m rig (9 cameras, 8 LiDARs)"),
        rig5=rig_plot("calibration_box5_tag36h11.config.yaml", "5 m rig, top sensors at 2 m"),
        lay_spin=layout_plot(last("dataset_BoxW_a3_all_k9_2sub/config.yaml") if False else str(OUT / "dataset_BoxW_a3_all_k9_2sub/config.yaml"), "Spin: boards on walls + floor, 10x10 m room"),
        lay_corr=layout_plot(last("dataset_BoxC2_2*/config.yaml"), "Corridor: 26-frame straight drive (red)", traj=ISAAC / "trajectories/trajectory_corridor_north.csv", ax_lim=(13, 11)),
        lay_single=layout_plot(str(Path(ds_i) / "config.yaml"), "Single shot: 28 tag-sets (I2), 5 m rig"),
        cloud_ref=ref_uri, cloud_lid=lid_uri, samples=sample_images())
    rows = data(exps)
    html = (ROOT / "templates" / "report" / "slides_template.html").read_text()
    html = html.replace("/*IMG*/{}", json.dumps(img)).replace("/*ROWS*/[]", json.dumps(rows, default=lambda o: None if o != o else o))
    (make_report.REP / "report.html").write_text(html)
    print("wrote", make_report.REP / "report.html", len(html) // 1024, "KB;", sum(r["ok"] for r in rows), "converged of", len(rows))


if __name__ == "__main__":
    main()
