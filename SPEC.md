# SPEC: Calibration Input Data

Input contract for `scripts/lcba.py` (`--faro_file`, `--apriltag_file`, `--path_data`). Verified against `ipb_preprocessing.py`, `apriltag.py`, and the example data in `input/` and `reference/`.

Conventions: units are meters and degrees. Reference frame is the TLS map frame. Camera 0 (the first entry of `image_topics`) is the sensor reference frame.

The input folder can be generated from a ROS2 bag with `scripts/rosbag_to_calibration.py` (`config/rosbag_to_calibration.yaml`), which writes `calibration.yaml` + sensor folders as described below.

## 1. Reference data (`reference/`)

### `reference_pointcloud.ply`
- TLS scan of the calibration environment, read by `open3d.io.read_point_cloud`.
- Binary PLY with `x y z` float, optionally colors. Normals are recommended and should be precomputed.
- Downsample to about 0.5 cm in CloudCompare to keep compute time down.
- The metric frame must be the same one used for the AprilTag coordinates.

### `apriltag_coords.txt`
- Output of the tag extraction (`README_apriltag_extraction.md`).
- Line 1 is a header and is skipped (`skiprows=1`). The example has the row count, `476`.
- Then 4 rows per tag, each `<id*100+corner_idx> x y z`, whitespace separated. Example: `8201 -2.87457 -1.37728 0.22523` is tag 82, corner 01.
- Corner rows come in the order `01..04`, either TL, TR, BR, BL (as in `apriltag_coords.csv`) or TL, TR, BL, BR. `apriltag.py` detects the cyclic order (corner 3 on the diagonal of corner 1) and swaps corners 3 and 4, so both work.
- Tag family is set with `--tag-family` (required, any family the `apriltag` lib knows, e.g. `tag16h5`, `tag36h11`). The id lookup is sized from the ids in the file; detected ids missing from the file are ignored.
- Coordinates are in the same frame as the PLY. The default std is 2 mm (`--std_apriltags`).
- `apriltag_coords.csv` is a human-readable equivalent and is not read by `lcba.py`.

## 2. Recorded data (`input/`)

```
input/
  calibration.yaml
  <sensor_name>/   # one folder per entry in image_topics / point_cloud_topics
    0000.png | 0000.npy ...
```

### Folder names
- Folder name = topic with every `/` removed. `/camera/front/image_raw` becomes `camerafrontimage_raw`.
- Files are loaded in sorted filename order.

### Pose synchronization
- The N-th file of every sensor is the same stop-and-go pose.
- Every sensor therefore needs the same number of files, with matching zero-padded names (`0000`, `0001`, ...). Counts are not checked at load time; a mismatch silently misaligns poses.
- The example data has 64 files per sensor. The README recommends about 50 or more.
- Capture while stationary. Vary both rotation and translation, cover about 360°, and keep the sensors unblocked.

### Images (`*.png`)
- Any PNG that `matplotlib.pyplot.imread` reads as HxWx3 (or HxWx4). It is converted to gray with `mean(axis=-1)*255` (values expected in 0..1, as `plt.imread` returns), so it must be color.
- Example: 2064x1024 float32 RGB.
- Use raw, distorted images. Do not undistort them.
- The image size is read from the files, and one size per camera is assumed.
- AprilTags (family given by `--tag-family`) must be visible in the images; the run asserts if none is detected anywhere. The initial intrinsics and poses are estimated from these detections, so several tags per image are needed: `estimate_initial_k` builds Zhang views from the tags on each world plane and needs >= 2 tags (8 points) per plane; around 6+ tags in the largest plane per camera is needed for a reliable init. Images without detections are allowed.
- Camera 0 should share a pose with each other camera where tags are visible in both. Extrinsics are initialised from the pose where both see the most tags.

### LiDAR scans (`*.npy`)
- One file is one scan in the sensor frame.
- Either a plain array reshapeable to `[N,3]` (xyz), or a structured array with fields `x`, `y`, `z`. Extra fields (intensity, ring, ...) are ignored.
- The example is an Ouster structured array of shape (128, 2048). Any shape works, because it is flattened.
- NaN/invalid points are not filtered in the loader (`load_scan`). Remove them beforehand if the sensor emits them.
- Scans are randomly downsampled to `--max_scanpoints` (default 2500, `-1` keeps all).
- Each scan must see structure in the reference map. Profile scanners need enough structural elements.

## 3. `input/calibration.yaml`

```yaml
image_topics:                 # order matters: first = camera 0 = reference frame
- /camera/front/image_raw
- /camera/left/image_raw
point_cloud_topics:
- /lidar/horizontal/points
camera_models:                # optional; one per image_topics entry; default all "pinhole"
- pinhole
- fisheye
init_cami_to_cam0:            # key = folder name (slashes removed)
  camerafrontimage_raw: [0, 0, 0, 0, 0, 0]
  cameraleftimage_raw:  [0, -90, 0, -0.1, -0.28, -0.6]
init_lidari_to_cam0:          # key = folder name
  lidarhorizontalpoints: [0, -90, 90, 0, -0.35, -0.6]
```

### Initial pose format
- Format is `[rx, ry, rz, tx, ty, tz]`: Euler `xyz` in degrees (scipy `Rotation.from_euler("xyz", ..., degrees=True)`), then translation in meters.
- It builds a 4x4 `T = [R | t]` with no inversion.
- LiDAR: `init_lidari_to_cam0` is used as the initial LiDAR-to-camera-0 extrinsic. A rough hand measurement is enough, and the BA refines it.
- LiDAR keys are required for every entry in `point_cloud_topics`. Missing keys raise a `KeyError`.
- Camera: `init_cami_to_cam0` is parsed if present (if the key exists it needs an entry for every camera, else `KeyError`). It is **not** used for initialisation. The camera extrinsics are estimated from the tag detections, and the parsed yaml value (`T_cami_cam_yaml` in `lcba.py`) is unused afterwards.
- Keep it filled in anyway, because the example does and the parser can read it.

### Constraints
- `camera_models` values are exactly `pinhole` or `fisheye`. Any other value is counted as neither and triggers a warning.
- If `camera_models` is absent, all cameras are pinhole.
- Camera intrinsics (K, distortion) are **not** inputs. They are estimated from the tag detections (`estimate_initial_k`): up to 10 coplanar tag views per camera, taken round-robin over distinct world planes, then refined by the bundle adjustment (`--dist_degree`, `--division_model`).

## 4. Calibration output

This `calibration.yaml` format differs from the output format.

CLI options besides the three inputs (`scripts/lcba.py`): `--path_out`, `--experiment_name` (default `dev`), `--std_pix` (0.2), `--std_apriltags` (0.002), `--std_lidar` (0.01), `--max_scanpoints`, `--bias/--scale` (per-LiDAR), `--dist_degree` (3), `--division_model`, `--export/--no-export`, `--gt_file`, `--visualize`.

The result goes to `output/<experiment_name>/`:
- `results.yaml` (+ `results.yaml.pkl`): per-sensor entries keyed by topic name, with `extrinsics` (4x4), `K` and `distortion_coeff` plus `is_pinhole` for cameras, and `intrinsics` (offset, scale) for LiDARs; plus a `metrics` entry (reprojection stats, LiDAR point-to-plane, plane-to-plane, sigma relative to cam0, sigma0).
- `report.txt`: metrics report with metric meanings (also printed last to the log); with `--gt_file` it includes the GT vs result table.
- `data_report.txt`: input-quality report from `diagnostics.check_data` (also printed if the solve raises `LinAlgError`).
- `errors.pkl`, `args.yaml`.
- `verification/` (unless `--no-export`): `lidar_<sensor>_scans.ply` (scans in the map frame), `<camera>/<image>_proj.png` (LiDAR projected into the image) and `<image>_reproj.png` (detected vs projected tag corners, error x50).
- `report/report.html` (unless `--no-export`; needs `verification/`): single-run HTML deck in the style of `output/report/report.html` (no GT): summary + date, rig top/front/side view, metrics, intrinsics, extrinsics, per-camera LiDAR-projection/reprojection images (`report/images/`), 3D clouds (`report/lib/` = three.js), metric meanings. Self-contained folder, open from `file://`.
- `reprojection_corners.csv`: per-corner GT and result reprojection residuals, only with `--gt_file`.

`--gt_file` is an Isaac Sim rig config (`base_link.sensors` + cameras); it is not part of the input contract otherwise.

A sensor-name filter in `io.py` classifies keys by substring: camera if the name contains `image` or `cam`, LiDAR if it contains `points`, `laser` or `lidar`. Sensor/topic names must keep these substrings.
