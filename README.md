# Joint Intrinsic and Extrinsic Calibration of Perception Systems Utilizing a Calibration Environment


![image](https://github.com/user-attachments/assets/1f3d64c6-fd31-4bf7-adef-cfa7cda43e56)

<div style="width:500px">
Basically all multi-sensor systems must calibrate
their sensors to exploit their full potential for state estimation
such as mapping and localization. In this paper, we investigate
the problem of extrinsic and intrinsic calibration of perception
systems. Traditionally, targets in the form of checkerboards or
uniquely identifiable tags are used to calibrate those systems.
We propose to use a whole calibration environment as a target
that supports the intrinsic and extrinsic calibration of different
types of sensors. By doing so, we are able to calibrate multiple
perception systems with different configurations, sensor types,
and sensor modalities. Our approach does not rely on overlaps
between sensors which is often otherwise required when using
classical targets. The main idea is to relate the measurements for
each sensor to a precise model of the calibration environment.
For this, we can choose for each sensor a specific method that best
suits its calibration. Then, we estimate all intrinsics and extrinsics
jointly using least squares adjustment. For the final evaluation
of a LiDAR-to-camera calibration of our system, we propose an
evaluation method that is independent of the calibration. This
allows for quantitative evaluation between different calibration
methods. The experiments show that our proposed method is
able to provide reliable calibration.
</div>


# Perception System Calibration

This repository contains the full setup for the system calibration (relative
poses between the sensors + intrinsics) for multipe Cameras and/ or LiDAR sensors.

## Prerequisites

The first thing is to obtain the point cloud and Apriltag coordinates from the calibration environment.
Next we have to record with the setup that should be recorded. After recording we can use this repository for calibration.

### TLS point cloud + Apriltag coordinate extraction

We provide exemplary data here:
- Point Cloud and Apriltag Coordinates: [Download here](https://www.ipb.uni-bonn.de/html/projects/ipb_calibration/reference_data.zip)

#### Using own data

If you want to use your own data: Take a TLS and scan the room. Then use the apriltag coordinate extraction software we provide in this repository. This will
extract 3D coordinates of the apriltags. Use these coordinates later as reference data in folder `reference`. A detailed
description for the apriltag extraction is provided [here](README_apriltag_extraction.md).

For the system calibration: To reduce the compute time I suggest using cloud compare to downsample to half a centimeter resolution and precompute the normals. We assume the point cloud to be in PLY file format.

### Data recording

We need for the calibration from each sensor a certain amount of observations (e.g., images and point clouds) in the calibration environment. For this, one should measure in a stop-and-go manner, i.e., (i) move, (ii) stand still, (iii) take a picture or point cloud from each sensor, repeat (i) - (iii) to have around 50 observations from each sensor. In the end it would be nice to be moved around 360 degrees with the setup such that each sensor saw most of the environment. When recording with 2D profile scanner, make sure that they see enough structural elements otherwise the calibration might fail. While taking the pictures/ point cloud please avoid blocking the sensors (try to find a blind spot to hide). Use a combination of rotation and translation changes for the sensor, not only pure rotation on the spot for best possible results.
For each sensor the data (.png for images, or .npy for the point clouds) should be placed in a folder with its name (can be chosen freely but must match later the ones used in the config file! We simply used the ros topic names without the /).

For the recorded data a `calibration.yaml` config file needs to be created. Please put all the topic names for the cameras and LiDAR, as well as roughly measure the initial guess. Additional, define which cameras are standard perspective cameras ("pinhole") or fisheye cameras ("fisheye"), e.g.

```
camera_models:
- pinhole
- pinhole
- pinhole
- pinhole
- fisheye
- fisheye
```

If no information is given in the yaml file, perspective cameras are assumed.
An example can be of how the yaml should look like can be found in the provided test data.

We provide exemplary data here:
- Images, Point Clouds, configuration file: [Download here](https://www.ipb.uni-bonn.de/html/projects/ipb_calibration/calibration_data.zip)

### Compute Calibration

To run the calibration use `lcba.py` for calibrating the ipb-car. One can probably leave most of the default parameters as they are.
In the end the point clouds should be well aligned to the reference map and the camera rays should intersect the apriltag corners.

## Docker for system calibration

A configuration for docker is available. We provide a Dockerfile together with a docker compose configuration which mounts necessary directories
from the hosts system for input and output. To prepare a run in a docker container provide the following:

- **Reference data** in folder `reference`: 
  - provide the reference point cloud (see above) as file `reference/reference_pointcloud.ply`
  - provide the Apriltag coordinates (see above) as file `reference/apriltag_coords.txt`
- **Input data** in folder `input`: provide the data recording (see above) together with the `calibration.yaml` config file in folder `input`.

The docker image can be build and the calibration can be started with
(working directory should be root directory of the repository):
```
docker compose up --build
```
(To separate building and start use `docker build .` and `docker compose up`.)

After the calibration, the output is available in folder `output`:
- `result1.log`: Logfile for the console output of the calibration run
- `result1/results.yaml`: (also available as pickle file): The actual
  calibration result with extrinsics and intrinsics together
  with their a-posteriori covariance matrices of the cameras and lasers.
  `result1/args.yaml`: The settings used for the calibration. 
- `result1/verification/`: Headless check of the result (disable with `--no-export`):
  - `lidar_<name>_scans.ply`: all scans of a LiDAR in the reference map frame (colored by pose index), to overlay with `reference_pointcloud.ply` e.g. in CloudCompare.
  - `<camera>/<image>_proj.png`: each camera image with the LiDAR scan of the same pose projected into it (colored by depth). Points should lie on the matching structures. Uses the full, non-downsampled scans.

**Notes**:
- If you need to change some parameters of the calibration program, 
  modify the start script `entrypoint.bash`
- if you need visualization (e.g. using --visualize in `entrypoint.bash`),
  then allow X11 connections for user root with
```
xhost local:root
```
  before starting the container.
- The compose file assumes that a NVIDIA GPU is available. If no visualization is needed 
  (and that is the default !) and you have no nvidia GPU, the "deploy" section the
  `compose.yaml` can be commented out.


## Run the pipeline from a ROS2 bag

End to end: ROS2 bag -> `input/` folder -> calibration (docker) -> `output/<experiment>/`. The reference data (TLS cloud + AprilTag coordinates) does not come from the bag, see [Prerequisites](#prerequisites).

### 1. What the bag must contain

One recording in stop-and-go style (see [Data recording](#data-recording)), ROS2 bag in `sqlite3` or `mcap` format, with these topics:

| Topic (names are free, you list them in the yaml) | Type | Needed |
|---|---|---|
| one image topic per camera | `sensor_msgs/msg/CompressedImage` (jpeg/png) | yes |
| one info topic per camera | `sensor_msgs/msg/CameraInfo` | yes, only `distortion_model` is used |
| one point cloud topic per LiDAR | `sensor_msgs/msg/PointCloud2` (`PointXYZIR` or `PointXYZIRADT` layout) | yes (the converter needs at least one LiDAR entry) |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | yes: every camera and LiDAR frame, all in one connected tree |

Requirements:
- The `header.frame_id` of each image and point cloud message is the **sensor name**. It must be unique per sensor and, because `io.py` classifies sensors by name, it must contain `cam` or `image` (cameras), or `lidar`, `points` or `laser` (LiDARs). The converter warns otherwise. For cameras use the optical frame that the image header carries; that frame must exist in `/tf_static`.
- `CameraInfo.distortion_model`: `plumb_bob` / `rational_polynomial` -> `pinhole`; `equidistant` / `fisheye` -> `fisheye`. Anything else stops the conversion. K and distortion are **not** used (the calibration estimates them).
- Images are decoded to colour PNG, so they must be raw/distorted images, not rectified.
- Header stamps must be increasing per topic. Every sensor needs a message within `max_sync_dt` of the reference camera image for a pose to be kept.
- Point cloud layouts (offsets are checked against the message, a wrong `point_format` raises an error):
  - `PointXYZIR`: x,y,z float32 at 0/4/8, intensity float32 at 16, ring uint16 at 20, `point_step` 32.
  - `PointXYZIRADT`: the above plus azimuth f32 at 24, distance f32 at 28, return_type u8 at 32, time_stamp f64 at 40, `point_step` 48.
  - Another layout: add it to `POINT_FORMATS`/`POINT_SIZES` in `src/ipb_calibration/rosbag_converter.py`.

### 2. Converter configuration (yaml)

Copy `config/rosbag_to_calibration.yaml` and edit it:

```yaml
bag: /path/to/bag_dir            # ROS2 bag directory (sqlite3 or mcap)
output: input                    # written: calibration.yaml + one folder per sensor
tf_static_topic: /tf_static
point_format: PointXYZIR         # PointXYZIR | PointXYZIRADT; per LiDAR override with `point_format`
every_n: 1                       # use every n-th image of the first camera
max_sync_dt: 0.05                # [s] max stamp difference of any sensor to the reference image, else the pose is dropped
max_poses: 64                    # optional; subsample evenly
use_header_stamp: true           # false = bag receive time
cameras:                         # first entry = camera 0 = reference frame of all extrinsics
  - image_topic: /sensing/camera/camera0/image_raw/compressed
    info_topic: /sensing/camera/camera0/camera_info
  - image_topic: /sensing/camera/camera1/image_raw/compressed
    info_topic: /sensing/camera/camera1/camera_info
lidars:
  - topic: /sensing/lidar/top/pointcloud_raw
  - topic: /sensing/lidar/left/pointcloud_raw
    point_format: PointXYZIRADT
```

Poses are chosen by time-sync: every `every_n`-th image of camera 0 and the nearest message of each other sensor. A sensor message is never used for two poses. The converter does **not** detect whether the rig stood still: record stop-and-go, or choose `every_n`/`max_poses` so that only still moments remain. Aim for about 50 poses covering ~360 deg of rotation plus some translation.

### 3. Convert

```bash
pip install -e .[rosbag]         # adds the `rosbags` package, no ROS install needed
python scripts/rosbag_to_calibration.py config/rosbag_to_calibration.yaml
```

It writes, in `output:` (use `input/` to feed docker directly):

```
input/
  calibration.yaml            # image_topics / point_cloud_topics (= frame_ids), camera_models,
                              # init_cami_to_cam0, init_lidari_to_cam0 (from /tf_static)
  <camera frame_id>/0000.png ...
  <lidar frame_id>/0000.npy ...   # float32 [N,3] in the sensor frame, NaN/inf removed
```

Folder and yaml names are the frame_ids with `/` removed. The initial poses are `[rx, ry, rz, tx, ty, tz]` (xyz Euler in degrees, metres) of each sensor in the camera 0 frame; a rough value is enough, the bundle adjustment refines it. Check `calibration.yaml` and a few PNGs before the run. Details of the input contract: `SPEC.md`.

### 4. Reference data and docker parameters

1. Put `reference_pointcloud.ply` and `apriltag_coords.txt` in `reference/` (TLS scan of the same environment, in the same frame as the tag coordinates).
2. Mounts in `compose.yaml` (`systemcalib` service): set the left side of each volume to your folders
   - `<input dir>:/root/input:ro`: the converted folder from step 3.
   - `<reference dir>:/root/reference:ro`
   - `$PWD/output:/root/output`
   - optional ground truth `<rig config>.yaml:/root/gt.yaml:ro` (only for Isaac Sim data; if `/root/gt.yaml` is missing `entrypoint.bash` skips it).
3. Parameters in `entrypoint.bash` (the `lcba.py` call):
   - `--tag-family` (**must** match the tags in the images and in `apriltag_coords.txt`, e.g. `tag36h11`, `tag16h5`).
   - `--std_pix` (0.2 px detection noise), `--std_apriltags` (0.002 m tag coordinates), `--std_lidar` (0.01 m LiDAR noise): used as weights, set them close to the real noise.
   - `--max_scanpoints` (2500 points per scan, `-1` = all), `--dist_degree` (3), `--division_model`, `--bias/--scale` (per-LiDAR bias/scale), `--no-export` (skip `verification/`), `--visualize` (needs X11, see notes above).
   - `--experiment_name` defaults to the timestamp, so runs do not overwrite each other.
4. Run (default image is Ubuntu 24.04):

```bash
docker compose up --build
```

The apriltag python module is installed in the image's venv. If you run `docker run` yourself as a non-root user, use a writable working directory (the code writes a `cache/` folder there) and keep `PYTHONPATH` including `/usr/local/lib/python3.12/site-packages`.

### 5. Output

Everything goes to `output/<experiment_name>/` (the console log is `output/<experiment_name>.log`):

| File | Content |
|---|---|
| `report.txt` | the metrics table below, plus meanings (also the last lines of the log; with `--gt_file` also a GT vs result table) |
| `results.yaml` (+ `.pkl`) | per sensor: `extrinsics` (4x4, relative to camera 0), `K`, `distortion_coeff`, `is_pinhole` (cameras), `intrinsics` (LiDAR), covariances, and the `metrics` entry |
| `data_report.txt` | input-quality check (e.g. tag coverage, how far the initial extrinsics are off); also printed when the solver fails |
| `args.yaml`, `errors.pkl` | settings used, per-iteration errors |
| `verification/lidar_<name>_scans.ply` | all scans of a LiDAR in the reference map frame, coloured by pose; overlay with `reference_pointcloud.ply` in CloudCompare |
| `verification/<camera>/<image>_proj.png` | image with the LiDAR scan of the same pose projected in (coloured by depth) |
| `verification/<camera>/<image>_reproj.png` | detected vs projected tag corners, error x50 |

A run is converged when the log contains `adding LiDARs` (camera stage converged) and ends with `converged`. If the camera stage runs out of its 150 iterations, `report.txt` is still written but the LiDARs stay at their initial values.

### 6. Metrics (`report.txt`)

| Metric | Meaning | Healthy |
|---|---|---|
| reproj [px] | distance between detected tag corners and the same corners projected through the estimated camera model and pose (rmse, mean, median, p95, max) | rmse about `--std_pix` (0.2); large max with small median = single bad detections |
| worst frames | frames with the highest per-frame reprojection RMSE (0-based index in the sorted image files, so 9 = `0010.png`) | check their `_reproj.png` for blur, partial tags, wrong detections |
| point-to-plane [m] | distance of LiDAR points to the local plane of the reference map | close to the sensor noise, well below `--std_lidar` |
| inlier ratio | share of LiDAR points with a map plane within 3x`std_lidar` (mean, worst frame) | about 1.0; low = scans misaligned or outside the map |
| plane-to-plane [m] | for map planes seen by two LiDARs, the difference of their mean plane offsets | about 0: the LiDARs agree with each other |
| sigma t / rot / K | 1-sigma precision from the inverse normal matrix, **relative to camera 0** (camera 0 reads ~0); says how well the data constrains a parameter, not the distance to the truth | small (sub-mm / 0.01 deg here); large = too few views, poor geometry |
| sigma0 | unit-weight standard deviation of the cameras and of the LiDARs | about 1; below 1 means the `std_*` are pessimistic, well above 1 optimistic or model errors |

Caveats:
- With no GT you can only judge consistency: reprojection about `std_pix`, point-to-plane in the mm range, inlier ratio about 1, sigma0 near 1, and clean `_proj.png` / `_reproj.png` images. Bias in the reference tag coordinates or reference cloud is not visible in these numbers.
- Isaac Sim data builds the reference cloud from the same scans, so LiDAR numbers there are optimistic. On real data expect LiDAR sigma0 near 1.
- Example (Ubuntu 24.04, converted Isaac bag, 36 poses, 9 cameras + 8 LiDARs): camera reproj RMSE 0.24 px, LiDAR point-to-plane 2.1 mm, inlier ratio >= 0.996, plane-to-plane 0.3 mm RMSE, sigma0 0.85 / 0.21.

## Usage without docker

We highly recommend using docker. In case it is not needed you can take a look at the Dockerfile to see what needs to be installed. 

Most importantly:
Install the [AprilTag library](https://github.com/AprilRobotics/apriltag/releases/tag/v3.4.2) (tested under 3.4.2).
Install this repository (e.g. `pip install -e .`)

## Experiment report and slides

After running experiments (`output/exp_Box*` with their `output/dataset_Box*`), build the HTML report and the slide deck with one command (host python with `numpy`, `pyyaml`, `Pillow`, `matplotlib`, and `pdftoppm` from poppler-utils):

```bash
python3 scripts/make_report.py output        # output dir is optional, default: output
```

This writes into `output/report/`:

- `index.html`: experiment explorer, metric comparison charts, 3D clouds.
- `report.html`: 8-slide deck; use its "Export PDF" button (or `google-chrome --headless --print-to-pdf=slides.pdf --no-pdf-header-footer report.html`).
- `images/`, `data/`, `lib/`: copied assets, so `output/report/` can be copied on its own to another machine.

To rebuild only the slides: `python3 scripts/make_slides.py output`. The HTML sources are in `templates/report/`; edit them and rerun the command.

## ToDos
- [x] Add scripts for extracting the Apriltag coordinates from the TLS point cloud
- [ ] Add evaluation script and data