import copy
from os.path import join
from pathlib import Path

import yaml
import ipdb
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import open3d as o3d
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm
from matplotlib import cm
from ipb_calibration import camera as cd
from ipb_calibration import lidar as li
from ipb_calibration.utils import homogenous, update_pose, np2o3d


def prior_error(T_est, T_prior):
    return np.concatenate([T_est[:3, -1]-T_prior[:3, -1], Rotation.from_matrix(T_prior[:3, :3].T @ T_est[:3, :3]).as_rotvec()])[:, None]


def compute_ray_error(x_img, K, T_calib, T_world, coords, distortion):
    """computes error

        Args:
            p_img (np.array): [N,2]
            T_calib (np.array): [4,4]
            T_world (np.array): [4,4]
            coords (np.array): [N,3]

        returns:
            errors (np.array): [N,2]
        """
    x_i = compute_ray(K, T_calib, T_world, coords, distortion)
    return x_i-x_img


def compute_ray(K, T_calib, T_world, coords, distortion):
    x_c = ((inv(T_calib) @ inv(T_world) @
            homogenous(coords).T)[:3]).T  # [3,N]
    x_n = x_c[:, :2]/x_c[:, -1:]
    x_d = distortion.distort(x_n)
    x_i = x_d @ K[:2, :2] + K[None, :2, -1]
    return x_i


def transform(points, T_cam_map=np.eye(4), T_os_cam=np.eye(4)):
    points = np.concatenate(
        [points, np.ones_like(points[..., :1])], axis=-1)  # [b,n,4]

    points_t = (T_cam_map @ T_os_cam @ points.T).T
    return points_t


def points2o3d(lidar_points, T_lidar_cam, T_cam_map):
    geoms = []
    cms = [cm.get_cmap("autumn"), cm.get_cmap("winter")]
    for lidar, T_calib, cm_ in zip(lidar_points, T_lidar_cam, cms):
        for t, p in enumerate(lidar):
            scan = np2o3d(p)
            scan.transform(T_calib)
            scan.transform(T_cam_map[t])
            scan.paint_uniform_color(cm_(t/len(lidar))[:3])
            # scan.paint_uniform_color(e0*(t == 27))
            geoms.append(scan)
    return geoms


class LCBundleAdjustment:
    def __init__(self,
                 cfg: dict,
                 num_poses,
                 outlier_mult=3,
                 ref_map: o3d.geometry.PointCloud = None,
                 ):
        self.ref_map = ref_map
        self.cfg = cfg

        self.outlier_mult = outlier_mult
        self.num_poses = num_poses
        self.num_cameras = 0
        self.num_lidar = 0
        self.num_lidar_intrinsics = 0
        self.num_tags = 0
        self.final_step = False
        self.converged = True
        #
        self.errors = {}

        self.key_mapping = None

    def add_cameras(
            self,
            p_img: list,
            ray2apriltag_idx: list,
            apriltag_coords: np.ndarray,
            k: np.ndarray,
            T_cam_map: np.ndarray,
            T_cami_cam: np.ndarray,
            std_pix: np.ndarray = 0.2,
            std_apriltags=0.002,
            dist_degree=3,
            division_model=True,
            init_coeff=None,
            cami_is_pinhole=True
    ):
        self.p_img = p_img
        self.ray2apriltag_idx = ray2apriltag_idx
        self.apriltag_coords = apriltag_coords

        self.T_cam_map = T_cam_map
        self.std_pix = std_pix
        self.std_apriltags = std_apriltags

        self.num_cameras = len(T_cami_cam)
        self.num_tags = len(apriltag_coords)

        # Some usefull things
        self.ray_mad = 1
        self.coords_prior = np.copy(apriltag_coords)
        self.k = k  # [xh,yh,c]
        self.K = [np.array([[k_[2], 0, k_[0]],
                           [0, k_[3], k_[1]],
                           [0, 0, 1]])for k_ in self.k]
        for i in range(self.num_cameras):
            self.errors[f"cam_{i}"] = [[] for _ in range(self.num_poses)]

        cami_is_pinhole = cami_is_pinhole if isinstance(
            cami_is_pinhole, list) else self.num_cameras*[cami_is_pinhole]
        self.cameras = [cd.Camera(self.K[i],
                                  T_cami_cam[i],
                                  is_pinhole=cami_is_pinhole[i],
                                  distortion=cd.CVDistortionModel(
            degree=dist_degree,
            division_model=division_model,
            cv2_coeff=init_coeff[i],
        )
        ) for i in range(len(self.K))]
        print(self.cameras)

    @property
    def num_params(self):
        return np.sum(self.param_sizes)

    @property
    def param_sizes(self) -> list:
        p = self.num_poses * [6]

        if self.num_cameras >= 1:
            for cam in self.cameras:
                p += [cam.num_params]

        if self.num_lidar >= 1:
            for lidar in self.lidars:
                p += [lidar.num_params]

        if self.num_cameras >= 1:
            p += [self.num_tags * 3]

        return p

    def param_idx(self, key):
        if self.key_mapping is None:
            csize = np.cumsum([0]+self.param_sizes)

            keys = [f"pose_{i}" for i in range(self.num_poses)]
            if self.num_cameras >= 1:
                for i, _ in enumerate(self.cameras):
                    keys += [f"cam_{i}"]
            if self.num_lidar >= 1:
                for i, _ in enumerate(self.lidars):
                    keys += [f"lidar_{i}"]
            if self.num_cameras >= 1:
                keys += ["tags"]

            self.key_mapping = {}
            for k, start, end in zip(keys, csize[:-1], csize[1:]):
                self.key_mapping[k] = (start, end)

        return self.key_mapping[key]

    def add_lidars(
        self,
        scans: list,
        T_lidar_cam: np.ndarray,
        GM_k=0.1,
        std_lidar=0.025,
        estimate_bias=False,
        estimate_scale=False
    ):
        self.scans = scans

        self.GM_k_lidar = GM_k

        self.std_lidar = std_lidar

        self.num_lidar = len(T_lidar_cam)
        self.points = []
        for lidar in self.scans:
            self.points.append(([np.array(s.points) for s in lidar]))

        self.map_kdtree = KDTree(np.asarray(
            self.ref_map.points), copy_data=True)

        self.map_points = np.asarray(self.ref_map.points)
        if not self.ref_map.has_normals():
            print("Estimate normals... may take some time")
            self.ref_map.estimate_normals()
        self.map_normals = np.asarray(self.ref_map.normals)

        for i in range(self.num_lidar):
            self.errors[f"lidar_{i}"] = [[] for _ in range(self.num_poses)]

        # Estimate scale and range offset
        self.lidars = [li.Lidar(T_lidar_cam[i],
                                li.LinearLidarIntrinsics(
            estimate_bias=estimate_bias,
            estimate_scale=estimate_scale)
        ) for i in range(self.num_lidar)]

    def visualize(self, out_dir=None, tag=""):
        """Headless alignment evidence: writes the reference map plus every
        transformed LiDAR scan (colored by timestamp) as one merged .pcd
        file, instead of opening an interactive OpenGL window. Runs fully
        on CPU, no display/GPU required.

        ponytail: camera ray/frame geometry (rays2o3d) is dropped here —
        it mixes point clouds, line sets and meshes, which can't be merged
        into one file as simply as point clouds can. Add a separate
        per-camera .ply export if that evidence is needed too.
        """
        if out_dir is None or self.num_lidar == 0:
            return

        combined = o3d.geometry.PointCloud()
        ref = copy.deepcopy(self.ref_map)
        ref.paint_uniform_color([0.5, 0.5, 0.5])
        combined += ref

        cm_ = cm.get_cmap("viridis")
        for scans, lidar in zip(self.points, self.lidars):
            for t, scan in enumerate(scans):
                pt = np2o3d(lidar.scan2world(scan, self.T_cam_map[t]))
                pt.paint_uniform_color(cm_(t/len(scans))[:3])
                combined += pt

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_file = join(out_dir, f"lidar_alignment_{tag}.pcd")
        o3d.io.write_point_cloud(out_file, combined)
        print(f"Wrote alignment evidence: {out_file}")

    def _add_ray_obs(self, N, g):
        # Add Image observations
        s0_sq = 0
        num_observations = 0
        num_invalids = 0
        errors = []
        errors_all = []

        for c, (camera, p_img_c) in enumerate(zip(self.cameras, self.p_img)):  # for each camera
            for t, (T_world, p_img) in enumerate(zip(self.T_cam_map, p_img_c)):  # for each timestamp
                p_img = self.p_img[c][t]
                num_rays = len(p_img)
                if num_rays > 0:
                    ap_idx = self.ray2apriltag_idx[c][t]

                    coords = self.apriltag_coords[ap_idx]

                    # Compute Error
                    error = camera.project(coords, T_world) - p_img
                    error = error.reshape([num_rays*2, 1])
                    errors_all.append(error)

                    # Compute Jacobians
                    J_pose, J_cam, J_tags = camera.jacobians(T_world,
                                                             coords,
                                                             self.ray2apriltag_idx[c][t],
                                                             self.num_tags)
                    J = np.zeros([num_rays*2, self.num_params])
                    idx_s, idx_e = self.param_idx(f"pose_{t}")
                    J[:, idx_s:idx_e] = J_pose
                    idx_s, idx_e = self.param_idx(f"cam_{c}")
                    J[:, idx_s:idx_e] = J_cam
                    idx_s, idx_e = self.param_idx("tags")
                    J[:, idx_s:idx_e] = J_tags

                    # Get Covariances of the observations
                    wweight = np.ones_like(error)/self.std_pix**2

                    # Update Normal Equations
                    N += J.T @ (wweight*J)
                    g -= J.T@(wweight * error)
                    num_observations += num_rays*2
                    s0_sq += error.T @ (wweight * error)

                    errors.append(error)
                    self.errors[f"cam_{c}"][t] = error
        errors = np.concatenate(errors)
        errors_all = np.concatenate(errors_all)
        self.ray_mad = np.median(np.abs(errors_all)) * 1.4826
        self.cam_sigma0 = float(np.squeeze((s0_sq/(num_observations-self.num_params))**0.5))
        print("Cameras: sigma0", self.cam_sigma0)
        print(
            f"Camera: Outliers: {num_invalids/(num_observations)*100:.2f}%")
        return s0_sq.sum()

    def _add_pose_prior(self, N, g):
        s0_pr_sq = 0
        error = prior_error(
            self.cameras[0].T_cam_ref, np.eye(4))  # xyz rx ry rz

        idx_s, _ = self.param_idx(f"cam_0")
        idx_e = idx_s+6

        wweight = np.eye(6)*1e6

        N[idx_s:idx_e, idx_s:idx_e] += wweight
        g[idx_s:idx_e] -= (wweight @ error)
        s0_pr_sq += (error.T @ wweight @ error)
        return s0_pr_sq

    def _add_lidar_obs(self, N, g):
        # Add lidars
        s0_points_sq = 0
        num_observations = 0

        for l, (scans, lidar) in enumerate(zip(self.points, self.lidars)):  # for each scan
            for t, (points, T_world) in enumerate(zip(scans, self.T_cam_map)):

                points_w = lidar.scan2world(points, T_world)
                threshold = self.std_lidar * \
                    self.outlier_mult if self.final_step else self.GM_k_lidar * self.outlier_mult

                d, indices = self.map_kdtree.query(
                    points_w[..., :3], workers=5, distance_upper_bound=self.GM_k_lidar * self.outlier_mult)
                valids = (d < threshold)
                indices[~valids] -= 1

                points_corr = self.map_points[indices]
                normals_corr = self.map_normals[indices]

                error = np.einsum(
                    "nd,nd->n", normals_corr[valids], (points_w[valids, :3] - points_corr[valids]))[:, None]

                # Jacobians of Calibration params of lidar
                num_points = len(points_w)
                J_pose, J_lidar = lidar.jacobians(T_world,
                                                  points,
                                                  normals_corr)
                J = np.zeros([num_points, self.num_params])
                idx_s, idx_e = self.param_idx(f"pose_{t}")
                J[:, idx_s:idx_e] = J_pose
                idx_s, idx_e = self.param_idx(f"lidar_{l}")
                J[:, idx_s:idx_e] = J_lidar
                J = J[valids]

                wweight = 1/(self.std_lidar**2) if self.final_step else self.GM_k_lidar**2 / \
                    (self.GM_k_lidar**2+error**2)**2

                N += J.T @ (wweight*J)
                g -= (J*wweight).T @ error
                s0_points_sq += error.T @ (wweight * error)
                num_observations += num_points

                self.errors[f"lidar_{l}"][t] = error
        self.lidar_sigma0 = float(np.squeeze((s0_points_sq /
              (num_observations-self.num_params))**0.5))
        print("LiDAR: sigma0", self.lidar_sigma0)
        return s0_points_sq.sum()

    def _add_apriltag_priors(self, N, g):
        error = (self.apriltag_coords-self.coords_prior).flatten()
        print(
            f"Apriltag: max diff: {np.abs(error).max():.4f}, mean: {np.mean(error):.4f}")
        g[-3*self.num_tags:, 0] -= error/self.std_apriltags**2
        N[-3*self.num_tags:, -3 *
          self.num_tags:] += np.eye(3*self.num_tags)/self.std_apriltags**2
        return np.sum((error/self.std_apriltags)**2)

    def _update_poses(self, d_w):
        d_t, d_r = [], []
        for t, (dt_wi, dr_wi) in enumerate(zip(*np.split(d_w.reshape([-1, 6]), [3], axis=-1))):
            self.T_cam_map[t] = update_pose(
                self.T_cam_map[t], dt=dt_wi, dr=dr_wi)
            dr, dt = np.linalg.norm(dr_wi), np.linalg.norm(dt_wi)
            self.converged = self.converged and (
                np.abs(dr) < 1e-5) and (np.abs(dt) < 1e-5)
            d_t.append(dt)
            d_r.append(dr)

        d_t = np.array(d_t)
        d_r = np.array(d_r)
        change = d_t + d_r
        i = np.argmax(change)
        print(f"Pose: {i}: change: {d_r[i]/np.pi*180:0.4}deg, {d_t[i]:0.4}m")

    def _update_camera_extrinsics(self, d_c):
        for c, dparams in enumerate(d_c.reshape([self.num_cameras, -1])):
            dr, dt = self.cameras[c].update_params(dparams)

            self.converged = self.converged and (
                np.abs(dr) < 1e-5) and (np.abs(dt) < 1e-5)
            print(f"Camera: {c}: change: {dr/np.pi*180:0.4}deg, {dt:0.4}m")

    def _update_lidar_extrinsics(self, d_l):
        for l, dparams in enumerate(d_l.reshape([self.num_lidar, -1])):
            dr, dt = self.lidars[l].update_params(dparams)
            self.converged = self.converged and (
                np.abs(dr) < 1e-5) and (np.abs(dt) < 1e-5)
            print(f"LiDAR: {l}: change: {dr/np.pi*180:0.4}deg, {dt:0.4}m")

    def _split_cov(self, cov, sizes):
        sizes = np.concatenate([[0], sizes]).reshape([-1, 2])
        covs_all = []
        for s in sizes:
            covs_i = []
            for t in range(self.num_poses):
                i = t * 3 + s[0]
                j = t * 3 + s[1]
                ind = [i, i+1, i+2, j, j+1, j+2,]
                covs_i.append(cov[ind][:, ind])
            covs_all.append(np.stack(covs_i, axis=0))
        return covs_all

    def result2dict(self, cov):
        out = {}
        poses = self.T_cam_map
        pose_cov = np.zeros([self.num_poses, 6, 6])
        for key in self.key_mapping:
            ind = self.param_idx(key)
            ind = list(range(ind[0], ind[1]))
            if "pose" in key:
                idx = int(key.split("_")[1])
                pose_cov[idx] = cov[ind][:, ind]
            if "cam" in key:
                idx = int(key.split("_")[1])
                out_key = self.cfg["image_topics"][idx].replace("/", "")
                out[out_key] = self.cameras[idx].get_param_dict(
                    cov=cov[ind][:, ind])
            if "lidar" in key:
                idx = int(key.split("_")[1])
                out_key = self.cfg["point_cloud_topics"][idx].replace("/", "")
                out[out_key] = self.lidars[idx].get_param_dict(
                    cov=cov[ind][:, ind])
        out["frameposes"] = {"poses": poses, "cov": pose_cov}
        return out

    def optimize(self, num_iter=50, visualize=False, out_dir=None):
        self.stats = {"iterations": []}
        if visualize:
            self.visualize(out_dir=out_dir, tag="initial")

        for i in tqdm(range(num_iter)):
            # Param Order: t_w, r_w, t_c, r_c, t_l, r_l, p
            N = np.zeros([self.num_params, self.num_params])
            g = np.zeros([self.num_params, 1])

            # Add observations to normal equations
            s0 = 0
            if self.num_cameras > 0:
                s0 += self._add_pose_prior(N, g)
                s0 += self._add_apriltag_priors(N, g)
                s0 += self._add_ray_obs(N, g)
            if self.num_lidar > 0:
                s0 += self._add_lidar_obs(N, g)
            s0 = s0.squeeze()
            print(f"squared Error: {s0:.5}")
            self.stats["iterations"].append({
                "iter": i,
                "squared_error": float(s0),
                "camera_sigma0": getattr(self, "cam_sigma0", None),
                "lidar_sigma0": getattr(self, "lidar_sigma0", None),
                "robust_kernel": not self.final_step,
            })

            # Solve
            cov = np.linalg.inv(N)
            dx = cov @ g

            # Update Params
            sizes_c = np.cumsum([self.num_poses*6,
                                 self.num_cameras *
                                 self.cameras[0].num_params,
                                 self.num_lidar * self.lidars[0].num_params])
            d_w, d_c, d_l, d_p = np.split(
                dx.squeeze(), sizes_c)

            self.converged = True
            self._update_poses(d_w)
            self._update_camera_extrinsics(d_c)
            self._update_lidar_extrinsics(d_l)
            self.apriltag_coords += d_p.reshape(self.apriltag_coords.shape)

            # Convergence
            print('With robust Kernel:', not self.final_step)
            if self.converged:
                # break
                if self.final_step:
                    print("converged")
                    break
                print("!!! without robust kernel", i)
                self.final_step = True
        self.stats["num_iterations_run"] = i + 1
        self.stats["converged"] = bool(self.converged and self.final_step)
        if visualize:
            self.visualize(out_dir=out_dir, tag="final")
        print(self.cameras)
        print(self.lidars)
        return self.result2dict(cov), self.errors


def read_calibration_yaml_file(filename: str):
    """
    Read a calibration written as yaml file and initialize corresponding
    laser and camera objects

    Args:
      filename: 
        The (complete) filename of the yaml file
    Returns:
      A tuple (cams, lasers, cam_names, laser_names) with
        cams: a dict of camera objects
        lasers: a dict of laser objects
    """
    with open(filename) as f:
        calib = yaml.load(f, Loader=yaml.SafeLoader)

    cam_names = set()
    laser_names = set()
    for key in calib.keys():
        if key.startswith('cam'):
            cam_names.add(key)
        if key.startswith('lidar') or key.startswith('os') or key.startswith('scan'):
            laser_names.add(key)
    cams, lasers = init_calibration_objects(calib, cam_names, laser_names)

    if "frameposes" in calib:
        poses = np.array(calib["frameposes"]["poses"])
    else:
        poses = None
    return cams, lasers, poses


def init_calibration_objects(calib, cam_names: set, laser_names: set):
    """ 
    For all cameras and lasers initialize objects for calculating.

    Args:
      calib: 
        A dict with all the calibration
      cam_names:
        a set of all the camera names (in the calibration dict)
      laser_names:
        a set of all the laser names (in the calibration dict)

    Returns:
      A tuple (cams, lasers) with
      cams: a dict of camera objects
      lasers: a dict of laser objects
    """
    cams = dict()
    lasers = dict()
    for n in cam_names:
        cams[n] = cd.Camera.fromdict(calib[n])
    for n in laser_names:
        lasers[n] = li.Lidar.fromdict(calib[n])

    return (cams, lasers)
