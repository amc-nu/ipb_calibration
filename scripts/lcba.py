import collections
import pickle
from ipb_calibration import lc_bundle_adjustment as ba
from pathlib import Path
import yaml
import numpy as np
import click
from os.path import join
from ipb_calibration import ipb_preprocessing as pp


def convert_dict(d, u=None):
    u = d if u is None else u
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = convert_dict(d.get(k, {}), v)
        else:
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
    return d


@click.command()
@click.option("--apriltag_file", "-a", type=str, help="File with the surveyed apriltag coordinates: either the legacy .txt (tag id + 3d corner, 4 rows/tag) or a reference-style .csv (tag_id + top_left/top_right/bottom_right/bottom_left_{x,y,z}_3d columns). Ignored for an Isaac Sim capture (path_data containing config.yaml): apriltag ground truth is read instead from that directory's gt/apriltag_<frame_id>.csv files.")
@click.option("--path_data", "-p", type=str, help="Path to the data directory. Either an ipb calibration.yaml-style directory, or an Isaac Sim calibration_room.py capture directory (contains config.yaml, camera/, lidar/, gt/).")
@click.option("--map-path", "-m", "map_path", type=str, help="Reference point cloud map: a single .ply/.pcd file, or a directory of .pcd files to concatenate into one map.")
@click.option("--path_out", "-o", type=str, help="Directory in which the results will be stored.")
@click.option("--std_pix", "-sp", default=0.2, help="Standard deviation of the apriltag detection in the image [pix]. (default=0.2)")
@click.option("--std_apriltags", "-sa", default=0.002, help="Standard deviation of the 3D coordinates of the Apriltags.(default=0.002)")
@click.option("--std_lidar", "-sl", default=0.01, help="Standard deviation of the LiDAR points. (default=0.01)")
@click.option("--max_scanpoints", "-mp", default=2500, type=int, help="Maximal number of used lidar points per scan. Give -1 to use all points. (default=2500)")
@click.option("--bias/--no-bias", default=False, help="Flag if one wants to estimate a bias for each LiDAR. (default=False)")
@click.option("--scale/--no-scale", default=False, help="Flag if one wants to estimate a scale for each LiDAR. (default=False)")
@click.option("--division_model/--no-division_model", default=False, help="Flag if to estimate the non-linear distortion with the brownsche distortion model or the division model. (default=False)")
@click.option("--dist_degree", default=3, help="Polynomial degree of the non-linear distortion model. (default=3)")
@click.option("--experiment_name", "-e", default="dev", help="Will be extended to the out_path to enable different experiments without overriding. (default=dev)")
@click.option("--visualize/--no-visualize", default=False, help="Flag if the initial guess and the final optimization should be visualized. (default=False)")
def main(apriltag_file,
         path_data,
         map_path,
         path_out,
         std_pix,
         std_apriltags,
         std_lidar,
         max_scanpoints,
         dist_degree,
         bias,
         scale,
         division_model,
         experiment_name,
         visualize):
    config = locals()
    print(30*"-")
    print(10*"-", experiment_name, 10*"-")
    print(30*"-")
    pcd_map = pp.load_reference_map(map_path)

    isaac_format = (Path(path_data) / "config.yaml").exists()

    if isaac_format:
        print("Detected Isaac Sim capture format (config.yaml found)")
        (imgpixel, indices, coords, observations, cam_is_pinhole,
         init_k, init_coeff, img_sizes, camera_names) = pp.parse_isaac_camera_data(path_data)
        lidar_points, T_os_cam, lidar_names = pp.parse_isaac_lidar_data(
            path_data, max_scanpoints)
        cfg = {"image_topics": camera_names, "point_cloud_topics": lidar_names}
    else:
        if not apriltag_file:
            raise click.UsageError(
                "--apriltag_file is required unless path_data is an Isaac Sim capture (must contain config.yaml)")
        imgpixel, indices, coords, observations, T_cami_cam_yaml, img_sizes, cam_is_pinhole = pp.parse_camera_data(
            apriltag_file, path_data)
        init_k, init_coeff = pp.estimate_initial_k(
            observations, cam_is_pinhole, max_num_images=10, image_size=img_sizes)
        # Initial guess for Lidar
        lidar_points, T_os_cam, cfg = pp.parse_lidar_data(path_data, max_scanpoints)

    init_K = [np.array([[k_[2], 0, k_[0]],
                        [0, k_[3], k_[1]],
                        [0, 0, 1]]) for k_ in init_k]
    print(init_k)
    print(init_coeff)

    T_cam_map, T_cami_cam = pp.estimate_initial_guess(
        observations, cam_is_pinhole, init_K=init_K, dist_coeff=init_coeff)

    # Execute BA
    lcba = ba.LCBundleAdjustment(ref_map=pcd_map,
                                 num_poses=len(T_cam_map),
                                 outlier_mult=3,
                                 cfg=cfg)
    lcba.add_cameras(p_img=imgpixel,
                     std_pix=std_pix,
                     ray2apriltag_idx=indices,
                     apriltag_coords=coords,
                     T_cam_map=T_cam_map,
                     T_cami_cam=T_cami_cam,
                     k=init_k,
                     std_apriltags=std_apriltags,
                     dist_degree=dist_degree,
                     division_model=division_model,
                     init_coeff=init_coeff,
                     cami_is_pinhole=cam_is_pinhole)

    lcba.add_lidars(scans=lidar_points,
                    T_lidar_cam=T_os_cam,
                    GM_k=0.1,
                    std_lidar=std_lidar,
                    estimate_bias=bias,
                    estimate_scale=scale)

    path_out = Path(join(path_out, experiment_name))
    path_out.mkdir(exist_ok=True, parents=True)

    # Headless: --visualize writes alignment evidence (.pcd files) to
    # path_out instead of opening an interactive OpenGL window; runs
    # entirely on CPU, no display/GPU required.
    results, errors = lcba.optimize(
        num_iter=100, visualize=visualize, out_dir=str(path_out))

    with open(join(path_out, "results.yaml.pkl"), "wb") as f:
        pickle.dump(results, f)
    with open(join(path_out, "results.yaml"), "w") as outfile:
        yaml.safe_dump(convert_dict(results), outfile,
                       default_flow_style=False)
    with open(join(path_out, "errors.pkl"), "wb") as f:
        pickle.dump(errors, f)
    with open(join(path_out, "args.yaml"), "w") as outfile:
        yaml.safe_dump(config, outfile)

    final = lcba.stats["iterations"][-1] if lcba.stats["iterations"] else {}
    summary = {
        "converged": lcba.stats.get("converged"),
        "num_iterations_run": lcba.stats.get("num_iterations_run"),
        "final_squared_error": final.get("squared_error"),
        "final_camera_sigma0_pix": final.get("camera_sigma0"),
        "final_lidar_sigma0_m": final.get("lidar_sigma0"),
        "iterations": lcba.stats["iterations"],
    }
    with open(join(path_out, "summary.yaml"), "w") as outfile:
        yaml.safe_dump(summary, outfile, default_flow_style=False)
    print(30*"-")
    print("Summary:", {k: v for k, v in summary.items() if k != "iterations"})
    print(30*"-")


if __name__ == "__main__":
    main()
