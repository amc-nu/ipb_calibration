import glob
from os.path import join, splitext
from pathlib import Path

import click
import numpy as np

PCD_TYPE = {"f": "F", "u": "U", "i": "I"}


def write_pcd(points: np.ndarray, out_file: str):
    """Writes a structured numpy array to a binary .pcd file, one field per named dtype entry."""
    points = points.reshape(-1)
    names = points.dtype.names or ("x", "y", "z")
    if points.dtype.names is None:
        points = points.astype([(n, "<f4") for n in names])

    sizes = [points.dtype.fields[n][0].itemsize for n in names]
    types = [PCD_TYPE[points.dtype.fields[n][0].kind] for n in names]

    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        f"FIELDS {' '.join(names)}",
        f"SIZE {' '.join(map(str, sizes))}",
        f"TYPE {' '.join(types)}",
        f"COUNT {' '.join(['1'] * len(names))}",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        "DATA binary",
        "",
    ])
    with open(out_file, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(points).tobytes())


@click.command()
@click.option("--path_data", "-p", required=True, type=str, help="Directory containing per-LiDAR topic folders with .npy scans (same layout as calibration input).")
@click.option("--path_out", "-o", required=True, type=str, help="Directory in which the .pcd files will be stored, mirroring the input folder structure.")
def main(path_data, path_out):
    for npy_file in sorted(glob.glob(join(path_data, "**", "*.npy"), recursive=True)):
        points = np.load(npy_file)
        out_file = join(path_out, splitext(npy_file[len(path_data):].lstrip("/\\"))[0] + ".pcd")
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        write_pcd(points, out_file)
        print(out_file)


if __name__ == "__main__":
    main()
