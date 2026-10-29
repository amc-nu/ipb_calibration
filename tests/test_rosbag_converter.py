"""python tests/test_rosbag_converter.py"""
from types import SimpleNamespace as NS

import numpy as np

from ipb_calibration.rosbag_converter import decode_cloud, select_poses, init_pose


def test_decode_xyzir():
    dt = np.dtype({"names": ["x", "y", "z", "i", "r"], "formats": ["<f4"] * 4 + ["<u2"],
                   "offsets": [0, 4, 8, 16, 20], "itemsize": 32})
    a = np.zeros(4, dt)
    a["x"] = [1, np.nan, 3, 4]
    a["y"], a["z"], a["r"] = 2, 3, 7
    f = lambda n, o: NS(name=n, offset=o)
    msg = NS(point_step=32, row_step=128, height=1, width=4, is_bigendian=False, data=np.frombuffer(a.tobytes(), np.uint8),
             fields=[f("x", 0), f("y", 4), f("z", 8), f("intensity", 16), f("ring", 20)])
    xyz = decode_cloud(msg, "PointXYZIR")
    assert xyz.shape == (3, 3) and xyz[0].tolist() == [1, 2, 3]
    try:
        decode_cloud(msg, "PointXYZIRADT")
        assert False
    except ValueError:
        pass


def test_select():
    ref = np.arange(10) * 1.0
    other = np.arange(10) * 1.0 + 0.01
    rows = select_poses(ref, [other], every_n=3, max_sync_dt=0.05)
    assert rows.tolist() == [[0, 0], [3, 3], [6, 6], [9, 9]]
    assert len(select_poses(ref, [other + 0.5], 1, 0.05)) == 0


def test_init_pose():
    T = np.eye(4)
    T[:3, 3] = [1, 0, 0]
    tree = {"cam0": ("base", np.eye(4)), "lidar": ("base", T)}
    assert np.allclose(init_pose(tree, "cam0", "lidar"), [0, 0, 0, 1, 0, 0])


if __name__ == "__main__":
    test_decode_xyzir(); test_select(); test_init_pose(); print("ok")
