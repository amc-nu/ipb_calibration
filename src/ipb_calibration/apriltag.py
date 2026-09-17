import csv
from pathlib import Path

import numpy as np
import apriltag
import ipdb
from matplotlib import pyplot as plt


def compute_border_threshold(corners):
    mean = np.mean(corners, axis=0, keepdims=True)
    diff = corners - mean
    max_diag = np.max(np.linalg.norm(diff, axis=-1)) * 2
    return max_diag / 2**0.5 / 8


_CORNER_COLS = ("top_left", "top_right", "bottom_right", "bottom_left")


class Apriltags:
    def __init__(self, file) -> None:
        self.detector = apriltag.apriltag("tag36h11")
        max_num_tags = 587

        if Path(file).suffix.lower() == ".csv":
            self.apriltag_ids, self.apriltag_coords = self._load_csv(file)
        else:
            apriltags = np.loadtxt(file, skiprows=1).reshape(-1, 4, 4)
            self.apriltag_coords = apriltags[:, :, 1:]
            self.apriltag_ids = (apriltags[:, 0, 0]/100).astype("int")

        self.tag_id2list_idx = np.full(
            max_num_tags, fill_value=max_num_tags, dtype=np.int64)
        self.tag_id2list_idx[self.apriltag_ids] = np.arange(
            len(self.apriltag_ids))

    @staticmethod
    def _load_csv(file):
        """Reads a reference/apriltag_coords.csv-style file (tag_id + the
        surveyed 3D corners, columns top_left/top_right/bottom_right/
        bottom_left). Reordered to top_left, top_right, bottom_left,
        bottom_right to match process_detection's corner convention."""
        ids = []
        coords = []
        with open(file, newline="") as f:
            for row in csv.DictReader(f):
                ids.append(int(row["tag_id"]))
                tl, tr, br, bl = ([float(row[f"{col}_x_3d"]),
                                  float(row[f"{col}_y_3d"]),
                                  float(row[f"{col}_z_3d"])]
                                 for col in _CORNER_COLS)
                coords.append([tl, tr, bl, br])
        return np.array(ids), np.array(coords)

    def process_detection(self, detection, img):
        list_idx = self.tag_id2list_idx[detection["id"]]
        if list_idx >= len(self.apriltag_coords):
            return None, None, None

        return detection["id"], detection["lb-rb-rt-lt"], self.apriltag_coords[list_idx]

    def detect(self, img: np.array, pixel2ray=None):
        """Detect apriltags in Image and 

        Args:
            img (_type_): Image
            pixel2ray (function, optional): function which converts pixel to camera rays.

        Returns:
            list: for each tag dict with: tag_id, corners [4,2], coords [4,3], (rays [4,3]) 
        """
        observations = []
        results = self.detector.detect(img)
        for detection in results:
            tag_id, corners, coords = self.process_detection(
                detection, img)
            if tag_id is None:
                continue

            # needed for pip apriltag
            # corners = corners[[0, 1, 3, 2]]

            # needed for apriltag https://github.com/AprilRobotics/apriltag#python
            corners = corners[[3, 2, 0, 1]]

            border = compute_border_threshold(corners)
            within_image = np.all(corners >= border) and np.all(
                corners <= (np.array(img.T.shape)-1-border))

            out = {"tag_id": tag_id,
                   "corners": corners,
                   "coords": coords}
            if pixel2ray is not None:
                out["rays"] = pixel2ray(corners)

            if within_image:
                observations.append(
                    out)
        return observations
