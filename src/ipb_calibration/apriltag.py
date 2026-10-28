import numpy as np
import apriltag
import ipdb
from matplotlib import pyplot as plt


def compute_border_threshold(corners):
    mean = np.mean(corners, axis=0, keepdims=True)
    diff = corners - mean
    max_diag = np.max(np.linalg.norm(diff, axis=-1)) * 2
    return max_diag / 2**0.5 / 8


class Apriltags:
    def __init__(self, file, family="tag36h11") -> None:
        self.detector = apriltag.apriltag(family)

        apriltags = np.loadtxt(file, skiprows=1).reshape(-1, 4, 4)
        self.apriltag_coords = apriltags[:, :, 1:]
        # The code expects corner order tl,tr,bl,br. Files listing corners around the tag (tl,tr,br,bl)
        # have corner 3 on the diagonal from corner 1; swap corners 3 and 4 for those.
        c = self.apriltag_coords
        cyclic = np.linalg.norm(c[:, 2]-c[:, 0], axis=-1) > 1.2*np.linalg.norm(c[:, 1]-c[:, 0], axis=-1)
        c[cyclic] = c[cyclic][:, [0, 1, 3, 2]]

        self.apriltag_ids = (apriltags[:, 0, 0]/100).astype("int")
        max_num_tags = self.apriltag_ids.max() + 1  # any family: size the lookup by the ids in the file
        self.tag_id2list_idx = np.full(
            max_num_tags, fill_value=max_num_tags, dtype=np.int64)
        self.tag_id2list_idx[self.apriltag_ids] = np.arange(
            len(self.apriltag_ids))

    def process_detection(self, detection, img):
        if detection["id"] >= len(self.tag_id2list_idx):
            return None, None, None
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
            if detection["margin"] < 10:  # low-margin decodes are false positives (frequent with tag16h5)
                continue
            tag_id, corners, coords = self.process_detection(
                detection, img)
            if tag_id is None:
                continue

            # needed for pip apriltag
            # corners = corners[[0, 1, 3, 2]]

            # needed for apriltag https://github.com/AprilRobotics/apriltag#python
            corners = corners[[1, 0, 2, 3]]

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
