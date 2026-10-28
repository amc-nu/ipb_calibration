"""Frame subset of an IPB dataset as a new dataset dir (hardlinks, same filesystem): K evenly spaced frames.
usage: python make_subset.py <dataset_dir> <K> <out_dir>   (K must divide the frame count)"""
import os, shutil, sys
from pathlib import Path

src, K, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
files = lambda d: sorted(p for p in d.iterdir() if p.suffix in (".png", ".npy"))
sensors = [d for d in (src / "input").iterdir() if d.is_dir()]
F = len(files(sensors[0]))
assert F % K == 0, f"{K} does not divide {F} frames"
idx = range(0, F, F // K)
(out / "input").mkdir(parents=True)
shutil.copy(src / "input" / "calibration.yaml", out / "input" / "calibration.yaml")
for d in sensors:
    (out / "input" / d.name).mkdir()
    fs = files(d)
    for k, i in enumerate(idx):
        os.link(fs[i], out / "input" / d.name / f"{k:04d}{fs[i].suffix}")
shutil.copytree(src / "reference", out / "reference", copy_function=os.link)
shutil.copy(src / "config.yaml", out / "config.yaml")
print(f"{out.name}: {K} of {F} frames {list(idx)}")
