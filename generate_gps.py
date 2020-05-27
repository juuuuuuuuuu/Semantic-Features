import json
import numpy as np
import os
import pykitti
import glob
from tools import utils

basedir = 'content/kitti_dataset/dataset'
sequence = '04'
dataset = pykitti.odometry(basedir, sequence)
# World is twisted so we need to transform.
T_w0_w = np.array([[ 0., 0., 1., 0.],
                   [-1., 0., 0., 0.],
                   [ 0.,-1., 0., 0.],
                   [ 0., 0., 0., 1.]])
T_cam0_cam2 = np.linalg.inv(dataset.calib.T_cam0_velo).dot(dataset.calib.T_cam2_velo)

ROOT_DIR = os.path.abspath('../')
DATASET_SFM = 'OpenSfM/data/kitti_08/'

image_paths = glob.glob(os.path.join(ROOT_DIR, DATASET_SFM, 'images', '*.png'))
# coordinates of some point in Karlsruhe
long, lat = (49.010136, 8.401784)
gps_data = {}
for image_path in image_paths:
    _, frame_id = os.path.split(image_path)
    frame_id_int = int(frame_id[:-4])
    transform = T_w0_w.dot(dataset.poses[frame_id_int].dot(T_cam0_cam2))
    x, y, z = transform[0:3, 3]
    d_lat, d_long = utils.metric_diff_to_coord_diff(x, y, lat)
    lat = d_lat + lat
    long = long + d_long
    gps_data[frame_id] = {
            "gps": {
            "latitude": lat,
            "longitude": long,
            "altitude": z
             }
    }

filename = os.path.join(ROOT_DIR, DATASET_SFM, 'exif_overrides.json')
with open(filename, 'w') as f:
   json.dump(gps_data, f)