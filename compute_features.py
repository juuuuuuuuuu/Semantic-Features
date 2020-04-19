import numpy as np
import os
import cv2
import pykitti
import json
import matplotlib.pyplot as plt
import imageio
from tools import utils

if __name__ == '__main__':
    basedir = 'content/kitti_dataset/dataset'
    sequence = '04'

    dataset = pykitti.odometry(basedir, sequence)
    baseline = dataset.calib.b_rgb
    P_cam2 = dataset.calib.P_rect_20
    f_x = P_cam2[0, 0]
    f_y = P_cam2[1, 1]
    u_0 = P_cam2[0, 2]
    v_0 = P_cam2[1, 2]

    # World is twisted so we need to transform.
    T_w0_w = np.array([[ 0., 0., 1., 0.],
                       [-1., 0., 0., 0.],
                       [ 0.,-1., 0., 0.],
                       [ 0., 0., 0., 1.]])
    # The poses are saved as T_w_cam0, for this reason we need a transform between cam0 and cam2.
    T_cam0_cam2 = np.linalg.inv(dataset.calib.T_cam0_velo).dot(dataset.calib.T_cam2_velo)
    P_cam2 = np.vstack([P_cam2, np.array([0, 0, 0, 1])])

    frame_count = len(dataset.poses)

    with open("results.json") as json_file:
        all_data = json.load(json_file)

    path = "content/kitti_dataset/dataset/sequences/04"
    out_path = "results"

    results = []

    for data in all_data['results']:
        frame_id = int(data['image_id'])
        print("Processing frame " + data['image_id'] + '.')

        path_to_mask = os.path.join(path, "instances_2", "L{}.png".format(data['image_id']))
        path_to_disp = os.path.join(path, "disparity_2", "{}raw.png".format(frame_id))

        mask_image = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
        disp_image = cv2.imread(path_to_disp, cv2.IMREAD_UNCHANGED)

        depth_image_stereo = utils.disparity_to_depth(disp_image, f_x, baseline)
        depth_image_stereo = cv2.GaussianBlur(depth_image_stereo, (3, 3), 0)

        velo = dataset.get_velo(frame_id)
        depth_image = utils.pcl_to_image(velo[:, :3], dataset.calib.T_cam0_velo,
                                         P_cam2, (mask_image.shape[0], mask_image.shape[1]))
        depth_image[:100, :] = depth_image_stereo[:100, :]

        class_ids = data['classes']
        for i in range(len(class_ids)):
            mask = np.where(np.logical_and(mask_image == i + 1, np.logical_not(np.isnan(depth_image))))
            point_cloud = np.zeros((4, mask[0].shape[0]))
            for j in range(mask[0].shape[0]):
                u = mask[1][j]
                v = mask[0][j]
                z = depth_image[v, u]

                point_cloud[:, j] = np.array([[z * (u - u_0) / f_x],
                                              [z * (v - v_0) / f_y],
                                              [z],
                                              [1.]], dtype=float).reshape((4,))

            transform = T_w0_w.dot(dataset.poses[frame_id].dot(T_cam0_cam2))
            # Ground truth poses are T_w_cam0
            point_cloud = transform.dot(point_cloud)

            # plt.imshow(depth_image)
            # plt.show()
            # exit(0)

            pcl_path = os.path.join(out_path, "landmark_f{}_i{}".format(data['image_id'], i))
            np.save(pcl_path, point_cloud)

            results.append([frame_id, i, class_ids[i], pcl_path, transform[:3, 3]])

    results.sort()

    with open(os.path.join(out_path, "_results.txt"), 'w') as f:
        for result in results:
            f.write(str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + " " + str(result[3]) + " " +
                    str(result[4][0]) + " " + str(result[4][1]) + " " + str(result[4][2]) + "\n")

    print("Finished.")



