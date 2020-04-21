import numpy as np
import os
import cv2
import pykitti
import json
import matplotlib.pyplot as plt
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
        depth_image = utils.pcl_to_image(velo[:, :3], dataset.calib.T_cam2_velo,
                                         P_cam2, (mask_image.shape[0], mask_image.shape[1]))
        depth_image[:100, :] = depth_image_stereo[:100, :]
        
        # Create structuring element for erosion
        kernel = np.ones((5, 5), np.uint8) 

        class_ids = data['classes']
        for i in range(len(class_ids)):
            mask_i = cv2.erode(np.array(mask_image == i + 1, dtype=np.uint8), kernel)
            mask = np.where(np.logical_and(mask_i, np.logical_not(np.isnan(depth_image))))
            
            u = []
            v = []
            z = []
            for j in range(mask[0].shape[0]):
                u.append(mask[1][j])
                v.append(mask[0][j])
                z.append(depth_image[v[j], u[j]])
            
            #Continue if there are less then 5 points per object
            if len(z)<5:
                continue

            u = np.asarray(u)
            v = np.asarray(v)
            z = np.asarray(z)
            
            #upper limit for z
            upperlim = np.quantile(z, 0.4)

            #cut of all z that are bigger than upperlim
            z = np.delete(z, np.where(z > upperlim))
            u = np.delete(u, np.where(z > upperlim))
            v = np.delete(v, np.where(z > upperlim))

            point_cloud = np.zeros((4, len(z)))
            for j in range(len(z)):
                point_cloud[:, j] = np.array([[z[j] * (u[j] - u_0) / f_x],
                                          [z[j] * (v[j] - v_0) / f_y],
                                          [z[j]],
                                          [1.]], dtype=float).reshape((4,))

            transform = T_w0_w.dot(dataset.poses[frame_id].dot(T_cam0_cam2))
        
            # Ground truth poses are T_w_cam0
            point_cloud = transform.dot(point_cloud)




            min_coord = np.min(point_cloud, axis=1)
            max_coord = np.max(point_cloud, axis=1)
            bbox = np.array([min_coord[0:3], max_coord[0:3]]).reshape((6,))
            # plt.imshow(depth_image)
            # plt.show()
            # exit(0)

            pcl_path = os.path.join(out_path, "landmark_f{}_i{}".format(data['image_id'], i))
            np.save(pcl_path, point_cloud)
            bbox_path = os.path.join(out_path, "bbox_f{}_i{}".format(data['image_id'], i))
            np.save(bbox_path, bbox)
            results.append([frame_id, i, class_ids[i], pcl_path, bbox_path, transform[:3, 3]])

    results.sort()

    with open(os.path.join(out_path, "_results.txt"), 'w') as f:
        for result in results:
            f.write(str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + " " + str(result[3]) + " " +
                    str(result[4]) + " " + str(result[5][0]) + " " + str(result[5][1]) + " " + str(result[5][2]) + "\n")

    print("Finished.")



