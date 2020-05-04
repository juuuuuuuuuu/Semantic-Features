import numpy as np
import cv2
from tools import utils
import pykitti
import matplotlib.pyplot as plt
import pandas


def reprojection_error(depth_1, depth_2):
    errors = []
    error_matr = np.zeros((370,1226))
    l = 0
    i = 0
    k = 0
    for i in range(0, 1225):
        for k in range(0, 369):
            if depth_1[k,i] > 1 and depth_2[k,i] > 1:
                errors.append(np.abs(depth_1[k,i] - depth_2[k,i]))
                error_matr[k,i] = np.abs(depth_1[k,i] - depth_2[k,i])
            l = l+1
    errors = np.array(errors)
    errors.sort()
    image = np.abs(depth_1 - depth_2)
#    image[np.logical_not(indices)] = 0
    #print("Number of errors < 0.5 m: {}".format(np.array(np.where(errors < 0.5)).shape[1]))
    plt.hist(errors, bins=1000)
    plt.show("Histogram of errors")
    #errors_mod = errors[~pandas.isnull(errors)]
    errors_mod = errors[~np.isnan(errors)]
    mse = np.mean(np.square(errors_mod[:5000]))
    mean = np.mean(errors_mod[:5000])
    median = np.median(errors_mod[:5000])
#    print(errors_mod)
    print("Mean error is: {}".format(mean))
    print("Median error is: {}".format(median))
    print("Mean square error is: {}".format(mse))
    print("Root mean square error is: {}".format(np.sqrt(mse)))

    return image, error_matr


if __name__ == '__main__':
    basedir = 'content/kitti_dataset/dataset'

    # Specify the dataset to load
    sequence = '04'

    # Load the data. Optionally, specify the frame range to load.
    # dataset = pykitti.odometry(basedir, sequence)
    dataset = pykitti.odometry(basedir, sequence)

    # Grab some data
    second_pose = dataset.poses[219]
    first_rgb = dataset.get_rgb(219)
    first_cam2 = dataset.get_cam2(219)
    velo = dataset.get_velo(219)
    baseline = dataset.calib.b_gray

#    path = "/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/disparity_2/219raw.png"
    path = "/Users/berkn/Desktop/ETH/Master/Semester_2/3D Vision/dataset/sequences/04/disparity/219.png"
    disparity = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    P_cam2 = dataset.calib.P_rect_00
    T_cam2_velo = dataset.calib.T_cam0_velo

    fx = P_cam2[0, 0]

    rgb_img = np.asarray(first_cam2)
    P_cam2 = np.vstack([P_cam2, [0., 0., 0., 1.]])

    depth_img = utils.pcl_to_image(velo[:, :3], T_cam2_velo, P_cam2, (rgb_img.shape[0], rgb_img.shape[1]))

    depth_from_disp = utils.disparity_to_depth(disparity)
    #print(type(disparity))
    depth_from_disp = cv2.GaussianBlur(depth_from_disp, (3, 3), 0)
    errors, error_matr = reprojection_error(depth_img, depth_from_disp)

    f, ax = plt.subplots(2, 2, figsize=(15, 5))

    ax[0, 0].imshow(first_cam2)
    ax[0, 0].set_title('Left RGB Image (cam2)')

    ax[0, 1].imshow(depth_img)
    ax[0, 1].set_title('Depth from LIDAR')

#    depth_from_disp[np.where(depth_img > 0)] = depth_img[np.where(depth_img > 0)]
    ax[1, 1].imshow(depth_from_disp)
    ax[1, 1].set_title('Depth from disparity')

    ax[1, 0].imshow(error_matr)
    ax[1, 0].set_title('Reprojection error of disparity image.')

    plt.show()
