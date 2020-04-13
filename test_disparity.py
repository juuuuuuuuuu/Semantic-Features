import numpy as np
import cv2
import utils
import pykitti
import matplotlib.pyplot as plt

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
    baseline = dataset.calib.b_rgb

    disparity = cv2.imread("/home/felix/vision_ws/Semantic-Features/content/kitti_dataset/dataset/sequences/04/rawpng/219raw.png",
                           cv2.IMREAD_UNCHANGED)

    P_cam2 = dataset.calib.P_rect_20
    T_cam2_velo = dataset.calib.T_cam2_velo

    rgb_img = np.asarray(first_cam2)
    P_cam2 = np.vstack([P_cam2, [0., 0., 0., 1.]])
    print(P_cam2.shape)
    depth_img = utils.pcl_to_image(velo[:, :3], T_cam2_velo, P_cam2, (rgb_img.shape[0], rgb_img.shape[1]))

    print(depth_img.shape)
    cv2.imwrite('depth.png', depth_img * 1000)

    # Display some of the data
    np.set_printoptions(precision=4, suppress=True)
    print('\nSequence: ' + str(dataset.sequence))
    print('\nFrame range: ' + str(dataset.frames))

    print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
    print('\nSecond ground truth pose:\n' + str(second_pose))

    f, ax = plt.subplots(3, 1, figsize=(15, 5))

    ax[0].imshow(first_cam2)
    ax[0].set_title('Left RGB Image (cam2)')

    ax[1].imshow(depth_img)
    ax[1].set_title('Depth from LIDAR')

    ax[2].imshow(baseline * 721 / disparity / 48)
    ax[2].set_title('Disparity of left RGB Image')

    plt.show()