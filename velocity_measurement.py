import numpy as np
import pykitti
import scipy.spatial.transform.rotation as r


def get_delta_rot(rot_1, rot_2):
    return r.Rotation.from_matrix(np.dot(rot_1.T, rot_2)).as_rotvec()


def get_gt_velocities(poses):
    linear_velocities = []
    angular_velocities = []
    for i in range(len(poses) - 1):
        linear_velocities.append(poses[i + 1][:3, 3] - poses[i][:3, 3])
        angular_velocities.append(get_delta_rot(poses[i + 1][:3, :3], poses[i][:3, :3]))

    return linear_velocities, angular_velocities


if __name__ == '__main__':
    basedir = 'content/kitti_dataset/dataset'
    sequence = '08'

    dataset = pykitti.odometry(basedir, sequence)

    v_lin, v_ang = get_gt_velocities(dataset.poses[:100])

    for i in range(len(v_lin)):
        print("Step {}:".format(i))
        print("Linear velocity:")
        print(v_lin[i])
        print("Angular velocity:")
        print(v_ang[i])
