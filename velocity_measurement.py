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


def get_gt_velocities_vehicle(poses):
    linear_velocities = []
    angular_velocities = []
    for i in range(len(poses) - 1):
        linear_velocities.append(
            np.linalg.inv(poses[i][:3, :3]).dot(poses[i + 1][:3, 3] - poses[i][:3, 3]))
        w_ = get_delta_rot(poses[i][:3, :3], poses[i + 1][:3, :3])
        angular_velocities.append(w_)

    return linear_velocities, angular_velocities


def transform_poses(poses):
    T_w0_w = np.array([[0., 0., 1., 0.],
                       [-1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 0., 1.]])
    return [T_w0_w.dot(pose) for pose in poses]


def process_particle(pose, rel_v, rel_w, noise_v, noise_w):
    v_world = np.squeeze(pose[:3, :3].dot((rel_v + noise_v).reshape(3, 1)))
    w_ = rel_w + noise_w
    old_pos = pose[:3, 3]
    old_r = pose[:3, :3]
    new_pos = old_pos + v_world
    new_r = old_r.dot(r.Rotation.from_rotvec(w_).as_matrix())
    out_pose = np.eye(4)
    out_pose[:3, :3] = new_r
    out_pose[:3, 3] = new_pos
    return out_pose


if __name__ == '__main__':
    basedir = 'content/kitti_dataset/dataset'
    sequence = '08'

    dataset = pykitti.odometry(basedir, sequence)

    poses = transform_poses(dataset.poses[:100])

    v_lin, v_ang = get_gt_velocities_vehicle(poses)

    current_pose = poses[0]
    for i in range(len(v_lin)):
        current_pose = process_particle(current_pose, v_lin[i], v_ang[i], np.zeros((3,)), np.zeros((3,)))
        print("=====")
        print(poses[i + 1])
        print(current_pose)
        # print("Step {}:".format(i))
        # print("Linear velocity:")
        # print(v_lin[i])
        # print("Angular velocity:")
        # print(poses[i][:3, :3].dot(v_ang[i].reshape(3, 1)))
        # print(v_ang[i])

